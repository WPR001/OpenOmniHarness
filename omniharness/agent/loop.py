"""The agent loop. Each question runs as a generator that yields generation requests and receives outputs. `run` keeps
a fixed number of questions in flight, each at its own pace: a turn is queued in the engine as soon as the question's
tools are done (vLLM batches whatever is queued), and a finished question is replaced by the next one at once.

Per turn the model thinks, then calls tools (bash, view, watch, or any tool in cfg.tools) or answer. The loop ends on
answer, after two replies without a tool call (the text is then read for a letter), or when the step or time budget
runs out (one forced answer turn follows).
"""
import itertools
import os
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .. import rules
from ..backends import Batched, Request
from ..media.video import probe
from ..parsing import extract_letter, mmss, split_output
from ..sandbox import Sandbox
from ..tools import ToolBox
from . import prompts

MAX_CALLS_PER_TURN = 4


class Agent:
    def __init__(self, backend, cfg, workers=16):
        self.backend = backend
        self.cfg = cfg
        self.workers = workers

    # ------------------------------------------------------------------ driver
    def run(self, samples, concurrency=16, on_result=None):
        """Answer the samples with up to `concurrency` questions in flight. Each finished record goes to on_result at
        once; the records are also returned in sample order."""
        engine = self.backend if hasattr(self.backend, 'submit') else Batched(self.backend)
        samples = list(samples)
        results, flows = [None] * len(samples), {}
        queued, busy = {}, set()          # request key -> question; futures of questions running their tools
        order, keys = iter(range(len(samples))), itertools.count()

        def advance(k, value):
            try:
                return k, flows[k].send(value), None
            except StopIteration as stop:
                return k, None, stop.value

        with ThreadPoolExecutor(max(1, self.workers)) as pool:
            def start():
                k = next(order, None)
                if k is not None:
                    flows[k] = self._safe(samples[k])
                    busy.add(pool.submit(advance, k, None))

            for _ in range(max(1, concurrency)):
                start()
            while busy or queued:
                for future in [f for f in busy if f.done()]:
                    busy.discard(future)
                    k, request, result = future.result()
                    if request is None:        # finished: hand it over and take the next question
                        results[k] = result
                        del flows[k]
                        if on_result:
                            on_result(result)
                        start()
                    else:
                        key = f'q{k}t{next(keys)}'
                        queued[key] = k
                        engine.submit(key, request)
                for key, out in (engine.poll() if queued else []):
                    busy.add(pool.submit(advance, queued.pop(key), out))
                if busy and not queued:
                    wait(busy, return_when=FIRST_COMPLETED)
        return results

    def _safe(self, sample):
        holder = {}
        flow = self._flow(sample, holder)
        try:
            reply = None
            while True:
                reply = yield flow.send(reply)
        except StopIteration as stop:
            return stop.value
        except Exception as e:
            record = holder.get('record') or {'uid': sample.uid, 'answer': sample.answer, 'meta': sample.meta}
            record.update(pred=record.get('pred'), stop='error', error=f'{type(e).__name__}: {e}',
                          traceback=traceback.format_exc(limit=6))
            return record

    # ------------------------------------------------------------------ one question
    def _flow(self, sample, holder):
        cfg = self.cfg
        t_start = time.time()
        sandbox = Sandbox(sample, cfg)
        record = holder['record'] = {'uid': sample.uid, 'answer': sample.answer, 'meta': sample.meta, 'pred': None,
                                     'stop': None, 'turns': 0, 'trace': [], 'prompt_tokens': [], 'errors': []}
        try:
            tools = ToolBox(sandbox, cfg)
            duration, size, transcript = first_message_facts(sandbox)
            system = prompts.system_prompt(cfg, tools.offered)
            question = prompts.QUESTION.format(question=sample.question, options='\n'.join(sample.options),
                                               duration=duration, size=size, transcript=transcript)
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': question}]
            letters = sample.letters or list('ABCD')
            idle = 0
            forced = 0
            while True:
                over = (record['turns'] >= cfg.max_turns or time.time() - t_start > cfg.max_seconds)
                if over and not forced:
                    messages.append({'role': 'user', 'content': prompts.FORCE})
                if over:
                    forced += 1
                out = yield Request(messages, cfg.answer_tokens if over else cfg.turn_tokens)
                record['turns'] += 1
                record['prompt_tokens'].append(out.prompt_tokens)
                if out.error:
                    record['errors'].append(out.error)
                    record['stop'] = 'backend_error'
                    break
                thinking, content, calls = split_output(out.text)
                calls = calls[:MAX_CALLS_PER_TURN]
                messages.append({'role': 'assistant', 'content': content, 'reasoning_content': thinking,
                                 'tool_calls': [{'type': 'function', 'function': c} for c in calls] or None})
                step = {'turn': record['turns'], 'thinking_chars': len(thinking), 'text': content[:600], 'calls': []}
                record['trace'].append(step)
                answer = next((c for c in calls if c['name'] == 'answer'), None)
                if answer is not None and (over or not rules.refuse_call(cfg, 'answer', answer['arguments'], tools)):
                    letter = extract_letter(str(answer['arguments'].get('answer', '')), letters, sample.options)
                    record.update(pred=letter, reason=str(answer['arguments'].get('reason', ''))[:1000],
                                  stop='forced_answer' if over else 'answer')
                    step['calls'].append({'tool': 'answer', 'args': answer['arguments']})
                    break
                if not calls:
                    idle += 1
                    if idle >= 2 or forced:
                        record.update(pred=extract_letter(content or thinking, letters, sample.options),
                                      stop='text_after_force' if forced else 'no_tool_call')
                        break
                    messages.append({'role': 'user', 'content': prompts.REMIND})
                    continue
                idle = 0
                if forced >= 2:   # the model keeps working after being told to answer
                    record.update(pred=extract_letter(content or thinking, letters, sample.options), stop='ignored_force')
                    break
                for call in calls:
                    result, info = tools.run(call['name'], call['arguments'])
                    step['calls'].append(info)
                    messages.append({'role': 'tool', 'content': result})
                if out.prompt_tokens > cfg.context_limit:
                    self._drop_old_media(messages)
            record['tool_counts'] = {}
            for s in record['trace']:
                for c in s['calls']:
                    record['tool_counts'][c['tool']] = record['tool_counts'].get(c['tool'], 0) + 1
            record['images_viewed'] = tools.images
            record['denied'] = [x for x in sandbox.log if 'denied' in x]
            record['refused'] = tools.refused
            record['commands'] = len(sandbox.log)
        finally:
            sandbox.close()
        record['correct'] = bool(sample.answer) and record['pred'] == sample.answer
        record['seconds'] = round(time.time() - t_start, 1)
        return record

    @staticmethod
    def _drop_old_media(messages, keep_last=4):
        """Replace media in all but the last few tool messages with a short note (bounds the context)."""
        tool_idx = [i for i, m in enumerate(messages) if m['role'] == 'tool' and isinstance(m['content'], list)]
        for i in tool_idx[:-keep_last]:
            if any(c['type'] in ('image', 'video') for c in messages[i]['content']):
                messages[i] = {'role': 'tool', 'content': [c if c['type'] == 'text' else {'type': 'text', 'text': prompts.DROPPED}
                                                           for c in messages[i]['content']]}


def first_message_facts(sandbox):
    """Duration, size and transcript summary for the first user message."""
    from ..media.transcript import Transcript
    info = probe(sandbox.video)
    size = f'{info.width}x{info.height}, {info.fps:.0f} fps'
    if os.path.exists(sandbox.asr_json):
        tr = Transcript(sandbox.asr_json)
        transcript = ('none (the video has no audio track)' if not tr.d['has_audio'] else
                      f'{len(tr.lines)} lines, language {tr.d["language"] or "none detected"}')
    else:
        transcript = 'not available'
    return mmss(info.duration), size, transcript
