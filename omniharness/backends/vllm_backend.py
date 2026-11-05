"""vLLM backend: Qwen3.8-27B in the offline engine. The agent queues each question's turn on its own (submit) and steps
the engine (poll), so a question goes on with its tools as soon as its own turn is generated; the engine batches
whatever is queued. The prefix cache reuses each conversation's history across its turns."""
import os

from . import Output
from .render import Renderer


class VLLMBackend:
    name = 'vllm'

    def __init__(self, model_path, cfg, tools, tensor_parallel_size=1, gpu_memory_utilization=0.92, max_model_len=131072,
                 max_num_seqs=16, enable_prefix_caching=True, enforce_eager=False, seed=0):
        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
        # FlashInfer's sampler is JIT-built at warm-up and needs ninja + nvcc, which these envs do not have.
        os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
        from transformers import AutoProcessor
        from vllm import LLM
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.renderer = Renderer(self.processor, cfg, tools)
        self.cfg = cfg
        self.max_model_len = max_model_len
        limits = {'image': {'count': cfg.max_images + 4, 'width': 1920, 'height': 1080},
                  'video': {'count': max(cfg.max_watch, 1), 'num_frames': 32, 'width': 640, 'height': 360}}
        self.llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size,
                       gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                       max_num_seqs=max_num_seqs, enable_prefix_caching=enable_prefix_caching,
                       enforce_eager=enforce_eager, seed=seed, limit_mm_per_prompt=limits)
        self._ready = []   # requests that failed at submit, handed back by the next poll

    def build(self, messages, tools=None):
        text, images, videos = self.renderer.render(messages, tools)
        data = {}
        if images:
            data['image'] = images
        if videos:
            data['video'] = videos
        return {'prompt': text, 'multi_modal_data': data}

    def _params(self, request):
        from vllm import SamplingParams
        return SamplingParams(temperature=self.cfg.temperature, top_p=self.cfg.top_p, top_k=self.cfg.top_k,
                              max_tokens=request.max_new_tokens, skip_special_tokens=False)

    @staticmethod
    def _out(result):
        text = result.outputs[0].text if result.outputs else ''
        return Output(text, len(result.prompt_token_ids or []))

    def submit(self, key, request):
        """Queue one request in the engine under the id `key`; poll() returns its output when it is finished."""
        from vllm.sampling_params import RequestOutputKind
        try:
            params = self._params(request)
            params.output_kind = RequestOutputKind.FINAL_ONLY
            # what LLM.generate does for each prompt (vllm/entrypoints/common/offline.py), without waiting for a batch
            prompt = self.llm._preprocess_cmpl_one(self.build(request.messages, request.tools), None, mm_processor_kwargs=None)
            self.llm.llm_engine.add_request(str(key), prompt, params)
        except Exception as e:   # e.g. over max_model_len
            self._ready.append((key, Output('', 0, f'{type(e).__name__}: {e}')))

    def poll(self):
        """One engine step. Returns [(key, Output)] for the requests that finished."""
        done, self._ready = self._ready, []
        if self.llm.llm_engine.has_unfinished_requests():
            done += [(r.request_id, self._out(r)) for r in self.llm.llm_engine.step() if r.finished]
        return done

    def generate(self, requests):
        outputs, prompts = [None] * len(requests), []
        for k, r in enumerate(requests):
            try:
                prompts.append((k, self.build(r.messages, r.tools)))
            except Exception as e:
                outputs[k] = Output('', 0, f'{type(e).__name__}: {e}')
        if prompts:
            try:
                results = self.llm.generate([p for _, p in prompts], [self._params(requests[k]) for k, _ in prompts],
                                            use_tqdm=False)
                for (k, _), result in zip(prompts, results):
                    outputs[k] = self._out(result)
            except Exception:   # isolate the request that broke the batch (e.g. over max_model_len)
                for k, p in prompts:
                    try:
                        outputs[k] = self._out(self.llm.generate([p], [self._params(requests[k])], use_tqdm=False)[0])
                    except Exception as e:
                        outputs[k] = Output('', 0, f'{type(e).__name__}: {e}')
        return outputs
