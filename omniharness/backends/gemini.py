"""Official Gemini Developer API adapter, with harness-controlled tool execution.

References (Google):
https://googleapis.github.io/python-genai/#manually-declare-and-invoke-a-function-for-function-calling
https://ai.google.dev/gemini-api/docs/generate-content/audio
https://ai.google.dev/gemini-api/docs/generate-content/function-calling

Only the SDK's public API is used. No gateway, custom endpoint, credential file,
automatic tool execution, or implicit proxy configuration is accepted here.
"""
import copy
import io
import math
import os
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from . import Output

OFFICIAL_BASE_URL = 'https://generativelanguage.googleapis.com'


class ProtocolError(ValueError):
    """A local, credential-free conversation validation error."""
    pass


def _sdk():
    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise ImportError('Gemini requires requirements-omni.txt (the official google-genai SDK).') from error
    return genai, types


def _blocks(content):
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}]
    if content is None:
        return []
    if not isinstance(content, list):
        raise ProtocolError('Message content must be text or a list of media blocks')
    return content


def media_parts(content, *, allow_audio=True):
    """Convert canonical observations to official SDK Parts; preserve source times."""
    _, types = _sdk()
    parts = []
    for block in _blocks(content):
        kind = block.get('type')
        if kind == 'text':
            if block.get('text'):
                parts.append(types.Part.from_text(text=str(block['text'])))
        elif kind == 'image':
            from PIL import Image
            image = block['image']
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            with io.BytesIO() as encoded:
                image.convert('RGB').save(encoded, format='JPEG', quality=90)
                parts.append(types.Part.from_bytes(data=encoded.getvalue(), mime_type='image/jpeg'))
        elif kind == 'video':
            frames = block['video']
            metadata = block.get('metadata') or {}
            fps = float(metadata.get('fps', 0))
            indices = metadata.get('frames_indices', [])
            if not math.isfinite(fps) or fps <= 0 or len(indices) != len(frames):
                raise ProtocolError('Video observations require source fps and one absolute index per frame')
            for index, frame in zip(indices, frames):
                seconds = float(index) / fps
                if not math.isfinite(seconds) or seconds < 0:
                    raise ProtocolError('Invalid absolute video frame time')
                parts.append(types.Part.from_text(text=f'[Video frame; source time {seconds:.3f} seconds]'))
                parts.extend(media_parts([{'type': 'image', 'image': frame}], allow_audio=allow_audio))
        elif kind == 'audio':
            if not allow_audio:
                raise ProtocolError('Raw audio requires the listen_audio tool to be enabled')
            raw, mime = block.get('audio'), block.get('mime_type', 'audio/wav')
            if not isinstance(raw, bytes) or not raw:
                raise ProtocolError('Audio observations require nonempty encoded audio bytes')
            if mime != 'audio/wav':
                raise ProtocolError('listen_audio observations must be audio/wav')
            t0, t1 = float(block['t0']), float(block['t1'])
            if not math.isfinite(t0) or not math.isfinite(t1) or not 0 <= t0 < t1:
                raise ProtocolError('Audio observations require a valid absolute time interval')
            parts.append(types.Part.from_text(text=(
                f'[Original audio; source interval {t0:g}-{t1:g} seconds; '
                f'audio offset 0 corresponds to source time {t0:g} seconds]')))
            parts.append(types.Part.from_bytes(data=raw, mime_type=mime))
        else:
            raise ProtocolError(f'Unsupported observation type: {kind}')
    return parts


def messages_to_contents(messages, *, allow_audio=True):
    """Keep model Content intact, pair every call, then attach media observations.

    Audio is a top-level user Part following function responses, not nested in a
    FunctionResponse whose multimodal MIME support differs between model versions.
    """
    _, types = _sdk()
    systems, contents, responses, attachments = [], [], [], []
    pending, responded = {}, set()

    def flush_tools():
        if pending and set(pending) != responded:
            raise ProtocolError('Every Gemini function call must have a matching tool response')
        if responses:
            contents.append(types.Content(role='user', parts=responses[:] + attachments[:]))
        pending.clear()
        responded.clear()
        responses.clear()
        attachments.clear()

    for message in messages:
        role = message['role']
        if role != 'tool':
            flush_tools()
        if role == 'system':
            blocks = _blocks(message.get('content'))
            if any(block.get('type') != 'text' for block in blocks):
                raise ProtocolError('System instructions must contain text only')
            systems.extend(media_parts(blocks, allow_audio=False))
        elif role == 'assistant':
            original = message.get('provider_content')
            canonical = message.get('tool_calls') or []
            if original is not None:
                original = (original.model_copy(deep=True) if isinstance(original, types.Content)
                            else types.Content.model_validate(copy.deepcopy(original)))
                original.role = 'model'
                native = [part.function_call for part in original.parts or [] if part.function_call is not None]
                if len(native) != len(canonical):
                    raise ProtocolError('Preserved Gemini function calls do not match the conversation')
                for call, local in zip(native, canonical):
                    local_id, function = local.get('id'), local['function']
                    if not local_id or local_id in pending or call.name != function['name']:
                        raise ProtocolError('Invalid or mismatched Gemini tool call identity')
                    if call.id and call.id != local_id:
                        raise ProtocolError('A Gemini native function call ID was modified')
                    if dict(call.args or {}) != function['arguments']:
                        raise ProtocolError('Gemini function call arguments were modified')
                    pending[local_id] = (call.name, call.id)
                contents.append(original)
            else:
                if canonical:
                    raise ProtocolError('Gemini function-call history requires its original provider_content')
                contents.append(types.Content(role='model', parts=media_parts(message.get('content'), allow_audio=allow_audio)))
        elif role == 'tool':
            call_id = message.get('tool_call_id')
            if call_id not in pending or call_id in responded:
                raise ProtocolError('Unknown or duplicate Gemini tool response ID')
            name, native_id = pending[call_id]
            if message.get('name') != name:
                raise ProtocolError('Gemini tool response name does not match its function call')
            observations = media_parts(message.get('content'), allow_audio=allow_audio)
            text = '\n'.join(part.text for part in observations if part.text)
            kwargs = {'name': name, 'response': {'output': text or '(empty tool output)'}}
            if native_id:
                kwargs['id'] = native_id
            responses.append(types.Part(function_response=types.FunctionResponse(**kwargs)))
            responded.add(call_id)
            if any(part.inline_data is not None for part in observations):
                attachments.append(types.Part.from_text(text=f'[Tool observation: {name}; tool_call_id={call_id}]'))
                attachments.extend(observations)
        elif role == 'user':
            contents.append(types.Content(role='user', parts=media_parts(message.get('content'), allow_audio=allow_audio)))
        else:
            raise ProtocolError(f'Unsupported conversation role: {role}')
    flush_tools()
    return systems, contents


def response_usage(response):
    """Usage belongs to the request, including refusals and malformed answers."""
    metadata = response.usage_metadata
    prompt = int(getattr(metadata, 'prompt_token_count', None) or 0)
    candidate_tokens = int(getattr(metadata, 'candidates_token_count', None) or 0)
    thoughts = int(getattr(metadata, 'thoughts_token_count', None) or 0)
    return {'reported': metadata is not None, 'input_tokens': prompt,
            'output_tokens': candidate_tokens + thoughts, 'reasoning_tokens': thoughts,
            'cache_read_input_tokens': int(getattr(metadata, 'cached_content_token_count', None) or 0),
            'total_tokens': int(getattr(metadata, 'total_token_count', None) or prompt + candidate_tokens + thoughts),
            'provider_usage': metadata.model_dump(mode='json', exclude_none=True) if metadata else {}}


def parse_response(response):
    """Read visible text and structured calls, retaining opaque model metadata."""
    candidates = response.candidates or []
    if not candidates:
        raise ProtocolError('Gemini returned no candidate (possibly blocked by the provider)')
    candidate = next((c for c in candidates if c.index in (0, None)), candidates[0])
    reason = getattr(candidate.finish_reason, 'value', candidate.finish_reason)
    if reason not in ('STOP', 'MAX_TOKENS'):
        raise ProtocolError(f'Gemini did not finish normally: {reason or "missing finish_reason"}')
    content = candidate.content
    if content is None or not content.parts:
        raise ProtocolError('Gemini returned no content parts')
    texts, calls = [], []
    for part in content.parts:
        if part.text and not part.thought:
            texts.append(part.text)
        call = part.function_call
        if call is not None:
            if not call.name or (call.args is not None and not isinstance(call.args, dict)):
                raise ProtocolError('Gemini returned a malformed function call')
            calls.append({'id': call.id or 'gemini_' + uuid.uuid4().hex,
                          'name': call.name, 'arguments': copy.deepcopy(call.args or {})})
    if len({call['id'] for call in calls}) != len(calls):
        raise ProtocolError('Gemini returned duplicate function call IDs')
    if not texts and not calls:
        raise ProtocolError('Gemini returned neither visible text nor a function call')
    usage = response_usage(response)
    usage['finish_reason'] = reason
    return Output(''.join(texts), prompt_tokens=usage['input_tokens'], tool_calls=calls, usage=usage,
                  provider_content=content.model_copy(deep=True))


class GeminiBackend:
    name = 'gemini'
    native_tools = True
    supports_raw_audio = True

    def __init__(self, model, cfg, tools, *, workers=4, timeout=180, max_retries=2, thinking_level=None):
        genai, types = _sdk()
        api_key = os.environ.get('GEMINI_API_KEY', '').strip()
        if not api_key:
            raise ValueError('Set GEMINI_API_KEY to a Google Gemini Developer API key before selecting Gemini.')
        if not isinstance(model, str) or not model.strip() or '://' in model:
            raise ValueError('Gemini requires an official model ID, not an endpoint URL')
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError('Gemini workers must be a positive integer')
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or not 0 <= max_retries <= 10:
            raise ValueError('Gemini max_retries must be an integer between 0 and 10')
        if isinstance(timeout, bool) or not math.isfinite(float(timeout)) or float(timeout) <= 0:
            raise ValueError('Gemini timeout must be positive and finite')
        if thinking_level not in (None, 'minimal', 'low', 'medium', 'high'):
            raise ValueError('Gemini thinking level must be minimal, low, medium, high, or unset')
        if thinking_level and 'thinking_level' not in types.ThinkingConfig.model_fields:
            raise ValueError('Upgrade google-genai to use Gemini thinking_level')
        self.model, self.cfg, self.tools = model, cfg, tools
        self.max_retries, self.thinking_level = max_retries, thinking_level
        self.audio_enabled = any(tool['function']['name'] == 'listen_audio' for tool in tools)
        self.client = genai.Client(
            api_key=api_key, vertexai=False,
            http_options=types.HttpOptions(
                base_url=OFFICIAL_BASE_URL, api_version='v1beta', timeout=int(float(timeout) * 1000),
                client_args={'trust_env': False}, async_client_args={'trust_env': False},
                retry_options=types.HttpRetryOptions(attempts=1)))
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='omniharness-gemini')
        self.pending = {}
        self._closed = False

    def _request_config(self, request):
        _, types = _sdk()
        systems, contents = messages_to_contents(request.messages, allow_audio=self.audio_enabled)
        specs = self.tools if request.tools is None else request.tools
        declarations = [types.FunctionDeclaration(name=tool['function']['name'],
                        description=tool['function'].get('description', ''),
                        parameters_json_schema=copy.deepcopy(tool['function'].get('parameters', {'type': 'object'})))
                        for tool in specs]
        options = dict(system_instruction=types.Content(parts=systems) if systems else None,
                       temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                       max_output_tokens=request.max_new_tokens, candidate_count=1,
                       response_modalities=['TEXT'],
                       automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        if declarations:
            options['tools'] = [types.Tool(function_declarations=declarations)]
            options['tool_config'] = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode='AUTO'))
        if self.thinking_level:
            options['thinking_config'] = types.ThinkingConfig(thinking_level=self.thinking_level.upper())
        return contents, types.GenerateContentConfig(**options)

    @staticmethod
    def _status(error):
        code = getattr(error, 'code', None) or getattr(getattr(error, 'response', None), 'status_code', None)
        try:
            return int(code)
        except (TypeError, ValueError):
            return None

    def _one(self, request):
        started, waited, attempts = time.monotonic(), 0.0, 0
        received_usage = {'reported': False}
        try:
            contents, config = self._request_config(request)
            for attempt in range(self.max_retries + 1):
                attempts += 1
                try:
                    response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
                    received_usage = response_usage(response)
                    out = parse_response(response)
                    out.usage.update(api_attempts=attempts, request_seconds=round(time.monotonic() - started, 3),
                                     retry_wait_seconds=round(waited, 3))
                    return out
                except Exception as error:
                    status = self._status(error)
                    transient = status in (408, 429, 500, 502, 503, 504) or type(error).__name__ in (
                        'TimeoutException', 'ConnectTimeout', 'ReadTimeout', 'WriteTimeout', 'PoolTimeout',
                        'ConnectError', 'ReadError', 'RemoteProtocolError')
                    if not transient or attempt >= self.max_retries:
                        raise
                    headers = getattr(getattr(error, 'response', None), 'headers', {}) or {}
                    try:
                        delay = float(headers.get('retry-after', 2 ** attempt))
                        if not math.isfinite(delay) or delay < 0:
                            delay = 2 ** attempt
                    except (TypeError, ValueError):
                        delay = 2 ** attempt
                    # Both retries and waiting are bounded; never launch another tool extraction here.
                    delay = min(delay, 60.0)
                    time.sleep(delay)
                    waited += delay
        except Exception as error:
            status = self._status(error)
            # API exceptions can contain request URLs/headers. Log only a status and class.
            detail = (f'Gemini request failed (HTTP {status}, {type(error).__name__})' if status else
                      f'Gemini request failed ({type(error).__name__})')
            if isinstance(error, ProtocolError) and not status:
                detail = str(error)[:300]  # Our local validation errors contain no request or credentials.
            return Output('', prompt_tokens=received_usage.get('input_tokens', 0), error=detail, tool_calls=[], usage={
                **received_usage, 'api_attempts': attempts, 'request_seconds': round(time.monotonic() - started, 3),
                'retry_wait_seconds': round(waited, 3)})

    def submit(self, key, request):
        if self._closed:
            raise RuntimeError('Gemini backend is closed')
        if key in self.pending.values():
            raise ValueError('Duplicate pending request key')
        self.pending[self.pool.submit(self._one, request)] = key

    def poll(self):
        if not self.pending:
            return []
        done, _ = wait(tuple(self.pending), timeout=0.1, return_when=FIRST_COMPLETED)
        return [(self.pending.pop(future), future.result()) for future in done]

    def generate(self, requests):
        if self._closed:
            raise RuntimeError('Gemini backend is closed')
        futures = [self.pool.submit(self._one, request) for request in requests]
        return [future.result() for future in futures]

    def close(self):
        if not self._closed:
            self._closed = True
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.client.close()
