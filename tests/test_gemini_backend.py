"""Offline protocol checks against the real SDK and an in-process HTTP transport."""
import base64
import json
import time

import pytest

genai = pytest.importorskip('google.genai')
httpx = pytest.importorskip('httpx')
from google.genai import types
from PIL import Image

from omniharness.backends import Request
from omniharness.backends.gemini import (
    GeminiBackend, OFFICIAL_BASE_URL, ProtocolError, media_parts, messages_to_contents, parse_response,
)
from omniharness.config import AgentConfig

TOOLS = [{'type': 'function', 'function': {
    'name': 'listen_audio', 'description': 'Listen to a selected interval.',
    'parameters': {'type': 'object', 'properties': {'t0': {'type': 'number'}, 't1': {'type': 'number'}},
                   'required': ['t0', 't1'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'bash', 'description': 'Read transcript.',
     'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}]


def response(parts=None, *, reason='STOP', usage=True):
    body = {'candidates': [{'index': 0, 'finishReason': reason,
                           'content': {'role': 'model', 'parts': parts or [{'text': 'A'}]}}]}
    if usage:
        body['usageMetadata'] = {'promptTokenCount': 11, 'candidatesTokenCount': 7,
                                 'thoughtsTokenCount': 3, 'cachedContentTokenCount': 5, 'totalTokenCount': 21}
    return body


def field(mapping, camel, snake):
    # Protobuf JSON accepts both spellings; SDK versions use both on nested fields.
    return mapping[camel] if camel in mapping else mapping[snake]


@pytest.fixture
def make_backend(monkeypatch):
    clients = []
    original_client = genai.Client
    monkeypatch.setenv('GEMINI_API_KEY', 'test-only-not-a-real-key')
    monkeypatch.setenv('GOOGLE_API_KEY', 'ignored-test-key')
    monkeypatch.setenv('GOOGLE_GEMINI_BASE_URL', 'https://invalid.example/gateway')
    monkeypatch.setenv('GOOGLE_GENAI_USE_VERTEXAI', 'true')
    monkeypatch.setenv('HTTPS_PROXY', 'http://invalid.example:9999')
    monkeypatch.setenv('HTTP_PROXY', 'http://invalid.example:9999')

    def factory(replies=None, **options):
        captured, requests = {}, []
        remaining = list(replies or [(200, response(), {})])

        def handler(request):
            requests.append(request)
            status, body, headers = remaining.pop(0)
            return httpx.Response(status, json=body, headers=headers)

        def sdk_factory(**kwargs):
            captured.update(kwargs)
            http_options = kwargs['http_options']
            http_options.httpx_client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)
            return original_client(**kwargs)

        monkeypatch.setattr(genai, 'Client', sdk_factory)
        backend = GeminiBackend('gemini-3.8-flash', AgentConfig(), TOOLS, **options)
        clients.append(backend)
        return backend, requests, captured

    yield factory
    for backend in clients:
        backend.close()


def test_official_transport_ignores_gateway_proxy_and_secondary_key(make_backend):
    backend, requests, config = make_backend()
    out = backend.generate([Request([{'role': 'user', 'content': 'Which option?'}])])[0]
    assert not out.error
    assert config['vertexai'] is False
    assert config['api_key'] == 'test-only-not-a-real-key'
    assert config['http_options'].base_url == OFFICIAL_BASE_URL
    assert config['http_options'].client_args['trust_env'] is False
    assert config['http_options'].async_client_args['trust_env'] is False
    assert config['http_options'].retry_options.attempts == 1
    assert str(requests[0].url).startswith(OFFICIAL_BASE_URL + '/v1beta/models/')
    assert requests[0].headers['x-goog-api-key'] == 'test-only-not-a-real-key'
    body = json.loads(requests[0].content)
    assert field(body['toolConfig'], 'functionCallingConfig', 'function_calling_config')['mode'] == 'AUTO'
    declaration = body['tools'][0]['functionDeclarations'][0]
    assert field(declaration, 'parametersJsonSchema', 'parameters_json_schema')['additionalProperties'] is False


def test_multiturn_keeps_signature_native_ids_and_audio_after_all_responses(make_backend):
    signature = base64.b64encode(b'opaque-sdk-signature').decode()
    first = response([
        {'thought': True, 'text': 'private thought', 'thoughtSignature': signature},
        {'functionCall': {'name': 'listen_audio', 'args': {'t0': 40, 't1': 55}, 'id': 'native_1'}},
        {'functionCall': {'name': 'bash', 'args': {'command': 'cat asr.txt'}}},
    ])
    backend, requests, _ = make_backend([(200, first, {}), (200, response(), {})])
    messages = [{'role': 'user', 'content': 'Which sound follows the door?'}]
    out = backend.generate([Request(messages)])[0]
    assert not out.error and out.text == ''
    assert out.tool_calls[0]['id'] == 'native_1'
    assert out.tool_calls[1]['id'].startswith('gemini_')
    messages.append({'role': 'assistant', 'content': out.text, 'provider_content': out.provider_content,
                     'tool_calls': [{'id': call['id'], 'type': 'function',
                                     'function': {'name': call['name'], 'arguments': call['arguments']}}
                                    for call in out.tool_calls]})
    raw = b'RIFF-offline-audio-observation'
    for i, call in enumerate(out.tool_calls):
        content = [{'type': 'text', 'text': 'tool result'}]
        if i == 0:
            content.append({'type': 'audio', 'audio': raw, 'mime_type': 'audio/wav', 't0': 40, 't1': 55})
        messages.append({'role': 'tool', 'name': call['name'], 'tool_call_id': call['id'], 'content': content})
    final = backend.generate([Request(messages)])[0]
    assert not final.error and final.text == 'A'
    body = json.loads(requests[1].content)
    assert body['contents'][1]['parts'][0]['thoughtSignature'] == signature
    observations = body['contents'][2]['parts']
    assert observations[0]['functionResponse']['id'] == 'native_1'
    assert 'id' not in observations[1]['functionResponse']
    assert all('functionResponse' in part for part in observations[:2])
    audio = next(part['inlineData'] for part in observations if 'inlineData' in part)
    assert field(audio, 'mimeType', 'mime_type') == 'audio/wav'
    assert audio['data'] == base64.b64encode(raw).decode()
    assert any('source time 40' in part.get('text', '') for part in observations)
    assert all('inlineData' not in part['functionResponse'] for part in observations[:2])


def test_frame_observations_preserve_absolute_source_times_and_image_encoding():
    frame = Image.new('RGB', (8, 8), 'green')
    parts = media_parts([{'type': 'video', 'video': [frame, frame],
                          'metadata': {'fps': 2, 'frames_indices': [10, 30]}}])
    assert [part.text for part in parts if part.text] == [
        '[Video frame; source time 5.000 seconds]', '[Video frame; source time 15.000 seconds]']
    assert all(part.inline_data.mime_type == 'image/jpeg' for part in parts if part.inline_data)
    with pytest.raises(ProtocolError, match='one absolute index'):
        media_parts([{'type': 'video', 'video': [frame], 'metadata': {'fps': 2, 'frames_indices': []}}])
    with pytest.raises(ProtocolError, match='Raw audio'):
        media_parts([{'type': 'audio', 'audio': b'wave', 't0': 0, 't1': 1}], allow_audio=False)


def test_incomplete_or_duplicate_tool_response_is_not_submitted():
    native = types.Content(role='model', parts=[types.Part.from_function_call(name='bash', args={'command': 'ls'})])
    messages = [{'role': 'assistant', 'provider_content': native, 'tool_calls': [
        {'id': 'local1', 'function': {'name': 'bash', 'arguments': {'command': 'ls'}}}]}]
    with pytest.raises(ProtocolError, match='Every Gemini'):
        messages_to_contents(messages)
    tool = {'role': 'tool', 'name': 'bash', 'tool_call_id': 'local1', 'content': 'result'}
    with pytest.raises(ProtocolError, match='duplicate'):
        messages_to_contents(messages + [tool, tool])
    messages[0]['tool_calls'][0]['function']['arguments'] = {'command': 'changed'}
    with pytest.raises(ProtocolError, match='arguments were modified'):
        messages_to_contents(messages + [tool])


def test_usage_counts_thoughts_once_and_keeps_cached_input_subset(make_backend):
    backend, _, _ = make_backend()
    out = backend.generate([Request([{'role': 'user', 'content': 'Question'}])])[0]
    assert out.prompt_tokens == 11
    assert out.usage['reported'] is True
    assert out.usage['input_tokens'] == 11
    assert out.usage['output_tokens'] == 10
    assert out.usage['reasoning_tokens'] == 3
    assert out.usage['cache_read_input_tokens'] == 5
    assert out.usage['total_tokens'] == 21
    assert out.usage['api_attempts'] == 1


@pytest.mark.parametrize('body', [response(reason='SAFETY'), {'usageMetadata': response()['usageMetadata']}])
def test_usage_survives_refusal_or_missing_candidate(make_backend, body):
    backend, requests, _ = make_backend([(200, body, {})])
    out = backend.generate([Request([{'role': 'user', 'content': 'Question'}])])[0]
    assert out.error
    assert out.usage['reported'] is True
    assert out.usage['total_tokens'] == 21
    assert out.prompt_tokens == 11
    assert len(requests) == 1


def test_bounded_retry_keeps_payload_and_does_not_retry_unauthorized(make_backend, monkeypatch):
    waits = []
    monkeypatch.setattr('omniharness.backends.gemini.time.sleep', waits.append)
    limited = {'error': {'code': 429, 'message': 'rate limited', 'status': 'RESOURCE_EXHAUSTED'}}
    backend, requests, _ = make_backend([(429, limited, {'Retry-After': '5'}), (200, response(), {})], max_retries=1)
    out = backend.generate([Request([{'role': 'user', 'content': 'Question'}])])[0]
    assert not out.error and len(requests) == 2 and waits == [5.0]
    assert requests[0].content == requests[1].content
    assert out.usage['api_attempts'] == 2 and out.usage['retry_wait_seconds'] == 5
    denied = {'error': {'code': 403, 'message': 'test-only-not-a-real-key must not be logged', 'status': 'PERMISSION_DENIED'}}
    backend2, requests2, _ = make_backend([(403, denied, {})], max_retries=2)
    failed = backend2.generate([Request([{'role': 'user', 'content': 'Question'}])])[0]
    assert 'HTTP 403' in failed.error and 'test-only' not in failed.error
    assert len(requests2) == 1 and failed.usage['reported'] is False


def test_submit_poll_preserves_request_ids_and_close(make_backend):
    backend, _, _ = make_backend([(200, response(), {}), (200, response(), {})], workers=2)
    request = Request([{'role': 'user', 'content': 'Question'}])
    backend.submit('one', request)
    backend.submit('two', request)
    with pytest.raises(ValueError, match='Duplicate'):
        backend.submit('two', request)
    outputs = {}
    until = time.monotonic() + 5
    while len(outputs) < 2 and time.monotonic() < until:
        outputs.update(backend.poll())
    assert set(outputs) == {'one', 'two'} and all(not out.error for out in outputs.values())
    backend.close()
    with pytest.raises(RuntimeError, match='closed'):
        backend.submit('three', request)


def test_gemini_thinking_level_uses_native_config_when_supported(make_backend):
    if 'thinking_level' not in types.ThinkingConfig.model_fields:
        pytest.skip('This installed SDK predates Gemini thinking_level')
    backend, requests, _ = make_backend(thinking_level='high')
    out = backend.generate([Request([{'role': 'user', 'content': 'Question'}])])[0]
    assert not out.error
    config = json.loads(requests[0].content)['generationConfig']
    thinking = field(config, 'thinkingConfig', 'thinking_config')
    assert field(thinking, 'thinkingLevel', 'thinking_level') == 'HIGH'


def test_key_is_required_without_loading_any_credential_file(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.setenv('GOOGLE_API_KEY', 'not-used-as-a-fallback')
    with pytest.raises(ValueError, match='GEMINI_API_KEY'):
        GeminiBackend('gemini-3.8-flash', AgentConfig(), TOOLS)
