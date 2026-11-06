"""Optional raw-audio tool. The reasoning model decides whether and where to listen."""
import math
import re
import time

from ..media.audio import AudioInputError, SAMPLE_RATE, extract_audio, probe_audio
from ..parsing import mmss, parse_time
from .base import ToolInputError, register

_TIME = re.compile(r'(?:\d+(?:\.\d+)?|\d+:[0-5]?\d(?:\.\d+)?|\d+:[0-5]?\d:[0-5]?\d(?:\.\d+)?)\Z')


def _time(value):
    if isinstance(value, bool):
        raise ToolInputError('t0 and t1 must be finite times in seconds or mm:ss')
    if isinstance(value, str):
        if len(value) > 64 or not _TIME.fullmatch(value.strip()):
            raise ToolInputError('t0 and t1 must be finite times in seconds or mm:ss')
        result = parse_time(value.strip())
    elif isinstance(value, (int, float)):
        try:
            result = float(value)
        except OverflowError:
            raise ToolInputError('t0 and t1 must be finite times in seconds or mm:ss') from None
    else:
        raise ToolInputError('t0 and t1 must be finite times in seconds or mm:ss')
    if not math.isfinite(result):
        raise ToolInputError('t0 and t1 must be finite times in seconds or mm:ss')
    return result


def trace_arguments(args):
    """Keep validated times in traces; never echo arbitrary/path arguments into tool records."""
    if not isinstance(args, dict):
        return {'invalid_arguments': True}
    safe = {}
    for key in ('t0', 't1'):
        if key in args:
            try:
                safe[key] = _time(args[key])
            except (ToolInputError, OverflowError, ValueError):
                safe[key] = '[invalid time]'
    if set(args) - {'t0', 't1'}:
        safe['unsupported_arguments'] = True
    return safe


@register('listen_audio',
          'Listen to the original audio in a selected time interval. Use it when the transcript and images do not '
          'provide the sound evidence needed for the question. Only t0 and t1 are accepted.',
          {'type': 'object', 'properties': {
              't0': {'type': 'string', 'description': 'start on the original timeline, seconds or mm:ss'},
              't1': {'type': 'string', 'description': 'end on the original timeline, seconds or mm:ss'}},
           'required': ['t0', 't1'], 'additionalProperties': False})
def listen_audio(box, args):
    started = time.monotonic()
    calls = box.state.setdefault('audio_listen_calls', [])
    info = {'status': 'rejected', 'requested_t0': None, 'requested_t1': None,
            'actual_t0': None, 'actual_t1': None, 'audio_seconds': 0.0,
            'bytes': 0, 'sample_rate': SAMPLE_RATE, 'channels': 1, 'budget_charged': False}
    calls.append(info)
    try:
        if not isinstance(args, dict) or set(args) != {'t0', 't1'}:
            raise ToolInputError('listen_audio accepts only the two required arguments t0 and t1')
        t0, t1 = _time(args['t0']), _time(args['t1'])
        info.update(requested_t0=t0, requested_t1=t1)
        if t0 < 0 or t1 <= t0:
            raise ToolInputError('the audio interval must satisfy 0 <= t0 < t1')
        limit = box.cfg.listen_seconds
        if (isinstance(limit, bool) or not isinstance(limit, (int, float))
                or not math.isfinite(limit) or limit <= 0):
            raise ToolInputError('the listen_audio duration budget is not configured correctly')
        if t1 - t0 > limit:
            raise ToolInputError(f'each audio interval may be at most {limit:g} seconds')
        if (t1 - t0) * SAMPLE_RATE < 1:
            raise ToolInputError('the requested audio interval is too short')
        used = box.state.get('audio_listen_count', 0)
        if isinstance(box.cfg.max_listen, bool) or not isinstance(box.cfg.max_listen, int) or box.cfg.max_listen < 0:
            raise ToolInputError('the listen_audio call budget is not configured correctly')
        if used >= box.cfg.max_listen:
            raise ToolInputError(f'the listen_audio budget ({box.cfg.max_listen}) is used up')
        source = box.sb._audio_source
        bounds = probe_audio(source)
        if t0 < bounds.start or t1 > bounds.end:
            raise ToolInputError(f'the audio interval must stay within {bounds.start:g}-{bounds.end:g} seconds')
        # Invalid parameters, out-of-range windows and missing audio do not spend a listen.
        # A decoder attempt does spend it, including a decoder failure.
        box.state['audio_listen_count'] = used + 1
        info.update(status='decoding', budget_charged=True)
        clip = extract_audio(source, t0, t1)
        actual_end = t0 + clip.seconds
        info.update(status='ok', actual_t0=t0, actual_t1=actual_end, audio_seconds=clip.seconds,
                    bytes=len(clip.data), sample_rate=clip.sample_rate, channels=clip.channels)
        return [
            {'type': 'text', 'text': f'Original audio {mmss(t0)}-{mmss(actual_end)} '
             f'({clip.seconds:.3f} seconds). Audio offset 0 s corresponds to original timeline {t0:.6f} s; '
             'add that offset to any time heard within this clip.'},
            {'type': 'audio', 'audio': clip.data, 'mime_type': 'audio/wav', 't0': t0, 't1': actual_end},
        ]
    except ToolInputError:
        raise
    except AudioInputError as error:
        info['status'] = 'decode_error' if info['budget_charged'] else 'source_error'
        raise ToolInputError(str(error)) from None
    except Exception:
        info['status'] = 'decode_error' if info['budget_charged'] else 'source_error'
        raise ToolInputError('audio processing is unavailable for this interval') from None
    finally:
        info['seconds'] = round(time.monotonic() - started, 3)
