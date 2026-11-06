"""Host-side audio inspection and bounded PCM extraction for the listen_audio tool.

Source paths and decoder stderr stay on the host. Only WAV bytes and timing metadata
leave this module; temporary files are never placed in the model's sandbox.
"""
import io
import json
import math
import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2


class AudioInputError(ValueError):
    """A message safe to return to the model (no source paths or subprocess output)."""


@dataclass(frozen=True)
class AudioInfo:
    start: float
    end: float


@dataclass(frozen=True)
class AudioClip:
    data: bytes
    seconds: float
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS


def _number(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def probe_audio(source):
    """Audio bounds in seconds from the source timeline's start, without exposing its path."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
             'stream=index,start_time,duration:format=start_time,duration', '-of', 'json', os.fspath(source)],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=True)
        data = json.loads(result.stdout)
    except Exception:
        raise AudioInputError('could not inspect the original audio source') from None
    if not isinstance(data, dict):
        raise AudioInputError('audio timing metadata is unavailable')
    streams = data.get('streams')
    if not isinstance(streams, list) or not streams:
        raise AudioInputError('the original source has no audio track')
    stream = streams[0]
    container = data.get('format') or {}
    if not isinstance(stream, dict) or not isinstance(container, dict):
        raise AudioInputError('audio timing metadata is unavailable')
    container_start = _number(container.get('start_time'), 0.0)
    audio_start = max(0.0, _number(stream.get('start_time'), container_start) - container_start)
    container_duration = _number(container.get('duration'))
    audio_duration = _number(stream.get('duration'))
    audio_end = audio_start + audio_duration if audio_duration is not None else container_duration
    if container_duration is not None and audio_end is not None:
        audio_end = min(audio_end, container_duration)
    if audio_end is None or audio_end <= audio_start:
        raise AudioInputError('audio duration is unavailable')
    return AudioInfo(audio_start, audio_end)


def _read_pcm_wav(path, requested_seconds):
    """Validate decoder output and trim at most a rounding sample at the requested end."""
    max_frames = math.floor(requested_seconds * SAMPLE_RATE + 1e-7)
    if max_frames < 1:
        raise AudioInputError('the requested audio interval is too short')
    try:
        with wave.open(os.fspath(path), 'rb') as reader:
            if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getcomptype()) != (
                    CHANNELS, SAMPLE_WIDTH, SAMPLE_RATE, 'NONE'):
                raise AudioInputError('the decoded audio has an unsupported format')
            frames = reader.getnframes()
            # Truncated sources and failed seeks must not masquerade as the requested evidence interval.
            if frames < 1 or abs(frames / SAMPLE_RATE - requested_seconds) > 0.05:
                raise AudioInputError('the decoded audio does not cover the requested interval')
            frames = min(frames, max_frames)
            pcm = reader.readframes(frames)
            if len(pcm) != frames * SAMPLE_WIDTH * CHANNELS:
                raise AudioInputError('the decoded audio is incomplete')
        output = io.BytesIO()
        with wave.open(output, 'wb') as writer:
            writer.setnchannels(CHANNELS)
            writer.setsampwidth(SAMPLE_WIDTH)
            writer.setframerate(SAMPLE_RATE)
            writer.writeframes(pcm)
        return AudioClip(output.getvalue(), frames / SAMPLE_RATE)
    except AudioInputError:
        raise
    except Exception:
        raise AudioInputError('could not read the decoded audio') from None


def extract_audio(source, t0, t1):
    """Decode one validated interval as 16 kHz mono PCM16 WAV, cleaning temporary files on every exit."""
    if (isinstance(t0, bool) or isinstance(t1, bool) or not isinstance(t0, (int, float))
            or not isinstance(t1, (int, float)) or not math.isfinite(t0) or not math.isfinite(t1)
            or t0 < 0 or t1 <= t0):
        raise AudioInputError('invalid audio interval')
    try:
        with tempfile.TemporaryDirectory(prefix='omniharness-audio-', dir='/tmp') as directory:
            path = os.path.join(directory, 'clip.wav')
            subprocess.run(
                ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                 '-ss', f'{t0:.9f}', '-i', os.fspath(source), '-t', f'{t1 - t0:.9f}',
                 '-map', '0:a:0', '-vn', '-sn', '-dn', '-ac', str(CHANNELS),
                 '-ar', str(SAMPLE_RATE), '-c:a', 'pcm_s16le', path],
                stdin=subprocess.DEVNULL, capture_output=True, timeout=90, check=True)
            return _read_pcm_wav(path, t1 - t0)
    except AudioInputError:
        raise
    except Exception:
        raise AudioInputError('could not decode the requested audio interval') from None
