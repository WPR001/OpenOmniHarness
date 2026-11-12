"""Opt-in audio signal analysis; scientific libraries load only in CPU workers."""
import math

VERSION = 'audio-math-v1'
TOOL_NAMES = ('audio_analyze', 'audio_python')


def validate_config(cfg):
    version = getattr(cfg, 'harness_version', 'original')
    if version not in ('original', VERSION):
        raise ValueError('harness_version must be original or audio-math-v1')
    if version == 'original':
        if any(name in cfg.tools for name in TOOL_NAMES):
            raise ValueError('audio_analyze/audio_python require --harness-version audio-math-v1')
        return
    for name in ('audio_analysis_seconds', 'audio_analysis_timeout'):
        value = getattr(cfg, name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(name + ' must be finite and positive')
    if cfg.audio_analysis_seconds > 120:
        raise ValueError('audio_analysis_seconds must be at most 120; use multiple windows for longer audio')
    if (isinstance(cfg.audio_analysis_calls, bool) or not isinstance(cfg.audio_analysis_calls, int)
            or cfg.audio_analysis_calls < 1):
        raise ValueError('audio_analysis_calls must be a positive integer')


def normalize_config(cfg):
    """Select the same tools for CLI and direct Python callers before building backend schemas."""
    validate_config(cfg)
    if getattr(cfg, 'harness_version', 'original') == VERSION:
        names = list(cfg.tools)
        for name in TOOL_NAMES:
            if name not in names:
                names.insert(names.index('answer') if 'answer' in names else len(names), name)
        cfg.tools = tuple(names)
    return cfg
