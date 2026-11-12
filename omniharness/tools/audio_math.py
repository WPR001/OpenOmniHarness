"""Signal measurements returned as text and files for the existing view/bash tools."""
from ..audio_math import VERSION
from .base import ToolInputError, register

STAMP = {'anyOf': [{'type': 'number'}, {'type': 'string'}],
         'description': 'Absolute source time in seconds or mm:ss.'}
PARAMS = {'type': 'object', 'properties': {
    'band_hz': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2},
    'channel': {'anyOf': [{'type': 'string', 'enum': ['mix', 'left', 'right']}, {'type': 'integer', 'minimum': 0}]},
    'min_peak_dbfs': {'type': 'number'}, 'envelope_smooth_hz': {'type': 'number'},
    'window_ms': {'type': 'number'}, 'hop_ms': {'type': 'number'},
    'min_gap_ms': {'type': 'number'}, 'threshold_above_local_floor_db': {'type': 'number'},
    'comparison_window': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}
}, 'additionalProperties': False}


def session(box):
    if box.cfg.harness_version != VERSION:
        raise ToolInputError('enable audio-math-v1 to use signal analysis')
    if 'audio_math' not in box.state:
        from ..audio_math.runtime import AudioMathSession
        box.state['audio_math'] = AudioMathSession(box.sb, box.cfg)
    return box.state['audio_math']


@register('audio_analyze',
          'Measure a localized audio interval after initial visual/ASR observations, when more acoustic evidence '
          'is needed. Returns timestamped measurements and PNG/CSV/JSON paths. Use view to inspect plots.',
          {'type': 'object', 'properties': {
              't0': STAMP, 't1': STAMP,
              'mode': {'type': 'string', 'enum': ['overview', 'events', 'spectrum', 'rhythm', 'compare']},
              'params': PARAMS}, 'required': ['t0', 't1', 'mode'], 'additionalProperties': False})
def audio_analyze(box, args):
    return session(box).analyze(args)


@register('audio_python',
          'Compute custom CPU signal measurements on a localized audio interval. Preloaded: '
          'x (float32 samples x channels), sr, t (absolute seconds), t0, t1, np, scipy, plt, math. '
          'Set result to a compact JSON-compatible value. Save files with save_json(name,value), '
          'save_csv(name,rows,header=None), save_plot(name); view displays the resulting plots.',
          {'type': 'object', 'properties': {
              't0': STAMP, 't1': STAMP,
              'code': {'type': 'string', 'description': 'Python signal-analysis code; keep absolute times and units.'}},
           'required': ['t0', 't1', 'code'], 'additionalProperties': False})
def audio_python(box, args):
    return session(box).python(args)
