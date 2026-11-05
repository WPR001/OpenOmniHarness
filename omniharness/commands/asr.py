"""asr: the precomputed speech transcript of the video (omniharness/media/transcript.py), read through
OH_ASR_JSON, which the sandbox sets."""
import argparse
import os

from ..parsing import mmss
from . import command
from .video import _time


@command('asr', positional=('cmd', 't0', 't1', 'query'))
def asr(argv=None):
    from ..media.transcript import Transcript
    p = argparse.ArgumentParser(prog='asr', description='The speech transcript of this video (Qwen3-ASR, computed '
                                'beforehand; it is the only access to the audio). Lines carry their start and end times; '
                                '[no words a-b] marks stretches where no speech was recognized (silence, music, noise '
                                'or crowd: the transcript cannot tell which).')
    sub = p.add_subparsers(dest='cmd')
    sub.add_parser('full', help='the whole transcript (default)')
    w = sub.add_parser('window', help='lines between T0 and T1')
    w.add_argument('t0', type=_time)
    w.add_argument('t1', type=_time)
    s = sub.add_parser('search', help='lines that contain the query words, best first')
    s.add_argument('query')
    s.add_argument('-k', type=int, default=20)
    ws = sub.add_parser('words', help='every word with its start and end time between T0 and T1')
    ws.add_argument('t0', type=_time)
    ws.add_argument('t1', type=_time)
    sub.add_parser('gaps', help='stretches without recognized words')
    sub.add_parser('info', help='language, coverage and whether word times exist')
    a = p.parse_args(argv)
    path = os.environ.get('OH_ASR_JSON', '')
    if not path or not os.path.exists(path):
        print('No transcript is available for this video.')
        return
    tr = Transcript(path)
    if not tr.d['has_audio']:
        print('This video has no audio track, so there is no transcript.')
        return
    cmd = a.cmd or 'full'
    if cmd == 'full':
        out = tr.render()
    elif cmd == 'window':
        out = tr.render(a.t0, a.t1)
    elif cmd == 'search':
        out = tr.search(a.query, a.k)
    elif cmd == 'words':
        out = tr.words(a.t0, a.t1) if tr.d['aligned'] else '(no word-level times for this video; use asr window)'
    elif cmd == 'gaps':
        out = '\n'.join(f'[no words {mmss(x)}-{mmss(y)}]' for x, y in tr.d['gaps'])
    else:
        out = tr.info()
    print(out or '(nothing)')
