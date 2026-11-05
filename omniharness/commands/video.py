"""Video commands: vprobe, vsheet, vframes, vcrop. The video is ./video.mp4 (no audio track). Times are seconds or
mm:ss. Images are written into the sandbox and shown to the model with the `view` tool."""
import argparse
import os

from ..parsing import mmss, parse_time
from . import command

VIDEO = os.environ.get('OH_VIDEO', 'video.mp4')


def _time(text):
    try:
        return parse_time(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f'not a time: {text}')


def _window(t0, t1, duration):
    t0 = min(max(t0, 0.0), max(duration - 0.05, 0.0))
    t1 = min(max(t1, t0 + 0.05), duration)
    return t0, t1


@command('vprobe', positional=('video',))
def vprobe(argv=None):
    from ..media.video import probe
    p = argparse.ArgumentParser(prog='vprobe', description='Duration, frame rate and size of the video.')
    p.add_argument('video', nargs='?', default=VIDEO)
    a = p.parse_args(argv)
    info = probe(a.video)
    print(f'{a.video}: duration {mmss(info.duration)} ({info.duration:.2f} s), {info.width}x{info.height}, '
          f'{info.fps:.2f} fps, {info.num_frames} frames. Audio: not in this file; use `asr` for what is said or sung.')


@command('vsheet', positional=('t0', 't1'))
def vsheet(argv=None):
    import numpy as np
    from ..media.video import contact_sheet, probe
    p = argparse.ArgumentParser(prog='vsheet', description='Contact sheet: frames from T0 to T1 every N seconds, '
                                'each stamped with its time (at most 64 cells; a long window widens the interval).')
    p.add_argument('t0', type=_time)
    p.add_argument('t1', type=_time)
    p.add_argument('--every', type=float, default=10.0, help='seconds between frames (>= 0.2)')
    p.add_argument('--cols', type=int, default=0)
    p.add_argument('--width', type=int, default=1440, help='sheet width in pixels (<= 1920)')
    p.add_argument('-o', '--out', default='')
    p.add_argument('--video', default=VIDEO)
    a = p.parse_args(argv)
    info = probe(a.video)
    t0, t1 = _window(a.t0, a.t1, info.duration)
    every = max(a.every, 0.2)
    times = list(np.arange(t0, t1, every)) or [t0]
    note = ''
    if len(times) > 64:
        times = list(np.linspace(t0, t1, 64, endpoint=False))
        note = f' (interval widened to {(t1 - t0) / 64:.1f} s to fit 64 cells)'
    sheet, stamps = contact_sheet(a.video, times, min(a.width, 1920), a.cols or None)
    out = a.out or f'sheet_{mmss(t0).replace(":", "-")}_{mmss(t1).replace(":", "-")}.jpg'
    if os.path.splitext(out)[1].lower() not in ('.jpg', '.jpeg', '.png'):
        out += '.jpg'
    sheet.save(out, quality=90)
    print(f'{out}: {len(stamps)} frames from {mmss(stamps[0])} to {mmss(stamps[-1])}{note}, {sheet.width}x{sheet.height}. '
          'Show it with view.')


@command('vframes', positional=('t0', 't1'))
def vframes(argv=None):
    import numpy as np
    from PIL import Image
    from ..media.video import decode, probe, stamp
    p = argparse.ArgumentParser(prog='vframes', description='Single frames from T0 to T1 at a given fps (at most 32 files).')
    p.add_argument('t0', type=_time)
    p.add_argument('t1', type=_time)
    p.add_argument('--fps', type=float, default=1.0)
    p.add_argument('--full', action='store_true', help='native resolution (default: longest side 960)')
    p.add_argument('-o', '--out', default='frames')
    p.add_argument('--video', default=VIDEO)
    a = p.parse_args(argv)
    info = probe(a.video)
    t0, t1 = _window(a.t0, a.t1, info.duration)
    n = int(min(32, max(1, round((t1 - t0) * a.fps))))
    times = list(np.linspace(t0, t1, n, endpoint=False)) if n > 1 else [t0]
    array, stamps = decode(a.video, times)
    os.makedirs(a.out, exist_ok=True)
    names = []
    for frame, t in zip(array, stamps):
        image = Image.fromarray(frame)
        if not a.full:
            image.thumbnail((960, 960), Image.BICUBIC)
        name = os.path.join(a.out, f'f_{mmss(t).replace(":", "-")}.jpg')
        stamp(image, mmss(t)).save(name, quality=92)
        names.append(name)
    print(f'{len(names)} frames ({"native" if a.full else "max side 960"}):\n' + '\n'.join(names))


@command('vcrop', positional=('image',))
def vcrop(argv=None):
    from PIL import Image
    from ..media.video import decode, probe, stamp
    p = argparse.ArgumentParser(prog='vcrop', description='Crop a region and enlarge it. Box x0,y0,x1,y1 on a 0-1000 scale '
                                '(fractions also work). Crop an image file, or the native-resolution frame at --t.')
    p.add_argument('image', nargs='?', default='')
    p.add_argument('--t', type=_time, default=None, help='crop the video frame at this time instead of an image file')
    p.add_argument('--box', required=True)
    p.add_argument('--side', type=int, default=1024, help='longest side of the output')
    p.add_argument('-o', '--out', default='')
    p.add_argument('--video', default=VIDEO)
    a = p.parse_args(argv)
    if a.t is not None:
        array, stamps = decode(a.video, [min(max(a.t, 0.0), probe(a.video).duration - 0.05)])
        image, label = Image.fromarray(array[0]), mmss(stamps[0])
    elif a.image:
        image, label = Image.open(a.image).convert('RGB'), ''
    else:
        p.error('give an image file or --t')
    box = [float(v) for v in a.box.replace(' ', '').split(',')]
    if len(box) != 4:
        p.error('--box needs four numbers')
    if max(box) > 1.0:
        box = [v / 1000.0 for v in box]
    x0, y0, x1, y1 = [min(max(v, 0.0), 1.0) for v in box]
    if x1 - x0 < 0.01 or y1 - y0 < 0.01:
        p.error('box too small')
    w, h = image.size
    region = image.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))
    scale = a.side / max(region.size)
    region = region.resize((max(1, round(region.width * scale)), max(1, round(region.height * scale))), Image.BICUBIC)
    if label:
        stamp(region, label)
    out = a.out or (f'crop_{label.replace(":", "-")}.jpg' if label else f'crop_{os.path.splitext(os.path.basename(a.image))[0]}.jpg')
    if os.path.splitext(out)[1].lower() not in ('.jpg', '.jpeg', '.png'):
        out += '.jpg'
    region.save(out, quality=92)
    print(f'{out}: {region.width}x{region.height}' + (f', frame at {label}' if label else ''))
