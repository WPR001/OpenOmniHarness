"""watch: show a segment of the video (at most 30 s) as video input, within the watch budget."""
from ..media.video import frames_for_video, probe
from ..parsing import mmss, parse_time
from .base import ToolInputError, register


@register('watch',
          'See a segment of the video (at most 30 s, at most 8 fps, up to 32 frames) as a video, with its '
          'time positions, for motion, direction and the order of events.',
          {'type': 'object', 'properties': {
              't0': {'type': 'string', 'description': 'start, seconds or mm:ss'},
              't1': {'type': 'string', 'description': 'end, seconds or mm:ss'},
              'fps': {'type': 'number', 'description': 'frames per second, 1-8 (default 4)'}},
           'required': ['t0', 't1']})
def watch(box, args):
    if box.watches >= box.cfg.max_watch:
        raise ToolInputError(f'the watch budget ({box.cfg.max_watch}) is used up')
    try:
        t0, t1 = parse_time(args.get('t0', 0)), parse_time(args.get('t1', 0))
    except ValueError as e:
        raise ToolInputError(f'bad time: {e}')
    duration = probe(box.sb.video).duration
    t0 = min(max(t0, 0.0), max(duration - 0.5, 0.0))
    t1 = min(max(t1, t0 + 0.5), duration, t0 + 30.0)
    fps = min(max(float(args.get('fps') or 4), 1.0), 8.0)
    frames, metadata, stamps = frames_for_video(box.sb.video, t0, t1, fps)
    box.watches += 1
    return [{'type': 'text', 'text': f'video {mmss(stamps[0])}-{mmss(stamps[-1])}, {len(stamps)} frames:'},
            {'type': 'video', 'video': frames, 'metadata': metadata}]
