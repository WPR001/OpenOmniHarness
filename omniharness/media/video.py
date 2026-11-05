"""Video helpers for the sandbox commands and the `watch` tool: probing, the muted copy, exact-time frame decoding,
contact sheets and crops. Heavy imports are local so that the ASR stage can import the package."""
import fcntl
import json
import math
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache

from ..parsing import mmss


@dataclass(frozen=True)
class Info:
    path: str
    duration: float
    fps: float
    num_frames: int
    width: int
    height: int
    has_audio: bool


def _rate(text):
    num, _, den = (text or '0/1').partition('/')
    den = float(den or 1)
    return float(num) / den if den else 0.0


@lru_cache(maxsize=4096)
def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)],
                         capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    video = next((s for s in data['streams'] if s.get('codec_type') == 'video'), {})
    audio = any(s.get('codec_type') == 'audio' for s in data['streams'])
    duration = float(data.get('format', {}).get('duration') or video.get('duration') or 0.0)
    fps = _rate(video.get('avg_frame_rate')) or _rate(video.get('r_frame_rate')) or 25.0
    frames = int(video.get('nb_frames') or 0) or int(round(duration * fps))
    return Info(str(path), duration, fps, frames, int(video.get('width') or 0), int(video.get('height') or 0), audio)


def muted_copy(video, cache_dir, key):
    """The video without its audio track (stream copy, no re-encoding), made once per video under a file lock."""
    os.makedirs(cache_dir, exist_ok=True)
    target = os.path.join(cache_dir, f'{key}{os.path.splitext(video)[1] or ".mp4"}')
    if os.path.exists(target):
        return target
    with open(target + '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not os.path.exists(target):
            tmp = target + '.part' + os.path.splitext(target)[1]
            subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', video, '-map', '0:v:0', '-an', '-sn', '-dn',
                            '-c', 'copy', tmp], check=True)
            os.replace(tmp, target)
    return target


def decode(path, times, workers=6):
    """Frames at the given times as uint8 (T, H, W, 3), with their presentation times. Many times are split over
    several decoders in parallel (each seek decodes from the previous keyframe)."""
    if len(times) > 8 and workers > 1:
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor
        order = sorted(range(len(times)), key=lambda i: times[i])
        parts = [order[k::workers] for k in range(workers)]
        parts = [sorted(p, key=lambda i: times[i]) for p in parts if p]
        with ThreadPoolExecutor(len(parts)) as pool:
            done = list(pool.map(lambda p: (p, decode(path, [times[i] for i in p], workers=1)), parts))
        frames, stamps = [None] * len(times), [None] * len(times)
        for p, (array, st) in done:
            for i, f, t in zip(p, array, st):
                frames[i], stamps[i] = f, t
        return np.stack(frames), stamps
    try:
        from torchcodec.decoders import VideoDecoder
        decoder = VideoDecoder(str(path), seek_mode='exact', dimension_order='NHWC')
        end = (decoder.metadata.duration_seconds or probe(path).duration) - 1e-3
        batch = decoder.get_frames_played_at(seconds=[min(max(t, 0.0), max(end, 0.0)) for t in times])
        return batch.data.numpy(), [float(t) for t in batch.pts_seconds]
    except Exception:
        import av
        import numpy as np
        frames, stamps = [], []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            for t in times:
                container.seek(int(max(t, 0) / stream.time_base), stream=stream, backward=True)
                picked = None
                for frame in container.decode(stream):
                    picked = frame
                    if frame.time is not None and frame.time >= t - 0.02:
                        break
                frames.append(picked.to_ndarray(format='rgb24'))
                stamps.append(float(picked.time or t))
        return np.stack(frames), stamps


def _font(size):
    from PIL import ImageFont
    for name in ('DejaVuSans-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def stamp(image, text):
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    draw.text((4, 3), text, fill=(255, 225, 0), font=_font(max(12, image.height // 9)), stroke_width=2, stroke_fill='black')
    return image


def contact_sheet(path, times, width=1440, cols=None):
    from PIL import Image
    array, stamps = decode(path, times)
    n = len(stamps)
    h0, w0 = array.shape[1:3]
    cols = cols or max(1, min(n, math.ceil(math.sqrt(n * 16 / 9 * h0 / max(w0, 1)))))
    rows = math.ceil(n / cols)
    cell_w = max(64, width // cols)
    cell_h = max(36, round(cell_w * h0 / max(w0, 1)))
    sheet = Image.new('RGB', (cols * cell_w, rows * cell_h), 'black')
    for k, (frame, t) in enumerate(zip(array, stamps)):
        cell = stamp(Image.fromarray(frame).resize((cell_w, cell_h), Image.BICUBIC), mmss(t))
        sheet.paste(cell, ((k % cols) * cell_w, (k // cols) * cell_h))
    return sheet, stamps


def frames_for_video(path, t0, t1, fps, max_frames=32, max_side=640):
    """Frames of [t0, t1] at fps for the `watch` tool, plus the metadata Qwen3-VL processors use for timestamps
    (absolute frame indices of the source video, so the model sees true video times)."""
    import numpy as np
    from PIL import Image
    info = probe(path)
    n = int(min(max_frames, max(2, round((t1 - t0) * fps))))
    n -= n % 2
    times = list(np.linspace(t0, t1, n, endpoint=False))
    array, stamps = decode(path, times)
    scale = min(1.0, max_side / max(array.shape[1:3]))
    h = max(32, int(array.shape[1] * scale) // 32 * 32)
    w = max(32, int(array.shape[2] * scale) // 32 * 32)
    array = np.stack([np.asarray(Image.fromarray(f).resize((w, h), Image.BICUBIC)) for f in array])
    metadata = {'fps': info.fps, 'duration': info.duration, 'total_num_frames': info.num_frames,
                'frames_indices': [int(round(t * info.fps)) for t in stamps], 'video_backend': 'torchcodec',
                'do_sample_frames': False}
    return array, metadata, stamps
