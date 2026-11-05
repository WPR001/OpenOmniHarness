"""WorldSense (3,172 questions on 1,662 videos), rewritten from VLMEvalKit `vlmeval/dataset/worldsense.py`
(`build_prompt` with video_llm=True) and `dataset/utils/worldsense.py` (answer extraction).

VLMEvalKit sends one user message [SYS, video, (audio), template, 'Question: ...\\nAnswer: '], whose text parts the
chat template joins with nothing in between. Here SYS is the Sample prefix (before the media) and the template plus
question form the prompt (after the media), which renders to the same text.

Settings (--bench-opt setting=...), matching the WorldSense_*_audio / _subs / plain variants:
- audio  (default) FRAMES_TMPL_AUDIO. VLMEvalKit adds the extracted wav as a second audio item while Qwen3-Omni
         already hears the track through use_audio_in_video; here the audio is given once (the video's own track).
- nosub  FRAMES_TMPL_NOSUB (combine with --audio-mode none for frames only).
- sub    FRAMES_TMPL_SUB with the subtitles of WorldSense/subtitles/<video>.srt, read the VLMEvalKit way: one
         lookup per second of video, the cue showing at that moment, repeats kept.
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from ..sample import Sample
from ..media.video import probe

from .base import Benchmark

BASE_SYS = 'Carefully watch this video and pay attention to every detail. '
SYS = BASE_SYS + 'Based on your observations, select the best option that accurately addresses the question.'

FRAMES_TMPL_NOSUB = """
These are the frames of a video. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

FRAMES_TMPL_SUB = """
These are the frames of a video. \
This video's subtitles are listed below:
{}
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

FRAMES_TMPL_AUDIO = """
These are the frames of a video and the corresponding audio. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

# Kept as in VLMEvalKit, including the two pairs that a missing comma glues together.
ANSWER_PREFIXES = [
    'The best answer is',
    'The correct answer is',
    'The answer is',
    'The answer',
    'The best option is'
    'The correct option is',
    'Best answer:'
    'Best option:',
    'Answer:',
    'Option:',
]


def extract_characters_regex(s):
    """VLMEvalKit dataset/utils/worldsense.py. '' when no letter can be read."""
    s = s.strip()
    for answer_prefix in ANSWER_PREFIXES:
        s = s.replace(answer_prefix, '')
    if len(s.split()) > 10 and not re.search('[ABCD]', s):
        return ''
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ''
    return matches[0]


def _srt_ms(stamp):
    h, m, rest = stamp.strip().split(':')
    sec, ms = rest.replace('.', ',').split(',')
    return ((int(h) * 60 + int(m)) * 60 + int(sec)) * 1000 + int(ms)


def read_srt(path):
    """(start_ms, end_ms, text) cues; multi-line cue text joined by spaces (pysubs2 \\N then replaced by ' ')."""
    cues = []
    with open(path, encoding='utf-8', errors='replace') as f:
        blocks = re.split(r'\n\s*\n', f.read().replace('\r\n', '\n').strip())
    for block in blocks:
        lines = [x for x in block.split('\n') if x.strip()]
        k = next((i for i, x in enumerate(lines) if '-->' in x), None)
        if k is None:
            continue
        start, _, end = lines[k].partition('-->')
        try:
            cues.append((_srt_ms(start), _srt_ms(end.split()[0]), ' '.join(lines[k + 1:])))
        except (ValueError, IndexError):
            continue
    return cues


def subtitles_text(video_path, srt_path):
    """VLMEvalKit build_prompt with video_llm=True: frame indices 0, int(fps), 2*int(fps), ... (one per second)."""
    if not os.path.exists(srt_path):
        return ''
    info = probe(video_path)
    cues = read_srt(srt_path)
    step = max(1, int(info.fps))
    lines = []
    for frame in range(0, max(info.num_frames, 1) - 1, step):
        cur = round(frame / info.fps * 1000)
        text = next((t for s, e, t in cues if s < cur < e), '')
        if text.strip():
            lines.append(text)
    return '\n'.join(lines)


class WorldSense(Benchmark):
    name = 'worldsense'
    subdir = 'WorldSense'
    dims = ('duration', 'domain', 'task_domain', 'task_type', 'audio_class')
    options = {'setting': 'audio'}

    def load(self):
        setting = self.opts['setting']
        if setting not in ('audio', 'nosub', 'sub'):
            raise ValueError(f'WorldSense setting must be audio, nosub or sub, got {setting}')
        data = json.load(open(self.path('worldsense_qa.json'), encoding='utf-8'))
        subs = {}
        if setting == 'sub':
            videos = list(data)
            with ThreadPoolExecutor(16) as pool:
                texts = pool.map(lambda v: subtitles_text(self.path('videos', f'{v}.mp4'), self.path('subtitles', f'{v}.srt')), videos)
            subs = dict(zip(videos, texts))
        template = {'audio': FRAMES_TMPL_AUDIO, 'nosub': FRAMES_TMPL_NOSUB}.get(setting)
        samples = []
        for video, entry in data.items():
            head = template if template is not None else FRAMES_TMPL_SUB.format(subs.get(video, ''))
            for task in sorted((k for k in entry if k.startswith('task')), key=lambda k: int(k[4:] or 0)):
                qa = entry[task]
                candidates = list(qa['candidates'])
                question_str = qa['question'] + '\n' + '\n'.join(candidates)
                samples.append(Sample(
                    uid=f'{video}_{task}', video=self.path('videos', f'{video}.mp4'), question=qa['question'],
                    options=candidates, prefix=SYS, prompt=head + 'Question: {}\nAnswer: '.format(question_str),
                    answer=str(qa['answer']).strip().upper(),
                    meta={'video_id': video, 'duration': entry.get('duration'), 'domain': entry.get('domain'),
                          'sub_category': entry.get('sub_category'), 'audio_class': list(entry.get('audio_class') or []),
                          'task_domain': qa.get('task_domain'), 'task_type': qa.get('task_type'), 'setting': setting}))
        return samples

    def extract(self, text, sample):
        # VLMEvalKit falls back to option-text matching on `question`, which holds no options for WorldSense,
        # so under exact_matching an unreadable reply scores 0; returning None does the same.
        return extract_characters_regex(text or '') or None
