"""OmniVideoBench (1,000 questions on 628 videos, up to 30 min). Prompt and system text from the official
`eval/qwen3_omni_eval.py`; the gold label is `correct_option` (as in its dataloader)."""
import os
import re
import string

import pandas as pd

from ..sample import Sample

from .base import QWEN_SYSTEM, Benchmark, letter_options


def to_seconds(text):
    seconds = 0.0
    for part in str(text).split(':'):
        seconds = seconds * 60 + float(part)
    return seconds


def duration_bucket(seconds):
    for limit, name in ((60, '<1min'), (300, '1-5min'), (600, '5-10min'), (1800, '10-30min')):
        if seconds < limit:
            return name
    return '>30min'


def ovb_prompt(question, options):
    options_text = '\n'.join(options)
    return ('You are given a video. Based on the content of the video, answer the following question:\n\n'
            f'Question:\n{question}\n\n'
            f'Options:\n{options_text}\n\n'
            "Answer with the option's letter directly(e.g., A, B, C, or D)."
            'If your access to the video content is limited, at least one option that is more likely than the others must be chosen.'
            "Mustn't give any other reason for can not choose!")


class OmniVideoBench(Benchmark):
    name = 'omnivideobench'
    subdir = 'OmniVideoBench'
    dims = ('question_type', 'audio_type', 'video_type', 'duration_bucket')

    def load(self):
        df = pd.read_parquet(self.path('data.parquet'))
        samples = []
        for k, row in enumerate(df.to_dict('records')):
            options = [str(o) for o in row['options']]
            seconds = to_seconds(row['duration'])
            name = os.path.splitext(os.path.basename(str(row['video'])))[0]   # the field reads 'videos/video_1.mp4'
            samples.append(Sample(
                uid=f'{k:04d}_{name}', video=self.path('videos', f'{name}.mp4'), question=row['question'],
                options=letter_options(options), prompt=ovb_prompt(row['question'], options), system=QWEN_SYSTEM,
                answer=str(row['correct_option']).strip().upper(),
                meta={'video_id': name, 'question_type': row.get('question_type'), 'audio_type': row.get('audio_type'),
                      'video_type': row.get('video_type'), 'duration': row.get('duration'),
                      'duration_bucket': duration_bucket(seconds), 'answer_text': row.get('answer')}))
        return samples

    def extract(self, text, sample):
        """utils/utils.py: extract_model_answer, then clean_text(answer) == clean_text(correct_option), which only
        holds when the cleaned reply is a single letter."""
        answer = (text or '').split('assistant')[-1].strip()
        for pattern in (r'/box\{([^}]+)\}', r'\\boxed\{([^}]+)\}'):
            m = re.search(pattern, answer)
            if m:
                answer = m.group(1).strip()
        cleaned = clean_text(answer)
        return cleaned.upper() if len(cleaned) == 1 and cleaned.upper() in sample.letters else None


def clean_text(text):
    """utils/utils.py clean_text."""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    box = re.search(r'/box\{([^}]+)\}', text)
    if box:
        text = box.group(1)
    if text and len(text) > 1 and text[1] == '.':
        text = text[0]
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())
