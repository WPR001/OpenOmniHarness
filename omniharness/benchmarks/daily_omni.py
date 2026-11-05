"""Daily-Omni (1,197 questions on 684 clips of 30 s / 60 s). Prompt from the official Qwen3-Omni test script
(`test_model/Qwen3-Omni/testmodel.py`, input_mode=all). Each clip ships an mp4 with audio and a separate wav;
the mp4's own track is used so that audio stays aligned with the frames."""
import json
import re

from ..sample import Sample

from .base import Benchmark


def daily_prompt(question, choices):
    return ('Your task is to accurately answer multiple-choice questions based on the given video and audio together.\n'
            'Select the single most accurate answer from the given choices.\n'
            f'Question: {question}\n'
            f'Choices: {choices}\n'
            'Your answer should be a capital letter representing your choice: A, B, C, or D. Don\'t generate any other text.\n')


class DailyOmni(Benchmark):
    name = 'daily_omni'
    subdir = 'Daily-Omni'
    dims = ('qa_type', 'video_category', 'video_duration')

    def load(self):
        items = json.load(open(self.path('qa.json'), encoding='utf-8'))
        samples = []
        for k, item in enumerate(items):
            vid = item['video_id']
            samples.append(Sample(
                uid=f'{k:04d}_{vid}', video=self.path('Videos', vid, f'{vid}_video.mp4'), question=item['Question'],
                options=list(item['Choice']), prompt=daily_prompt(item['Question'], item['Choice']),
                answer=item['Answer'].strip().upper(),
                meta={'video_id': vid, 'qa_type': item.get('Type'), 'video_category': item.get('video_category'),
                      'video_duration': item.get('video_duration'),
                      'wav': self.path('Videos', vid, f'{vid}_audio.wav')}))
        return samples

    def extract(self, text, sample):
        """testmodel.py extract_choice_letter: the first character if it is A-D, else the first standalone A-D."""
        s = (text or '').strip()
        if not s:
            return None
        if s[0] in 'ABCD':
            return s[0]
        m = re.search(r'\b([ABCD])\b', s)
        return m.group(1) if m else None
