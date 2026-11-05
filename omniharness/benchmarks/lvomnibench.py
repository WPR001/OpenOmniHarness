"""LVOmniBench (1,014 questions on 275 videos of 10-90 min). Prompt from the official OmniEval
`framework/prompt_utils.py` and system text from `models/qwen3_omni.py`."""
import ast
import json
import re

from ..sample import Sample

from .base import QWEN_SYSTEM, Benchmark

SYSTEM = QWEN_SYSTEM + ' Please analyze the video carefully and select the most appropriate answer from the given options.'


def lv_prompt(question, options):
    options_text = '\n'.join(options)
    return (f'Question: {question}\n'
            f'Options:\n{options_text}\n\n'
            'Select the best answer from the options above. '
            'Directly provide the letter representing your choice and nothing else. '
            'Do not include the full text of the option, do not provide any explanation.')


class LVOmniBench(Benchmark):
    name = 'lvomnibench'
    subdir = 'LVOmniBench'
    dims = ('question_type', 'audio_type', 'difficulty', 'video_category')

    def load(self):
        items = json.load(open(self.path('data.json'), encoding='utf-8'))
        samples = []
        for item in items:
            options = item['options']
            if isinstance(options, str):
                options = ast.literal_eval(options)
            options = [str(o) for o in options]
            samples.append(Sample(
                uid=f'{item["question_id"]}_{item["video_id"]}', video=self.path('videos', f'{item["video_id"]}.mp4'),
                question=item['question'], options=options, prompt=lv_prompt(item['question'], options), system=SYSTEM,
                answer=str(item['correct_option']).strip().upper(),
                meta={'video_id': item['video_id'], 'question_type': item.get('question_type'),
                      'audio_type': item.get('audio_type'), 'difficulty': item.get('difficulty'),
                      'video_category': item.get('video_category'), 'duration': item.get('duration')}))
        return samples

    def extract(self, text, sample, valid_options=('A', 'B', 'C', 'D', 'E', 'F')):
        """OmniEval framework/prompt_utils.py extract_choice_answer."""
        if not text:
            return None
        resp_upper = text.upper().strip()
        for opt in valid_options:
            if resp_upper.startswith(opt) and (len(resp_upper) == 1 or resp_upper[1] in ['.', ')', ':', ' ', ',']):
                return opt
        answer_pattern = re.search(r'(?:THE\s+)?ANSWER\s*(?:IS|:)\s*([A-F])', resp_upper)
        if answer_pattern and answer_pattern.group(1) in valid_options:
            return answer_pattern.group(1)
        for opt in valid_options:
            if re.search(rf'\b{opt}\b', resp_upper):
                return opt
        for char in resp_upper:
            if char in valid_options:
                return char
        return None
