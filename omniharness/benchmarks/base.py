"""Benchmark loader interface: read the official annotation files and emit Samples with the benchmark's own prompt,
and read answers with the benchmark's own extraction rule (`extract`)."""
import os

from ..config import REPO
from ..parsing import extract_letter

DATA_ROOT = os.environ.get('OMNI_BENCH_ROOT', os.path.join(REPO, 'data'))   # one folder per benchmark

# System prompt Qwen3-Omni ships with (used by the OmniVideoBench and LVOmniBench reference scripts).
QWEN_SYSTEM = ('You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving '
               'auditory and visual inputs, as well as generating text and speech.')


class Benchmark:
    name = ''
    subdir = ''
    dims = ()            # meta keys to break accuracy down by; list-valued keys count once per value
    options = {}         # benchmark-specific settings accepted through --bench-opt key=value, with defaults

    def __init__(self, root=None, **opts):
        self.root = root or os.path.join(DATA_ROOT, self.subdir)
        unknown = set(opts) - set(self.options)
        if unknown:
            raise ValueError(f'{self.name} does not take options {sorted(unknown)}; known: {self.options}')
        self.opts = {**self.options, **opts}

    def load(self):
        raise NotImplementedError

    def extract(self, text, sample):
        """Answer letter from a reply, by the benchmark's official rule (default: the lenient shared one)."""
        return extract_letter(text, sample.letters, sample.options)

    def path(self, *parts):
        return os.path.join(self.root, *parts)


def letter_options(options):
    """['A. x', 'B. y'] from a dict {'A': 'x'} or a list that may lack the letters."""
    if isinstance(options, dict):
        return [f'{k}. {v}' for k, v in sorted(options.items())]
    out = []
    for k, option in enumerate(options):
        text = str(option).strip()
        letter = chr(ord('A') + k)
        out.append(text if text[:1] == letter and text[1:2] in '.):' else f'{letter}. {text}')
    return out
