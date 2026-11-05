"""Per-question sandbox: a work directory with the muted video and question.md, and a guarded bash runner.

Commands run with cwd and HOME in the work directory and a minimal PATH (the sandbox commands first); before a
command runs, omniharness/rules checks it (the built-in guard and the active optional rules). Every command is logged.
"""
import os
import re
import shutil
import subprocess
import sys
import uuid

from .. import rules
from ..commands import BIN
from ..config import ASR_CACHE, MUTED_CACHE, REPO
from ..media.transcript import store_path, video_key
from ..media.video import muted_copy


class Sandbox:
    def __init__(self, sample, cfg, asr_cache=ASR_CACHE, muted_cache=MUTED_CACHE):
        self.cfg = cfg
        name = re.sub(r'[^\w.-]', '_', sample.uid)[:80]
        self.dir = os.path.join(cfg.sandbox_root, f'{name}_{uuid.uuid4().hex[:6]}')
        os.makedirs(self.dir)
        source = sample.audio or sample.video
        self.video = muted_copy(sample.video, muted_cache, video_key(sample.video))
        os.symlink(self.video, os.path.join(self.dir, 'video.mp4'))
        self.asr_json = store_path(asr_cache, source)
        with open(os.path.join(self.dir, 'question.md'), 'w', encoding='utf-8') as f:
            f.write(sample.question + '\n\n' + '\n'.join(sample.options) + '\n')
        self.write_transcript()
        self.env = {
            'PATH': f'{BIN}:{os.path.dirname(sys.executable)}:/usr/bin:/bin',
            'HOME': self.dir, 'TMPDIR': self.dir, 'PYTHONPATH': REPO, 'LANG': 'C.UTF-8',
            'OH_VIDEO': 'video.mp4', 'OH_ASR_JSON': self.asr_json,
            'OMP_NUM_THREADS': '2', 'MPLBACKEND': 'Agg',
            'TORCHINDUCTOR_CACHE_DIR': os.path.join(cfg.sandbox_root, '.torchinductor'),   # keep caches out of the work dir
        }
        self.log = []

    def write_transcript(self):
        """asr.txt (lines and no-word stretches in time order) and asr_words.tsv (start, end, word) from the
        precomputed transcript, so the model can read them with cat, grep or python."""
        from ..media.transcript import Transcript
        if not os.path.exists(self.asr_json):
            text, words = 'No transcript is available for this video.\n', ''
        else:
            tr = Transcript(self.asr_json)
            if not tr.d['has_audio']:
                text, words = 'This video has no audio track.\n', ''
            else:
                text = f'# {tr.info()}\n' + tr.render() + '\n'
                words = ('start\tend\tword\n' + ''.join(f'{a:.2f}\t{b:.2f}\t{w}\n' for a, b, w in tr.d['words'])
                         if tr.d['words'] else '')
        with open(os.path.join(self.dir, 'asr.txt'), 'w', encoding='utf-8') as f:
            f.write(text)
        if words:
            with open(os.path.join(self.dir, 'asr_words.tsv'), 'w', encoding='utf-8') as f:
                f.write(words)

    def path(self, name):
        """Resolve a file name given by the model; it must stay inside the sandbox."""
        full = os.path.realpath(os.path.join(self.dir, name))
        if not full.startswith(os.path.realpath(self.dir) + os.sep):
            raise ValueError(f'{name} is outside the sandbox')
        return full

    def run(self, command):
        reason = rules.refuse_command(self.cfg, command)
        if reason:
            self.log.append({'command': command, 'denied': reason})
            return f'[not run] {reason}', None
        try:
            proc = subprocess.run(['bash', '-c', command], cwd=self.dir, env=self.env, capture_output=True, text=True,
                                  timeout=self.cfg.bash_timeout)
            out, code = (proc.stdout or '') + (('\n[stderr]\n' + proc.stderr) if proc.stderr.strip() else ''), proc.returncode
        except subprocess.TimeoutExpired as e:
            out, code = (e.stdout or '') if isinstance(e.stdout, str) else '', None
            out += f'\n[timed out after {self.cfg.bash_timeout} s]'
        self.log.append({'command': command, 'exit': code})
        return self.clip(out.strip()), code

    def clip(self, text):
        limit = self.cfg.output_chars
        if len(text) <= limit:
            return text
        return text[:limit // 2] + f'\n... [{len(text) - limit} characters cut] ...\n' + text[-limit // 2:]

    def close(self):
        if not self.cfg.keep_sandbox:
            shutil.rmtree(self.dir, ignore_errors=True)
