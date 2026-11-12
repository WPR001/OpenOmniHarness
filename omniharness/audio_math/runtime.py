"""Per-question audio assets, bounded CPU workers and their numerical observations."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from ..sandbox import Sandbox
from ..tools.base import ToolInputError
from ..tools.listen_audio import _time


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


class AudioMathSandbox(Sandbox):
    """Visual/ASR sandbox with a lazily prepared copy of the source audio stream."""
    def __init__(self, sample, cfg, **kwargs):
        super().__init__(sample, cfg, **kwargs)
        self.dir = os.path.abspath(self.dir)
        self.env.update(HOME=self.dir, TMPDIR=self.dir)
        self.audio_file = Path(self.dir) / 'audio_source.mka'
        self.audio_meta = {'version': cfg.harness_version, 'asset': self.audio_file.name,
                           'status': 'available_on_first_audio_call',
                           'time_reference': 'original_source_seconds', 'visual_file': 'video.mp4',
                           'visual_file_is_muted': True}
        dump(Path(self.dir) / 'audio_source.json', self.audio_meta)
        # Include the installed decoder locations (e.g. Homebrew) in the worker's minimal PATH.
        decoder_dirs = dict.fromkeys(str(Path(p).parent) for p in
                                    (shutil.which('ffmpeg'), shutil.which('ffprobe')) if p)
        if decoder_dirs:
            self.env['PATH'] += os.pathsep + os.pathsep.join(decoder_dirs)

    def prepare_audio(self):
        if self.audio_file.exists():
            return
        tmp = self.audio_file.with_name('audio_source.partial.mka')
        try:
            proc = subprocess.run(
                ['ffmpeg', '-v', 'error', '-nostdin', '-y', '-copyts', '-start_at_zero',
                 '-i', self._audio_source, '-map', '0:a:0', '-vn', '-sn', '-dn', '-c:a', 'copy', str(tmp)],
                capture_output=True, timeout=self.cfg.audio_analysis_timeout, env=self.env)
            if proc.returncode:
                raise ValueError('could not copy the source audio track; check that the source contains audio')
            proc = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(tmp)],
                capture_output=True, timeout=15, check=True, env=self.env)
            meta = json.loads(proc.stdout)
            tmp.replace(self.audio_file)
            self.audio_meta.update(status='ready', stream_copy=True,
                                   stream_start_time=meta['streams'][0].get('start_time'),
                                   format_start_time=meta.get('format', {}).get('start_time'),
                                   note='Analysis tools use the original source timeline; inspect these start times '
                                        'when using the supplementary MKA directly.')
        except Exception:
            tmp.unlink(missing_ok=True)
            self.audio_meta['status'] = 'unavailable'
            raise
        finally:
            dump(Path(self.dir) / 'audio_source.json', self.audio_meta)


class AudioMathSession:
    def __init__(self, sandbox, cfg):
        self.sb, self.cfg = sandbox, cfg
        self.root = Path(sandbox.dir) / 'audio_math'
        self.root.mkdir(exist_ok=True)
        self.log, self.cache = [], {}

    def interval(self, args):
        if not isinstance(args, dict) or not {'t0', 't1'} <= args.keys():
            raise ToolInputError('give t0 and t1')
        a, b = _time(args['t0']), _time(args['t1'])
        if not 0 <= a < b:
            raise ToolInputError('require 0 <= t0 < t1')
        if b - a > self.cfg.audio_analysis_seconds:
            raise ToolInputError(f'window exceeds {self.cfg.audio_analysis_seconds:g}s; select a shorter interval')
        return a, b

    def analyze(self, args):
        if not isinstance(args, dict) or set(args) - {'mode', 't0', 't1', 'params'}:
            raise ToolInputError('use mode, t0, t1 and optional params')
        if args.get('mode') not in ('overview', 'events', 'spectrum', 'rhythm', 'compare'):
            raise ToolInputError('mode must be overview, events, spectrum, rhythm, or compare')
        params = args.get('params', {})
        if not isinstance(params, dict):
            raise ToolInputError('params must be an object')
        keys = {'channel', 'min_peak_dbfs', 'envelope_smooth_hz', 'band_hz', 'window_ms', 'hop_ms',
                'min_gap_ms', 'threshold_above_local_floor_db', 'comparison_window'}
        if set(params) - keys:
            raise ToolInputError('unknown params: ' + ', '.join(sorted(set(params) - keys)))
        normalized = dict(args, params=dict(params))
        if args['mode'] == 'compare':
            other = params.get('comparison_window')
            if not isinstance(other, list) or len(other) != 2:
                raise ToolInputError('compare requires params.comparison_window=[start,end]')
            normalized['params']['comparison_window'] = list(self.interval(dict(zip(('t0', 't1'), other))))
        return self._run('analyze', normalized)

    def python(self, args):
        if not isinstance(args, dict) or set(args) != {'t0', 't1', 'code'}:
            raise ToolInputError('use t0, t1 and code')
        if not isinstance(args['code'], str) or not args['code'].strip() or len(args['code']) > 24000:
            raise ToolInputError('code must contain 1..24000 characters')
        return self._run('python', args)

    def _run(self, kind, args):
        a, b = self.interval(args)
        args = dict(args, t0=a, t1=b)
        key = hashlib.sha256(json.dumps([kind, args], sort_keys=True, allow_nan=False).encode()).hexdigest()
        name = 'audio_analyze' if kind == 'analyze' else 'audio_python'
        if key in self.cache:
            result = dict(self.cache[key], cached=True)
            self.log.append({'tool': name, 'args': args, 'status': 'ok', 'cached': True,
                             'analysis_id': result['analysis_id'], 'result_file': result['result_file'], 'seconds': 0.0})
            return self.content(result)
        used = sum(not item.get('cached') for item in self.log)
        if used >= self.cfg.audio_analysis_calls:
            raise ToolInputError('audio analysis call budget exhausted; use existing evidence or answer')
        directory = self.root / f'{used + 1:03d}_{kind}'
        directory.mkdir()
        entry = {'tool': name, 'args': args, 'analysis_id': directory.name, 'status': 'running', 'cached': False}
        self.log.append(entry)
        started = time.monotonic()
        try:
            self.sb.prepare_audio()
            request = {'audio_path': self.sb._audio_source, 't0': a, 't1': b,
                       'mode': args.get('mode'), 'params': args.get('params', {}), 'code': args.get('code', ''),
                       'output_dir': str(directory.resolve()), 'max_seconds': self.cfg.audio_analysis_timeout}
            env = dict(self.sb.env)
            env.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MPLCONFIGDIR=str(directory / '.mplconfig'), MPLBACKEND='Agg')
            worker = 'python_worker' if kind == 'python' else 'analysis_worker'
            # The request contains the host's source path; keep it outside the question's sandbox.
            with tempfile.TemporaryDirectory(prefix='oh_audio_request_') as private:
                request_file = Path(private) / 'request.json'
                dump(request_file, request)
                with (directory / 'worker.log').open('w') as log:
                    proc = subprocess.Popen([sys.executable, '-m', 'omniharness.audio_math.' + worker,
                                             str(request_file)], cwd=directory, env=env,
                                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        proc.wait(timeout=self.cfg.audio_analysis_timeout)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        proc.wait()
                        raise TimeoutError(f'audio CPU worker exceeded {self.cfg.audio_analysis_timeout:g}s') from None
            result_file = directory / 'result.json'
            if not result_file.exists():
                raise RuntimeError('audio worker exited without results; inspect ' +
                                   str((directory / 'worker.log').relative_to(self.sb.dir)))
            result = json.loads(result_file.read_text())
            if not isinstance(result, dict):
                raise ValueError('audio worker result must be a JSON object')
            result['analysis_id'] = directory.name
            result['result_file'] = str(result_file.relative_to(self.sb.dir))
            artifacts = result.get('artifacts', [])
            if isinstance(artifacts, dict):
                artifacts = [{'name': k, 'path': v} for k, v in artifacts.items()]
            result['artifacts'] = []
            for item in artifacts:
                item = {'path': item} if isinstance(item, str) else dict(item)
                path = (directory / item['path']).resolve()
                if directory.resolve() in path.parents and path.is_file():
                    item['path'] = str(path.relative_to(Path(self.sb.dir).resolve()))
                    result['artifacts'].append(item)
            entry['status'] = result.get('status', 'ok')
            if proc.returncode and entry['status'] == 'ok':
                result.update(status='error', error=f'audio worker exited with code {proc.returncode}')
                entry['status'] = 'error'
            if entry['status'] == 'ok':
                self.cache[key] = result
        except Exception as exc:
            result = {'status': 'error', 'analysis_id': directory.name,
                      'error': f'{type(exc).__name__}: {exc}',
                      'result_file': str((directory / 'result.json').relative_to(self.sb.dir)), 'artifacts': []}
            entry['status'] = 'error'
        entry['seconds'] = round(time.monotonic() - started, 3)
        entry['result'] = result
        entry['result_file'] = result['result_file']
        # Persist the same normalized paths returned to the model.
        dump(directory / 'result.json', result)
        content = self.content(result)
        entry['observation'] = content[0]['text']
        dump(directory / 'trace.json', entry)
        return content

    def content(self, result):
        text = json.dumps(result, ensure_ascii=False, allow_nan=False)
        if len(text) > self.cfg.output_chars:
            keys = ('status', 'analysis_id', 'summary', 'error', 'result_file', 'artifacts', 'window',
                    'time_reference', 'cached', 'sample_rate_hz', 'channels', 'decoded_duration_s')
            summary = {k: result[k] for k in keys if k in result}
            summary['details'] = 'Read result_file with bash for the complete numerical results.'
            text = json.dumps(summary, ensure_ascii=False, allow_nan=False)
        return [{'type': 'text', 'text': text}]
