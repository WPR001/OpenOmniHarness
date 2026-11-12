"""CPU signal tests with generated samples; no model, API, GPU or FFmpeg needed."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
    from omniharness.audio_math import dsp
except ImportError:
    dsp = None


@unittest.skipIf(dsp is None, 'install requirements-audio-math.txt')
class AudioMathWorkerTests(unittest.TestCase):
    def test_decoder_preserves_channels_rate_and_timeline_padding(self):
        info = {'duration': 2.0, 'sample_rate': 8000, 'channels': 2,
                'audio_start_s': .25, 'audio_duration_s': 1.0}
        original = np.tile(np.array([[.25, -.5]], dtype='<f4'), (8000, 1))
        with patch.object(dsp, 'probe_audio', return_value=info), \
                patch.object(dsp.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, original.tobytes())) as run:
            x, sr = dsp.decode_window('/source.mp4', .1, 1.6)
        self.assertEqual(sr, 8000)
        self.assertEqual(x.shape, (12000, 2))
        np.testing.assert_array_equal(x[:1200], 0)
        np.testing.assert_array_equal(x[1200:9200], original)
        np.testing.assert_array_equal(x[9200:], 0)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index('-ss') + 1], '0.250000000')
        self.assertEqual(command[command.index('-t') + 1], '1.000000000')

    def test_spectrum_and_artifact_timestamps_use_selected_channel(self):
        sr, t0, t1 = 16000, 10.0, 11.0
        t = np.arange(sr) / sr
        x = np.stack((.4 * np.sin(2*np.pi*440*t), .4 * np.sin(2*np.pi*880*t)), axis=1).astype('float32')
        info = {'sample_rate': sr, 'channels': 2, 'duration': 12,
                'time_reference': 'seconds from container start'}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(dsp, 'probe_audio', return_value=info), \
                patch.object(dsp, 'decode_window', return_value=(x, sr)):
            out = Path(directory)
            result = dsp.analyze('/source.mp4', t0, t1, 'spectrum',
                                 {'channel': 'left', 'window_ms': 50}, out)
            self.assertAlmostEqual(result['spectrum']['peaks'][0]['frequency_hz'], 440, delta=1)
            self.assertEqual(result['window'], [10.0, 11.0])
            self.assertEqual(result['source']['channels'], 2)
            self.assertAlmostEqual(result['rms_dbfs'], 20*np.log10(.4/np.sqrt(2)), places=4)
            times = np.genfromtxt(out / 'features.csv', delimiter=',', names=True)['time_s']
            self.assertGreaterEqual(float(times.min()), t0)
            self.assertLessEqual(float(times.max()), t1)
            self.assertTrue((out / result['artifacts']['overview_plot']).stat().st_size > 1000)
            json.dumps(result, allow_nan=False)

    def test_rejects_malformed_parameters_before_decoding(self):
        for params in ([], 'channel=left', {'unsupported': 1}, {'features': ['rms']}):
            with self.subTest(params=params), patch.object(dsp, 'decode_window') as decode:
                with self.assertRaises(ValueError):
                    dsp.analyze('/source.mp4', 0, 1, params=params, output_dir='/unused')
                decode.assert_not_called()
        for value in (True, float('nan'), float('inf'), 10**1000):
            with self.subTest(value=str(value)[:30]), self.assertRaises(ValueError):
                dsp._finite(value, 'test')

    def test_python_worker_writes_measurements_and_plot(self):
        # The worker installs process-local I/O and CPU limits, so execute it in a subprocess.
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            code = '''from scipy import signal
result = {"start": float(t[0]), "end": float(t[-1]), "rate": sr,
          "channels": x.shape[1], "rms": np.sqrt(np.mean(x*x,axis=0))}
save_csv("samples.csv", zip(t[:4],x[:4,0]), header=["time_s","amplitude"])
plt.plot(t, x[:,0])
plt.xlabel("Original time (seconds)")
save_plot("signal.png")
print("analysis complete")
'''
            request = {'audio_path': '/source.mp4', 't0': 12.5, 't1': 12.6,
                       'code': code, 'max_seconds': 30, 'output_dir': str(out)}
            request_file = out / 'request.json'
            request_file.write_text(json.dumps(request))
            runner = '''import json,sys,numpy as np
from omniharness.audio_math import dsp,python_worker
dsp.decode_window=lambda *a:(np.tile(np.array([[.2,.4]],dtype=np.float32),(2205,1)),22050)
python_worker.run(json.load(open(sys.argv[1])))
'''
            repo = Path(__file__).resolve().parents[1]
            env = dict(os.environ, PYTHONPATH=str(repo), MPLCONFIGDIR=str(out / '.mplconfig'),
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            proc = subprocess.run([sys.executable, '-c', runner, str(request_file)], cwd=repo,
                                  env=env, capture_output=True, text=True, timeout=45)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads((out / 'result.json').read_text())
            self.assertEqual(result['status'], 'ok', result)
            self.assertEqual(result['result']['channels'], 2)
            self.assertEqual(result['result']['rate'], 22050)
            self.assertEqual(result['result']['start'], 12.5)
            self.assertAlmostEqual(result['result']['end'], 12.6-1/22050)
            np.testing.assert_allclose(result['result']['rms'], [.2,.4], atol=1e-5)
            self.assertEqual((out / 'analysis_code.py').read_text(), code)
            paths = {a['path'] for a in result['artifacts']}
            self.assertTrue({'samples.csv','signal.png','analysis_code.py'} <= paths)
            self.assertIn('analysis complete', result['stdout'])


if __name__ == '__main__':
    unittest.main()
