"""Tool lifecycle and multi-turn observations with local fixture workers."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omniharness import rules
from omniharness.agent.loop import Agent
from omniharness.audio_math.runtime import AudioMathSession
from omniharness.backends import Output
from omniharness.config import AgentConfig
from omniharness.sample import Sample
from omniharness.tools import ToolBox, specs


class FixtureSandbox:
    def __init__(self, root, cfg):
        self.dir, self.cfg = root, cfg
        self._audio_source = '/host/private/source.mp4'
        self.env = dict(os.environ)
        self.log = []
        self.prepared = 0

    def prepare_audio(self):
        self.prepared += 1

    def close(self):
        pass

    def path(self, name):
        return str(Path(self.dir) / name)


class FixtureProcess:
    returncode = 0
    pid = 12345

    def __init__(self, argv, **kwargs):
        request = json.loads(Path(argv[-1]).read_text())
        root = Path(request['output_dir'])
        from PIL import Image
        Image.new('RGB', (24, 16), '#335577').save(root / 'spectrum.png')
        result = {'status': 'ok', 'window': [request['t0'], request['t1']],
                  'result': {'frequency_hz': 440}, 'artifacts': {'spectrum': 'spectrum.png'}}
        (root / 'result.json').write_text(json.dumps(result))

    def wait(self, timeout=None):
        return 0


class AudioMathRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = AgentConfig(harness_version='audio-math-v1', audio_analysis_calls=1, keep_sandbox=True)
        self.sb = FixtureSandbox(self.tmp.name, self.cfg)
        self.box = ToolBox(self.sb, self.cfg)

    def test_cached_calls_preserve_paths_and_leave_budget_for_new_work(self):
        args = {'mode': 'spectrum', 't0': 0, 't1': 1}
        with patch('omniharness.audio_math.runtime.subprocess.Popen', FixtureProcess):
            first, trace = self.box.run('audio_analyze', args)
            second, _ = self.box.run('audio_analyze', dict(args, t0='00:00', t1='00:01'))
        payload = json.loads(first[0]['text'])
        self.assertEqual(payload['status'], 'ok')
        self.assertTrue(json.loads(second[0]['text'])['cached'])
        self.assertEqual(self.sb.prepared, 1)
        self.assertEqual(json.loads(Path(self.sb.path(payload['result_file'])).read_text()), payload)
        content, image_trace = self.box.run('view', {'paths': [payload['artifacts'][0]['path']]})
        self.assertEqual(image_trace['images'], 1)
        self.assertEqual(next(x for x in content if x['type'] == 'image')['image'].size, (24, 16))
        self.assertEqual(trace['output'], first[0]['text'])
        self.assertNotIn(self.sb._audio_source, json.dumps(trace))
        failed, _ = self.box.run('audio_analyze', dict(args, t1=2))
        self.assertIn('budget exhausted', failed[0]['text'])

    def test_invalid_compare_window_never_starts_worker(self):
        with patch('omniharness.audio_math.runtime.subprocess.Popen') as worker:
            content, _ = self.box.run('audio_analyze', {'mode': 'compare', 't0': 0, 't1': 1,
                                                       'params': {'comparison_window': [0, 121]}})
        self.assertIn('exceeds', content[0]['text'])
        worker.assert_not_called()

    def test_audio_commands_only_change_in_enabled_version(self):
        command = 'ffmpeg -i audio_source.mka -vn clip.wav'
        self.assertIsNotNone(rules.refuse_command(AgentConfig(), command))
        self.assertIsNone(rules.refuse_command(self.cfg, command))
        self.assertIsNotNone(rules.refuse_command(self.cfg, 'curl https://example.com'))

    def test_visual_backend_receives_audio_measurements_then_plot(self):
        backend = SimpleNamespace(native_tools=True, supports_raw_audio=False)
        sample = Sample('test', 'source.mp4', 'Which note?', ['A. 440 Hz', 'B. 880 Hz'], '', answer='A')
        self.assertIn('audio_python', [s['function']['name'] for s in specs(self.cfg)])
        with patch('omniharness.audio_math.runtime.AudioMathSandbox', return_value=self.sb), \
                patch('omniharness.agent.loop.first_message_facts', return_value=('00:02', '64x64', '1 line')), \
                patch('omniharness.audio_math.runtime.subprocess.Popen', FixtureProcess):
            flow = Agent(backend, self.cfg)._flow(sample, {})
            next(flow)
            request = flow.send(Output('', tool_calls=[{'id': 'measure', 'name': 'audio_python', 'arguments': {
                't0': 0, 't1': 1, 'code': 'result = {"frequency_hz": 440}'}}]))
            observation = request.messages[-1]
            self.assertEqual((observation['name'], observation['tool_call_id']), ('audio_python', 'measure'))
            result = json.loads(observation['content'][0]['text'])
            request = flow.send(Output('', tool_calls=[{'id': 'plot', 'name': 'view', 'arguments': {
                'paths': [result['artifacts'][0]['path']]}}]))
            self.assertTrue(any(c['type'] == 'image' for c in request.messages[-1]['content']))
            self.assertFalse(any(c['type'] == 'audio' for c in request.messages[-1]['content']))
            with self.assertRaises(StopIteration) as end:
                flow.send(Output('', tool_calls=[{'id': 'final', 'name': 'answer', 'arguments': {'answer': 'A'}}]))
        record = end.exception.value
        self.assertTrue(record['correct'])
        self.assertEqual(record['audio_math']['calls'][0]['args']['code'], 'result = {"frequency_hz": 440}')
        self.assertEqual(record['harness_version'], 'audio-math-v1')


if __name__ == '__main__':
    unittest.main()
