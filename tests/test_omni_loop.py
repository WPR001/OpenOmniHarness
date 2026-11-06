"""Offline checks for mode selection and multi-turn protocol preservation; no model/API/GPU access."""
import argparse
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from infer.common import add_common_args, build_config, config_dict
from omniharness.agent import prompts
from omniharness.agent.loop import Agent
from omniharness.backends import Output
from omniharness.config import AGENT_MODEL, AgentConfig


class DummySandbox:
    def __init__(self, sample, cfg):
        self.log = []

    def close(self):
        pass


class DummyTools:
    executed = []

    def __init__(self, sandbox, cfg):
        self.offered, self.images, self.refused = [], 0, []
        self.state = {'audio_listen_calls': []}

    def run(self, name, args):
        self.executed.append(name)
        block = {'type': 'audio', 'audio': b'fake-test-wav', 'mime_type': 'audio/wav', 't0': 1.0, 't1': 2.0}
        self.state['audio_listen_calls'].append({'t0': 1.0, 't1': 2.0, 'status': 'ok'})
        return [{'type': 'text', 'text': 'source 1-2 seconds'}, block], {
            'tool': name, 'args': args, 'output': 'source 1-2 seconds', 'audios': 1}


class ModeTests(unittest.TestCase):
    def config(self, *flags):
        parser = add_common_args(argparse.ArgumentParser())
        with tempfile.TemporaryDirectory() as scratch:
            args = parser.parse_args([*flags, '--scratch', scratch])
            return args, build_config(args)

    def test_default_visual_behavior(self):
        args, cfg = self.config()
        self.assertEqual((args.backend, args.model), ('vllm', AGENT_MODEL))
        self.assertEqual(cfg.tools, ('bash', 'view', 'watch', 'answer'))
        expected = prompts.SYSTEM.format(max_turns=50, max_images=30, max_minutes=30)
        self.assertEqual(prompts.system_prompt(cfg), expected)

    def test_omni_defaults_and_prompt(self):
        args, cfg = self.config('--agent-mode', 'omni')
        self.assertEqual((args.backend, args.model), ('gemini', 'gemini-3.8-flash'))
        self.assertIn('listen_audio', cfg.tools)
        self.assertNotIn('only access to the audio', prompts.system_prompt(cfg))
        self.assertIn('not evidence of silence', prompts.system_prompt(cfg))
        self.assertEqual((cfg.max_listen, cfg.listen_seconds), (4, 30.0))
        self.assertNotIn('api_key', json.dumps(config_dict(args, cfg)).lower())

    def test_omni_audio_can_be_disabled(self):
        _, cfg = self.config('--agent-mode', 'omni', '--max-listen', '0')
        self.assertNotIn('listen_audio', cfg.tools)

    def test_visual_gemini_is_available(self):
        args, cfg = self.config('--backend', 'gemini')
        self.assertEqual(cfg.agent_mode, 'visual')
        self.assertNotIn('listen_audio', cfg.tools)
        self.assertEqual(args.model, 'gemini-3.8-flash')

    def test_invalid_modes_and_budgets(self):
        for flags in [('--agent-mode', 'omni', '--backend', 'hf'),
                      ('--tools', 'listen_audio'), ('--listen-seconds', 'nan'),
                      ('--max-listen', '-1'), ('--api-workers', '0'),
                      ('--backend', 'gemini', '--reasoning-effort', 'xhigh'),
                      ('--backend', 'gemini', '--no-preserve-thinking')]:
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.config(*flags)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.patches = [patch('omniharness.agent.loop.Sandbox', DummySandbox),
                        patch('omniharness.agent.loop.ToolBox', DummyTools),
                        patch('omniharness.agent.loop.first_message_facts', return_value=('01:00', '640x480', 'ASR'))]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        DummyTools.executed = []
        self.cfg = AgentConfig(agent_mode='omni', tools=('listen_audio', 'answer'))
        self.backend = SimpleNamespace(native_tools=True, supports_raw_audio=True)
        self.sample = SimpleNamespace(uid='q', question='Question?', options=['A. One', 'B. Two'],
                                      answer='B', letters=['A', 'B'], meta={})

    def flow(self):
        return Agent(self.backend, self.cfg)._flow(self.sample, {})

    def test_native_audio_followup_keeps_call_id_and_provider_content(self):
        flow = self.flow()
        first = next(flow)
        self.assertEqual(len(first.messages), 2)
        provider = {'role': 'model', 'parts': [{'thought_signature': b'opaque-test-signature'}]}
        request = flow.send(Output('', 100, tool_calls=[{'id': 'c1', 'name': 'listen_audio',
                                                       'arguments': {'t0': 1, 't1': 2}}],
                                   provider_content=provider,
                                   usage={'input_tokens': 100, 'output_tokens': 12, 'total_tokens': 112}))
        self.assertIs(request.messages[-2]['provider_content'], provider)
        result = request.messages[-1]
        self.assertEqual((result['tool_call_id'], result['name']), ('c1', 'listen_audio'))
        self.assertEqual(result['content'][1]['type'], 'audio')
        with self.assertRaises(StopIteration) as stopped:
            flow.send(Output('', 200, tool_calls=[{'id': 'c2', 'name': 'answer',
                                                   'arguments': {'answer': 'B'}}]))
        record = stopped.exception.value
        self.assertTrue(record['correct'])
        self.assertEqual(record['usage']['input_tokens'], 100)
        self.assertTrue(record['audio_listen']['offered'])
        self.assertNotIn('fake-test-wav', json.dumps(record))
        self.assertNotIn('opaque-test-signature', json.dumps(record))

    def test_parallel_calls_over_limit_get_matching_response(self):
        flow = self.flow()
        next(flow)
        calls = [{'id': f'c{i}', 'name': 'listen_audio', 'arguments': {'t0': 1, 't1': 2}} for i in range(5)]
        calls[-1]['arguments']['path'] = '/private/should-not-appear-in-record'
        request = flow.send(Output('', tool_calls=calls, provider_content={'role': 'model', 'parts': []}))
        responses = [m for m in request.messages if m['role'] == 'tool']
        self.assertEqual([m['tool_call_id'] for m in responses], [c['id'] for c in calls])
        self.assertEqual(len(DummyTools.executed), 4)
        self.assertIn('[not run]', responses[-1]['content'][0]['text'])
        with self.assertRaises(StopIteration) as stopped:
            flow.send(Output('', tool_calls=[{'id': 'answer', 'name': 'answer', 'arguments': {'answer': 'B'}}]))
        self.assertNotIn('/private/should-not-appear-in-record', json.dumps(stopped.exception.value))

    def test_media_eviction_keeps_audio_time_and_tool_identity(self):
        messages = [{'role': 'tool', 'tool_call_id': f'c{i}', 'name': 'listen_audio', 'content': [
            {'type': 'text', 'text': 'absolute source 1-2 seconds'},
            {'type': 'audio', 'audio': b'not-stored', 'mime_type': 'audio/wav'}]} for i in range(6)]
        Agent._drop_old_media(messages, keep_last=4)
        self.assertEqual(messages[0]['tool_call_id'], 'c0')
        self.assertEqual(messages[0]['name'], 'listen_audio')
        self.assertEqual(messages[0]['content'][0]['text'], 'absolute source 1-2 seconds')
        self.assertTrue(all(c['type'] == 'text' for c in messages[0]['content']))
        self.assertEqual(messages[-1]['content'][1]['type'], 'audio')

    def test_local_backend_cannot_claim_omni_capability(self):
        with self.assertRaises(ValueError):
            Agent(SimpleNamespace(), self.cfg)


if __name__ == '__main__':
    unittest.main()
