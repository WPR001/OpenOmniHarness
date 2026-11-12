"""CLI mode selection and benchmark resume compatibility; no model or API calls."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

from infer.common import add_common_args, build_config, check_audio_math_resume, config_dict
from omniharness.agent import prompts
from omniharness.config import AGENT_MODEL


class AudioMathConfigTests(unittest.TestCase):
    def config(self, *flags):
        parser = add_common_args(argparse.ArgumentParser())
        with tempfile.TemporaryDirectory() as scratch:
            args = parser.parse_args([*flags, '--scratch', scratch])
            return args, build_config(args)

    def test_default_prompt_and_tools_are_unchanged(self):
        args, cfg = self.config()
        self.assertEqual(cfg.harness_version, 'original')
        self.assertEqual(cfg.tools, ('bash', 'view', 'watch', 'answer'))
        self.assertEqual((args.backend, args.model), ('vllm', AGENT_MODEL))
        self.assertEqual(prompts.system_prompt(cfg),
                         prompts.SYSTEM.format(max_turns=50, max_images=30, max_minutes=30))

    def test_visual_audio_math_uses_local_backend(self):
        args, cfg = self.config('--harness-version', 'audio-math-v1')
        self.assertEqual(args.backend, 'vllm')
        self.assertEqual(cfg.agent_mode, 'visual')
        self.assertEqual(cfg.tools, ('bash', 'view', 'watch', 'audio_analyze', 'audio_python', 'answer'))
        self.assertNotIn('listen_audio', cfg.tools)
        self.assertEqual((cfg.audio_analysis_calls, cfg.audio_analysis_seconds, cfg.audio_analysis_timeout),
                         (12, 120.0, 90.0))
        self.assertEqual(args.tools, list(cfg.tools))

    def test_omni_can_use_both_listening_and_measurements(self):
        args, cfg = self.config('--harness-version', 'audio-math-v1', '--agent-mode', 'omni',
                                '--audio-analysis-calls', '6', '--audio-analysis-seconds', '45',
                                '--audio-analysis-timeout', '30')
        self.assertEqual(args.backend, 'gemini')
        self.assertTrue({'listen_audio', 'audio_analyze', 'audio_python'} <= set(cfg.tools))
        self.assertEqual((cfg.audio_analysis_calls, cfg.audio_analysis_seconds, cfg.audio_analysis_timeout),
                         (6, 45.0, 30.0))

    def test_audio_tools_require_enabled_version(self):
        for tool in ('audio_analyze', 'audio_python'):
            with self.subTest(tool=tool), self.assertRaises(ValueError):
                self.config('--tools', 'bash', tool, 'answer')

    def test_invalid_analysis_budgets(self):
        for flags in [('--audio-analysis-calls', '0'), ('--audio-analysis-calls', '-1'),
                      ('--audio-analysis-seconds', 'nan'), ('--audio-analysis-seconds', '0'),
                      ('--audio-analysis-timeout', 'inf'), ('--audio-analysis-timeout', '-1')]:
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.config('--harness-version', 'audio-math-v1', *flags)

    def write_config(self, directory, args, cfg, name='config_shard0of1.json'):
        (Path(directory) / name).write_text(json.dumps(config_dict(args, cfg)), encoding='utf-8')

    def test_same_audio_math_run_resumes(self):
        args, cfg = self.config('--harness-version', 'audio-math-v1')
        with tempfile.TemporaryDirectory() as out:
            self.write_config(out, args, cfg)
            (Path(out) / 'results_shard0of1.jsonl').write_text('{"uid":"done"}\n')
            check_audio_math_resume(out, args, cfg)

    def test_changed_audio_budget_or_tools_requires_new_output(self):
        args, cfg = self.config('--harness-version', 'audio-math-v1')
        with tempfile.TemporaryDirectory() as out:
            self.write_config(out, args, cfg)
            for flags in [('--audio-analysis-calls', '7'), ('--audio-analysis-seconds', '30'),
                          ('--audio-analysis-timeout', '45'), ('--tools', 'bash', 'view', 'answer')]:
                with self.subTest(flags=flags):
                    new_args, new_cfg = self.config('--harness-version', 'audio-math-v1', *flags)
                    with self.assertRaises(ValueError):
                        check_audio_math_resume(out, new_args, new_cfg)

    def test_original_and_audio_math_outputs_cannot_mix(self):
        orig_args, orig_cfg = self.config()
        math_args, math_cfg = self.config('--harness-version', 'audio-math-v1')
        for saved, current in [((orig_args, orig_cfg), (math_args, math_cfg)),
                               ((math_args, math_cfg), (orig_args, orig_cfg))]:
            with self.subTest(version=saved[1].harness_version), tempfile.TemporaryDirectory() as out:
                self.write_config(out, *saved)
                with self.assertRaises(ValueError):
                    check_audio_math_resume(out, *current)

    def test_old_original_config_is_recognized(self):
        orig_args, orig_cfg = self.config()
        math_args, math_cfg = self.config('--harness-version', 'audio-math-v1')
        previous = config_dict(orig_args, orig_cfg)
        previous['config'].pop('harness_version')
        with tempfile.TemporaryDirectory() as out:
            (Path(out) / 'config_shard0of1.json').write_text(json.dumps(previous))
            check_audio_math_resume(out, orig_args, orig_cfg)
            with self.assertRaises(ValueError):
                check_audio_math_resume(out, math_args, math_cfg)

    def test_unversioned_result_shard_cannot_be_resumed_as_audio_math(self):
        args, cfg = self.config('--harness-version', 'audio-math-v1')
        with tempfile.TemporaryDirectory() as out:
            self.write_config(out, args, cfg, name='config_shard0of2.json')
            (Path(out) / 'results_shard1of2.jsonl').write_text('{"uid":"done"}\n')
            with self.assertRaises(ValueError):
                check_audio_math_resume(out, args, cfg)


if __name__ == '__main__':
    unittest.main()
