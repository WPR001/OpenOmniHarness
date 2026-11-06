"""Audio-tool boundary tests: no model, API, GPU or installed FFmpeg is needed."""
import io
import json
import os
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omniharness.media.audio import AudioClip, AudioInfo, AudioInputError, extract_audio, probe_audio
from omniharness.sandbox.sandbox import Sandbox
from omniharness.tools import ToolBox


def wav_bytes(seconds=1.0):
    data = io.BytesIO()
    with wave.open(data, 'wb') as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b'\x00\x00' * int(seconds * 16000))
    return data.getvalue()


def box(max_listen=4, listen_seconds=30.0, enabled=True):
    cfg = SimpleNamespace(tools=('listen_audio',) if enabled else ('answer',), rules=(),
                          max_listen=max_listen, listen_seconds=listen_seconds)
    sandbox = SimpleNamespace(_audio_source='/private/original-audio.wav', video='/sandbox/muted.mp4')
    return ToolBox(sandbox, cfg)


class ListenAudioTests(unittest.TestCase):
    def test_uses_original_source_and_returns_wav_and_absolute_time(self):
        toolbox = box()
        clip = AudioClip(wav_bytes(1.5), 1.5)
        with patch('omniharness.tools.listen_audio.probe_audio', return_value=AudioInfo(0, 10)) as probe, \
                patch('omniharness.tools.listen_audio.extract_audio', return_value=clip) as decode:
            content, trace = toolbox.run('listen_audio', {'t0': '00:02', 't1': '00:03.5'})
        probe.assert_called_once_with(toolbox.sb._audio_source)
        decode.assert_called_once_with(toolbox.sb._audio_source, 2.0, 3.5)
        self.assertEqual(content[1], {'type': 'audio', 'audio': clip.data, 'mime_type': 'audio/wav',
                                      't0': 2.0, 't1': 3.5})
        self.assertIn('2.000000 s', content[0]['text'])
        self.assertEqual(toolbox.state['audio_listen_count'], 1)
        self.assertEqual(trace['audios'], 1)
        self.assertEqual(trace['audio']['audio_seconds'], 1.5)
        self.assertEqual(trace['audio']['bytes'], len(clip.data))
        self.assertNotIn(toolbox.sb._audio_source, json.dumps(trace))
        self.assertNotIn('RIFF', json.dumps(trace))

    def test_invalid_intervals_do_not_probe_decode_or_spend_budget(self):
        values = [(-1, 1), (2, 2), (3, 2), (True, 2), (0, False), (float('nan'), 2),
                  (0, float('inf')), (10**1000, 2), ('1:99', 100), ('/private/source', 2), (0, 31)]
        for t0, t1 in values:
            with self.subTest(t0=str(t0)[:40], t1=t1):
                toolbox = box()
                with patch('omniharness.tools.listen_audio.probe_audio') as probe, \
                        patch('omniharness.tools.listen_audio.extract_audio') as decode:
                    content, trace = toolbox.run('listen_audio', {'t0': t0, 't1': t1})
                self.assertIn('[error]', content[0]['text'])
                probe.assert_not_called()
                decode.assert_not_called()
                self.assertEqual(toolbox.state.get('audio_listen_count', 0), 0)
                self.assertFalse(trace['audio']['budget_charged'])

    def test_unknown_path_parameter_is_rejected_without_echoing_it(self):
        toolbox = box()
        with patch('omniharness.tools.listen_audio.probe_audio') as probe:
            content, trace = toolbox.run('listen_audio', {'t0': 0, 't1': 1, 'path': '/private/secret.wav'})
        probe.assert_not_called()
        self.assertIn('[error]', content[0]['text'])
        self.assertNotIn('/private/secret.wav', json.dumps(trace))
        self.assertTrue(trace['args']['unsupported_arguments'])

    def test_audio_bounds_and_missing_track_do_not_spend_budget(self):
        for bounds, args in [(AudioInfo(0, 5), {'t0': 4, 't1': 6}),
                             (AudioInfo(0.5, 5), {'t0': 0, 't1': 1})]:
            toolbox = box()
            with patch('omniharness.tools.listen_audio.probe_audio', return_value=bounds), \
                    patch('omniharness.tools.listen_audio.extract_audio') as decode:
                content, trace = toolbox.run('listen_audio', args)
            decode.assert_not_called()
            self.assertIn('within', content[0]['text'])
            self.assertFalse(trace['audio']['budget_charged'])
        toolbox = box()
        with patch('omniharness.tools.listen_audio.probe_audio',
                   side_effect=AudioInputError('the original source has no audio track')):
            content, trace = toolbox.run('listen_audio', {'t0': 0, 't1': 1})
        self.assertIn('no audio track', content[0]['text'])
        self.assertEqual(trace['audio']['status'], 'source_error')
        self.assertEqual(toolbox.state.get('audio_listen_count', 0), 0)

    def test_decoder_failure_is_charged_once_and_never_leaks_source(self):
        toolbox = box(max_listen=1)
        with patch('omniharness.tools.listen_audio.probe_audio', return_value=AudioInfo(0, 10)), \
                patch('omniharness.tools.listen_audio.extract_audio',
                      side_effect=RuntimeError(toolbox.sb._audio_source)) as decode:
            content, trace = toolbox.run('listen_audio', {'t0': 0, 't1': 1})
            second, second_trace = toolbox.run('listen_audio', {'t0': 0, 't1': 1})
        decode.assert_called_once()
        self.assertNotIn(toolbox.sb._audio_source, content[0]['text'] + json.dumps(trace))
        self.assertEqual(trace['audio']['status'], 'decode_error')
        self.assertTrue(trace['audio']['budget_charged'])
        self.assertEqual(toolbox.state['audio_listen_count'], 1)
        self.assertIn('budget', second[0]['text'])
        self.assertFalse(second_trace['audio']['budget_charged'])

    def test_tool_is_not_available_when_not_selected(self):
        toolbox = box(enabled=False)
        with patch('omniharness.tools.listen_audio.probe_audio') as probe:
            content, trace = toolbox.run('listen_audio', {'t0': 0, 't1': 1})
        probe.assert_not_called()
        self.assertIn('there is no tool', content[0]['text'])
        self.assertEqual(trace['audios'], 0)
        self.assertNotIn('audio_listen_count', toolbox.state)


class AudioExtractionTests(unittest.TestCase):
    def test_ffmpeg_writes_valid_wav_in_host_tmp_and_tempfile_is_removed(self):
        outputs = []

        def ffmpeg(command, **kwargs):
            self.assertEqual(command[0], 'ffmpeg')
            self.assertEqual(command[command.index('-i') + 1], '/private/original.mp4')
            self.assertEqual(command[command.index('-map') + 1], '0:a:0')
            self.assertEqual(command[command.index('-ss') + 1], '0.250000000')
            self.assertEqual(command[command.index('-t') + 1], '1.000000000')
            self.assertTrue(command[-1].startswith('/tmp/omniharness-audio-'))
            outputs.append(command[-1])
            Path(command[-1]).write_bytes(wav_bytes(1))
            return subprocess.CompletedProcess(command, 0, b'', b'')

        with patch('omniharness.media.audio.subprocess.run', side_effect=ffmpeg):
            clip = extract_audio('/private/original.mp4', 0.25, 1.25)
        self.assertFalse(Path(outputs[0]).exists())
        self.assertFalse(Path(outputs[0]).parent.exists())
        with wave.open(io.BytesIO(clip.data), 'rb') as reader:
            self.assertEqual((reader.getnchannels(), reader.getsampwidth(), reader.getframerate()), (1, 2, 16000))
            self.assertEqual(reader.getnframes(), 16000)
        self.assertEqual(clip.seconds, 1)

    def test_decoder_error_is_sanitized_and_tempfile_removed(self):
        directories = []

        def failed(command, **kwargs):
            directories.append(Path(command[-1]).parent)
            raise subprocess.CalledProcessError(1, command, stderr=b'/private/original.mp4: secret error')

        with patch('omniharness.media.audio.subprocess.run', side_effect=failed):
            with self.assertRaises(AudioInputError) as caught:
                extract_audio('/private/original.mp4', 0, 1)
        self.assertNotIn('/private', str(caught.exception))
        self.assertFalse(directories[0].exists())

    def test_truncated_audio_is_not_returned_as_full_evidence(self):
        def short(command, **kwargs):
            Path(command[-1]).write_bytes(wav_bytes(0.5))
            return subprocess.CompletedProcess(command, 0, b'', b'')

        with patch('omniharness.media.audio.subprocess.run', side_effect=short):
            with self.assertRaisesRegex(AudioInputError, 'does not cover'):
                extract_audio('/private/original.mp4', 0, 1)

    def test_probe_uses_audio_stream_bounds_relative_to_container(self):
        payload = {'streams': [{'index': 1, 'start_time': '1.7', 'duration': '2.0'}],
                   'format': {'start_time': '1.5', 'duration': '3.0'}}
        with patch('omniharness.media.audio.subprocess.run',
                   return_value=SimpleNamespace(stdout=json.dumps(payload))):
            bounds = probe_audio('/private/source.mp4')
        self.assertAlmostEqual(bounds.start, 0.2)
        self.assertAlmostEqual(bounds.end, 2.2)

    def test_probe_errors_never_echo_source_or_stderr(self):
        with patch('omniharness.media.audio.subprocess.run',
                   side_effect=OSError('/private/source.mp4 cannot open')):
            with self.assertRaises(AudioInputError) as caught:
                probe_audio('/private/source.mp4')
        self.assertNotIn('/private', str(caught.exception))


class SandboxSourceTests(unittest.TestCase):
    def test_raw_source_is_private_and_separate_audio_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, muted, separate = [root / name for name in ('original.mp4', 'muted.mp4', 'original.wav')]
            for path in (original, muted, separate):
                path.write_bytes(b'test')
            sample = SimpleNamespace(uid='question-1', video=str(original), audio=str(separate),
                                     question='Which sound?', options=['A. x', 'B. y'])
            cfg = SimpleNamespace(sandbox_root=str(root / 'sandbox'), keep_sandbox=False)
            Path(cfg.sandbox_root).mkdir()
            with patch('omniharness.sandbox.sandbox.muted_copy', return_value=str(muted)), \
                    patch('omniharness.sandbox.sandbox.store_path', return_value=str(root / 'absent.json')):
                sandbox = Sandbox(sample, cfg)
            try:
                self.assertEqual(sandbox._audio_source, str(separate))
                self.assertEqual(os.path.realpath(Path(sandbox.dir) / 'video.mp4'), os.path.realpath(muted))
                self.assertNotIn(str(separate), json.dumps(sandbox.env))
                for path in Path(sandbox.dir).iterdir():
                    if path.is_file() and not path.is_symlink():
                        self.assertNotIn(str(separate), path.read_text())
            finally:
                sandbox.close()


if __name__ == '__main__':
    unittest.main()
