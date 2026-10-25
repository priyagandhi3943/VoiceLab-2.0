"""Tests for voicelab.audio_utils.

Run from the project folder with either:
    python -m unittest discover -s tests -v
    python -m pytest            (if pytest is installed)
"""

import io
import os
import sys
import unittest

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from voicelab import audio_utils as au  # noqa: E402


def tone(seconds=1.0, sr=22050, freq=220.0, amp=0.5):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestSteps(unittest.TestCase):
    def test_to_mono_averages_channels(self):
        left = np.ones(100, dtype=np.float32)
        right = np.zeros(100, dtype=np.float32)
        mono = au.to_mono(np.stack([left, right], axis=1))
        self.assertEqual(mono.shape, (100,))
        np.testing.assert_allclose(mono, 0.5)

    def test_to_mono_leaves_mono_alone(self):
        y = tone()
        self.assertIs(au.to_mono(y), y)

    def test_normalize_hits_target_peak(self):
        y = tone(amp=0.1)
        out = au.normalize_peak(y, 0.95)
        self.assertAlmostEqual(float(np.max(np.abs(out))), 0.95, places=4)

    def test_normalize_silence_is_safe(self):
        y = np.zeros(1000, dtype=np.float32)
        np.testing.assert_array_equal(au.normalize_peak(y), y)

    def test_resample_changes_length(self):
        y = tone(seconds=1.0, sr=44100)
        out = au.resample(y, 44100, 22050)
        self.assertAlmostEqual(len(out), 22050, delta=2)

    def test_resample_preserves_frequency(self):
        y = tone(seconds=1.0, sr=44100, freq=440.0)
        out = au.resample(y, 44100, 22050)
        spectrum = np.abs(np.fft.rfft(out))
        peak_hz = np.argmax(spectrum) * 22050 / len(out)
        self.assertAlmostEqual(peak_hz, 440.0, delta=2.0)

    def test_trim_silence_removes_padding(self):
        sr = 22050
        silence = np.zeros(sr, dtype=np.float32)  # 1 s of silence
        y = np.concatenate([silence, tone(1.0, sr), silence])
        out = au.trim_silence(y, sr, top_db=30)
        # about 1 s of tone + ~0.2 s padding, well under the original 3 s
        self.assertLess(len(out) / sr, 1.5)
        self.assertGreater(len(out) / sr, 1.0)

    def test_trim_silence_keeps_all_speech(self):
        sr = 22050
        y = np.concatenate([np.zeros(sr, np.float32), tone(1.0, sr)])
        out = au.trim_silence(y, sr)
        self.assertGreaterEqual(len(out), sr)  # the full second of tone survives

    def test_wav_roundtrip(self):
        y = tone()
        data = au.to_wav_bytes(y, 22050)
        sr, raw = wavfile.read(io.BytesIO(data))
        self.assertEqual(sr, 22050)
        back = au._pcm_to_float(raw)
        np.testing.assert_allclose(back, y, atol=1e-3)


class TestPipeline(unittest.TestCase):
    def _wav_bytes(self, y, sr):
        return au.to_wav_bytes(y, sr)

    def test_full_pipeline_on_stereo_44k(self):
        sr = 44100
        speech = tone(8.0, sr, amp=0.2)
        stereo = np.stack([speech, speech * 0.5], axis=1)
        padded = np.concatenate(
            [np.zeros((sr, 2), np.float32), stereo, np.zeros((sr, 2), np.float32)]
        )
        result = au.preprocess(padded, sr)
        self.assertEqual(result.sr, au.TARGET_SR)
        self.assertEqual(result.audio.ndim, 1)
        self.assertEqual(result.original_channels, 2)
        self.assertLess(result.processed_seconds, result.original_seconds)
        self.assertAlmostEqual(float(np.max(np.abs(result.audio))), 0.95, places=3)
        self.assertEqual(result.warnings, [])

    def test_short_audio_warns(self):
        result = au.preprocess(tone(3.0), 22050)
        self.assertTrue(any("Voice cloning" in w for w in result.warnings))

    def test_too_short_raises(self):
        with self.assertRaises(au.AudioError):
            au.preprocess(tone(0.3), 22050)

    def test_silent_audio_raises(self):
        with self.assertRaises(au.AudioError):
            au.preprocess(np.zeros(22050 * 5, dtype=np.float32), 22050)

    def test_empty_audio_raises(self):
        with self.assertRaises(au.AudioError):
            au.preprocess(np.zeros(0, dtype=np.float32), 22050)

    def test_nan_raises(self):
        y = tone(5.0)
        y[10] = np.nan
        with self.assertRaises(au.AudioError):
            au.preprocess(y, 22050)

    def test_clipping_warns(self):
        y = np.clip(tone(8.0, amp=3.0), -1.0, 1.0)
        result = au.preprocess(y, 22050)
        self.assertTrue(any("clipped" in w for w in result.warnings))


class TestLoading(unittest.TestCase):
    def test_load_wav_bytes(self):
        data = au.to_wav_bytes(tone(1.0), 22050)
        y, sr = au.load_audio(data, "x.wav")
        self.assertEqual(sr, 22050)
        self.assertEqual(y.ndim, 1)

    def test_load_garbage_raises_friendly_error(self):
        with self.assertRaises(au.AudioError):
            au.load_audio(b"this is not audio", "x.wav")

    def test_load_empty_raises(self):
        with self.assertRaises(au.AudioError):
            au.load_audio(b"", "x.wav")


if __name__ == "__main__":
    unittest.main()
