"""Tests for voicelab.analysis, using synthetic signals with known answers.

Run from the project folder:
    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from voicelab import analysis as an  # noqa: E402

SR = 22050


def tone(freq, seconds=1.0, amp=0.5, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def pulse_voice(f0, seconds=1.0, sr=SR):
    """Harmonic-rich 'buzz' (sawtooth), closer to a voice than a pure sine."""
    t = np.arange(int(seconds * sr)) / sr
    return 0.4 * (2.0 * ((f0 * t) % 1.0) - 1.0)


class TestSpectrogram(unittest.TestCase):
    def test_shapes(self):
        spec = an.stft(tone(440), SR)
        self.assertEqual(spec.magnitude.shape[0], an.N_FFT // 2 + 1)
        self.assertEqual(spec.magnitude.shape[1], len(spec.times))
        self.assertEqual(len(spec.freqs), spec.magnitude.shape[0])

    def test_peak_at_tone_frequency(self):
        spec = an.stft(tone(1000), SR)
        mean_mag = spec.magnitude.mean(axis=1)
        peak_hz = spec.freqs[np.argmax(mean_mag)]
        self.assertAlmostEqual(peak_hz, 1000, delta=SR / an.N_FFT)

    def test_db_range(self):
        db = an.stft(tone(440), SR).db(top_db=80)
        self.assertLessEqual(db.max(), 1e-9)
        self.assertGreaterEqual(db.min(), -80 - 1e-9)

    def test_short_signal_does_not_crash(self):
        spec = an.stft(np.ones(100) * 0.1, SR)
        self.assertGreaterEqual(spec.magnitude.shape[1], 1)

    def test_centroid_higher_for_higher_tone(self):
        lo = an.spectral_centroid(an.stft(tone(300), SR))
        hi = an.spectral_centroid(an.stft(tone(3000), SR))
        self.assertGreater(hi, lo)
        self.assertAlmostEqual(lo, 300, delta=60)


class TestPitch(unittest.TestCase):
    def test_pure_tone(self):
        track = an.estimate_pitch(tone(200, 1.0), SR)
        stats = track.stats()
        self.assertAlmostEqual(stats["median_hz"], 200, delta=2.0)
        self.assertGreater(stats["voiced_fraction"], 0.8)

    def test_low_male_like_pitch(self):
        stats = an.estimate_pitch(tone(100, 1.0), SR).stats()
        self.assertAlmostEqual(stats["median_hz"], 100, delta=2.0)

    def test_harmonic_buzz_no_octave_error(self):
        for f0 in (110.0, 180.0, 260.0):
            stats = an.estimate_pitch(pulse_voice(f0), SR).stats()
            self.assertAlmostEqual(stats["median_hz"], f0, delta=f0 * 0.02, msg=f"f0={f0}")

    def test_noise_is_mostly_unvoiced(self):
        rng = np.random.default_rng(0)
        noise = 0.3 * rng.standard_normal(SR)
        stats = an.estimate_pitch(noise, SR).stats()
        self.assertLess(stats["voiced_fraction"], 0.2)

    def test_silence_is_unvoiced(self):
        track = an.estimate_pitch(np.zeros(SR), SR)
        self.assertEqual(track.stats()["voiced_fraction"], 0.0)
        self.assertTrue(np.isnan(track.stats()["median_hz"]))

    def test_glide_is_increasing(self):
        sr = SR
        t = np.arange(int(2.0 * sr)) / sr
        f = 120 + 80 * t / 2.0                      # 120 -> 200 Hz
        phase = 2 * np.pi * np.cumsum(f) / sr
        track = an.estimate_pitch(0.5 * np.sin(phase), sr)
        v = track.f0[track.voiced]
        self.assertGreater(v[-10:].mean(), v[:10].mean() + 40)

    def test_tone_with_silence_gaps(self):
        y = np.concatenate([tone(150, 0.6), np.zeros(int(0.4 * SR)), tone(150, 0.6)])
        track = an.estimate_pitch(y, SR)
        gap_mask = (track.times > 0.7) & (track.times < 0.9)
        self.assertTrue(np.all(np.isnan(track.f0[gap_mask])))
        self.assertAlmostEqual(track.stats()["median_hz"], 150, delta=2.0)

    def test_short_audio_does_not_crash(self):
        track = an.estimate_pitch(tone(200, 0.05), SR)
        self.assertEqual(len(track.f0), len(track.times))


class TestMfcc(unittest.TestCase):
    def test_filterbank_shape_and_range(self):
        fb = an.mel_filterbank(SR, an.N_FFT, 40)
        self.assertEqual(fb.shape, (40, an.N_FFT // 2 + 1))
        self.assertGreaterEqual(fb.min(), 0.0)
        self.assertLessEqual(fb.max(), 1.0 + 1e-9)
        self.assertTrue(np.all(fb.sum(axis=1) > 0))

    def test_filter_centres_increase(self):
        fb = an.mel_filterbank(SR, an.N_FFT, 40)
        centres = np.argmax(fb, axis=1)
        self.assertTrue(np.all(np.diff(centres) >= 0))

    def test_mel_roundtrip(self):
        f = np.array([100.0, 1000.0, 5000.0])
        np.testing.assert_allclose(an.mel_to_hz(an.hz_to_mel(f)), f, rtol=1e-9)

    def test_shape(self):
        res = an.mfcc(pulse_voice(150), SR, n_mfcc=13)
        self.assertEqual(res.coeffs.shape[0], 13)
        self.assertEqual(res.coeffs.shape[1], len(res.times))
        self.assertTrue(np.all(np.isfinite(res.coeffs)))

    def test_same_signal_zero_distance(self):
        a = an.mfcc(pulse_voice(150), SR)
        b = an.mfcc(pulse_voice(150), SR)
        self.assertAlmostEqual(an.mfcc_distance(a, b), 0.0, places=9)

    def test_different_timbre_is_farther_than_same(self):
        a = an.mfcc(pulse_voice(150), SR)
        same = an.mfcc(pulse_voice(150) * 0.5, SR)       # just quieter
        diff = an.mfcc(tone(150), SR)                     # buzz vs pure sine
        self.assertLess(an.mfcc_distance(a, same), an.mfcc_distance(a, diff))

    def test_loudness_only_changes_c0(self):
        a = an.mfcc(pulse_voice(150), SR)
        b = an.mfcc(pulse_voice(150) * 0.5, SR)
        self.assertGreater(abs(a.coeffs[0].mean() - b.coeffs[0].mean()), 1.0)
        np.testing.assert_allclose(
            a.coeffs[1:].mean(axis=1), b.coeffs[1:].mean(axis=1), atol=0.05
        )


class TestReportAndCompare(unittest.TestCase):
    def test_analyze_fields(self):
        rep = an.analyze(pulse_voice(140, 2.0), SR)
        self.assertAlmostEqual(rep.seconds, 2.0, places=2)
        self.assertGreater(rep.peak, 0.3)
        self.assertAlmostEqual(rep.pitch.stats()["median_hz"], 140, delta=3)
        self.assertEqual(len(rep.ltas_db), len(rep.spectrogram.freqs))

    def test_compare_pitch_shift_one_octave(self):
        a = an.analyze(pulse_voice(120, 2.0), SR)
        b = an.analyze(pulse_voice(240, 2.0), SR)
        result = an.compare(a, b)
        self.assertAlmostEqual(result["pitch_shift_semitones"], 12.0, delta=0.5)

    def test_compare_identical_signals(self):
        a = an.analyze(pulse_voice(150, 2.0), SR)
        b = an.analyze(pulse_voice(150, 2.0), SR)
        result = an.compare(a, b)
        self.assertAlmostEqual(result["mfcc_distance"], 0.0, places=6)
        self.assertAlmostEqual(result["pitch_shift_semitones"], 0.0, places=6)
        self.assertGreater(result["ltas_correlation"], 0.999)

    def test_baseline_present_for_long_signal(self):
        a = an.analyze(pulse_voice(150, 2.0), SR)
        self.assertIsNotNone(an.compare(a, a)["mfcc_baseline"])

    def test_compare_with_unvoiced_signal_gives_nan_shift(self):
        rng = np.random.default_rng(1)
        a = an.analyze(pulse_voice(150, 2.0), SR)
        b = an.analyze(0.3 * rng.standard_normal(2 * SR), SR)
        self.assertTrue(np.isnan(an.compare(a, b)["pitch_shift_semitones"]))


if __name__ == "__main__":
    unittest.main()
