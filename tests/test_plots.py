"""Smoke tests: every figure builds and renders to an image without errors."""

import io
import os
import sys
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from voicelab import analysis as an  # noqa: E402
from voicelab import plots  # noqa: E402

SR = 22050


def buzz(f0, seconds=2.0):
    t = np.arange(int(seconds * SR)) / SR
    return 0.4 * (2.0 * ((f0 * t) % 1.0) - 1.0)


class TestPlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = an.analyze(buzz(120), SR)
        cls.b = an.analyze(buzz(180), SR)
        cls.noise = an.analyze(0.2 * np.random.default_rng(0).standard_normal(2 * SR), SR)

    def _render(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=60)
        plt.close(fig)
        self.assertGreater(len(buf.getvalue()), 1000)

    def test_waveform(self):
        self._render(plots.waveform_figure([("a", buzz(120), SR, "#fff"), ("b", buzz(180), SR, "#0f0")]))

    def test_spectrogram_two_rows(self):
        self._render(plots.spectrogram_figure([("A", self.a.spectrogram), ("B", self.b.spectrogram)]))

    def test_pitch_two_tracks(self):
        self._render(plots.pitch_figure([("A", self.a.pitch, "#fff"), ("B", self.b.pitch, "#0f0")]))

    def test_pitch_with_no_voiced_frames(self):
        self._render(plots.pitch_figure([("noise", self.noise.pitch, "#fff")]))

    def test_mfcc_two_rows(self):
        self._render(plots.mfcc_figure([("A", self.a.mfcc), ("B", self.b.mfcc)]))

    def test_ltas(self):
        f = self.a.spectrogram.freqs
        self._render(plots.ltas_figure([("A", f, self.a.ltas_db, "#fff"), ("B", f, self.b.ltas_db, "#0f0")]))


if __name__ == "__main__":
    unittest.main()
