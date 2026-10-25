"""Smoke test for app.py using a FAKE Streamlit.

What this proves: the app's Python code runs from top to bottom without errors
in several scenarios (no upload, valid upload, bad upload, demo mode, Generate
button with and without consent) and shows the messages we expect.

What this does NOT prove: how the page looks in a browser. For that, run
`streamlit run app.py` and look at it.
"""

import os
import runpy
import sys
import types
import unittest
from collections import defaultdict
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from voicelab.audio_utils import to_wav_bytes  # noqa: E402


class FakeUpload:
    def __init__(self, data: bytes, name: str):
        self._data, self.name = data, name

    def getvalue(self):
        return self._data


class FakeStreamlit(types.ModuleType):
    """Records every call; returns scripted answers for input widgets."""

    def __init__(self, *, ref_upload=None, gen_upload=None, pressed=(), text="", consent=True,
                 session=None):
        super().__init__("streamlit")
        self.calls = defaultdict(list)
        self.session_state = dict(session or {})
        self.sidebar = MagicMock()
        self._ref_upload, self._gen_upload = ref_upload, gen_upload
        self._pressed, self._text, self._consent = set(pressed), text, consent

    # -- widgets with scripted answers
    def file_uploader(self, label, *a, **k):
        self.calls["file_uploader"].append(label)
        return self._gen_upload if k.get("key") == "generated_upload" else self._ref_upload

    def button(self, label, *a, **k):
        self.calls["button"].append(label)
        return any(p in label for p in self._pressed)

    def toggle(self, label, *a, value=False, **k):
        return value

    def slider(self, label, lo, hi, value, **k):
        return value

    def checkbox(self, label, *a, **k):
        return self._consent

    def text_area(self, *a, **k):
        return self._text

    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [MagicMock() for _ in range(n)]

    def tabs(self, labels):
        return [MagicMock() for _ in labels]

    def cache_data(self, *a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

    def progress(self, *a, **k):
        self.calls["progress"].append(a)
        return MagicMock()

    # -- everything else: record and return a harmless mock
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def recorder(*a, **k):
            self.calls[name].append((a, k))
            return MagicMock()

        return recorder

    def messages(self, kind):
        return [str(c[0][0]) for c in self.calls[kind] if c[0]]


def run_app(**kwargs) -> FakeStreamlit:
    fake = FakeStreamlit(**kwargs)
    saved = sys.modules.get("streamlit")
    sys.modules["streamlit"] = fake
    try:
        runpy.run_path(os.path.join(ROOT, "app.py"), run_name="__main__")
    finally:
        if saved is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = saved
    return fake


def voice_wav(seconds=8.0, sr=22050, f0=130.0):
    t = np.arange(int(seconds * sr)) / sr
    y = 0.4 * (2.0 * ((f0 * t) % 1.0) - 1.0) * (0.6 + 0.4 * np.sin(2 * np.pi * 1.5 * t))
    return to_wav_bytes(y.astype(np.float32), sr)


class TestAppSmoke(unittest.TestCase):
    def test_empty_state_runs(self):
        fake = run_app()
        self.assertTrue(any("Upload a recording" in m for m in fake.messages("info")))
        self.assertEqual(len(fake.calls["pyplot"]), 0)

    def test_valid_upload_shows_preview_and_analysis(self):
        fake = run_app(ref_upload=FakeUpload(voice_wav(), "me.wav"))
        self.assertEqual(fake.messages("error"), [])
        # 1 preview + waveform, spectrogram, pitch, mfcc = 5 figures (no comparison without generated)
        self.assertEqual(len(fake.calls["pyplot"]), 5)
        labels = [c[0][0] for c in fake.calls["download_button"]]
        self.assertTrue(any("cleaned reference" in l for l in labels))

    def test_upload_plus_generated_shows_comparison(self):
        fake = run_app(
            ref_upload=FakeUpload(voice_wav(f0=120), "me.wav"),
            gen_upload=FakeUpload(voice_wav(f0=160), "generated.wav"),
        )
        self.assertEqual(fake.messages("error"), [])
        self.assertEqual(len(fake.calls["pyplot"]), 6)  # + comparison spectrum plot
        self.assertEqual(len(fake.calls["dataframe"]), 1)

    def test_garbage_upload_gives_friendly_error(self):
        fake = run_app(ref_upload=FakeUpload(b"definitely not audio", "bad.wav"))
        errors = fake.messages("error")
        self.assertEqual(len(errors), 1)
        self.assertIn("Could not read", errors[0])
        self.assertEqual(len(fake.calls["pyplot"]), 0)

    def test_silent_upload_gives_friendly_error(self):
        silent = to_wav_bytes(np.zeros(22050 * 5, dtype=np.float32), 22050)
        fake = run_app(ref_upload=FakeUpload(silent, "silent.wav"))
        self.assertTrue(any("silent" in m for m in fake.messages("error")))

    def test_bad_generated_file_does_not_break_reference_analysis(self):
        fake = run_app(
            ref_upload=FakeUpload(voice_wav(), "me.wav"),
            gen_upload=FakeUpload(b"junk", "generated.wav"),
        )
        self.assertTrue(any(m.startswith("Generated audio:") for m in fake.messages("error")))
        self.assertEqual(len(fake.calls["pyplot"]), 5)

    def test_demo_mode_full_comparison(self):
        fake = run_app(session={"demo": True})
        self.assertEqual(fake.messages("error"), [])
        self.assertEqual(len(fake.calls["pyplot"]), 6)
        self.assertEqual(len(fake.calls["dataframe"]), 1)

    def test_load_demo_button_sets_flag(self):
        fake = run_app(pressed=["Load demo audio"])
        self.assertTrue(fake.session_state.get("demo"))

    def test_uploaded_reference_overrides_demo_for_generated(self):
        # With a real upload, the demo "generated" file must NOT be compared against it.
        fake = run_app(ref_upload=FakeUpload(voice_wav(), "me.wav"), session={"demo": True})
        self.assertEqual(len(fake.calls["pyplot"]), 5)
        self.assertEqual(len(fake.calls["dataframe"]), 0)

    def test_generate_without_upload_errors(self):
        fake = run_app(pressed=["Generate Speech"], text="hello")
        self.assertTrue(any("upload a valid reference" in m for m in fake.messages("error")))

    def test_generate_without_text_errors(self):
        fake = run_app(ref_upload=FakeUpload(voice_wav(), "me.wav"), pressed=["Generate Speech"], text="  ")
        self.assertTrue(any("enter some text" in m for m in fake.messages("error")))

    def test_generate_without_consent_errors(self):
        fake = run_app(
            ref_upload=FakeUpload(voice_wav(), "me.wav"),
            pressed=["Generate Speech"], text="hello", consent=False,
        )
        self.assertTrue(any("permission" in m for m in fake.messages("error")))

    def test_generate_with_everything_points_to_colab(self):
        fake = run_app(
            ref_upload=FakeUpload(voice_wav(), "me.wav"),
            pressed=["Generate Speech"], text="hello world", consent=True,
        )
        self.assertEqual(fake.messages("error"), [])
        self.assertTrue(any("Colab" in m for m in fake.messages("warning")))
        self.assertNotIn("generated", fake.session_state)  # nothing was fabricated


if __name__ == "__main__":
    unittest.main()
