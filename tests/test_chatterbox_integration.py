"""Integration tests for VoiceLab Chatterbox backend communication."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from voicelab import synthesis
from voicelab.audio_utils import to_wav_bytes


class TestChatterboxIntegration(unittest.TestCase):
    def setUp(self):
        # Clear environment and session caches for clean testing
        os.environ.pop("VOICELAB_API_URL", None)
        os.environ.pop("VOICELAB_API_TOKEN", None)

    def test_backend_status_unconfigured(self):
        ok, msg = synthesis.backend_status()
        self.assertFalse(ok)
        self.assertIn("not configured", msg)

    def test_backend_status_rejects_insecure_nonlocal_http(self):
        ok, msg = synthesis.backend_status(custom_url="http://insecure-remote.com", custom_token="token123")
        self.assertFalse(ok)
        self.assertIn("must use HTTPS", msg)

    def test_backend_status_allows_localhost_http(self):
        with patch("requests.get") as mock_get:
            mock_res = MagicMock()
            mock_res.json.return_value = {"ready": True, "model": "chatterbox-tts"}
            mock_res.raise_for_status.return_value = None
            mock_get.return_value = mock_res

            ok, msg = synthesis.backend_status(custom_url="http://localhost:8765", custom_token="token123")
            self.assertTrue(ok)
            self.assertIn("online", msg)

    def test_synthesize_validates_text_length(self):
        samples = np.zeros(22050, dtype=np.float32)
        with patch("voicelab.synthesis.backend_status", return_value=(True, "online")):
            # Empty text
            with self.assertRaises(ValueError):
                synthesis.synthesize("   ", samples, 22050)
            
            # Oversized text (>500 chars)
            with self.assertRaises(ValueError):
                synthesis.synthesize("a" * 501, samples, 22050)

    def test_synthesize_handles_offline_gracefully(self):
        samples = np.zeros(22050, dtype=np.float32)
        with self.assertRaises(synthesis.SynthesisUnavailable):
            synthesis.synthesize("Hello world", samples, 22050)

    def test_synthesize_payload_dispatch(self):
        samples = np.sin(np.linspace(0, 100, 22050)).astype(np.float32)
        expected_wav = to_wav_bytes(samples, 22050)

        with patch("voicelab.synthesis.backend_status", return_value=(True, "online")), \
             patch("voicelab.synthesis._settings", return_value=("https://test-colab.trycloudflare.com", "secret-token-12345")), \
             patch("requests.post") as mock_post:
            
            mock_response = MagicMock()
            mock_response.content = expected_wav
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            out_samples, out_sr = synthesis.synthesize("Test phrase", samples, 22050)
            self.assertEqual(out_sr, 22050)
            self.assertGreater(len(out_samples), 0)

            # Verify request parameters
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            self.assertEqual(call_kwargs["headers"]["Authorization"], "Bearer secret-token-12345")
            self.assertEqual(call_kwargs["data"]["text"], "Test phrase")
            self.assertEqual(call_kwargs["data"]["consent"], "true")
            self.assertIn("reference", call_kwargs["files"])


if __name__ == "__main__":
    unittest.main()
