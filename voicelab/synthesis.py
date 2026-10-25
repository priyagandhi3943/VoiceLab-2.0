"""Authenticated remote Chatterbox synthesis; no GPU required in Streamlit."""
from __future__ import annotations
import io
import os
import wave
import numpy as np
import requests

MAX_TEXT_CHARS = 500

class SynthesisUnavailable(Exception):
    """Backend is not configured or is offline."""


def _settings():
    url = os.getenv("VOICELAB_API_URL", "").strip().rstrip("/")
    token = os.getenv("VOICELAB_API_TOKEN", "").strip()
    try:
        import streamlit as st
        session_url = getattr(st, "session_state", {}).get("VOICELAB_API_URL", "")
        session_token = getattr(st, "session_state", {}).get("VOICELAB_API_TOKEN", "")

        secrets_url = ""
        secrets_token = ""
        if hasattr(st, "secrets"):
            try:
                secrets_url = str(st.secrets.get("VOICELAB_API_URL", ""))
                secrets_token = str(st.secrets.get("VOICELAB_API_TOKEN", ""))
            except Exception:
                pass

        url = str(session_url or secrets_url or url).strip().rstrip("/")
        token = str(session_token or secrets_token or token).strip()
    except (ImportError, FileNotFoundError, KeyError, AttributeError):
        pass
    return url, token


def backend_status(custom_url: str = "", custom_token: str = "") -> tuple[bool, str]:
    url, token = _settings()
    if custom_url:
        url = custom_url.strip().rstrip("/")
    if custom_token:
        token = custom_token.strip()

    if not url or not token:
        return False, "Chatterbox backend not configured. See deployment guide."
    if not (url.startswith("https://") or url.startswith("http://localhost") or url.startswith("http://127.0.0.1")):
        return False, "Backend URL must use HTTPS (or localhost for development)."
    try:
        r = requests.get(url + "/health", timeout=6)
        r.raise_for_status()
        data = r.json()
        if data.get("ready") is True:
            model_name = data.get("model", "chatterbox-tts")
            return True, f"Chatterbox GPU backend is online ({model_name})."
        return False, "Chatterbox backend is initializing on GPU..."
    except requests.Timeout:
        return False, "Chatterbox backend health check timed out. Verify Colab is still running."
    except (requests.RequestException, ValueError):
        return False, "Chatterbox GPU backend is offline. Start the Colab service."



def _wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sr))
        wav.writeframes(pcm)
    return buf.getvalue()


def synthesize(text: str, reference: np.ndarray, reference_sr: int) -> tuple[np.ndarray, int]:
    url, token = _settings()
    ok, reason = backend_status()
    if not ok:
        raise SynthesisUnavailable(reason)
    if not text.strip() or len(text) > MAX_TEXT_CHARS:
        raise ValueError("Enter between 1 and 500 characters.")
    audio = _wav_bytes(reference, reference_sr)
    try:
        r = requests.post(
            url + "/generate",
            headers={"Authorization": "Bearer " + token},
            files={"reference": ("reference.wav", audio, "audio/wav")},
            data={"text": text.strip(), "consent": "true"},
            timeout=180,
        )
        r.raise_for_status()
        from voicelab.audio_utils import load_audio
        samples, sr = load_audio(r.content, "generated.wav")
        return samples, sr
    except requests.Timeout as exc:
        raise SynthesisUnavailable("Generation timed out. Check the Colab GPU session.") from exc
    except requests.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f" (HTTP {exc.response.status_code})"
        raise SynthesisUnavailable("Chatterbox request failed" + detail + ". Check Colab logs.") from exc
