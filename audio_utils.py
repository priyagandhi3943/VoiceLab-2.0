"""Audio loading and preprocessing for VoiceLab.

Everything here uses only NumPy and SciPy, so it is easy to read and test.

The preprocessing pipeline (each step is explained in the README):

    load  ->  stereo to mono  ->  resample  ->  trim silence  ->  normalize

Audio is stored as a 1-D NumPy array of float32 numbers between -1.0 and +1.0.
Each number is one "sample" (the air-pressure value at one instant). The
sample rate (sr) tells us how many samples make up one second.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from math import gcd

import numpy as np
from scipy import signal
from scipy.io import wavfile

# Sample rate used by the whole pipeline. 22.05 kHz is plenty for speech
# (human speech energy is mostly below 8 kHz) and keeps the data small.
TARGET_SR = 22050

MIN_USABLE_SECONDS = 1.0     # below this we refuse to continue
RECOMMENDED_MIN_SECONDS = 6.0  # voice cloning works better with 6s or more
RECOMMENDED_MAX_SECONDS = 30.0


class AudioError(Exception):
    """Raised when audio cannot be read or is unusable.

    The message is written for the end user, so the app can show it directly.
    """


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_audio(data: bytes, filename: str = "") -> tuple[np.ndarray, int]:
    """Decode uploaded audio bytes into (samples, sample_rate).

    Returns float32 samples shaped (n,) for mono or (n, channels) for stereo.
    Tries the `soundfile` library first (WAV, FLAC, OGG and often MP3).
    If it is not installed, falls back to SciPy, which reads WAV only.
    """
    if not data:
        raise AudioError("The uploaded file is empty.")

    try:
        import soundfile as sf  # optional dependency
    except ImportError:
        sf = None

    if sf is not None:
        try:
            samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
            return np.asarray(samples, dtype=np.float32), int(sr)
        except Exception as exc:  # soundfile raises its own error types
            raise AudioError(
                f"Could not read '{filename or 'the file'}' as audio. "
                "Try a WAV or FLAC file recorded on your phone or laptop."
            ) from exc

    try:
        sr, raw = wavfile.read(io.BytesIO(data))
    except Exception as exc:
        raise AudioError(
            "Could not read this file. Without the 'soundfile' package only "
            "WAV files are supported. Please upload a .wav file."
        ) from exc
    return _pcm_to_float(raw), int(sr)


def _pcm_to_float(raw: np.ndarray) -> np.ndarray:
    """Convert integer PCM arrays (as returned by SciPy) to float32 in [-1, 1]."""
    if raw.dtype == np.int16:
        out = raw.astype(np.float32) / 32768.0
    elif raw.dtype == np.int32:
        out = raw.astype(np.float32) / 2147483648.0
    elif raw.dtype == np.uint8:
        out = (raw.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(raw.dtype, np.floating):
        out = raw.astype(np.float32)
    else:
        raise AudioError(f"Unsupported WAV sample format: {raw.dtype}.")
    return out


# --------------------------------------------------------------------------
# Individual processing steps
# --------------------------------------------------------------------------
def to_mono(y: np.ndarray) -> np.ndarray:
    """Stereo -> mono by averaging the channels.

    A stereo file has a left and right channel. Voice models want a single
    channel, so we average them: mono = (left + right) / 2.
    """
    if y.ndim == 1:
        return y
    return y.mean(axis=1).astype(np.float32)


def resample(y: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Change the sample rate using polyphase filtering (anti-aliased)."""
    if sr == target_sr:
        return y
    g = gcd(int(sr), int(target_sr))
    up, down = target_sr // g, sr // g
    return signal.resample_poly(y, up, down).astype(np.float32)


def remove_dc_offset(y: np.ndarray) -> np.ndarray:
    """Subtract the average value so the waveform is centred on zero."""
    return (y - y.mean()).astype(np.float32)


def trim_silence(
    y: np.ndarray,
    sr: int,
    top_db: float = 30.0,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    pad_ms: float = 100.0,
) -> np.ndarray:
    """Cut silence from the start and end of a recording.

    How it works:
      1. Slice the signal into short overlapping frames (25 ms each).
      2. Measure each frame's loudness as RMS (root-mean-square energy).
      3. Convert to decibels relative to the loudest frame.
      4. Keep everything from the first frame to the last frame that is
         louder than `-top_db` dB. A smaller top_db trims more aggressively.
      5. Add a short padding (100 ms) so word edges are not clipped.
    """
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if frame < 1 or hop < 1 or len(y) < frame:
        return y

    n_frames = 1 + (len(y) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    rms = np.sqrt(np.mean(y[idx].astype(np.float64) ** 2, axis=1))

    loudest = rms.max()
    if loudest < 1e-8:  # completely silent: nothing sensible to trim
        return y

    level_db = 20.0 * np.log10(np.maximum(rms, 1e-10) / loudest)
    active = np.where(level_db > -top_db)[0]
    if active.size == 0:
        return y

    pad = int(sr * pad_ms / 1000)
    start = max(0, active[0] * hop - pad)
    end = min(len(y), active[-1] * hop + frame + pad)
    return y[start:end]


def normalize_peak(y: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Scale the whole signal so its loudest sample equals `target_peak`.

    This makes quiet and loud recordings comparable without clipping
    (a peak of 1.0 is the maximum a WAV file can hold).
    """
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < 1e-8:
        return y
    return (y * (target_peak / peak)).astype(np.float32)


def to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    """Encode float audio as a 16-bit PCM WAV file (for playback/download)."""
    clipped = np.clip(y, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    wavfile.write(buffer, sr, pcm)
    return buffer.getvalue()


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------
@dataclass
class ProcessedAudio:
    """Result of preprocessing, plus facts the interface can display."""

    audio: np.ndarray                 # processed mono signal
    sr: int                           # its sample rate
    original: np.ndarray              # mono (but otherwise untouched) signal
    original_sr: int
    original_channels: int
    original_seconds: float
    processed_seconds: float
    peak_before: float
    peak_after: float
    steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def preprocess(
    y: np.ndarray,
    sr: int,
    *,
    target_sr: int = TARGET_SR,
    do_trim: bool = True,
    top_db: float = 30.0,
    do_normalize: bool = True,
    check_length: bool = True,
) -> ProcessedAudio:
    """Run the full preprocessing pipeline and describe what happened.

    `check_length=False` skips the "reference is too short/long" warnings. It is
    used for generated audio, where those recommendations do not apply.
    """
    if y.size == 0:
        raise AudioError("The recording contains no audio samples.")
    if not np.all(np.isfinite(y)):
        raise AudioError("The recording contains invalid (NaN/inf) values.")

    channels = 1 if y.ndim == 1 else y.shape[1]
    steps: list[str] = []
    warnings: list[str] = []

    mono = to_mono(y)
    if channels > 1:
        steps.append(f"Converted {channels} channels to mono")
    original_seconds = len(mono) / sr

    if float(np.max(np.abs(mono))) < 1e-4:
        raise AudioError(
            "This recording is silent (or almost silent). "
            "Please check your microphone and record again."
        )

    processed = remove_dc_offset(mono)
    steps.append("Removed DC offset")

    if sr != target_sr:
        processed = resample(processed, sr, target_sr)
        steps.append(f"Resampled {sr} Hz -> {target_sr} Hz")

    if do_trim:
        before = len(processed)
        processed = trim_silence(processed, target_sr, top_db=top_db)
        removed = (before - len(processed)) / target_sr
        steps.append(f"Trimmed {removed:.2f} s of silence")

    peak_before = float(np.max(np.abs(processed)))
    if do_normalize:
        processed = normalize_peak(processed)
        steps.append("Peak-normalized to 0.95")
    peak_after = float(np.max(np.abs(processed)))

    processed_seconds = len(processed) / target_sr
    if processed_seconds < MIN_USABLE_SECONDS:
        raise AudioError(
            f"Only {processed_seconds:.1f} s of speech remain after cleaning. "
            f"Please upload at least {RECOMMENDED_MIN_SECONDS:.0f} seconds of clear speech."
        )
    if check_length and processed_seconds < RECOMMENDED_MIN_SECONDS:
        warnings.append(
            f"Only {processed_seconds:.1f} s of speech. Voice cloning usually "
            f"works better with {RECOMMENDED_MIN_SECONDS:.0f}+ seconds."
        )
    if check_length and processed_seconds > RECOMMENDED_MAX_SECONDS:
        warnings.append(
            f"The reference is {processed_seconds:.0f} s long. "
            f"More than {RECOMMENDED_MAX_SECONDS:.0f} s rarely helps and slows things down."
        )
    if float(np.mean(np.abs(mono) > 0.99)) > 0.001:
        warnings.append(
            "The recording looks clipped (too loud), which can distort the cloned voice."
        )

    return ProcessedAudio(
        audio=processed,
        sr=target_sr,
        original=mono,
        original_sr=sr,
        original_channels=channels,
        original_seconds=original_seconds,
        processed_seconds=processed_seconds,
        peak_before=peak_before,
        peak_after=peak_after,
        steps=steps,
        warnings=warnings,
    )
