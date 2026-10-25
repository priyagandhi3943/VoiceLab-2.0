"""Signal-processing analysis for VoiceLab.

Everything is implemented with NumPy and SciPy (no librosa), so each step is
visible and explainable:

    STFT / spectrogram   how the frequency content changes over time
    Pitch (F0)           autocorrelation method, frame by frame
    MFCC                 mel filterbank + log + DCT (the "tone colour" features)
    Comparison           numbers that describe how two recordings differ

Audio is always a 1-D float array in [-1, 1] plus its sample rate `sr`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import dct

N_FFT = 1024
HOP = 256


# --------------------------------------------------------------------------
# Short-Time Fourier Transform (spectrogram)
# --------------------------------------------------------------------------
@dataclass
class Spectrogram:
    freqs: np.ndarray       # (n_freq,)  Hz
    times: np.ndarray       # (n_frames,) seconds
    magnitude: np.ndarray   # (n_freq, n_frames) linear magnitude

    def db(self, top_db: float = 80.0) -> np.ndarray:
        """Magnitude in decibels, relative to the loudest cell, floored at -top_db."""
        ref = max(float(self.magnitude.max()), 1e-10)
        out = 20.0 * np.log10(np.maximum(self.magnitude, 1e-10) / ref)
        return np.maximum(out, -top_db)


def stft(y: np.ndarray, sr: int, n_fft: int = N_FFT, hop: int = HOP) -> Spectrogram:
    """Short-Time Fourier Transform.

    1. Cut the signal into overlapping frames of `n_fft` samples.
    2. Multiply each frame by a Hann window (smooth fade in/out) so the
       frame edges do not create false frequencies.
    3. Take the FFT of every frame. The magnitude tells how much energy
       each frequency has at that moment.
    """
    y = np.asarray(y, dtype=np.float64)
    pad = n_fft // 2
    y = np.pad(y, pad, mode="constant")  # so frame 0 is centred on t = 0
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    frames = sliding_window_view(y, n_fft)[::hop]
    window = np.hanning(n_fft)
    spectrum = np.fft.rfft(frames * window, axis=1)
    magnitude = np.abs(spectrum).T  # (n_freq, n_frames)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    times = np.arange(magnitude.shape[1]) * hop / sr
    return Spectrogram(freqs=freqs, times=times, magnitude=magnitude)


def long_term_average_spectrum(spec: Spectrogram) -> np.ndarray:
    """Average power over time, in dB: the 'overall colour' of the voice."""
    power = np.mean(spec.magnitude**2, axis=1)
    ref = max(float(power.max()), 1e-20)
    return 10.0 * np.log10(np.maximum(power, 1e-20) / ref)


def spectral_centroid(spec: Spectrogram) -> float:
    """Centre of mass of the spectrum in Hz (higher = 'brighter' sound)."""
    power = np.mean(spec.magnitude**2, axis=1)
    total = power.sum()
    if total <= 0:
        return 0.0
    return float(np.sum(spec.freqs * power) / total)


# --------------------------------------------------------------------------
# Pitch (fundamental frequency, F0) by autocorrelation
# --------------------------------------------------------------------------
@dataclass
class PitchTrack:
    times: np.ndarray       # (n_frames,) seconds
    f0: np.ndarray          # (n_frames,) Hz, NaN where the frame is unvoiced
    strength: np.ndarray    # (n_frames,) periodicity 0..1 (how clearly pitched)

    @property
    def voiced(self) -> np.ndarray:
        return ~np.isnan(self.f0)

    def stats(self) -> dict:
        """Summary numbers over the voiced frames."""
        v = self.f0[self.voiced]
        if v.size == 0:
            return {
                "median_hz": float("nan"),
                "mean_hz": float("nan"),
                "low_hz": float("nan"),
                "high_hz": float("nan"),
                "range_semitones": float("nan"),
                "voiced_fraction": 0.0,
            }
        low, high = np.percentile(v, [5, 95])
        return {
            "median_hz": float(np.median(v)),
            "mean_hz": float(np.mean(v)),
            "low_hz": float(low),
            "high_hz": float(high),
            "range_semitones": float(12 * np.log2(high / low)),
            "voiced_fraction": float(self.voiced.mean()),
        }


def estimate_pitch(
    y: np.ndarray,
    sr: int,
    fmin: float = 70.0,
    fmax: float = 400.0,
    frame_ms: float = 40.0,
    hop_ms: float = 10.0,
    voicing_threshold: float = 0.4,
    energy_floor_db: float = -35.0,
) -> PitchTrack:
    """Estimate F0 with the autocorrelation method.

    Idea: a voiced sound repeats every T seconds (T = 1 / F0). If we compare a
    frame with a delayed copy of itself, the match is strongest when the delay
    equals one period. So for every 40 ms frame:

      1. Compute the autocorrelation (similarity vs. delay), normalised so a
         delay of 0 scores 1.0.
      2. Look for the first strong peak between 1/fmax and 1/fmin seconds.
      3. Pitch = sample_rate / peak_delay (refined with a parabola for sub-sample
         accuracy).
      4. Mark the frame "unvoiced" (NaN) if the peak is weak (noise, 's', 'f')
         or the frame is nearly silent.
      5. Smooth with a median filter and drop one- or two-frame blips.
    """
    y = np.asarray(y, dtype=np.float64)
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    min_lag = int(np.floor(sr / fmax))
    max_lag = int(np.ceil(sr / fmin))
    if frame <= max_lag + 2:
        raise ValueError("Pitch frame is too short for the requested fmin.")

    padded = np.pad(y, frame // 2)
    if len(padded) < frame:
        padded = np.pad(padded, (0, frame - len(padded)))
    frames = sliding_window_view(padded, frame)[::hop]
    frames = frames - frames.mean(axis=1, keepdims=True)
    n_frames = frames.shape[0]
    times = np.arange(n_frames) * hop / sr

    # Autocorrelation of every frame at once, via the FFT (Wiener-Khinchin).
    nfft = 1 << int(np.ceil(np.log2(2 * frame)))
    spectrum = np.fft.rfft(frames, nfft, axis=1)
    ac = np.fft.irfft(np.abs(spectrum) ** 2, nfft, axis=1)[:, : max_lag + 2]
    energy = ac[:, 0].copy()
    ac = ac / np.maximum(energy[:, None], 1e-12)

    # Energy gate: ignore frames far quieter than the loudest one.
    rms = np.sqrt(energy / frame)
    rms_db = 20 * np.log10(np.maximum(rms, 1e-12) / max(rms.max(), 1e-12))
    loud_enough = rms_db > energy_floor_db

    # Peak picking: take the FIRST local maximum that is nearly as high as the
    # best one. (Plain argmax can jump to a multiple of the period = octave error.)
    seg = ac[:, min_lag : max_lag + 1]
    best = seg.max(axis=1)
    inner = seg[:, 1:-1]
    is_peak = (inner >= seg[:, :-2]) & (inner > seg[:, 2:])
    good = is_peak & (inner >= 0.9 * best[:, None])
    has_good = good.any(axis=1)
    first = np.argmax(good, axis=1) + 1
    chosen = np.where(has_good, first, np.argmax(seg, axis=1))
    lag = min_lag + chosen
    strength = seg[np.arange(n_frames), chosen]

    # Parabolic interpolation around the peak for sub-sample accuracy.
    rows = np.arange(n_frames)
    y0, y1, y2 = ac[rows, lag - 1], ac[rows, lag], ac[rows, lag + 1]
    denom = y0 - 2 * y1 + y2
    ok = np.abs(denom) > 1e-12
    delta = np.where(ok, 0.5 * (y0 - y2) / np.where(ok, denom, 1.0), 0.0)
    delta = np.clip(delta, -1.0, 1.0)
    f0 = sr / (lag + delta)

    voiced = loud_enough & (strength >= voicing_threshold) & (f0 >= fmin) & (f0 <= fmax)
    f0 = np.where(voiced, f0, np.nan)
    f0 = _median_smooth(f0, width=5)
    f0 = _drop_short_runs(f0, min_len=3)
    return PitchTrack(times=times, f0=f0, strength=np.clip(strength, 0.0, 1.0))


def _median_smooth(f0: np.ndarray, width: int = 5) -> np.ndarray:
    """Median-filter voiced values only; unvoiced frames stay NaN."""
    half = width // 2
    padded = np.pad(f0, half, constant_values=np.nan)
    windows = sliding_window_view(padded, width)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(windows, axis=1)
    return np.where(np.isnan(f0), np.nan, med)


def _drop_short_runs(f0: np.ndarray, min_len: int = 3) -> np.ndarray:
    """Remove voiced stretches shorter than `min_len` frames (isolated blips)."""
    out = f0.copy()
    voiced = ~np.isnan(out)
    i, n = 0, len(out)
    while i < n:
        if voiced[i]:
            j = i
            while j < n and voiced[j]:
                j += 1
            if j - i < min_len:
                out[i:j] = np.nan
            i = j
        else:
            i += 1
    return out


# --------------------------------------------------------------------------
# MFCC (Mel-Frequency Cepstral Coefficients)
# --------------------------------------------------------------------------
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int = N_FFT, n_mels: int = 40) -> np.ndarray:
    """Triangular filters spaced evenly on the mel scale.

    Human hearing is more sensitive to differences at low frequencies than at
    high ones. The mel scale models that: filters are narrow and close together
    at low frequencies and wide and far apart at high frequencies.
    Returns an array of shape (n_mels, n_fft // 2 + 1).
    """
    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2.0), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(1, n_mels + 1):
        left, centre, right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        rising = (freqs - left) / max(centre - left, 1e-9)
        falling = (right - freqs) / max(right - centre, 1e-9)
        fb[m - 1] = np.maximum(0.0, np.minimum(rising, falling))
    return fb


@dataclass
class MfccResult:
    coeffs: np.ndarray   # (n_mfcc, n_frames)
    times: np.ndarray    # (n_frames,)


def mfcc(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_mels: int = 40,
    n_fft: int = N_FFT,
    hop: int = HOP,
    pre_emphasis: float = 0.97,
) -> MfccResult:
    """Compute MFCCs.

    Steps:
      1. Pre-emphasis: boost high frequencies a little, y[n] - 0.97*y[n-1].
      2. STFT -> power spectrum of each frame.
      3. Mel filterbank: add up the power inside each triangular filter.
      4. Take the logarithm (hearing is roughly logarithmic in loudness).
      5. DCT (discrete cosine transform) of the log energies. The first few
         coefficients describe the smooth overall shape of the spectrum,
         which is mostly the shape of the vocal tract = the voice's tone colour.
    """
    y = np.asarray(y, dtype=np.float64)
    if len(y) > 1:
        y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    spec = stft(y, sr, n_fft=n_fft, hop=hop)
    power = spec.magnitude**2
    mel_energy = mel_filterbank(sr, n_fft, n_mels) @ power
    log_mel = 10.0 * np.log10(np.maximum(mel_energy, 1e-10))
    coeffs = dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]
    return MfccResult(coeffs=coeffs, times=spec.times)


# --------------------------------------------------------------------------
# One-stop analysis of a recording
# --------------------------------------------------------------------------
@dataclass
class SignalReport:
    seconds: float
    peak: float
    rms_db: float                 # average loudness in dBFS
    spectrogram: Spectrogram
    ltas_db: np.ndarray
    centroid_hz: float
    pitch: PitchTrack
    mfcc: MfccResult


def analyze(y: np.ndarray, sr: int) -> SignalReport:
    """Run every analysis on one recording."""
    y = np.asarray(y, dtype=np.float64)
    spec = stft(y, sr)
    rms = float(np.sqrt(np.mean(y**2))) if y.size else 0.0
    return SignalReport(
        seconds=len(y) / sr,
        peak=float(np.max(np.abs(y))) if y.size else 0.0,
        rms_db=20.0 * np.log10(max(rms, 1e-10)),
        spectrogram=spec,
        ltas_db=long_term_average_spectrum(spec),
        centroid_hz=spectral_centroid(spec),
        pitch=estimate_pitch(y, sr),
        mfcc=mfcc(y, sr),
    )


# --------------------------------------------------------------------------
# Comparing two recordings
# --------------------------------------------------------------------------
def mean_mfcc(result: MfccResult, skip_c0: bool = True) -> np.ndarray:
    """Average MFCC vector over time. c0 (overall energy) is skipped because
    we care about spectral shape, not loudness."""
    start = 1 if skip_c0 else 0
    return result.coeffs[start:].mean(axis=1)


def mfcc_distance(a: MfccResult, b: MfccResult) -> float:
    """Euclidean distance between the average MFCC vectors (0 = identical)."""
    return float(np.linalg.norm(mean_mfcc(a) - mean_mfcc(b)))


def _half_baseline(report: SignalReport) -> float | None:
    """Distance between the first and second half of the SAME recording.

    This is a yardstick: even one speaker's voice changes a bit between
    different words, so some distance is normal.
    """
    c = report.mfcc.coeffs
    if c.shape[1] < 40:
        return None
    mid = c.shape[1] // 2
    first = c[1:, :mid].mean(axis=1)
    second = c[1:, mid:].mean(axis=1)
    return float(np.linalg.norm(first - second))


def compare(reference: SignalReport, generated: SignalReport) -> dict:
    """Numbers describing how `generated` differs from `reference`.

    These are simple signal measurements. They are NOT a proof that the voices
    sound the same to a listener.
    """
    ref_p, gen_p = reference.pitch.stats(), generated.pitch.stats()

    if np.isnan(ref_p["median_hz"]) or np.isnan(gen_p["median_hz"]):
        pitch_shift = float("nan")
    else:
        pitch_shift = float(12 * np.log2(gen_p["median_hz"] / ref_p["median_hz"]))

    n = min(len(reference.ltas_db), len(generated.ltas_db))
    freqs = reference.spectrogram.freqs[:n]
    band = freqs <= 8000.0
    a, b = reference.ltas_db[:n][band], generated.ltas_db[:n][band]
    if a.std() < 1e-9 or b.std() < 1e-9:
        ltas_corr = float("nan")
    else:
        ltas_corr = float(np.corrcoef(a, b)[0, 1])

    return {
        "duration_ref_s": reference.seconds,
        "duration_gen_s": generated.seconds,
        "pitch_ref": ref_p,
        "pitch_gen": gen_p,
        "pitch_shift_semitones": pitch_shift,
        "centroid_ref_hz": reference.centroid_hz,
        "centroid_gen_hz": generated.centroid_hz,
        "rms_ref_db": reference.rms_db,
        "rms_gen_db": generated.rms_db,
        "mfcc_distance": mfcc_distance(reference.mfcc, generated.mfcc),
        "mfcc_baseline": _half_baseline(reference),
        "ltas_correlation": ltas_corr,
    }
