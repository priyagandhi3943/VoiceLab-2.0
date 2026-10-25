"""Create two synthetic demo recordings so you can try VoiceLab without a microphone.

    python tools/make_demo_audio.py

Output (in the samples/ folder):
    demo_reference_voice.wav   stereo, 44.1 kHz, ~120 Hz buzz-like "voice" with silence at both ends
    demo_generated_voice.wav   mono, 24 kHz, ~150 Hz, slightly different vowel colours

IMPORTANT: these are artificial vowel-like sounds made with a simple source-filter
model. They are NOT real speech and NOT cloned voices. They only exist to let you
test the interface and the SIP Analysis tab.

Source-filter model (a nice thing to explain in your presentation):
  * source: a sawtooth wave at the pitch F0 (like vibrating vocal cords)
  * filter: three resonances ("formants") that shape each vowel, like the mouth
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter

# (F1, F2, F3) in Hz for five vowels, typical textbook values
FORMANTS = {
    "a": (730, 1090, 2440),
    "e": (530, 1840, 2480),
    "i": (270, 2290, 3010),
    "o": (570, 840, 2410),
    "u": (300, 870, 2240),
}


def resonator(x, fc, bw, sr):
    """Second-order resonant filter centred at fc Hz with bandwidth bw Hz."""
    r = np.exp(-np.pi * bw / sr)
    theta = 2 * np.pi * fc / sr
    a = [1.0, -2 * r * np.cos(theta), r * r]
    b = [1.0 - r]
    return lfilter(b, a, x)


def synth(seconds, sr, f0_base, formant_scale, vowels, seed):
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr

    # slowly varying pitch: gentle wobble plus a fall at the end of the phrase
    f0 = f0_base * (1 + 0.06 * np.sin(2 * np.pi * 0.7 * t) + 0.04 * np.sin(2 * np.pi * 2.1 * t))
    f0 *= 1 - 0.12 * np.clip((t - seconds * 0.8) / (seconds * 0.2), 0, 1)
    phase = np.cumsum(f0) / sr
    source = 2.0 * (phase % 1.0) - 1.0

    out = np.zeros(n)
    seg_len = 0.55  # seconds per vowel
    gap = 0.12      # silence between vowels
    start, k = 0.3, 0
    while start + seg_len < seconds - 0.2:
        f1, f2, f3 = (f * formant_scale for f in FORMANTS[vowels[k % len(vowels)]])
        voiced = source.copy()
        for fc, bw in ((f1, 80), (f2, 100), (f3, 140)):
            voiced = resonator(voiced, fc, bw, sr) if fc < sr / 2 - 200 else voiced
        # smooth on/off envelope for this vowel
        env = np.zeros(n)
        i0, i1 = int(start * sr), int((start + seg_len) * sr)
        # give every vowel the same loudness (the filters alone make some much quieter)
        voiced = voiced / max(float(np.sqrt(np.mean(voiced[i0:i1] ** 2))), 1e-9)
        ramp = int(0.04 * sr)
        env[i0:i1] = 1.0
        env[i0 : i0 + ramp] = np.linspace(0, 1, ramp)
        env[i1 - ramp : i1] = np.linspace(1, 0, ramp)
        out += voiced * env
        start += seg_len + gap
        k += 1

    out = out / np.max(np.abs(out))
    out += 0.003 * rng.standard_normal(n)  # faint background noise (about -50 dB)
    return out / np.max(np.abs(out))


def save(path, audio, sr):
    pcm = (np.clip(audio, -1, 1) * 0.8 * 32767).astype(np.int16)
    wavfile.write(path, sr, pcm)
    print(f"wrote {path}  ({pcm.shape[0] / sr:.1f} s, {sr} Hz, {pcm.ndim if pcm.ndim == 1 else pcm.shape[1]} ch)")


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, "samples")
    os.makedirs(out_dir, exist_ok=True)

    sr_ref = 44100
    ref_mono = synth(9.0, sr_ref, f0_base=120, formant_scale=1.0, vowels="aeiou", seed=1)
    ref = np.stack([ref_mono, 0.9 * ref_mono], axis=1)  # stereo
    save(os.path.join(out_dir, "demo_reference_voice.wav"), ref, sr_ref)

    sr_gen = 24000  # the sample rate XTTS-v2 outputs
    gen = synth(8.0, sr_gen, f0_base=150, formant_scale=1.1, vowels="eioua", seed=2)
    save(os.path.join(out_dir, "demo_generated_voice.wav"), gen, sr_gen)


if __name__ == "__main__":
    sys.exit(main())
