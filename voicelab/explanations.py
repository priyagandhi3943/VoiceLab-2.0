"""Plain-language explanations shown in the app (and handy for your presentation)."""

WAVEFORM = """
**What you are looking at:** the raw signal. The horizontal axis is time and the
vertical axis is amplitude (how far the air pressure moves). Loud parts are tall,
quiet parts are flat.

**Why it matters:** it is the quickest check of a recording: is it too quiet, is it
clipped (flat tops at ±1), how long are the pauses? Comparing original and
generated waveforms shows differences in rhythm, pauses and loudness, but it
cannot tell you much about *voice quality*. The next plots do that.
"""

SPECTROGRAM = """
**What you are looking at:** a picture of sound. Time runs left to right, frequency
runs bottom to top, and colour shows how much energy there is at that frequency and
moment (bright = strong).

**How it is made:** the Short-Time Fourier Transform (STFT). The signal is cut into
short overlapping frames, each frame is multiplied by a Hann window, and the Fourier
transform of each frame is taken.

**What to look for:** horizontal stripes are *harmonics* of the voice (multiples of
the pitch). Thick bright bands that move are *formants*, the resonances of the
mouth and throat that make vowels sound different. A cloudy vertical smear is a
consonant like "s" or "sh".
"""

PITCH = """
**What you are looking at:** the fundamental frequency (F0), which is how fast the
vocal cords vibrate, measured in Hertz. It is what we hear as the "note" of the voice.
Typical ranges: adult male about 85-180 Hz, adult female about 165-255 Hz.

**How it is measured:** autocorrelation. A voiced sound repeats every period T. If
you compare a short frame with a delayed copy of itself, the match is best when the
delay equals T, so F0 = 1 / T. Frames that are noisy or silent have no clear
period and are left blank ("unvoiced").

**What to look for:** the generated voice should sit in a similar pitch range to
the original. The *shape* of the line is the intonation (rise and fall), which
usually follows the text rather than the reference speaker.
"""

MFCC = """
**What you are looking at:** MFCCs (Mel-Frequency Cepstral Coefficients), a compact
description of the *shape* of the spectrum. Each column is one moment in time; each
row is one coefficient.

**How they are made:** spectrum, then a mel filterbank (spacing that mimics human
hearing), then log, then a DCT (a kind of Fourier transform of the spectrum itself).
The low coefficients capture the broad spectral shape, which mostly reflects the
vocal tract. That is why MFCCs are the classic features for speaker and speech
recognition.

**How to read it:** row 0 follows loudness. Rows 1-12 describe tone colour. Two
recordings of the same voice tend to have similar *average* MFCC values.
"""

COMPARISON = """
**How to read these numbers:** they are simple measurements, not a verdict on how the
voice sounds to a human listener. Always listen to both files too.

- **Pitch shift** in semitones: 0 means the median pitch matches; 12 is one octave.
- **MFCC distance** between the two average MFCC vectors: smaller means more similar
  tone colour. The *yardstick* is the same distance measured between the first and
  second half of the reference itself. Distances close to or below that are
  "within normal variation of the speaker".
- **Spectrum correlation** (long-term average spectrum, 0-8 kHz): closer to 1 means the
  overall spectral shape is alike.
- **Brightness** (spectral centroid): the centre of mass of the spectrum in Hz.
"""
