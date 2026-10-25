> **VoiceLab 2.0 update:** See [DEPLOYMENT.md](DEPLOYMENT.md) for Chatterbox integration and public deployment. The older XTTS-v2 instructions below are preserved as a backup.

# 🎙️ VoiceLab

A Signal and Image Processing (SIP) college project. You give it a short recording of
**your own voice** (or a volunteer who agreed), generate new speech in a similar voice,
and then **analyse both signals** with waveforms, spectrograms, pitch tracking and MFCCs.

---

## Project status (read this first)

| Part | Status |
|---|---|
| Streamlit interface (upload, text, playback, downloads, error messages) | Written. Its Python logic is tested with a stand-in Streamlit (13 scenarios). **Not yet seen in a real browser.** |
| Preprocessing (stereo→mono, resample, trim silence, normalize) | Done. Unit-tested. |
| SIP analysis (waveform, spectrogram, pitch, MFCC, comparison) | Done. Unit-tested on signals with known answers. |
| Demo audio (synthetic, to try everything without a microphone) | Done. |
| Voice cloning with XTTS-v2 in Google Colab | Notebook written and syntax-checked. **Never run end to end.** |
| Generating speech directly inside VoiceLab (no Colab hand-off) | **Not built yet** (planned next step). The Generate button says so. |
| Windows launcher (`run_windows.bat`) | Written. **Not yet tested on Windows.** |

Automated checks so far: 63 tests pass on Linux with Python 3.13. Nothing has been tested on
Windows yet, and the MP3/FLAC reading path (needs the `soundfile` package) has not been exercised.

---

## 1. Run VoiceLab on Windows, step by step

You need: Windows 10 or 11, an internet connection, and a web browser. Nothing else is assumed.

### Step 1: Install Python (one time only)

1. Open <https://www.python.org/downloads/> and click the big yellow **Download Python 3.x** button.
   (Python 3.12 or 3.13 is fine. If you prefer 3.11, note that python.org may only offer it as source code
   for newer releases. Using 3.12 or 3.13 is easier.)
2. Double-click the downloaded file to start the installer.
3. **IMPORTANT:** on the first screen, tick **“Add python.exe to PATH”** at the bottom. Then click **Install Now**.
4. When it says “Setup was successful”, click **Close**.

*Already have Python 3.11, 3.12 or 3.13? You can skip this step.*

### Step 2: Unzip the project

1. Find the file `VoiceLab.zip` (probably in your Downloads folder).
2. Right-click it → **Extract All…** → **Extract**.
3. Open the new folder. You should see files such as `app.py`, `run_windows.bat`, and `README.md`.

> Do not run the project from *inside* the ZIP. It must be extracted first.

### Step 3: Start VoiceLab

1. Double-click **`run_windows.bat`**.
2. If Windows shows **“Windows protected your PC”**, click **More info → Run anyway**. (The file is a plain
   text script, and you can open it in Notepad to read it.)
3. A black window opens and prints some text. **The first time, it downloads and installs packages, which can take
   several minutes. This is normal.** Leave it open.
4. When it says **“Local URL: http://localhost:8501”**, your browser opens VoiceLab automatically.
   If a message in the black window asks for an *Email address*, just press **Enter**.
5. To stop VoiceLab, close the black window.

Next time, double-click `run_windows.bat` again. It starts in a few seconds.

### Step 4: Try it with the demo audio (no microphone needed)

1. In the **left sidebar**, click **Load demo audio**.
2. Open the **📈 SIP Analysis** tab. You will see the waveform, spectrogram, pitch, MFCC and Comparison sub-tabs.
3. Click **“In simple words”** under each plot for an explanation you can use in your presentation.

The demo files are **artificial vowel-like sounds, not real speech and not a cloned voice**. They exist only so you can
explore the interface.

### Step 5: Use your own voice

1. Record 10–30 seconds of clear speech on your phone or with the Windows **Voice Recorder** app. Speak normally, in a quiet room.
2. In VoiceLab’s **🎛️ Studio** tab, upload the recording. WAV is best.
3. Check the numbers and plots. VoiceLab shows what it did (mono, resampling, silence trimming, normalizing).
4. Click **Download cleaned reference (for Colab)**. This saves `reference_clean.wav`.

### Step 6: Generate cloned speech in Google Colab

Colab is a free Google service that lends you a powerful computer. Your laptop does not have an NVIDIA GPU, so this is the practical way.

1. Go to <https://colab.research.google.com> and sign in with a Google account.
2. Choose **File → Upload notebook** and select `colab/VoiceLab_XTTS_v2.ipynb` from the VoiceLab folder.
3. Choose **Runtime → Change runtime type → T4 GPU → Save**.
4. Follow the steps written inside the notebook, running each cell **one at a time** with its ▶ button.
5. At the end, download `generated.wav`.

### Step 7: Compare the original with the generated speech

1. Back in VoiceLab, open the **📈 SIP Analysis** tab.
2. Under **Generated speech**, upload `generated.wav`.
3. Look at every sub-tab, especially **Comparison**.

---

## 2. If the launcher does not work: the manual way

Use **Command Prompt** (press the Windows key, type `cmd`, press Enter), not PowerShell.

```
cd Downloads\VoiceLab
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Change `Downloads\VoiceLab` to wherever you extracted the folder. If `py` is not recognised, try `python` instead of `py -3`.

### Common problems

| Problem | What to do |
|---|---|
| “Python was not found” in the black window | Do Step 1 again and make sure you tick **Add python.exe to PATH** |
| The black window closes immediately | Open Command Prompt and use the manual way above, so you can read the error |
| `pip install` fails | Check your internet connection, then run the launcher again. Copy the error text if it keeps failing |
| Browser does not open | Copy the **Local URL** from the black window into your browser |
| “Could not read this file” in the app | Upload a WAV file. Other formats need the `soundfile` package (it is in `requirements.txt`) |
| Charts do not appear | Click **Load demo audio** to test. If that also fails, copy the error text |

---

## 3. Project structure

```
VoiceLab/
├── app.py                      # the Streamlit interface (what you see)
├── run_windows.bat             # double-click launcher for Windows
├── requirements.txt            # Python packages
├── README.md                   # this file
├── .streamlit/config.toml      # dark theme, upload size limit
├── voicelab/                   # the project's own code
│   ├── audio_utils.py          # loading + preprocessing
│   ├── analysis.py             # STFT, pitch, MFCC, comparison (NumPy/SciPy only)
│   ├── plots.py                # all figures
│   ├── explanations.py         # plain-language texts shown in the app
│   └── synthesis.py            # voice-model interface (not connected yet)
├── colab/
│   └── VoiceLab_XTTS_v2.ipynb  # voice cloning notebook for Google Colab
├── samples/                    # synthetic demo audio
├── tools/make_demo_audio.py    # regenerates the demo audio
└── tests/                      # automatic tests
```

Run the tests (after Step 3, in Command Prompt inside the folder, with `.venv\Scripts\activate` done):

```
python -m unittest discover -s tests -v
```

---

## 4. How it works (explanations for your presentation)

### The pipeline

```
reference recording ─► preprocessing ─┬─► Colab: XTTS-v2 ─► generated.wav ─┐
                                      │                                    ├─► SIP analysis + comparison
                                      └────────────────────────────────────┘
```

### What is digital audio?

Sound is air pressure changing over time. A microphone measures it thousands of times per second.
Each measurement is a **sample**, and the number of samples per second is the **sample rate**
(for example 44,100 Hz). VoiceLab stores audio as a list of numbers between −1 and +1.

### Preprocessing (`audio_utils.py`)

1. **Stereo → mono.** Average the left and right channels, `(L + R) / 2`.
2. **Remove DC offset.** Subtract the mean so the waveform is centred on zero.
3. **Resample to 22,050 Hz.** Speech has little useful energy above about 8 kHz. SciPy’s polyphase resampler
   low-pass filters first so no false frequencies (aliasing) appear.
4. **Trim silence.** Cut the signal into 25 ms frames, measure each frame’s RMS energy in decibels, and remove
   the start and end where everything is more than 30 dB below the loudest frame. A 100 ms margin protects the word edges.
5. **Normalize.** Scale so the loudest sample is 0.95: comparable loudness, no clipping.

### Voice cloning (XTTS-v2)

A **pretrained** neural network (XTTS-v2) reads a few seconds of the reference voice, turns it into a
“voice fingerprint”, and then speaks new text using that fingerprint. We reuse a pretrained model because
training one needs many hours of data and powerful GPUs. XTTS-v2 runs in Google Colab for speed. It is
released under the Coqui Public Model License, which is reported to be non-commercial. Check the model page.

### SIP analysis (`analysis.py`)

All implemented with NumPy/SciPy, so every step is visible:

- **Waveform:** amplitude vs time.
- **Spectrogram (STFT):** frames of 1024 samples × Hann window → FFT → magnitude in dB. Shows harmonics
  (horizontal stripes) and formants (moving bright bands).
- **Pitch (F0):** autocorrelation of each 40 ms frame. The first strong peak between 1/400 s and 1/70 s gives the period
  *T*, so F0 = 1/T. Weak or silent frames are “unvoiced”. A median filter smooths the track.
- **MFCCs:** pre-emphasis → STFT → mel filterbank (40 triangular filters) → log → DCT, keeping 13 coefficients.
  They describe the broad spectral shape = vocal tract = tone colour.
- **Comparison:** pitch shift in semitones, distance between average MFCC vectors (with a same-speaker
  “yardstick”: first half vs second half of the reference), spectrum correlation, brightness (spectral centroid).

### Honest limitations to mention

- The comparison numbers are **simple signal measurements**, not a proof that the voices sound alike. Listening matters.
- The MFCC distance compares *average* tone colour and ignores timing. A stricter method (DTW alignment or a
  speaker-embedding model) could be future work.
- Pitch tracking is tuned for speech between 70 and 400 Hz and can mistake a harmonic for the fundamental on some frames.
- Voice cloning quality depends heavily on a clean 10–30 s reference.

### Responsible use

Only clone a voice you own or whose owner has clearly agreed. The app and the notebook both ask you to confirm this.

---

## 5. Next steps (roadmap)

1. **Generate inside VoiceLab (your final goal).** Run the Colab notebook as a small web service behind a public tunnel,
   and have the **Generate Speech** button call it. The interface already handles the “not available” case, so this plugs into
   `voicelab/synthesis.py`.
2. Add an optional CPU mode (slow) for computers without Colab.
3. Add DTW-aligned MFCC comparison.


## VoiceLab 2.0 visual refresh
The Streamlit interface has a purple orbital hero, glass-inspired cards, responsive layout, and direct audio playback in the comparison tab. The voice generation endpoint is unchanged and requires a separately available GPU service.
