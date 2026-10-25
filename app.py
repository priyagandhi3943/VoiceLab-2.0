"""VoiceLab 2.0 - Immersive Futuristic Signal Intelligence & Speech Studio.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from voicelab import analysis as an
from voicelab import explanations, synthesis
from voicelab.audio_utils import (
    TARGET_SR,
    AudioError,
    load_audio,
    preprocess,
    to_wav_bytes,
)
from voicelab.plots import (
    ACCENT,
    ACCENT_2,
    ltas_figure,
    mfcc_figure,
    pitch_figure,
    spectrogram_figure,
    waveform_figure,
)
from voicelab.frontend.sculpture import render_hero_sculpture
from voicelab.frontend.generation import render_generation_journey
from voicelab.frontend.audio_player import render_wave_player
from voicelab.frontend.comparison import render_comparison_deck

ROOT = Path(__file__).resolve().parent
DEMO_REF = ROOT / "samples" / "demo_reference_voice.wav"
DEMO_GEN = ROOT / "samples" / "demo_generated_voice.wav"
AUDIO_TYPES = ["wav", "flac", "ogg", "mp3"]

st.set_page_config(
    page_title="VoiceLab 2.0 // Neural Signal Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- luxury styling
st.markdown(
    r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,700;1,400&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  color-scheme: dark;
}

/* Global App Container */
html, body, [data-testid="stAppViewContainer"] {
  font-family: 'DM Sans', sans-serif !important;
  background: radial-gradient(ellipse 70% 40% at 75% 2%, rgba(76, 29, 149, 0.28) 0%, transparent 70%),
              radial-gradient(ellipse 50% 50% at 15% 90%, rgba(59, 7, 100, 0.3) 0%, transparent 65%),
              linear-gradient(160deg, #07040e 0%, #110820 45%, #090514 100%) !important;
  color: #f5f0ff !important;
}

[data-testid="stHeader"] {
  background: transparent !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #120924 0%, #0a0614 100%) !important;
  border-right: 1px solid rgba(192, 132, 252, 0.16) !important;
}

.block-container {
  max-width: 1360px !important;
  padding-top: 1.2rem !important;
  padding-bottom: 5rem !important;
}

h1, h2, h3, h4 {
  font-family: 'Space Grotesk', sans-serif !important;
  letter-spacing: -0.035em !important;
  color: #ffffff !important;
}

p, label, .stCaption {
  color: #cbbad9 !important;
}

/* Custom Tab styling */
[data-testid="stTabs"] [role="tablist"] {
  gap: 0.6rem;
  flex-wrap: wrap;
  padding: 6px;
  background: rgba(18, 9, 32, 0.6);
  border: 1px solid rgba(192, 132, 252, 0.18);
  border-radius: 100px;
  backdrop-filter: blur(12px);
  margin-bottom: 1.8rem;
}

button[data-baseweb="tab"] {
  background: transparent !important;
  border: 1px solid transparent !important;
  border-radius: 100px !important;
  padding: 0.65rem 1.45rem !important;
  color: #d1b8eb !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.02em !important;
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

button[data-baseweb="tab"]:hover {
  color: #ffffff !important;
  background: rgba(147, 51, 234, 0.15) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, #a855f7 0%, #7e22ce 100%) !important;
  color: #ffffff !important;
  border-color: rgba(233, 213, 255, 0.35) !important;
  box-shadow: 0 4px 18px rgba(168, 85, 247, 0.45) !important;
}

/* Widgets */
[data-testid="stFileUploader"], [data-testid="stTextArea"], [data-testid="stDataFrame"], [data-testid="stAudioInput"] {
  border-radius: 18px !important;
  background: rgba(22, 12, 38, 0.5) !important;
  border: 1px solid rgba(192, 132, 252, 0.2) !important;
  backdrop-filter: blur(12px) !important;
}

[data-testid="stMetric"], [data-testid="stExpander"] {
  background: rgba(24, 12, 42, 0.6) !important;
  border: 1px solid rgba(192, 132, 252, 0.2) !important;
  border-radius: 18px !important;
  backdrop-filter: blur(10px) !important;
}

/* Buttons */
div.stButton > button[kind="primary"], div.stDownloadButton > button {
  background: linear-gradient(115deg, #a855f7 0%, #d946ef 100%) !important;
  color: #ffffff !important;
  border: 1px solid rgba(255, 255, 255, 0.25) !important;
  box-shadow: 0 8px 25px rgba(168, 85, 247, 0.38) !important;
  border-radius: 100px !important;
  padding: 0.75rem 1.6rem !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: 0.03em !important;
  transition: transform 0.2s, box-shadow 0.2s !important;
}

div.stButton > button {
  border-radius: 100px !important;
  padding: 0.65rem 1.3rem !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 500 !important;
  border: 1px solid rgba(192, 132, 252, 0.28) !important;
  background: rgba(30, 15, 52, 0.5) !important;
  color: #e9d5ff !important;
  transition: transform 0.2s, box-shadow 0.2s !important;
}

div.stButton > button:hover {
  transform: translateY(-2px) !important;
  border-color: #d8b4fe !important;
  box-shadow: 0 8px 22px rgba(168, 85, 247, 0.3) !important;
}

/* Metric HUD row */
.metric-row {
  display: flex;
  gap: 0.8rem;
  flex-wrap: wrap;
  margin: 0.8rem 0 1.4rem;
}

.metric {
  flex: 1;
  min-width: 125px;
  padding: 0.9rem 1.1rem;
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(32, 16, 56, 0.75) 0%, rgba(18, 9, 32, 0.85) 100%);
  border: 1px solid rgba(192, 132, 252, 0.22);
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3), inset 0 1px rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
}

.metric .label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.64rem;
  color: #a78bfa;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.metric .value {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.35rem;
  font-weight: 600;
  color: #ffffff;
  margin-top: 2px;
}

/* Luxury Top Branding Bar */
.vl-nav-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.9rem 0 1.5rem;
  border-bottom: 1px solid rgba(192, 132, 252, 0.16);
  margin-bottom: 1.6rem;
}

.vl-brand {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 1.3rem;
  letter-spacing: -0.03em;
  color: #ffffff;
  display: flex;
  align-items: center;
  gap: 10px;
}

.vl-brand span.highlight {
  background: linear-gradient(135deg, #f5d0fe 0%, #c084fc 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.vl-brand span.version {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: #c084fc;
  border: 1px solid rgba(192, 132, 252, 0.3);
  padding: 2px 7px;
  border-radius: 100px;
}

.vl-nav-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.status-pill {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.08em;
  padding: 5px 12px;
  border-radius: 100px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.status-pill.online {
  background: rgba(52, 211, 153, 0.15);
  border: 1px solid rgba(52, 211, 153, 0.4);
  color: #a7f3d0;
}

.status-pill.offline {
  background: rgba(245, 158, 11, 0.15);
  border: 1px solid rgba(245, 158, 11, 0.35);
  color: #fde68a;
}

.dot-indicator {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: currentColor;
  box-shadow: 0 0 6px currentColor;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- helpers
def metric_row(items: list[tuple[str, str]]) -> None:
    cells = "".join(
        f'<div class="metric"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="metric-row">{cells}</div>', unsafe_allow_html=True)


def show(fig) -> None:
    """Display a matplotlib figure, then free its memory."""
    st.pyplot(fig)
    plt.close(fig)


def fmt(value, digits: int = 1, unit: str = "") -> str:
    """Format a number for display; NaN/None become a dash."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    return f"{value:.{digits}f}{unit}"


@st.cache_data(show_spinner=False)
def cached_report(audio: np.ndarray, sr: int) -> an.SignalReport:
    """Analysis is the slow part, so remember the result between reruns."""
    return an.analyze(audio, sr)


# ---------------------------------------------------------------- sidebar setup
with st.sidebar:
    st.markdown("### ⌁ Audio Preprocessing")
    do_trim = st.toggle("Trim silence", value=True)
    top_db = st.slider(
        "Silence threshold (dB)",
        10,
        60,
        30,
        help="Parts quieter than this, relative to the loudest moment, count as silence. "
        "A smaller number trims more aggressively.",
        disabled=not do_trim,
    )
    do_normalize = st.toggle("Normalize volume", value=True)
    st.caption(f"Audio is converted to mono at {TARGET_SR} Hz.")

    st.divider()
    st.markdown("### ◇ Demo Workspace")
    if DEMO_REF.exists() and DEMO_GEN.exists():
        if st.session_state.get("demo"):
            st.success("Demo audio is loaded.")
            if st.button("Stop using demo audio"):
                st.session_state["demo"] = False
                st.rerun()
        else:
            if st.button("Load demo audio"):
                st.session_state["demo"] = True
                st.rerun()
        st.caption(
            "Two synthetic vowel sounds (not real speech, not a cloned voice) "
            "so you can explore every plot."
        )
    else:
        st.caption("Demo files missing. Run: python tools/make_demo_audio.py")

    st.divider()
    backend_ok, backend_reason = synthesis.backend_status()
    st.markdown("### ◉ Chatterbox Connection")
    if backend_ok:
        st.success(backend_reason)
    else:
        st.warning(backend_reason)

    with st.expander("⚙ Backend Tunnel Settings", expanded=not backend_ok):
        st.caption("Connect your Colab T4 GPU tunnel:")
        curr_u, curr_t = synthesis._settings()
        input_url = st.text_input("Backend URL (HTTPS)", value=curr_u, placeholder="https://xxxx.trycloudflare.com", key="cfg_backend_url")
        input_token = st.text_input("API Token", value=curr_t, type="password", key="cfg_backend_token")
        if st.button("Apply & Test"):
            st.session_state["VOICELAB_API_URL"] = input_url.strip()
            st.session_state["VOICELAB_API_TOKEN"] = input_token.strip()
            st.rerun()


def run_preprocessing(data: bytes, name: str, *, check_length: bool):
    raw, sr = load_audio(data, name)
    return preprocess(
        raw,
        sr,
        do_trim=do_trim,
        top_db=float(top_db),
        do_normalize=do_normalize,
        check_length=check_length,
    )


# ---------------------------------------------------------------- top navigation
backend_status_pill = (
    '<div class="status-pill online"><div class="dot-indicator"></div>GPU ONLINE</div>'
    if backend_ok
    else '<div class="status-pill offline"><div class="dot-indicator"></div>GPU STANDBY</div>'
)

st.markdown(
    f"""
<div class="vl-nav-bar">
  <div class="vl-brand">
    ◉ VOICE<span class="highlight">LAB</span> <span class="version">2.0</span>
  </div>
  <div class="vl-nav-right">
    {backend_status_pill}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- tabs
tab_exp, tab_studio, tab_analysis, tab_compare, tab_about = st.tabs(
    [
        "✦ Experience",
        "🎙 Voice Studio",
        "⌁ Signal Intelligence",
        "⚖ Voice Comparison",
        "◇ Methodology",
    ]
)

reference = None      # ProcessedAudio of reference voice
ref_source = None     # "upload" or "demo" or "mic"

# ================================================================= TAB 1: EXPERIENCE
with tab_exp:
    render_hero_sculpture(height=540)

    st.markdown("#### The Architecture of Sound")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            """
            **01 // Neural Voice Cloning**  
            Zero-shot acoustic cloning powered by Chatterbox. Extracts speaker embeddings from short audio clips to condition speech synthesis.
            """
        )
    with col_b:
        st.markdown(
            """
            **02 // Signal Intelligence**  
            Mathematical DSP laboratory using pure NumPy/SciPy: Short-Time Fourier Transforms, Mel Filterbanks, Autocorrelation F0, and MFCCs.
            """
        )
    with col_c:
        st.markdown(
            """
            **03 // Comparative Analysis**  
            Measure pitch drift in semitones, Euclidean cepstral distance against baseline yardsticks, and Long-Term Average Spectrum correlation.
            """
        )

# ================================================================= TAB 2: VOICE STUDIO
with tab_studio:
    left, right = st.columns(2, gap="large")

    # ---- Left Column: Reference voice upload & Recording
    with left:
        st.subheader("01 / Reference Voice")
        uploaded = st.file_uploader(
            "Upload a short clear recording (6–30 seconds, one speaker)",
            type=AUDIO_TYPES,
            help="WAV works everywhere. Other formats need the soundfile package.",
        )

        mic_rec = None
        if hasattr(st, "audio_input"):
            mic_rec = st.audio_input("Or record with your microphone (6–30 seconds)", key="mic_input")

        ref_bytes, ref_name = None, ""
        if uploaded is not None:
            ref_bytes, ref_name, ref_source = uploaded.getvalue(), uploaded.name, "upload"
            st.session_state["ref_audio_store"] = (ref_bytes, ref_name, "upload")
        elif mic_rec is not None and hasattr(mic_rec, "getvalue") and len(mic_rec.getvalue()) > 0:
            ref_bytes, ref_name, ref_source = mic_rec.getvalue(), "mic_recording.wav", "mic"
            st.session_state["ref_audio_store"] = (ref_bytes, ref_name, "mic")
        elif st.session_state.get("ref_audio_store") is not None and not st.session_state.get("demo"):
            ref_bytes, ref_name, ref_source = st.session_state["ref_audio_store"]
        elif st.session_state.get("demo"):
            try:
                ref_bytes, ref_name, ref_source = DEMO_REF.read_bytes(), DEMO_REF.name, "demo"
            except OSError as exc:
                st.error(f"Could not read the demo file: {exc}")

        if ref_bytes is not None:
            try:
                with st.spinner("Cleaning up the recording..."):
                    reference = run_preprocessing(ref_bytes, ref_name, check_length=True)
            except AudioError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Something unexpected went wrong while reading the audio: {exc}")
        else:
            st.info("Upload a recording, or click “Load demo audio” in the sidebar.")

        if reference is not None:
            if ref_source == "demo":
                st.caption("Using the synthetic demo reference.")
            elif ref_source == "mic":
                st.caption("Using live microphone recording.")
            for w in reference.warnings:
                st.warning(w)

            metric_row(
                [
                    ("Original", f"{reference.original_seconds:.1f} s"),
                    ("Cleaned", f"{reference.processed_seconds:.1f} s"),
                    ("Channels", f"{reference.original_channels} → 1"),
                    ("Sample rate", f"{reference.original_sr} → {reference.sr} Hz"),
                ]
            )

            clean_wav = to_wav_bytes(reference.audio, reference.sr)
            render_wave_player(reference.audio, reference.sr, clean_wav, title="Cleaned Reference Signal", accent_color="#c084fc")

            st.download_button(
                "↓ Download cleaned reference",
                data=clean_wav,
                file_name="reference_clean.wav",
                mime="audio/wav",
            )
            with st.expander("What was done to the recording?"):
                for step in reference.steps:
                    st.write(f"✓ {step}")

    # ---- Right Column: Text & Generate
    with right:
        st.subheader("02 / Speech Synthesis")
        saved_text = st.session_state.get("text_prompt", "")
        text = st.text_area(
            "What should the cloned voice say?",
            value=saved_text,
            placeholder="Type any sentence here...",
            max_chars=synthesis.MAX_TEXT_CHARS,
            height=160,
            key="studio_text_input",
        )
        if text != saved_text:
            st.session_state["text_prompt"] = text

        consent = st.checkbox(
            "This is my own voice, or the speaker has given me permission to clone it."
        )

        generate = st.button("✦ Generate Speech", type="primary")

        if generate:
            if reference is None:
                st.error("Please upload a valid reference recording first.")
            elif not text.strip():
                st.error("Please enter some text.")
            elif not consent:
                st.error("Please confirm you have permission to use this voice.")
            else:
                st.session_state["journey_state"] = "generating"
                progress = st.progress(10, text="Checking the voice model...")
                try:
                    progress.progress(40, text="Generating speech with Chatterbox GPU...")
                    audio, out_sr = synthesis.synthesize(
                        text.strip(), reference.audio, reference.sr
                    )
                    progress.progress(100, text="Done")
                    st.session_state["generated"] = (audio, out_sr, text.strip())
                    st.session_state["journey_state"] = "complete"
                    st.session_state["journey_error"] = ""
                    st.rerun()
                except synthesis.SynthesisUnavailable as exc:
                    progress.empty()
                    st.session_state["journey_state"] = "error"
                    st.session_state["journey_error"] = str(exc)
                    st.warning("Chatterbox is unavailable. Start the Colab backend and check its URL and token.")
                except Exception as exc:
                    progress.empty()
                    st.session_state["journey_state"] = "error"
                    st.session_state["journey_error"] = str(exc)
                    st.error(f"Speech generation failed: {exc}")

        # Active or generated output state
        generated_now = st.session_state.get("generated")
        if generated_now is not None:
            gen_audio, out_sr, spoken_text = generated_now
            st.success("Generated speech successfully created.")
            wav = to_wav_bytes(gen_audio, out_sr)
            render_wave_player(gen_audio, out_sr, wav, title="Neural Cloned Speech", accent_color="#34d399")
            st.download_button(
                "↓ Download generated WAV",
                data=wav,
                file_name="voicelab_output.wav",
                mime="audio/wav",
            )

        with st.expander("How live Chatterbox generation works", expanded=False):
            st.markdown(
                """
                1. Open `colab/VoiceLab_Chatterbox_Backend.ipynb` in Google Colab on a T4 GPU.
                2. Run all cells. It will output `VOICELAB_API_URL` and `VOICELAB_API_TOKEN`.
                3. Paste them into the sidebar **Backend Tunnel Settings** or your `.streamlit/secrets.toml`.
                4. Keep Colab running while generating speech in Voice Studio.
                """
            )

    # ---- Cinematic Generation Journey Stage
    if reference is not None:
        st.divider()
        st.subheader("Interactive Synthesis Transformation")
        journey_state = st.session_state.get("journey_state", "idle")
        journey_err = st.session_state.get("journey_error", "")
        gen_audio_journey = None
        gen_sr_journey = 22050
        if st.session_state.get("generated") is not None:
            journey_state = "complete"
            gen_audio_journey, gen_sr_journey, _ = st.session_state["generated"]

        render_generation_journey(
            state=journey_state,
            text=text.strip() if text else "Linguistic Latent Stream",
            reference_audio=reference.audio,
            reference_sr=reference.sr,
            generated_audio=gen_audio_journey,
            generated_sr=gen_sr_journey,
            error_msg=journey_err,
            height=380,
        )

        st.subheader("Before and after preprocessing")
        show(
            waveform_figure(
                [
                    ("Original (mono)", reference.original, reference.original_sr, ACCENT),
                    ("Preprocessed", reference.audio, reference.sr, ACCENT_2),
                ]
            )
        )

# ============================================================= TAB 3: SIP ANALYSIS
with tab_analysis:
    st.subheader("Signal Intelligence Laboratory")
    if reference is None:
        st.info(
            "Add a reference voice in the Studio tab first, or click “Load demo audio” "
            "in the sidebar."
        )
    else:
        st.caption(
            "Optional: upload the speech you generated (for example `generated.wav` "
            "from the Colab notebook) to compare it with the original."
        )
        gen_upload = st.file_uploader(
            "Generated speech",
            type=AUDIO_TYPES,
            key="generated_upload",
        )

        gen_bytes, gen_name = None, ""
        if gen_upload is not None:
            gen_bytes, gen_name = gen_upload.getvalue(), gen_upload.name
        elif st.session_state.get("generated") is not None:
            generated_audio, generated_sr, _ = st.session_state["generated"]
            gen_bytes, gen_name = to_wav_bytes(generated_audio, generated_sr), "voicelab_generated.wav"
        elif ref_source == "demo":
            try:
                gen_bytes, gen_name = DEMO_GEN.read_bytes(), DEMO_GEN.name
            except OSError as exc:
                st.error(f"Could not read the demo file: {exc}")

        generated = None
        if gen_bytes is not None:
            try:
                generated = run_preprocessing(gen_bytes, gen_name, check_length=False)
            except AudioError as exc:
                st.error(f"Generated audio: {exc}")
            except Exception as exc:
                st.error(f"Something unexpected went wrong while reading the generated audio: {exc}")

        ref_report = gen_report = None
        try:
            with st.spinner("Analysing the signals..."):
                ref_report = cached_report(reference.audio, reference.sr)
                if generated is not None:
                    gen_report = cached_report(generated.audio, generated.sr)
        except Exception as exc:
            st.error(f"The analysis failed: {exc}")

        if ref_report is not None:
            if generated is not None and ref_source == "demo" and gen_upload is None:
                st.caption("Comparing the two synthetic demo signals.")

            reports = [("Original voice", ref_report, ACCENT)]
            if gen_report is not None:
                reports.append(("Generated speech", gen_report, ACCENT_2))

            t_wave, t_spec, t_pitch, t_mfcc, t_cmp = st.tabs(
                ["Waveform", "Spectrogram", "Pitch", "MFCC", "Comparison"]
            )

            # ---- waveform
            with t_wave:
                rows = [("Original voice", reference.audio, reference.sr, ACCENT)]
                if generated is not None:
                    rows.append(("Generated speech", generated.audio, generated.sr, ACCENT_2))
                show(waveform_figure(rows))
                with st.expander("In simple words"):
                    st.markdown(explanations.WAVEFORM)

            # ---- spectrogram
            with t_spec:
                show(spectrogram_figure([(name, r.spectrogram) for name, r, _ in reports]))
                with st.expander("In simple words"):
                    st.markdown(explanations.SPECTROGRAM)

            # ---- pitch
            with t_pitch:
                cards = []
                for name, r, _ in reports:
                    s = r.pitch.stats()
                    cards.append((f"{name}: median", fmt(s["median_hz"], 0, " Hz")))
                    cards.append((f"{name}: range", fmt(s["range_semitones"], 1, " st")))
                    cards.append((f"{name}: voiced", fmt(100 * s["voiced_fraction"], 0, " %")))
                metric_row(cards)
                show(pitch_figure([(name, r.pitch, colour) for name, r, colour in reports]))
                with st.expander("In simple words"):
                    st.markdown(explanations.PITCH)

            # ---- MFCC
            with t_mfcc:
                show(mfcc_figure([(name, r.mfcc) for name, r, _ in reports]))
                with st.expander("In simple words"):
                    st.markdown(explanations.MFCC)

            # ---- comparison
            with t_cmp:
                if gen_report is None:
                    st.info(
                        "Upload generated speech above to see the comparison. "
                        "(Or load the demo audio from the sidebar.)"
                    )
                else:
                    st.markdown("#### Listen side by side")
                    listen_left, listen_right = st.columns(2)
                    with listen_left:
                        st.caption("Original reference")
                        st.audio(to_wav_bytes(reference.audio, reference.sr), format="audio/wav")
                    with listen_right:
                        st.caption("Generated / comparison recording")
                        st.audio(to_wav_bytes(generated.audio, generated.sr), format="audio/wav")
                    cmp = an.compare(ref_report, gen_report)
                    rp, gp = cmp["pitch_ref"], cmp["pitch_gen"]

                    metric_row(
                        [
                            ("Pitch shift", fmt(cmp["pitch_shift_semitones"], 1, " st")),
                            ("MFCC distance", fmt(cmp["mfcc_distance"], 1)),
                            ("Yardstick", fmt(cmp["mfcc_baseline"], 1)),
                            ("Spectrum correlation", fmt(cmp["ltas_correlation"], 2)),
                        ]
                    )
                    table = pd.DataFrame(
                        {
                            "Measurement": [
                                "Duration",
                                "Median pitch",
                                "Pitch range (5th-95th pct)",
                                "Voiced frames",
                                "Brightness (spectral centroid)",
                                "Average level",
                            ],
                            "Original": [
                                fmt(cmp["duration_ref_s"], 1, " s"),
                                fmt(rp["median_hz"], 0, " Hz"),
                                fmt(rp["range_semitones"], 1, " st"),
                                fmt(100 * rp["voiced_fraction"], 0, " %"),
                                fmt(cmp["centroid_ref_hz"], 0, " Hz"),
                                fmt(cmp["rms_ref_db"], 1, " dBFS"),
                            ],
                            "Generated": [
                                fmt(cmp["duration_gen_s"], 1, " s"),
                                fmt(gp["median_hz"], 0, " Hz"),
                                fmt(gp["range_semitones"], 1, " st"),
                                fmt(100 * gp["voiced_fraction"], 0, " %"),
                                fmt(cmp["centroid_gen_hz"], 0, " Hz"),
                                fmt(cmp["rms_gen_db"], 1, " dBFS"),
                            ],
                        }
                    )
                    st.dataframe(table, hide_index=True)
                    st.caption(
                        "These are signal measurements, not proof that the voices sound alike. "
                        "Always listen to both recordings as well."
                    )

                    freqs = ref_report.spectrogram.freqs
                    gen_freqs = gen_report.spectrogram.freqs
                    show(
                        ltas_figure(
                            [
                                ("Original voice", freqs, ref_report.ltas_db, ACCENT),
                                ("Generated speech", gen_freqs, gen_report.ltas_db, ACCENT_2),
                            ]
                        )
                    )
                    with st.expander("How to read these numbers"):
                        st.markdown(explanations.COMPARISON)

# ============================================================= TAB 4: VOICE COMPARISON
with tab_compare:
    st.subheader("Dual-Deck Synchronized Acoustic Comparison")
    if reference is None or ref_report is None:
        st.info("Load or upload a reference recording first.")
    elif gen_report is None or generated is None:
        st.info("Upload generated speech in the Studio or Signal Intelligence tab to view comparison.")
    else:
        cmp = an.compare(ref_report, gen_report)
        rp, gp = cmp["pitch_ref"], cmp["pitch_gen"]

        ref_stats = {
            "median_hz": rp.get("median_hz", 0),
            "range_semitones": rp.get("range_semitones", 0),
            "voiced_fraction": rp.get("voiced_fraction", 0),
            "rms_db": cmp.get("rms_ref_db", 0),
            "centroid_hz": cmp.get("centroid_ref_hz", 0),
        }
        gen_stats = {
            "median_hz": gp.get("median_hz", 0),
            "range_semitones": gp.get("range_semitones", 0),
            "voiced_fraction": gp.get("voiced_fraction", 0),
            "rms_db": cmp.get("rms_gen_db", 0),
            "centroid_hz": cmp.get("centroid_gen_hz", 0),
        }

        render_comparison_deck(
            ref_audio=reference.audio,
            ref_sr=reference.sr,
            gen_audio=generated.audio,
            gen_sr=generated.sr,
            cmp_stats=cmp,
            ref_stats=ref_stats,
            gen_stats=gen_stats,
            height=430,
        )

# ================================================================= TAB 5: METHODOLOGY
with tab_about:
    st.subheader("The Science Behind VoiceLab 2.0")
    st.markdown(
        """
        VoiceLab 2.0 brings together mathematical signal processing and neural acoustic generation:

        #### 1. Signal Preprocessing
        * **Mono Conversion**: Multichannel signals are downmixed to single-channel acoustic arrays.
        * **Anti-Aliasing Resampling**: Signals are resampled to 22,050 Hz (`TARGET_SR`) using polyphase filtering to guarantee consistent Nyquist frequencies.
        * **Energy Trimming**: Trailing and leading silences quieter than the configured dB threshold are trimmed.
        * **Peak Normalization**: Normalizes volume to prevent digital clipping while maximizing dynamic range.

        #### 2. Neural Speech Synthesis (Chatterbox)
        * Pretrained text-to-speech vocoder running on an external GPU backend (e.g. Google Colab T4).
        * Encodes speaker timbre and phonetic rhythm from the reference recording to synthesize the typed sentence in a matching voice.

        #### 3. Mathematical Acoustic Analysis
        * **STFT & Spectrogram**: Computes Short-Time Fourier Transform using Hann windows ($N_{\\text{FFT}}=1024, \\text{hop}=256$) to track frequency evolution over time.
        * **Autocorrelation Pitch Tracking**: Estimates fundamental frequency ($F_0$) frame-by-frame, extracting median pitch and pitch variance in semitones.
        * **Mel Filterbank & MFCCs**: Warps spectral power into the perceptual Mel scale and applies the Discrete Cosine Transform (DCT) to isolate vocal tract resonance formants.
        * **LTAS & Distance**: Measures Long-Term Average Spectrum correlation and computes Euclidean distances against empirical yardsticks.
        """
    )
    st.caption("Only clone voices you own or have explicit permission to use. Signal similarity scores are mathematical measurements, not proof of identity.")
