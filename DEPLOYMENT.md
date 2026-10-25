# VoiceLab 2.0 — Production & Free Cloud Deployment Guide

This guide explains how to run **VoiceLab 2.0** locally or deploy it to **Streamlit Community Cloud** (free) while connecting to the **Chatterbox GPU backend** hosted on a free Google Colab T4 instance.

---

## 1. Hosting the Chatterbox Model on Free Google Colab

VoiceLab's neural voice-cloning runs on **Chatterbox TTS** with PyTorch and CUDA. Because Streamlit Community Cloud provides CPU-only instances, Chatterbox runs in an external GPU container:

1. Open `colab/VoiceLab_Chatterbox_Backend.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Select **Runtime → Change runtime type → T4 GPU** (Hardware accelerator: GPU).
3. Select **Runtime → Run all**.
4. The notebook installs Chatterbox, loads the pretrained weights onto the T4 GPU, launches a FastAPI service, and automatically creates a secure HTTPS tunnel (via Cloudflare Tunnel or ngrok).
5. The final output block displays:
   ```text
   ============================================================
   🎉 VOICELAB CHATTERBOX BACKEND IS LIVE!
   ============================================================
   VOICELAB_API_URL   = https://xxxx.trycloudflare.com
   VOICELAB_API_TOKEN = 4f98a21bc34d88e0a...
   ============================================================
   ```
6. **Keep the Colab tab open** while synthesizing speech.

---

## 2. Running VoiceLab Locally

1. Open a terminal in the `VoiceLab` folder:
   ```powershell
   # Activate virtual environment
   .venv\Scripts\activate

   # Launch Streamlit app
   streamlit run app.py
   ```
   *(Or double-click `run_windows.bat`).*
2. Open `http://localhost:8501`.
3. In the left sidebar, expand **⚙ Backend Tunnel Settings**.
4. Paste the `VOICELAB_API_URL` and `VOICELAB_API_TOKEN` printed by Colab, then click **Apply & Test**.
5. The top right badge will illuminate as `● GPU ONLINE`.
6. You can now upload or record reference audio, type text, and generate cloned speech!

---

## 3. Free Deployment to Streamlit Community Cloud

You can host VoiceLab 2.0 100% free on Streamlit Community Cloud:

1. Push your VoiceLab project to a **GitHub repository** (public or private):
   ```bash
   git init
   git add .
   git commit -m "VoiceLab 2.0 futuristic platform"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/VoiceLab.git
   git push -u origin main
   ```
   *(The included `.gitignore` automatically prevents `.venv/`, `.streamlit/secrets.toml`, and temporary recordings from being uploaded).*

2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Click **Create app** and select:
   - **Repository**: `YOUR_USERNAME/VoiceLab`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Expand **Advanced settings → Secrets** and enter your default backend credentials:
   ```toml
   VOICELAB_API_URL = "https://your-colab-tunnel.trycloudflare.com"
   VOICELAB_API_TOKEN = "your-api-token"
   ```
5. Click **Deploy!**
6. Streamlit Community Cloud will automatically build and deploy your app.

> **Note**: Even if your Colab GPU instance is offline, your deployed VoiceLab website will remain online, fully functional for signal exploration (Waveforms, STFT Spectrograms, Autocorrelation Pitch, and MFCC analysis with the included demo audio). Voice generation will display a graceful GPU standby notice until Colab is started.

---

## 4. Security & Compliance

- **Consent Verification**: VoiceLab enforces user confirmation before transmitting reference speech to the model.
- **Data Privacy**: Uploaded reference WAVs are deleted from temporary memory immediately after vocoder inference.
- **Authentication**: All API requests between Streamlit and the Colab backend are secured with 256-bit Bearer token authorization over TLS/HTTPS.
