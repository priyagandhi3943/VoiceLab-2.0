"""Interactive Audio Waveform Player Component for VoiceLab 2.0.
Renders a luxury cybernetic audio player with real waveform scrubbing and synchronized playhead.
"""

from __future__ import annotations
import base64
import json
import numpy as np
import streamlit as st
from voicelab.frontend.generation import downsample_waveform


def _render_html(html: str, height: int):
    """Renders HTML via streamlit.components.v1.html with test mock fallback."""
    components = getattr(st, "components", None)
    if components and hasattr(components, "v1") and hasattr(components.v1, "html"):
        components.v1.html(html, height=height, scrolling=False)
    else:
        try:
            import streamlit.components.v1 as components_v1
            components_v1.html(html, height=height, scrolling=False)
        except Exception:
            st.markdown(f'<div class="vl-audio-player" style="height:{height}px"></div>', unsafe_allow_html=True)



def render_wave_player(
    audio: np.ndarray,
    sr: int,
    wav_bytes: bytes,
    title: str = "Audio Recording",
    accent_color: str = "#c084fc",
    height: int = 150,
):
    """Renders a responsive interactive waveform player."""
    if audio is None or len(audio) == 0:
        return

    samples = downsample_waveform(audio, num_points=160)
    duration = float(len(audio) / sr)
    b64_audio = base64.b64encode(wav_bytes).decode("ascii")
    samples_json = json.dumps(samples)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;600&family=JetBrains+Mono:wght@400;500&display=swap');

  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}

  body {{
    background: transparent;
    font-family: 'DM Sans', sans-serif;
    color: #f3e8ff;
    overflow: hidden;
    user-select: none;
  }}

  .player-deck {{
    position: relative;
    width: 100%;
    height: {height}px;
    background: linear-gradient(135deg, rgba(20, 10, 36, 0.85) 0%, rgba(13, 7, 24, 0.95) 100%);
    border: 1px solid rgba(192, 132, 252, 0.2);
    border-radius: 18px;
    padding: 14px 20px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4), inset 0 1px rgba(255, 255, 255, 0.1);
  }}

  .deck-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}

  .deck-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.84rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: #e9d5ff;
    display: flex;
    align-items: center;
    gap: 8px;
  }}

  .deck-title .tag {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    padding: 2px 8px;
    background: rgba(147, 51, 234, 0.2);
    border: 1px solid rgba(192, 132, 252, 0.3);
    border-radius: 100px;
    color: {accent_color};
  }}

  .deck-time {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #c4b5fd;
  }}

  /* Waveform canvas */
  .waveform-container {{
    position: relative;
    width: 100%;
    height: 52px;
    cursor: pointer;
  }}

  #wave-canvas {{
    width: 100%;
    height: 100%;
    display: block;
  }}

  .deck-controls {{
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}

  .play-btn {{
    display: flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg, {accent_color}, #9333ea);
    border: none;
    color: #ffffff;
    cursor: pointer;
    box-shadow: 0 4px 14px rgba(147, 51, 234, 0.4);
    transition: transform 0.2s, box-shadow 0.2s;
  }}

  .play-btn:hover {{
    transform: scale(1.08);
    box-shadow: 0 6px 18px rgba(192, 132, 252, 0.6);
  }}

  .spec-tag {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #a78bfa;
    display: flex;
    gap: 12px;
  }}
</style>
</head>
<body>

<div class="player-deck">
  <div class="deck-header">
    <div class="deck-title">
      <span>{title}</span>
      <span class="tag">REAL SIGNAL</span>
    </div>
    <div class="deck-time">
      <span id="curr-time">00:00.0</span> / <span>{duration:.1f}s</span>
    </div>
  </div>

  <div class="waveform-container" id="wave-container" title="Click anywhere to scrub playback position">
    <canvas id="wave-canvas"></canvas>
  </div>

  <div class="deck-controls">
    <button class="play-btn" id="play-btn" title="Play / Pause">
      <svg id="play-icon" width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
        <polygon points="5 3 19 12 5 21 5 3"></polygon>
      </svg>
    </button>
    <div class="spec-tag">
      <span>SR: {sr} Hz</span>
      <span>PCM 16-BIT</span>
      <span>MONO</span>
    </div>
  </div>
</div>

<audio id="audio-el" src="data:audio/wav;base64,{b64_audio}" preload="auto"></audio>

<script>
(function() {{
  const canvas = document.getElementById('wave-canvas');
  const ctx = canvas.getContext('2d');
  const audio = document.getElementById('audio-el');
  const playBtn = document.getElementById('play-btn');
  const playIcon = document.getElementById('play-icon');
  const currTimeEl = document.getElementById('curr-time');
  const waveContainer = document.getElementById('wave-container');

  const samples = {samples_json};
  const duration = {duration};
  const accent = "{accent_color}";

  let width, height, dpr;

  function resize() {{
    const rect = canvas.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    width = rect.width;
    height = rect.height;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    drawWave();
  }}

  window.addEventListener('resize', resize);

  function formatTime(s) {{
    const m = Math.floor(s / 60).toString().padStart(2, '0');
    const sec = (s % 60).toFixed(1).padStart(4, '0');
    return `${{m}}:${{sec}}`;
  }}

  function drawWave() {{
    ctx.clearRect(0, 0, width, height);

    const barW = width / samples.length;
    const midY = height * 0.5;
    const progress = duration > 0 ? (audio.currentTime / duration) : 0;
    const playheadX = progress * width;

    // Draw bars
    for (let i = 0; i < samples.length; i++) {{
      const amp = samples[i];
      const barH = Math.max(2, amp * (height * 0.44));
      const bx = i * barW;
      
      const isPlayed = bx <= playheadX;
      if (isPlayed) {{
        ctx.fillStyle = accent;
      }} else {{
        ctx.fillStyle = 'rgba(192, 132, 252, 0.25)';
      }}
      ctx.fillRect(bx, midY - barH, Math.max(1.2, barW - 1), barH * 2);
    }}

    // Playhead laser
    ctx.save();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.8;
    ctx.shadowColor = accent;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.moveTo(playheadX, 0);
    ctx.lineTo(playheadX, height);
    ctx.stroke();
    ctx.restore();
  }}

  // Play / Pause toggle
  playBtn.addEventListener('click', () => {{
    if (audio.paused) {{
      audio.play();
      playIcon.innerHTML = '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
    }} else {{
      audio.pause();
      playIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
    }}
  }});

  audio.addEventListener('timeupdate', () => {{
    currTimeEl.textContent = formatTime(audio.currentTime);
    drawWave();
  }});

  audio.addEventListener('ended', () => {{
    playIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
    drawWave();
  }});

  // Scrubbing
  waveContainer.addEventListener('click', (e) => {{
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const scrubRatio = Math.max(0, Math.min(1, clickX / rect.width));
    audio.currentTime = scrubRatio * duration;
    drawWave();
  }});

  setTimeout(resize, 50);
}})();
</script>
</body>
</html>"""

    _render_html(html, height=height)
