"""Interactive Voice Comparison Instrument Component for VoiceLab 2.0.
Renders an interactive dual-deck scientific comparison workspace with synchronized timeline cursors and acoustic delta HUDs.
"""

from __future__ import annotations
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
            st.markdown(f'<div class="vl-comparison-deck" style="height:{height}px"></div>', unsafe_allow_html=True)



def render_comparison_deck(
    ref_audio: np.ndarray,
    ref_sr: int,
    gen_audio: np.ndarray,
    gen_sr: int,
    cmp_stats: dict,
    ref_stats: dict,
    gen_stats: dict,
    height: int = 420,
):
    """Renders the interactive dual-deck audio comparison instrument."""
    ref_points = downsample_waveform(ref_audio, num_points=160)
    gen_points = downsample_waveform(gen_audio, num_points=160)

    payload = {
        "ref": {
            "points": ref_points,
            "duration": float(len(ref_audio) / ref_sr),
            "medianPitch": ref_stats.get("median_hz", 0),
            "pitchRange": ref_stats.get("range_semitones", 0),
            "voiced": ref_stats.get("voiced_fraction", 0) * 100,
            "rms": ref_stats.get("rms_db", 0),
            "centroid": ref_stats.get("centroid_hz", 0),
        },
        "gen": {
            "points": gen_points,
            "duration": float(len(gen_audio) / gen_sr),
            "medianPitch": gen_stats.get("median_hz", 0),
            "pitchRange": gen_stats.get("range_semitones", 0),
            "voiced": gen_stats.get("voiced_fraction", 0) * 100,
            "rms": gen_stats.get("rms_db", 0),
            "centroid": gen_stats.get("centroid_hz", 0),
        },
        "cmp": {
            "pitchShift": cmp_stats.get("pitch_shift_semitones", 0),
            "mfccDist": cmp_stats.get("mfcc_distance", 0),
            "yardstick": cmp_stats.get("mfcc_baseline", 0),
            "ltasCorr": cmp_stats.get("ltas_correlation", 0),
        },
    }

    payload_json = json.dumps(payload)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

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

  .instrument-deck {{
    position: relative;
    width: 100%;
    height: {height}px;
    background: radial-gradient(ellipse 70% 60% at 50% 40%, rgba(59, 7, 100, 0.28) 0%, transparent 80%),
                linear-gradient(135deg, rgba(14, 8, 26, 0.95) 0%, rgba(22, 11, 40, 0.9) 100%);
    border: 1px solid rgba(192, 132, 252, 0.22);
    border-radius: 22px;
    padding: 18px 24px;
    display: flex;
    flex-direction: column;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), inset 0 1px rgba(255, 255, 255, 0.1);
  }}

  .deck-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }}

  .deck-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 10px;
  }}

  .deck-title .mode-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    padding: 3px 9px;
    border-radius: 100px;
    background: rgba(147, 51, 234, 0.2);
    border: 1px solid rgba(192, 132, 252, 0.35);
    color: #c084fc;
  }}

  .toggle-group {{
    display: flex;
    gap: 6px;
    background: rgba(0, 0, 0, 0.4);
    padding: 3px;
    border-radius: 100px;
    border: 1px solid rgba(192, 132, 252, 0.2);
  }}

  .toggle-btn {{
    background: transparent;
    border: none;
    color: #a78bfa;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    padding: 5px 12px;
    border-radius: 100px;
    cursor: pointer;
    transition: all 0.2s;
  }}

  .toggle-btn.active {{
    background: #9333ea;
    color: #ffffff;
    box-shadow: 0 2px 10px rgba(147, 51, 234, 0.5);
  }}

  /* Dual Waveform Canvas */
  .canvas-stage {{
    position: relative;
    flex: 1;
    width: 100%;
    cursor: crosshair;
  }}

  #compare-canvas {{
    width: 100%;
    height: 100%;
    display: block;
  }}

  /* Acoustic Delta HUD */
  .delta-hud {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid rgba(192, 132, 252, 0.12);
  }}

  .hud-card {{
    background: rgba(22, 12, 38, 0.6);
    border: 1px solid rgba(192, 132, 252, 0.18);
    border-radius: 12px;
    padding: 9px 14px;
    display: flex;
    flex-direction: column;
    backdrop-filter: blur(8px);
    transition: border-color 0.2s;
  }}

  .hud-card:hover {{
    border-color: rgba(192, 132, 252, 0.4);
  }}

  .hud-card .label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #a78bfa;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }}

  .hud-card .val {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #ffffff;
    margin-top: 2px;
  }}

  .hud-card .sub {{
    font-size: 0.64rem;
    color: #c4b5fd;
    margin-top: 2px;
  }}
</style>
</head>
<body>

<div class="instrument-deck">
  <div class="deck-header">
    <div class="deck-title">
      <span>SYNCHRONIZED ACOUSTIC COMPARISON</span>
      <span class="mode-badge">SYNCHRONIZED TIMELINE</span>
    </div>
    <div class="toggle-group">
      <button class="toggle-btn active" id="btn-split">SPLIT DUAL-DECK</button>
      <button class="toggle-btn" id="btn-overlay">HARMONIC OVERLAY</button>
    </div>
  </div>

  <div class="canvas-stage" id="canvas-stage">
    <canvas id="compare-canvas"></canvas>
  </div>

  <div class="delta-hud">
    <div class="hud-card">
      <span class="label">PITCH SHIFT</span>
      <span class="val" id="hud-pitch">–</span>
      <span class="sub">Fundamental F0 Delta</span>
    </div>
    <div class="hud-card">
      <span class="label">MFCC DISTANCE</span>
      <span class="val" id="hud-mfcc">–</span>
      <span class="sub" id="hud-yardstick">vs Baseline: –</span>
    </div>
    <div class="hud-card">
      <span class="label">SPECTRUM CORR</span>
      <span class="val" id="hud-corr">–</span>
      <span class="sub">LTAS Spectral Shape</span>
    </div>
    <div class="hud-card">
      <span class="label">DURATION DELTA</span>
      <span class="val" id="hud-dur">–</span>
      <span class="sub">Temporal Length</span>
    </div>
  </div>
</div>

<script>
(function() {{
  const data = {payload_json};
  const canvas = document.getElementById('compare-canvas');
  const ctx = canvas.getContext('2d');
  const stage = document.getElementById('canvas-stage');
  const btnSplit = document.getElementById('btn-split');
  const btnOverlay = document.getElementById('btn-overlay');

  let mode = 'split'; // 'split' or 'overlay'
  let width, height, dpr;
  let mouseX = -1;

  // Fill HUD values
  const fmt = (v, d = 1, u = '') => (v === null || isNaN(v) ? '–' : v.toFixed(d) + u);
  
  document.getElementById('hud-pitch').textContent = fmt(data.cmp.pitchShift, 1, ' st');
  document.getElementById('hud-mfcc').textContent = fmt(data.cmp.mfccDist, 1);
  document.getElementById('hud-yardstick').textContent = 'Yardstick: ' + fmt(data.cmp.yardstick, 1);
  document.getElementById('hud-corr').textContent = fmt(data.cmp.ltasCorr, 2);
  const durDelta = (data.gen.duration - data.ref.duration);
  document.getElementById('hud-dur').textContent = (durDelta > 0 ? '+' : '') + fmt(durDelta, 2, ' s');

  function resize() {{
    const rect = canvas.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    width = rect.width;
    height = rect.height;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    draw();
  }}

  window.addEventListener('resize', resize);

  btnSplit.addEventListener('click', () => {{
    mode = 'split';
    btnSplit.classList.add('active');
    btnOverlay.classList.remove('active');
    draw();
  }});

  btnOverlay.addEventListener('click', () => {{
    mode = 'overlay';
    btnOverlay.classList.add('active');
    btnSplit.classList.remove('active');
    draw();
  }});

  stage.addEventListener('mousemove', (e) => {{
    const rect = canvas.getBoundingClientRect();
    mouseX = e.clientX - rect.left;
    draw();
  }});

  stage.addEventListener('mouseleave', () => {{
    mouseX = -1;
    draw();
  }});

  function draw() {{
    ctx.clearRect(0, 0, width, height);

    const refPts = data.ref.points;
    const genPts = data.gen.points;
    const maxDur = Math.max(data.ref.duration, data.gen.duration) || 1;

    if (mode === 'split') {{
      const deckH = (height - 18) * 0.5;

      // TOP DECK: Reference
      drawDeckTrack(
        refPts, 
        0, 
        deckH, 
        '#c084fc', 
        'ORIGINAL REFERENCE VOICE (' + data.ref.duration.toFixed(1) + 's)', 
        data.ref.duration / maxDur
      );

      // BOTTOM DECK: Generated
      drawDeckTrack(
        genPts, 
        deckH + 18, 
        deckH, 
        '#38bdf8', 
        'GENERATED / COMPARISON SPEECH (' + data.gen.duration.toFixed(1) + 's)', 
        data.gen.duration / maxDur
      );
    }} else {{
      // OVERLAY MODE
      drawOverlayMode(refPts, genPts, 0, height, maxDur);
    }}

    // Synchronized Crosshair Cursor
    if (mouseX >= 0 && mouseX <= width) {{
      ctx.save();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(mouseX, 0);
      ctx.lineTo(mouseX, height);
      ctx.stroke();

      // Tooltip timecode
      const curTime = (mouseX / width) * maxDur;
      ctx.fillStyle = '#ffffff';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillText(curTime.toFixed(2) + 's', mouseX + 8, 16);
      ctx.restore();
    }}
  }}

  function drawDeckTrack(points, y, h, color, title, widthRatio) {{
    ctx.save();
    
    // Background plate
    ctx.fillStyle = 'rgba(26, 14, 44, 0.45)';
    ctx.roundRect(0, y, width, h, 10);
    ctx.fill();

    // Title
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = color;
    ctx.fillText(title, 14, y + 16);

    const midY = y + h * 0.58;
    const trackW = width * widthRatio;
    const barW = trackW / points.length;

    for (let i = 0; i < points.length; i++) {{
      const amp = points[i];
      const barH = Math.max(2, amp * (h * 0.38));
      const bx = i * barW;
      
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.4 + amp * 0.6;
      ctx.fillRect(bx, midY - barH, Math.max(1.2, barW - 1), barH * 2);
    }}

    ctx.restore();
  }}

  function drawOverlayMode(refPts, genPts, y, h, maxDur) {{
    ctx.save();
    
    // Plate
    ctx.fillStyle = 'rgba(26, 14, 44, 0.45)';
    ctx.roundRect(0, y, width, h, 12);
    ctx.fill();

    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#c084fc';
    ctx.fillText('● REFERENCE (PURPLE)  vs  ● GENERATED (CYAN)', 16, y + 20);

    const midY = y + h * 0.55;
    const rW = width * (data.ref.duration / maxDur);
    const gW = width * (data.gen.duration / maxDur);

    // Reference curve
    ctx.beginPath();
    ctx.strokeStyle = '#c084fc';
    ctx.lineWidth = 1.6;
    for (let i = 0; i < refPts.length; i++) {{
      const bx = i * (rW / refPts.length);
      const by = midY - refPts[i] * (h * 0.35);
      if (i === 0) ctx.moveTo(bx, by);
      else ctx.lineTo(bx, by);
    }}
    ctx.stroke();

    // Generated curve
    ctx.beginPath();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.6;
    for (let i = 0; i < genPts.length; i++) {{
      const bx = i * (gW / genPts.length);
      const by = midY - genPts[i] * (h * 0.35);
      if (i === 0) ctx.moveTo(bx, by);
      else ctx.lineTo(bx, by);
    }}
    ctx.stroke();

    ctx.restore();
  }}

  setTimeout(resize, 40);
}})();
</script>
</body>
</html>"""

    _render_html(html, height=height)
