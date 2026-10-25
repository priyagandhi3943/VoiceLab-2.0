"""Voice Studio Generation Journey Component for VoiceLab 2.0.
Implements the 4-stage cinematic visual transformation for text-to-speech generation.
"""

from __future__ import annotations
import json
import numpy as np
import streamlit as st


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
            st.markdown(f'<div class="vl-generation-journey" style="height:{height}px"></div>', unsafe_allow_html=True)



def downsample_waveform(audio: np.ndarray, num_points: int = 140) -> list[float]:
    """Extract a balanced envelope of the real audio signal for visual rendering."""
    if audio is None or len(audio) == 0:
        return [0.0] * num_points
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    chunk_size = max(1, len(audio) // num_points)
    points = []
    for i in range(num_points):
        chunk = audio[i * chunk_size : (i + 1) * chunk_size]
        if len(chunk) > 0:
            # take peak amplitude in this window
            val = float(np.max(np.abs(chunk)))
            points.append(round(min(1.0, val), 3))
        else:
            points.append(0.0)
    return points


def get_generation_journey_html(
    state: str,
    text: str,
    ref_samples: list[float],
    ref_duration: float,
    gen_samples: list[float] | None = None,
    gen_duration: float = 0.0,
    error_msg: str = "",
    height: int = 440,
) -> str:
    """Returns the standalone HTML/Canvas application for the generation journey."""
    ref_json = json.dumps(ref_samples)
    gen_json = json.dumps(gen_samples or [])
    escaped_text = json.dumps(text or "Type your prompt to synthesize...")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}

  body {{
    background: transparent;
    color: #f3e8ff;
    font-family: 'DM Sans', sans-serif;
    overflow: hidden;
    user-select: none;
    -webkit-user-select: none;
  }}

  .journey-stage {{
    position: relative;
    width: 100%;
    height: {height}px;
    border-radius: 24px;
    background: radial-gradient(ellipse 60% 70% at 50% 50%, rgba(88, 28, 135, 0.25) 0%, transparent 80%),
                linear-gradient(135deg, rgba(13, 7, 24, 0.95) 0%, rgba(26, 12, 46, 0.9) 50%, rgba(10, 6, 18, 0.98) 100%);
    border: 1px solid rgba(192, 132, 252, 0.22);
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.55), inset 0 1px rgba(255, 255, 255, 0.12);
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }}

  /* Top Stage HUD */
  .stage-header {{
    position: relative;
    z-index: 10;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 18px 26px 12px;
    border-bottom: 1px solid rgba(192, 132, 252, 0.12);
    backdrop-filter: blur(8px);
  }}

  .stage-title {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.88rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #e9d5ff;
    text-transform: uppercase;
  }}

  .stage-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 100px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
  }}

  .badge-idle {{
    background: rgba(147, 51, 234, 0.15);
    border: 1px solid rgba(192, 132, 252, 0.3);
    color: #d8b4fe;
  }}

  .badge-processing {{
    background: rgba(236, 72, 153, 0.2);
    border: 1px solid rgba(244, 114, 182, 0.4);
    color: #fbcfe8;
    animation: badgeGlow 1.5s ease-in-out infinite alternate;
  }}

  .badge-complete {{
    background: rgba(52, 211, 153, 0.15);
    border: 1px solid rgba(52, 211, 153, 0.4);
    color: #a7f3d0;
  }}

  .badge-error {{
    background: rgba(239, 68, 68, 0.18);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #fca5a5;
  }}

  @keyframes badgeGlow {{
    0% {{ box-shadow: 0 0 4px rgba(244, 114, 182, 0.2); }}
    100% {{ box-shadow: 0 0 16px rgba(244, 114, 182, 0.5); }}
  }}

  .stage-timer {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    color: #a78bfa;
    display: flex;
    align-items: center;
    gap: 8px;
  }}

  /* Main Canvas Area */
  #journey-canvas {{
    flex: 1;
    width: 100%;
    height: 100%;
    display: block;
  }}

  /* Bottom status strip */
  .stage-footer {{
    position: relative;
    z-index: 10;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 26px;
    border-top: 1px solid rgba(192, 132, 252, 0.1);
    background: rgba(13, 7, 24, 0.6);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #c4b5fd;
  }}

  .stage-status {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}

  .status-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #c084fc;
    box-shadow: 0 0 8px #c084fc;
  }}

  .status-dot.active {{
    animation: dotPulse 1.2s ease-in-out infinite;
  }}

  @keyframes dotPulse {{
    0%, 100% {{ transform: scale(1); opacity: 1; }}
    50% {{ transform: scale(1.4); opacity: 0.5; }}
  }}

  /* Error modal overlay if failed */
  .error-card {{
    position: absolute;
    inset: 40px;
    z-index: 30;
    border-radius: 20px;
    background: rgba(22, 10, 32, 0.94);
    border: 1px solid rgba(239, 68, 68, 0.35);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8);
    backdrop-filter: blur(20px);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2.5rem;
    text-align: center;
  }}

  .error-card h3 {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    color: #fca5a5;
    margin-bottom: 0.8rem;
  }}

  .error-card p {{
    font-size: 0.9rem;
    color: #e2e8f0;
    max-width: 480px;
    line-height: 1.5;
    margin-bottom: 1.2rem;
  }}

  .error-card .hint {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #cbd5e1;
    background: rgba(0,0,0,0.4);
    padding: 6px 14px;
    border-radius: 8px;
  }}
</style>
</head>
<body>

<div class="journey-stage">
  <div class="stage-header">
    <div class="stage-title">
      <span id="header-icon">✦</span>
      <span id="header-text">ACOUSTIC SYNTHESIS STAGE</span>
      <div class="stage-badge" id="state-badge">IDLE</div>
    </div>
    <div class="stage-timer" id="timer-display">
      <span>LATENT CLOCK:</span>
      <strong id="timer-val">00:00.0</strong>
    </div>
  </div>

  <canvas id="journey-canvas"></canvas>

  <div class="stage-footer">
    <div class="stage-status">
      <div class="status-dot active" id="status-dot"></div>
      <span id="status-description">Ready for neural speech synthesis</span>
    </div>
    <div id="footer-stats">
      REF DURATION: {ref_duration:.2f}s | CHATTERBOX GPU
    </div>
  </div>

  <div class="error-card" id="error-card" style="display: {'flex' if state == 'error' else 'none'};">
    <h3>Chatterbox Backend Offline</h3>
    <p>{error_msg or 'GPU worker instance is currently disconnected. Please launch the Colab backend notebook to generate speech.'}</p>
    <div class="hint">Your reference recording and typed text remain saved. Start Colab & retry.</div>
  </div>
</div>

<script>
(function() {{
  const canvas = document.getElementById('journey-canvas');
  const ctx = canvas.getContext('2d');
  const stateBadge = document.getElementById('state-badge');
  const timerVal = document.getElementById('timer-val');
  const statusDesc = document.getElementById('status-description');
  const statusDot = document.getElementById('status-dot');

  const appState = "{state}"; // "idle", "generating", "complete", "error"
  const refWave = {ref_json};
  const genWave = {gen_json};
  const promptText = {escaped_text};

  let width, height, dpr;
  let startTime = performance.now();
  let time = 0;

  // Particle system for glowing text fragments
  const particles = [];
  const textChars = promptText.length > 0 ? promptText.split('') : ['S','y','n','t','h'];
  const maxParticles = 65;

  function initParticles() {{
    particles.length = 0;
    for (let i = 0; i < maxParticles; i++) {{
      particles.push({{
        x: Math.random() * 0.35, // starting on left
        y: 0.3 + Math.random() * 0.4,
        vx: 0.003 + Math.random() * 0.005,
        vy: (Math.random() - 0.5) * 0.002,
        char: textChars[i % textChars.length],
        alpha: 0.2 + Math.random() * 0.7,
        size: 9 + Math.random() * 6,
        orbitRadius: 20 + Math.random() * 35,
        phase: Math.random() * Math.PI * 2
      }});
    }}
  }}

  initParticles();

  // Setup UI badges based on state
  if (appState === 'generating') {{
    stateBadge.className = 'stage-badge badge-processing';
    stateBadge.textContent = 'SYNTHESIZING';
    statusDesc.textContent = 'Transforming linguistic tokens into acoustic latents...';
  }} else if (appState === 'complete') {{
    stateBadge.className = 'stage-badge badge-complete';
    stateBadge.textContent = 'SYNTHESIS COMPLETE';
    statusDesc.textContent = 'Generated speech synthesized successfully';
  }} else if (appState === 'error') {{
    stateBadge.className = 'stage-badge badge-error';
    stateBadge.textContent = 'BACKEND OFFLINE';
    statusDesc.textContent = 'Remote Colab GPU unavailable';
  }} else {{
    stateBadge.className = 'stage-badge badge-idle';
    stateBadge.textContent = 'STANDBY';
    statusDesc.textContent = 'Reference loaded. Press Generate to begin.';
  }}

  function resize() {{
    const rect = canvas.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    width = rect.width;
    height = rect.height;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
  }}

  window.addEventListener('resize', resize);
  resize();

  // Draw Waveform Panel
  function drawWaveform(points, x, y, w, h, color, label, progress = 1.0) {{
    ctx.save();
    
    // Panel backdrop
    const grad = ctx.createLinearGradient(x, y, x, y + h);
    grad.addColorStop(0, 'rgba(30, 16, 52, 0.4)');
    grad.addColorStop(1, 'rgba(16, 8, 28, 0.6)');
    ctx.fillStyle = grad;
    ctx.strokeStyle = 'rgba(192, 132, 252, 0.16)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 14);
    ctx.fill();
    ctx.stroke();

    // Label
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#a78bfa';
    ctx.fillText(label, x + 16, y + 22);

    // Centerline
    const midY = y + h * 0.55;
    ctx.strokeStyle = 'rgba(192, 132, 252, 0.12)';
    ctx.beginPath();
    ctx.moveTo(x + 12, midY);
    ctx.lineTo(x + w - 12, midY);
    ctx.stroke();

    // Waveform bars
    if (points && points.length > 0) {{
      const barW = (w - 32) / points.length;
      const count = Math.floor(points.length * progress);
      
      for (let i = 0; i < count; i++) {{
        const amp = points[i];
        const barH = Math.max(3, amp * (h * 0.42));
        const bx = x + 16 + i * barW;
        
        // Dynamic color
        const alpha = 0.35 + amp * 0.65;
        ctx.fillStyle = color.replace('ALPHA', alpha.toFixed(2));
        ctx.fillRect(bx, midY - barH, Math.max(1.5, barW - 1), barH * 2);
      }}

      // Glow sweep line
      if (appState === 'generating' || appState === 'idle') {{
        const sweepX = x + 16 + ((time * 0.4) % 1) * (w - 32);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.65)';
        ctx.shadowColor = '#c084fc';
        ctx.shadowBlur = 10;
        ctx.beginPath();
        ctx.moveTo(sweepX, y + 10);
        ctx.lineTo(sweepX, y + h - 10);
        ctx.stroke();
        ctx.shadowBlur = 0;
      }}
    }}
    ctx.restore();
  }}

  // Draw Center Neural Audio Core Sculpture
  function drawCoreSculpture(cx, cy, radius) {{
    ctx.save();
    
    // Core halo
    const halo = ctx.createRadialGradient(cx, cy, 5, cx, cy, radius * 1.5);
    const pulseIntensity = appState === 'generating' ? (0.35 + 0.15 * Math.sin(time * 6)) : 0.18;
    halo.addColorStop(0, 'rgba(232, 121, 249, ' + pulseIntensity + ')');
    halo.addColorStop(0.6, 'rgba(147, 51, 234, 0.1)');
    halo.addColorStop(1, 'transparent');
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(cx, cy, radius * 1.5, 0, Math.PI * 2);
    ctx.fill();

    // Outer rotating dimensional energy rings
    const ringCount = 3;
    for (let r = 0; r < ringCount; r++) {{
      const angle = time * (0.8 + r * 0.4) * (r % 2 === 0 ? 1 : -1);
      const rx = radius * (0.7 + r * 0.25);
      const ry = radius * (0.35 + r * 0.15);

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(angle);
      ctx.strokeStyle = r === 0 ? 'rgba(240, 171, 252, 0.8)' : 'rgba(192, 132, 252, 0.4)';
      ctx.lineWidth = r === 0 ? 1.8 : 1.0;
      ctx.beginPath();
      ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }}

    // Center crystalline node
    const coreGrad = ctx.createRadialGradient(cx - 5, cy - 5, 2, cx, cy, radius * 0.35);
    coreGrad.addColorStop(0, '#ffffff');
    coreGrad.addColorStop(0.3, '#f5d0fe');
    coreGrad.addColorStop(0.7, '#c084fc');
    coreGrad.addColorStop(1, '#581c87');
    ctx.fillStyle = coreGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, radius * 0.32, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }}

  // Draw particle text tokens moving toward center
  function drawParticles(leftX, rightX, cy) {{
    ctx.save();
    particles.forEach(p => {{
      // Update
      p.x += p.vx;
      if (p.x > 0.48) {{
        // Reset to left side
        p.x = 0.05 + Math.random() * 0.1;
        p.y = 0.25 + Math.random() * 0.5;
      }}

      // Calculate position
      const px = width * (0.28 + p.x);
      // Gravitational attraction toward center Y
      const targetY = cy + Math.sin(p.x * 12 + time * 3) * 35;
      const py = height * p.y + (targetY - height * p.y) * (p.x / 0.48);

      // Render glowing character token
      ctx.font = `${{p.size}}px "JetBrains Mono", monospace`;
      const distToCore = Math.abs(px - (width * 0.5));
      const fade = Math.min(1.0, distToCore / 60);
      
      ctx.fillStyle = `rgba(245, 208, 254, ${{p.alpha * fade}})`;
      ctx.fillText(p.char, px, py);

      // Connecting filament stream
      ctx.strokeStyle = `rgba(192, 132, 252, ${{0.15 * fade}})`;
      ctx.lineWidth = 0.75;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(width * 0.5, cy);
      ctx.stroke();
    }});
    ctx.restore();
  }}

  // Main animation frame
  function render() {{
    time += 0.02;

    // Timer calculation
    if (appState === 'generating') {{
      const elapsed = (performance.now() - startTime) / 1000;
      const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
      const secs = (elapsed % 60).toFixed(1).padStart(4, '0');
      timerVal.textContent = `${{mins}}:${{secs}}`;
      
      // Stage status steps
      if (elapsed < 3) {{
        statusDesc.textContent = "Encoding acoustic voice characteristics (22.05 kHz)...";
      }} else if (elapsed < 8) {{
        statusDesc.textContent = "Chatterbox GPU synthesizing neural speech tokens...";
      }} else {{
        statusDesc.textContent = "Decoding latent Mel-spectrogram into audio waveform...";
      }}
    }}

    ctx.clearRect(0, 0, width, height);

    const pad = 24;
    const panelW = Math.min(260, width * 0.3);
    const panelH = height - pad * 2;
    const centerY = height * 0.5;
    const centerX = width * 0.5;

    // 1. LEFT PANEL: Reference Waveform
    drawWaveform(
      refWave,
      pad,
      pad,
      panelW,
      panelH,
      'rgba(192, 132, 252, ALPHA)',
      '01 // REFERENCE VOICE'
    );

    // 2. CENTER: Neural Synthesis Sculpture & Particle Streaming
    if (appState === 'generating') {{
      drawParticles(pad + panelW, width - pad - panelW, centerY);
    }}
    drawCoreSculpture(centerX, centerY, 58);

    // 3. RIGHT PANEL: Generated Waveform
    const rightX = width - pad - panelW;
    if (appState === 'complete') {{
      drawWaveform(
        genWave,
        rightX,
        pad,
        panelW,
        panelH,
        'rgba(52, 211, 153, ALPHA)',
        '02 // GENERATED SPEECH'
      );
    }} else {{
      // Silhouette emerging waveform
      drawWaveform(
        appState === 'generating' ? refWave.map(v => v * (0.2 + 0.15 * Math.sin(time * 4))) : [],
        rightX,
        pad,
        panelW,
        panelH,
        'rgba(244, 114, 182, ALPHA)',
        appState === 'generating' ? '02 // EMERGENCE ZONE' : '02 // WAITING FOR GENERATE',
        appState === 'generating' ? ((time * 0.3) % 1.0) : 0
      );
    }}

    if (!document.hidden) {{
      requestAnimationFrame(render);
    }}
  }}

  requestAnimationFrame(render);

}})();
</script>
</body>
</html>"""


def render_generation_journey(
    state: str,
    text: str,
    reference_audio: np.ndarray | None,
    reference_sr: int = 22050,
    generated_audio: np.ndarray | None = None,
    generated_sr: int = 22050,
    error_msg: str = "",
    height: int = 440,
):
    """Renders the Voice Studio text-to-speech generation visual journey."""
    ref_samples = downsample_waveform(reference_audio, num_points=120)
    ref_duration = float(len(reference_audio) / reference_sr) if reference_audio is not None and len(reference_audio) > 0 else 0.0

    gen_samples = downsample_waveform(generated_audio, num_points=120) if generated_audio is not None else []
    gen_duration = float(len(generated_audio) / generated_sr) if generated_audio is not None and len(generated_audio) > 0 else 0.0

    html = get_generation_journey_html(
        state=state,
        text=text,
        ref_samples=ref_samples,
        ref_duration=ref_duration,
        gen_samples=gen_samples,
        gen_duration=gen_duration,
        error_msg=error_msg,
        height=height,
    )
    _render_html(html, height=height)
