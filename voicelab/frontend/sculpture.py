"""Hero 3D Audio Sculpture component for VoiceLab 2.0.
Renders an interactive, motion-rich 3D metallic filament & liquid ribbon sound sculpture.
"""

from __future__ import annotations
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
            st.markdown(f'<div class="vl-hero-sculpture" style="height:{height}px"></div>', unsafe_allow_html=True)



def get_sculpture_html(height: int = 560) -> str:
    """Returns HTML/JS for the 3D metallic sound sculpture."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
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

  .hero-container {{
    position: relative;
    width: 100%;
    height: {height}px;
    border-radius: 28px;
    background: radial-gradient(ellipse 65% 55% at 75% 45%, rgba(106, 27, 154, 0.28) 0%, transparent 75%),
                radial-gradient(ellipse 40% 40% at 20% 80%, rgba(59, 7, 100, 0.35) 0%, transparent 60%),
                linear-gradient(135deg, rgba(14, 8, 26, 0.95) 0%, rgba(24, 12, 42, 0.88) 50%, rgba(10, 6, 18, 0.98) 100%);
    border: 1px solid rgba(192, 132, 252, 0.18);
    box-shadow: 0 30px 80px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.12);
    overflow: hidden;
    display: flex;
    align-items: center;
  }}

  /* Ambient light aura */
  .aura {{
    position: absolute;
    right: 12%;
    top: 15%;
    width: 480px;
    height: 480px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.22) 0%, rgba(126, 34, 206, 0.08) 50%, transparent 70%);
    filter: blur(60px);
    pointer-events: none;
    animation: auraBreath 8s ease-in-out infinite alternate;
  }}

  @keyframes auraBreath {{
    0% {{ transform: scale(0.9) translate(0, 0); opacity: 0.7; }}
    100% {{ transform: scale(1.15) translate(-20px, 15px); opacity: 1; }}
  }}

  /* Micro grid background */
  .grid-pattern {{
    position: absolute;
    inset: 0;
    background-image: 
      linear-gradient(rgba(192, 132, 252, 0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(192, 132, 252, 0.04) 1px, transparent 1px);
    background-size: 32px 32px;
    mask-image: radial-gradient(ellipse 90% 80% at 50% 50%, black 40%, transparent 90%);
    pointer-events: none;
  }}

  /* Content overlay */
  .hero-content {{
    position: relative;
    z-index: 5;
    padding: 0 4rem;
    max-width: 540px;
    pointer-events: none;
  }}

  .badge-tag {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    background: rgba(147, 51, 234, 0.12);
    border: 1px solid rgba(192, 132, 252, 0.3);
    border-radius: 100px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    color: #e9d5ff;
    margin-bottom: 1.4rem;
    backdrop-filter: blur(10px);
  }}

  .badge-tag .pulse-dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #c084fc;
    box-shadow: 0 0 10px #c084fc;
    animation: dotPulse 2s ease-in-out infinite;
  }}

  @keyframes dotPulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.4; transform: scale(0.8); }}
  }}

  .hero-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.4rem, 4.2vw, 3.8rem);
    font-weight: 600;
    line-height: 1.05;
    letter-spacing: -0.04em;
    color: #ffffff;
    margin-bottom: 1.2rem;
  }}

  .hero-title em {{
    font-style: normal;
    background: linear-gradient(135deg, #f5d0fe 0%, #c084fc 45%, #e879f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}

  .hero-desc {{
    font-size: 0.98rem;
    line-height: 1.65;
    color: #c4b5fd;
    margin-bottom: 2rem;
  }}

  .hero-metrics {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    pointer-events: auto;
  }}

  .metric-pill {{
    display: flex;
    flex-direction: column;
    padding: 8px 16px;
    background: rgba(26, 16, 44, 0.65);
    border: 1px solid rgba(192, 132, 252, 0.2);
    border-radius: 12px;
    backdrop-filter: blur(12px);
    transition: transform 0.25s, border-color 0.25s;
  }}

  .metric-pill:hover {{
    transform: translateY(-2px);
    border-color: rgba(192, 132, 252, 0.45);
  }}

  .metric-pill .lbl {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    color: #a78bfa;
    text-transform: uppercase;
  }}

  .metric-pill .val {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #ffffff;
  }}

  /* 3D Canvas element */
  #sculpture-canvas {{
    position: absolute;
    top: 0;
    right: 0;
    width: 65%;
    height: 100%;
    z-index: 2;
    cursor: grab;
  }}

  #sculpture-canvas:active {{
    cursor: grabbing;
  }}

  /* Interactive floating HUD tags on the sculpture */
  .hud-node {{
    position: absolute;
    z-index: 6;
    padding: 8px 14px;
    background: rgba(22, 12, 38, 0.7);
    border: 1px solid rgba(192, 132, 252, 0.35);
    border-radius: 100px;
    backdrop-filter: blur(14px);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    color: #e9d5ff;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4), inset 0 1px rgba(255, 255, 255, 0.15);
    display: flex;
    align-items: center;
    gap: 8px;
    pointer-events: auto;
    transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.3s;
  }}

  .hud-node:hover {{
    transform: scale(1.05);
    border-color: #d8b4fe;
  }}

  .hud-node.top-right {{
    top: 14%;
    right: 6%;
  }}

  .hud-node.bottom-right {{
    bottom: 12%;
    right: 18%;
  }}

  /* Audio reactive indicator */
  .audio-indicator {{
    display: flex;
    align-items: center;
    gap: 2px;
    height: 12px;
  }}

  .audio-indicator span {{
    width: 2px;
    height: 100%;
    background: #c084fc;
    border-radius: 1px;
    animation: barBounce 1s ease-in-out infinite alternate;
  }}

  .audio-indicator span:nth-child(2) {{ animation-delay: -0.2s; }}
  .audio-indicator span:nth-child(3) {{ animation-delay: -0.5s; }}
  .audio-indicator span:nth-child(4) {{ animation-delay: -0.8s; }}

  @keyframes barBounce {{
    0% {{ height: 20%; }}
    100% {{ height: 100%; }}
  }}

  /* Interactive audio trigger button */
  .sound-btn {{
    position: absolute;
    bottom: 24px;
    right: 28px;
    z-index: 10;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 9px 18px;
    background: rgba(147, 51, 234, 0.25);
    border: 1px solid rgba(192, 132, 252, 0.45);
    border-radius: 100px;
    color: #ffffff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    cursor: pointer;
    backdrop-filter: blur(12px);
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(147, 51, 234, 0.25);
  }}

  .sound-btn:hover {{
    background: rgba(147, 51, 234, 0.45);
    border-color: #e9d5ff;
    transform: translateY(-2px);
    box-shadow: 0 6px 26px rgba(168, 85, 247, 0.4);
  }}

  @media (max-width: 900px) {{
    .hero-container {{
      flex-direction: column;
      height: 680px;
      padding-top: 2rem;
    }}
    .hero-content {{
      padding: 0 2rem;
      max-width: 100%;
    }}
    #sculpture-canvas {{
      top: auto;
      bottom: 0;
      width: 100%;
      height: 55%;
    }}
    .hud-node {{
      display: none;
    }}
  }}

  @media (prefers-reduced-motion: reduce) {{
    .aura, .audio-indicator span, .badge-tag .pulse-dot {{
      animation: none !important;
    }}
  }}
</style>
</head>
<body>

<div class="hero-container">
  <div class="grid-pattern"></div>
  <div class="aura"></div>

  <!-- Hero Content -->
  <div class="hero-content">
    <div class="badge-tag">
      <div class="pulse-dot"></div>
      VOICELAB 2.0 // SIGNAL ENGINE
    </div>
    <h1 class="hero-title">
      Pure Sound.<br>
      <em>Sculpted by AI.</em>
    </h1>
    <p class="hero-desc">
      An immersive scientific audio laboratory combining mathematical signal processing with neural speech synthesis. Interact directly with sound geometry.
    </p>
    <div class="hero-metrics">
      <div class="metric-pill">
        <span class="lbl">SAMPLING RATE</span>
        <span class="val">22,050 Hz</span>
      </div>
      <div class="metric-pill">
        <span class="lbl">DSP RESOLUTION</span>
        <span class="val">1024-STFT</span>
      </div>
      <div class="metric-pill">
        <span class="lbl">MODEL CORE</span>
        <span class="val">Chatterbox</span>
      </div>
    </div>
  </div>

  <!-- Interactive 3D Canvas -->
  <canvas id="sculpture-canvas"></canvas>

  <!-- Holographic Floating HUD Nodes -->
  <div class="hud-node top-right">
    <div class="audio-indicator">
      <span></span><span></span><span></span><span></span>
    </div>
    HARMONIC HARMONY: 98.4%
  </div>

  <div class="hud-node bottom-right">
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#c084fc" stroke-width="2.5"><path d="M12 2v20M2 12h20M4.93 4.93l14.14 14.14M4.93 19.07L19.07 4.93"/></svg>
    LATENT COHERENCE: ACTIVE
  </div>

  <button class="sound-btn" id="harmonic-pulse-btn" title="Click to pulse harmonic wave frequencies through the filaments">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 2 2-1.79 2-4V7h4V3h-6z"/></svg>
    <span id="sound-btn-text">PULSE HARMONICS</span>
  </button>
</div>

<script>
(function() {{
  const canvas = document.getElementById('sculpture-canvas');
  const ctx = canvas.getContext('2d');
  const soundBtn = document.getElementById('harmonic-pulse-btn');
  const soundBtnText = document.getElementById('sound-btn-text');

  let width, height, dpr;
  let mouseX = 0, mouseY = 0;
  let targetRotX = 0.25, targetRotY = 0.35;
  let rotX = 0.25, rotY = 0.35;
  let isDragging = false;
  let lastMouseX = 0, lastMouseY = 0;
  let time = 0;
  let audioHarmonicImpulse = 0;
  let audioSynth = null;
  let isAudioPlaying = false;

  // Adaptive performance
  let fps = 60;
  let lastFrameTime = performance.now();
  let frameCount = 0;
  let strandCount = 42; // Base metallic filament count
  let strandResolution = 110;

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

  // Mouse / Pointer Parallax
  window.addEventListener('mousemove', (e) => {{
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0) return;
    const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const ny = ((e.clientY - rect.top) / rect.height) * 2 - 1;
    if (isDragging) {{
      const dx = e.clientX - lastMouseX;
      const dy = e.clientY - lastMouseY;
      targetRotY += dx * 0.008;
      targetRotX += dy * 0.008;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
    }} else {{
      targetRotY = 0.35 + nx * 0.45;
      targetRotX = 0.25 - ny * 0.35;
    }}
  }});

  canvas.addEventListener('mousedown', (e) => {{
    isDragging = true;
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
  }});

  window.addEventListener('mouseup', () => {{
    isDragging = false;
  }});

  // Web Audio Synthetic Harmonic Pulse
  soundBtn.addEventListener('click', () => {{
    audioHarmonicImpulse = 1.0;
    try {{
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) return;
      if (!audioSynth) {{
        audioSynth = new AudioContext();
      }}
      if (audioSynth.state === 'suspended') {{
        audioSynth.resume();
      }}

      // Create rich chord (F0 harmonic series: 110Hz, 220Hz, 330Hz, 440Hz, 660Hz)
      const freqs = [110, 220, 277.18, 329.63, 440, 659.25];
      const now = audioSynth.currentTime;
      const masterGain = audioSynth.createGain();
      masterGain.gain.setValueAtTime(0.18, now);
      masterGain.gain.exponentialRampToValueAtTime(0.0001, now + 1.8);
      masterGain.connect(audioSynth.destination);

      freqs.forEach((f, idx) => {{
        const osc = audioSynth.createOscillator();
        const g = audioSynth.createGain();
        osc.type = idx % 2 === 0 ? 'sine' : 'triangle';
        osc.frequency.setValueAtTime(f, now);
        osc.frequency.exponentialRampToValueAtTime(f * (1 + 0.05 * Math.sin(idx)), now + 1.8);
        g.gain.setValueAtTime(1 / (idx + 1), now);
        osc.connect(g);
        g.connect(masterGain);
        osc.start(now);
        osc.stop(now + 1.8);
      }});

      soundBtnText.textContent = "HARMONIC WAVE ACTIVE";
      setTimeout(() => {{
        soundBtnText.textContent = "PULSE HARMONICS";
      }}, 1900);
    }} catch (e) {{
      console.warn("AudioContext not allowed without gesture", e);
    }}
  }});

  // 3D Point projection
  function project(x, y, z, cx, cy, fov, scale) {{
    // Rotate Y
    const cosY = Math.cos(rotY);
    const sinY = Math.sin(rotY);
    const x1 = x * cosY - z * sinY;
    const z1 = z * cosY + x * sinY;

    // Rotate X
    const cosX = Math.cos(rotX);
    const sinX = Math.sin(rotX);
    const y2 = y * cosX - z1 * sinX;
    const z2 = z1 * cosX + y * sinX;

    const distance = 420;
    const pz = z2 + distance;
    if (pz <= 1) return null;

    const f = (fov * scale) / pz;
    return {{
      x: cx + x1 * f,
      y: cy + y2 * f,
      depth: z2,
      scale: f
    }};
  }}

  // Render loop
  function draw() {{
    const now = performance.now();
    const dt = (now - lastFrameTime) / 1000;
    lastFrameTime = now;

    // Adaptive resolution governor
    frameCount++;
    if (frameCount % 60 === 0) {{
      fps = 1 / dt;
      if (fps < 40 && strandCount > 24) {{
        strandCount -= 4;
        strandResolution = Math.max(70, strandResolution - 10);
      }}
    }}

    // Damping mouse rotation
    rotX += (targetRotX - rotX) * 0.06;
    rotY += (targetRotY - rotY) * 0.06;

    // Time step
    time += 0.016;
    if (audioHarmonicImpulse > 0.01) {{
      audioHarmonicImpulse *= 0.95;
    }} else {{
      audioHarmonicImpulse = 0;
    }}

    ctx.clearRect(0, 0, width, height);

    const cx = width * 0.52;
    const cy = height * 0.5;
    const baseRadius = Math.min(width, height) * 0.32;
    const fov = 400;

    // Center ambient glow core
    const coreGrad = ctx.createRadialGradient(cx, cy, 5, cx, cy, baseRadius * 0.9);
    coreGrad.addColorStop(0, 'rgba(216, 180, 254, ' + (0.15 + audioHarmonicImpulse * 0.25) + ')');
    coreGrad.addColorStop(0.5, 'rgba(147, 51, 234, 0.08)');
    coreGrad.addColorStop(1, 'transparent');
    ctx.fillStyle = coreGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, baseRadius * 1.1, 0, Math.PI * 2);
    ctx.fill();

    // Render multi-strand metallic sound filaments
    // Parametric Toroidal / Hopf-harmonic ribbon math:
    // x(u) = (R + r*cos(q*u)) * cos(p*u)
    // y(u) = (R + r*cos(q*u)) * sin(p*u)
    // z(u) = r*sin(q*u) + wave(u, t)
    const p = 2; // Torus knots
    const q = 3;
    const strands = strandCount;
    const steps = strandResolution;

    // Sort strands or segments by average depth for metallic luster
    for (let s = 0; s < strands; s++) {{
      const strandOffset = (s / strands) * Math.PI * 2;
      const strandPhase = strandOffset + time * 0.45;
      
      // Strand harmonic color
      const hue = 275 + Math.sin(s * 0.2 + time) * 22; // plum to metallic lavender
      const lightness = 60 + Math.sin(s * 0.3 + time * 1.2) * 20;
      const baseAlpha = 0.22 + (s / strands) * 0.55;

      ctx.beginPath();
      let firstPoint = true;

      for (let i = 0; i <= steps; i++) {{
        const u = (i / steps) * Math.PI * 2;
        
        // Dynamic harmonic acoustic deformation
        const harmonicWave = Math.sin(u * 5 + time * 2.5 + strandOffset) * (8 + audioHarmonicImpulse * 32);
        const microMod = Math.cos(u * 9 - time * 3.5) * (3 + audioHarmonicImpulse * 14);
        
        // Ribbon radius modulation
        const r = (baseRadius * 0.38 + harmonicWave) * (1 + 0.12 * Math.sin(u * 2 + strandPhase));
        const R = baseRadius * 0.85 + microMod;

        const x = (R + r * Math.cos(q * u + strandPhase)) * Math.cos(p * u);
        const y = (R + r * Math.cos(q * u + strandPhase)) * Math.sin(p * u);
        const z = r * Math.sin(q * u + strandPhase) + Math.sin(u * 4 + time * 2) * 24;

        const proj = project(x, y, z, cx, cy, fov, 1.0);
        if (proj) {{
          if (firstPoint) {{
            ctx.moveTo(proj.x, proj.y);
            firstPoint = false;
          }} else {{
            ctx.lineTo(proj.x, proj.y);
          }}
        }}
      }}

      // Fine filament stroke
      const isLeadStrand = (s % 5 === 0);
      ctx.lineWidth = isLeadStrand ? (1.5 + audioHarmonicImpulse * 1.0) : 0.85;
      
      if (isLeadStrand) {{
        // Iridescent highlight
        ctx.strokeStyle = `hsla(${{hue + 15}}, 90%, ${{Math.min(95, lightness + 15)}}%, ${{Math.min(1.0, baseAlpha + 0.35 + audioHarmonicImpulse * 0.4)}})`;
      }} else {{
        ctx.strokeStyle = `hsla(${{hue}}, 75%, ${{lightness}}%, ${{baseAlpha + audioHarmonicImpulse * 0.25}})`;
      }}
      ctx.stroke();
    }}

    // Render floating luminous sound particle beads travelling along the filaments
    const beadCount = 18;
    for (let b = 0; b < beadCount; b++) {{
      const bProgress = (time * 0.18 + b / beadCount) % 1;
      const u = bProgress * Math.PI * 2;
      const bStrand = (b * 7) % strands;
      const strandPhase = (bStrand / strands) * Math.PI * 2 + time * 0.45;
      
      const r = baseRadius * 0.38;
      const R = baseRadius * 0.85;
      const x = (R + r * Math.cos(q * u + strandPhase)) * Math.cos(p * u);
      const y = (R + r * Math.cos(q * u + strandPhase)) * Math.sin(p * u);
      const z = r * Math.sin(q * u + strandPhase) + Math.sin(u * 4 + time * 2) * 24;

      const proj = project(x, y, z, cx, cy, fov, 1.0);
      if (proj && proj.scale > 0.4) {{
        const bSize = Math.max(1.2, 3.2 * proj.scale * (1 + audioHarmonicImpulse * 0.8));
        ctx.fillStyle = '#ffffff';
        ctx.shadowColor = '#d8b4fe';
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.arc(proj.x, proj.y, bSize, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0; // reset
      }}
    }}

    if (!document.hidden) {{
      requestAnimationFrame(draw);
    }}
  }}

  // Resume when page becomes visible
  document.addEventListener('visibilitychange', () => {{
    if (!document.hidden) {{
      lastFrameTime = performance.now();
      requestAnimationFrame(draw);
    }}
  }});

  // Check reduced motion
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {{
    strandCount = 20;
    draw(); // draw once statically
  }} else {{
    requestAnimationFrame(draw);
  }}

}})();
</script>
</body>
</html>"""


def render_hero_sculpture(height: int = 560):
    """Renders the 3D Hero Sound Sculpture into Streamlit."""
    _render_html(get_sculpture_html(height=height), height=height)

