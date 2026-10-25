"""Matplotlib figures used by the Streamlit app.

Colours are chosen to look good on the app's dark theme.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # draw to memory; no desktop window needed
import matplotlib.pyplot as plt
import numpy as np

BG = "#0d0718"
FG = "#f3e8ff"
GRID = "#2a1542"
ACCENT = "#c084fc"
ACCENT_2 = "#38bdf8"



def _style_axis(ax) -> None:
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)


def waveform_figure(signals: list[tuple[str, np.ndarray, int, str]]):
    """Plot one waveform per row.

    `signals` is a list of (title, samples, sample_rate, colour).
    All rows share the same amplitude range so they are easy to compare.
    """
    fig, axes = plt.subplots(
        len(signals), 1, figsize=(8, 2.2 * len(signals)), squeeze=False
    )
    fig.patch.set_facecolor(BG)
    for ax, (title, y, sr, colour) in zip(axes[:, 0], signals):
        t = np.arange(len(y)) / sr
        ax.plot(t, y, color=colour, linewidth=0.6)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(0, max(t[-1], 1e-3) if len(t) else 1)
        _style_axis(ax)
    axes[-1, 0].set_xlabel("Time (seconds)")
    fig.tight_layout()
    return fig


def _style_colorbar(cbar, label: str) -> None:
    cbar.ax.tick_params(colors=FG, labelsize=8)
    cbar.set_label(label, color=FG, fontsize=8)
    cbar.outline.set_edgecolor(GRID)


def spectrogram_figure(items, fmax: float = 8000.0, top_db: float = 80.0):
    """One spectrogram per row. `items` is a list of (title, Spectrogram)."""
    fig, axes = plt.subplots(len(items), 1, figsize=(8, 2.9 * len(items)), squeeze=False)
    fig.patch.set_facecolor(BG)
    for ax, (title, spec) in zip(axes[:, 0], items):
        keep = spec.freqs <= fmax
        db = spec.db(top_db)[keep]
        im = ax.imshow(
            db,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=-top_db,
            vmax=0,
            extent=[0, max(float(spec.times[-1]), 1e-3), 0, float(spec.freqs[keep][-1])],
        )
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("Frequency (Hz)")
        _style_axis(ax)
        ax.grid(False)
        _style_colorbar(fig.colorbar(im, ax=ax, pad=0.01), "dB")
    axes[-1, 0].set_xlabel("Time (seconds)")
    fig.tight_layout()
    return fig


def pitch_figure(items):
    """Overlay pitch tracks. `items` is a list of (label, PitchTrack, colour)."""
    fig, ax = plt.subplots(figsize=(8, 3.4))
    fig.patch.set_facecolor(BG)
    voiced_values = []
    for label, track, colour in items:
        v = track.f0[track.voiced]
        if v.size == 0:
            continue
        voiced_values.append(v)
        ax.plot(track.times, track.f0, "-", color=colour, linewidth=1.0, alpha=0.6)
        ax.plot(track.times, track.f0, ".", color=colour, markersize=3, label=label)
        ax.axhline(float(np.median(v)), color=colour, linestyle="--", linewidth=0.8, alpha=0.8)
    if voiced_values:
        allv = np.concatenate(voiced_values)
        ax.set_ylim(max(40.0, allv.min() * 0.8), min(500.0, allv.max() * 1.2))
        legend = ax.legend(loc="upper right", fontsize=8, facecolor=BG, edgecolor=GRID)
        for text in legend.get_texts():
            text.set_color(FG)
    else:
        ax.text(0.5, 0.5, "No voiced speech detected", transform=ax.transAxes,
                ha="center", va="center", color=FG)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Pitch F0 (Hz)")
    ax.set_title("Pitch contour (dashed line = median)", fontsize=10, loc="left")
    _style_axis(ax)
    fig.tight_layout()
    return fig


def mfcc_figure(items):
    """MFCC heatmaps (c1-c12; c0 is loudness and is left out).

    `items` is a list of (title, MfccResult). All rows share one colour scale so
    they can be compared directly.
    """
    arrays = [res.coeffs[1:] for _, res in items]
    stacked = np.concatenate([a.ravel() for a in arrays])
    limit = float(np.percentile(np.abs(stacked), 98)) or 1.0
    fig, axes = plt.subplots(len(items), 1, figsize=(8, 2.8 * len(items)), squeeze=False)
    fig.patch.set_facecolor(BG)
    for ax, (title, res), data in zip(axes[:, 0], items, arrays):
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            extent=[0, max(float(res.times[-1]), 1e-3), 0.5, data.shape[0] + 0.5],
        )
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("MFCC index")
        _style_axis(ax)
        ax.grid(False)
        _style_colorbar(fig.colorbar(im, ax=ax, pad=0.01), "value")
    axes[-1, 0].set_xlabel("Time (seconds)")
    fig.tight_layout()
    return fig


def ltas_figure(items, fmax: float = 8000.0):
    """Overlay long-term average spectra. `items` = (label, freqs, ltas_db, colour)."""
    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.patch.set_facecolor(BG)
    for label, freqs, ltas_db, colour in items:
        keep = freqs <= fmax
        ax.plot(freqs[keep], ltas_db[keep], color=colour, linewidth=1.2, label=label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Relative level (dB)")
    ax.set_title("Long-term average spectrum", fontsize=10, loc="left")
    _style_axis(ax)
    legend = ax.legend(loc="upper right", fontsize=8, facecolor=BG, edgecolor=GRID)
    for text in legend.get_texts():
        text.set_color(FG)
    fig.tight_layout()
    return fig
