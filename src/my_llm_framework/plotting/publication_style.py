from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


MORANDI = {
    "salmon": "#DE7C7D",
    "peach": "#FBCFBB",
    "dusty_blue": "#81AFD4",
    "pale_blue": "#C4E0E8",
    "deep_teal": "#26708B",
    "sage": "#B3D9CE",
    "mint": "#9CD0BD",
    "lavender": "#9A9CC8",
    "warm_gray": "#C9C9C4",
    "slate_blue": "#8FA1BD",
    "seafoam": "#ABD7DF",
    "dark_blue": "#1664A2",
    "dark_green": "#2D6A4F",
    "charcoal": "#333333",
}

DOMAIN_COLORS = {
    "source": MORANDI["dusty_blue"],
    "target_h": MORANDI["salmon"],
    "target_self": MORANDI["mint"],
}

METHOD_COLORS = {
    "raw_spectrum": MORANDI["dusty_blue"],
    "physics_summary": MORANDI["peach"],
    "domain_aware": MORANDI["mint"],
    "pnn_forward+pso": MORANDI["deep_teal"],
    "pnn_forward+cmaes": MORANDI["lavender"],
    "retrieval_then_pso_then_cmaes": MORANDI["dark_blue"],
    "inn_inverse+pnn_forward": MORANDI["warm_gray"],
}

POLICY_COLORS = {
    "Source-only": MORANDI["warm_gray"],
    "Rule": MORANDI["dusty_blue"],
    "DeepSeek": MORANDI["mint"],
    "Oracle": MORANDI["lavender"],
}


def set_publication_style(font_scale: str = "large") -> None:
    """Apply a consistent high-readable style for journal figures."""
    if font_scale == "compact":
        title, label, tick, legend, panel = 20, 18, 16, 16, 22
    else:
        title, label, tick, legend, panel = 24, 22, 18, 18, 24
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.titlesize": title,
            "axes.labelsize": label,
            "xtick.labelsize": tick,
            "ytick.labelsize": tick,
            "legend.fontsize": legend,
            "axes.linewidth": 1.6,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.unicode_minus": False,
            "_internal.classic_mode": False,
        }
    )
    plt.rcParams["figure.labelsize"] = panel


def style_axis(ax: plt.Axes, *, grid_axis: str = "y", ylim: tuple[float, float] | None = None) -> None:
    ax.grid(True, axis=grid_axis, alpha=0.24, linewidth=1.0, color="#9AA0A6")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(1.6)
    ax.tick_params(width=1.5, length=6)
    if ylim is not None:
        ax.set_ylim(*ylim)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=plt.rcParams["figure.labelsize"],
        fontweight="bold",
        ha="left",
        va="top",
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: Iterable[str] = ("png", "pdf")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.4)
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def color_cycle(n: int) -> list[str]:
    colors = [
        MORANDI["dusty_blue"],
        MORANDI["salmon"],
        MORANDI["mint"],
        MORANDI["peach"],
        MORANDI["lavender"],
        MORANDI["pale_blue"],
        MORANDI["deep_teal"],
        MORANDI["warm_gray"],
        MORANDI["slate_blue"],
        MORANDI["seafoam"],
    ]
    return [colors[i % len(colors)] for i in range(n)]


def bar_label(ax: plt.Axes, bars, *, fmt: str = "{:.2f}", fontsize: int | None = None) -> None:
    fontsize = fontsize or plt.rcParams["ytick.labelsize"]
    for bar in bars:
        height = bar.get_height()
        if height != height:
            continue
        label = fmt.format(height)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )
