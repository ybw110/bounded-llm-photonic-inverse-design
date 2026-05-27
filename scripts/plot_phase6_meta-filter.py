"""
Phase 6 on-demand meta-filter figure.

This plot follows the Phase 1-5 manuscript figure style. The figure keeps the
claim narrow: bounded selectors see the same finite meta-filter evidence, their
choices remain auditable across three physically retained requests, and representative selected
cylinder geometries can be exported back to CST spectra.

Panels:
  a. Three retained Gaussian-notch requests used in the meta-filter result.
  b. Selected candidate ranks for Manual, DeepSeek, and GLM-5.1.
  c. Offline-vs-CST target MSE for the four exported spectrum examples.
  d. MF01 low-MSE/deep-notch selection with a dimensioned cylinder inset.
  e. MF01 centre-aligned GLM selection with a dimensioned cylinder inset.
  f. MF02 lower-frequency request with a dimensioned cylinder inset.
  g. MF03 broader-notch request with a dimensioned cylinder inset.

The composite and standalone panels are exported as PNG/PDF and displayed with
plt.show() when this script is run directly.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parent
AUDIT_DIR = PROJECT_ROOT / "outputs" / "phase6_metafilter_audit4_v1"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "phase6_meta_filter"

REQUESTS_CSV = AUDIT_DIR / "phase6_metafilter_audit_requests.csv"
SELECTIONS_CSV = AUDIT_DIR / "phase6_metafilter_audit_selection_summary.csv"
PRESENTED_REQUEST_IDS = [
    "mf01_center45_bw5_d90",
    "mf02_center42_bw5_d85",
    "mf04_center45_bw8_d75",
]
DISPLAY_REQUEST_LABELS = {
    "mf01_center45_bw5_d90": "MF01",
    "mf02_center42_bw5_d85": "MF02",
    "mf04_center45_bw8_d75": "MF03",
}

EXPORT_CASES = [
    {
        "key": "mf01_rank0",
        "request_id": "mf01_center45_bw5_d90",
        "short_label": "MF01 rank 0",
        "panel_title": "MF01 low-MSE deep-notch selection",
        "selector_label": "Manual / DeepSeek",
        "spectra_csv": PROJECT_ROOT / "outputs" / "phase6_metafilter_deepseek_source_v1" / "phase6_metafilter_spectra.csv",
        "selected_csv": PROJECT_ROOT / "outputs" / "phase6_metafilter_deepseek_source_v1" / "phase6_metafilter_selected_candidate.csv",
        "cst_csv": PROJECT_ROOT
        / "outputs"
        / "phase6_metafilter_deepseek_source_v1"
        / "cst_results"
        / "phase6_metafilter_deepseek_designer_source_cst_spectrum.csv",
    },
    {
        "key": "mf01_rank5",
        "request_id": "mf01_center45_bw5_d90",
        "short_label": "MF01 rank 5",
        "panel_title": "MF01 centre-aligned GLM selection",
        "selector_label": "GLM-5.1",
        "spectra_csv": PROJECT_ROOT / "outputs" / "phase6_metafilter_glm51_source_v1" / "phase6_metafilter_spectra.csv",
        "selected_csv": PROJECT_ROOT / "outputs" / "phase6_metafilter_glm51_source_v1" / "phase6_metafilter_selected_candidate.csv",
        "cst_csv": PROJECT_ROOT
        / "outputs"
        / "phase6_metafilter_glm51_source_v1"
        / "cst_results"
        / "phase6_metafilter_glm_designer_source_cst_spectrum.csv",
    },
    {
        "key": "mf02_rank0",
        "request_id": "mf02_center42_bw5_d85",
        "short_label": "MF02 rank 0",
        "panel_title": "MF02 lower-frequency request",
        "selector_label": "Shared selection",
        "spectra_csv": AUDIT_DIR
        / "requests"
        / "mf02_center42_bw5_d85"
        / "manual"
        / "phase6_metafilter_spectra.csv",
        "selected_csv": AUDIT_DIR
        / "requests"
        / "mf02_center42_bw5_d85"
        / "manual"
        / "phase6_metafilter_selected_candidate.csv",
        "cst_csv": AUDIT_DIR
        / "requests"
        / "mf02_center42_bw5_d85"
        / "manual"
        / "cst_results_nonfast"
        / "phase6_metafilter_manual_match_source_cst_spectrum.csv",
    },
    {
        "key": "mf04_rank0",
        "request_id": "mf04_center45_bw8_d75",
        "short_label": "MF03 rank 0",
        "panel_title": "MF03 broader-notch request",
        "selector_label": "Shared selection",
        "spectra_csv": AUDIT_DIR
        / "requests"
        / "mf04_center45_bw8_d75"
        / "manual"
        / "phase6_metafilter_spectra.csv",
        "selected_csv": AUDIT_DIR
        / "requests"
        / "mf04_center45_bw8_d75"
        / "manual"
        / "phase6_metafilter_selected_candidate.csv",
        "cst_csv": AUDIT_DIR
        / "requests"
        / "mf04_center45_bw8_d75"
        / "manual"
        / "cst_results_nonfast"
        / "phase6_metafilter_manual_match_source_cst_spectrum.csv",
    },
]

SUCCESS_THRESHOLD = 0.03

SAVE_PNG = True
SAVE_PDF = True
SAVE_DPI = 600
SHOW_FIGURES = True

TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
VALUE_FONTSIZE = 12
LEGEND_FONTSIZE = 13
PANEL_FONTSIZE = 18

PALETTE = {
    "blue": "#81AFD4",
    "pale_blue": "#CAE0E8",
    "red": "#DE7C7D",
    "peach": "#FBCFBB",
    "gray": "#CBCBCB",
    "dark_gray": "#4D4D4D",
}

SELECTOR_ORDER = ["manual", "deepseek", "glm"]
SELECTOR_LABELS = {"manual": "Manual", "deepseek": "DeepSeek", "glm": "GLM-5.1"}
REQUEST_COLORS = [PALETTE["dark_gray"], PALETTE["blue"], PALETTE["red"], "#9D86A8"]


def setup_basic_matplotlib() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "Arial"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 6 meta-filter result file: {path}")
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, file_stem: str, *, tight: bool = True) -> None:
    if tight:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout.*",
            )
            fig.tight_layout()
    if SAVE_PNG:
        fig.savefig(OUTPUT_DIR / f"{file_stem}.png", dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.08)
    if SAVE_PDF:
        fig.savefig(OUTPUT_DIR / f"{file_stem}.pdf", dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.08)


def style_axis(
    ax: plt.Axes,
    *,
    x_grid: bool = False,
    y_grid: bool = True,
    y_lim: tuple[float, float] | None = None,
) -> None:
    if y_grid:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.28, zorder=0)
    if x_grid:
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.7, alpha=0.22, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=1.0, length=4)
    ax.xaxis.label.set_size(LABEL_FONTSIZE)
    ax.yaxis.label.set_size(LABEL_FONTSIZE)
    if y_lim is not None:
        ax.set_ylim(*y_lim)


def add_panel_label(ax: plt.Axes, label: str, *, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_FONTSIZE,
        fontweight="bold",
        ha="left",
        va="top",
    )


def format_mse(value: float) -> str:
    if value < 0.001:
        return f"{value:.2e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{value:.4f}"


def format_dim(value: float) -> str:
    return f"{float(value):.2f}"


def target_curve(center: float, bandwidth: float, depth: float, frequency: np.ndarray) -> np.ndarray:
    sigma = bandwidth / 2.355
    return 1.0 - depth * np.exp(-0.5 * ((frequency - center) / sigma) ** 2)


def display_request_label(request_id: str) -> str:
    return DISPLAY_REQUEST_LABELS.get(str(request_id), str(request_id).split("_")[0].upper())


def interpolate_to(x: np.ndarray, y: np.ndarray, ref_x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    return np.interp(ref_x, x[order], y[order])


def selected_row(path: Path) -> pd.Series:
    selected = read_csv_checked(path)
    if selected.empty:
        raise ValueError(f"Selected-candidate table is empty: {path}")
    return selected.iloc[0]


def build_export_case(case_spec: dict[str, object]) -> dict[str, object]:
    spectra = read_csv_checked(Path(case_spec["spectra_csv"]))
    selected = selected_row(Path(case_spec["selected_csv"]))
    cst = read_csv_checked(Path(case_spec["cst_csv"]))

    freq = spectra["frequency_thz"].to_numpy(dtype=float)
    target = spectra["target_amplitude"].to_numpy(dtype=float)
    offline = spectra["candidate_amplitude"].to_numpy(dtype=float)
    cst_freq = cst["frequency_thz"].to_numpy(dtype=float)
    cst_amp = cst["cst_amplitude"].to_numpy(dtype=float)
    target_on_cst = interpolate_to(freq, target, cst_freq)
    offline_on_cst = interpolate_to(freq, offline, cst_freq)
    offline_min_idx = int(np.nanargmin(offline))
    cst_min_idx = int(np.nanargmin(cst_amp))

    case = dict(case_spec)
    case.update(
        {
            "spectra": spectra,
            "selected": selected,
            "cst": cst,
            "candidate_rank": int(selected["candidate_rank"]),
            "offline_target_mse": float(np.mean((offline - target) ** 2)),
            "cst_target_mse": float(np.mean((cst_amp - target_on_cst) ** 2)),
            "cst_offline_mse": float(np.mean((cst_amp - offline_on_cst) ** 2)),
            "offline_min_amplitude": float(offline[offline_min_idx]),
            "offline_min_frequency_thz": float(freq[offline_min_idx]),
            "cst_min_amplitude": float(cst_amp[cst_min_idx]),
            "cst_min_frequency_thz": float(cst_freq[cst_min_idx]),
        }
    )
    return case


def load_phase6_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    requests = read_csv_checked(REQUESTS_CSV).copy()
    requests = requests[requests["request_id"].astype(str).isin(PRESENTED_REQUEST_IDS)].copy()
    requests["_presented_order"] = requests["request_id"].astype(str).map(
        {request_id: idx for idx, request_id in enumerate(PRESENTED_REQUEST_IDS)}
    )
    requests = requests.sort_values("_presented_order", kind="stable").drop(columns="_presented_order").reset_index(drop=True)
    selections = read_csv_checked(SELECTIONS_CSV).copy()
    selections = selections[selections["request_id"].astype(str).isin(PRESENTED_REQUEST_IDS)].copy()
    request_order = {request_id: idx for idx, request_id in enumerate(requests["request_id"].astype(str))}
    selector_order = {selector: idx for idx, selector in enumerate(SELECTOR_ORDER)}
    selections["request_order"] = selections["request_id"].astype(str).map(request_order)
    selections["selector_order"] = selections["selector"].astype(str).map(selector_order)
    selections = selections.sort_values(["request_order", "selector_order"], kind="stable").reset_index(drop=True)
    exports = {str(case_spec["key"]): build_export_case(case_spec) for case_spec in EXPORT_CASES}
    return requests, selections, exports


def draw_request_panel(ax: plt.Axes, requests: pd.DataFrame, *, panel_label: str | None = "a") -> None:
    frequency = np.linspace(30.0, 60.0, 301)
    for idx, row in enumerate(requests.itertuples(index=False)):
        ax.plot(
            frequency,
            target_curve(float(row.center_frequency_thz), float(row.bandwidth_thz), float(row.depth), frequency),
            linewidth=2.2,
            color=REQUEST_COLORS[idx % len(REQUEST_COLORS)],
            label=f"{display_request_label(str(row.request_id))} | {float(row.center_frequency_thz):g} THz | BW {float(row.bandwidth_thz):g}",
            zorder=3,
        )
    ax.set_title("Three retained notch requests", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=8)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Target amplitude")
    style_axis(ax, y_lim=(-0.03, 1.06), x_grid=True)
    legend = ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.90,
        fontsize=LEGEND_FONTSIZE,
        loc="lower right",
    )
    legend.set_zorder(20)
    if panel_label:
        add_panel_label(ax, panel_label)


def draw_rank_heatmap(ax: plt.Axes, requests: pd.DataFrame, selections: pd.DataFrame, *, panel_label: str | None = "b") -> None:
    matrix = np.full((len(requests), len(SELECTOR_ORDER)), np.nan)
    for row in selections.itertuples(index=False):
        matrix[int(row.request_order), int(row.selector_order)] = float(row.selected_rank)
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=max(1.0, float(np.nanmax(matrix))))
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            value_color = "white" if not np.isnan(value) and value >= 1 else PALETTE["dark_gray"]
            ax.text(
                x,
                y,
                "-" if np.isnan(value) else str(int(value)),
                fontsize=VALUE_FONTSIZE + 2,
                ha="center",
                va="center",
                color=value_color,
                fontweight="bold" if value_color == "white" else "normal",
            )
    ax.set_title("Auditable selected rank", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=8)
    ax.set_xticks(np.arange(len(SELECTOR_ORDER)))
    ax.set_xticklabels([SELECTOR_LABELS[selector] for selector in SELECTOR_ORDER], rotation=24, ha="right")
    ax.set_yticks(np.arange(len(requests)))
    ax.set_yticklabels([display_request_label(str(request_id)) for request_id in requests["request_id"].astype(str)])
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=0, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    im.set_clim(0.0, max(1.0, float(np.nanmax(matrix))))
    if panel_label:
        add_panel_label(ax, panel_label)


def draw_export_mse_panel(ax: plt.Axes, exports: dict[str, dict[str, object]], *, panel_label: str | None = "c") -> None:
    cases = [exports["mf01_rank0"], exports["mf01_rank5"], exports["mf02_rank0"], exports["mf04_rank0"]]
    labels = [str(case["short_label"]).replace(" ", "\n", 1) for case in cases]
    offline = [float(case["offline_target_mse"]) for case in cases]
    cst = [float(case["cst_target_mse"]) for case in cases]
    x = np.arange(len(cases), dtype=float)
    width = 0.34

    bars_offline = ax.bar(
        x - width / 2,
        offline,
        width,
        color=PALETTE["blue"],
        edgecolor="white",
        linewidth=0.8,
        label="Offline",
        zorder=3,
    )
    bars_cst = ax.bar(
        x + width / 2,
        cst,
        width,
        color=PALETTE["red"],
        edgecolor="white",
        linewidth=0.8,
        label="CST",
        zorder=3,
    )
    ax.axhline(SUCCESS_THRESHOLD, color=PALETTE["dark_gray"], linestyle=":", linewidth=1.2, zorder=2)
    ax.text(
        0.98,
        SUCCESS_THRESHOLD + 0.00035,
        "threshold = 0.03",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=VALUE_FONTSIZE,
        color=PALETTE["dark_gray"],
    )
    for bars in [bars_offline, bars_cst]:
        for bar in bars:
            value = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.0006,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=VALUE_FONTSIZE,
                rotation=28,
            )
    ax.set_title("Target-MSE after physical export", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=8)
    ax.set_ylabel("Amplitude MSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_FONTSIZE)
    style_axis(ax, y_lim=(0.0, 0.0345))
    # ax.legend(
    #     frameon=False,
    #     fontsize=LEGEND_FONTSIZE,
    #     loc="upper right",
    #     bbox_to_anchor=(0.985, 0.7),
    #     borderaxespad=0.0,
    # )
    # ===== custom vertical legend =====
    legend_x = 0.82
    rect_w = 0.075
    rect_h = 0.045

    # Offline color block
    ax.add_patch(
        Rectangle(
            (legend_x, 0.70),
            rect_w,
            rect_h,
            transform=ax.transAxes,
            facecolor=PALETTE["blue"],
            edgecolor="none",
            clip_on=False,
            zorder=10,
        )
    )
    ax.text(
        legend_x + rect_w / 2,
        0.70 - 0.025,
        "Offline",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=LEGEND_FONTSIZE,
    )

    # CST color block
    ax.add_patch(
        Rectangle(
            (legend_x, 0.54),
            rect_w,
            rect_h,
            transform=ax.transAxes,
            facecolor=PALETTE["red"],
            edgecolor="none",
            clip_on=False,
            zorder=10,
        )
    )
    ax.text(
        legend_x + rect_w / 2,
        0.54 - 0.025,
        "CST",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=LEGEND_FONTSIZE,
    )
    # ================================
    if panel_label:
        add_panel_label(ax, panel_label)


def draw_cylinder_structure_inset(ax: plt.Axes, selected: pd.Series, *, bounds: list[float] | None = None) -> None:
    bounds = [0.50, 0.045, 0.47, 0.50] if bounds is None else bounds
    inset = ax.inset_axes(bounds)
    inset.set_xlim(-0.24, 2.34)
    inset.set_ylim(-0.22, 1.18)
    inset.set_aspect("equal")
    inset.axis("off")

    radius = float(selected["radius_raw"])
    gap = float(selected["gap_raw"])
    thickness = float(selected["thickness_raw"])
    epsilon = float(selected["epsilon_raw"])
    period = 2.0 * radius + gap

    inset.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#F4EBDD",
            edgecolor=PALETTE["dark_gray"],
            linewidth=0.9,
            alpha=0.96,
        )
    )
    inset.add_patch(Circle((0.5, 0.5), 0.27, facecolor="#C9A34E", edgecolor="#7D6228", linewidth=0.8))
    inset.text(0.5, 1.08, "Top view", ha="center", va="bottom", fontsize=11.8, color="#333333")
    inset.annotate("", xy=(-0.10, 1.0), xytext=(-0.10, 0.0), arrowprops=dict(arrowstyle="<|-|>", lw=0.58, color="#333333"))
    inset.annotate("", xy=(1.0, -0.10), xytext=(0.0, -0.10), arrowprops=dict(arrowstyle="<|-|>", lw=0.58, color="#333333"))
    inset.text(-0.17, 0.5, f"p={period:.2f}", ha="center", va="center", fontsize=11.3, rotation=90)
    inset.text(0.5, -0.17, f"p={period:.2f}", ha="center", va="center", fontsize=11.3)
    inset.text(
        1.10,
        0.52,
        "p=" + format_dim(period) + " um\n"
        "r=" + format_dim(radius) + " um\n"
        "g=" + format_dim(gap) + " um\n"
        "t=" + format_dim(thickness) + " um\n"
        "eps=" + format_dim(epsilon),
        ha="left",
        va="center",
        fontsize=11.3,
        color="#333333",
        linespacing=1.10,
    )


def draw_spectrum_panel(
    ax: plt.Axes,
    case: dict[str, object],
    requests: pd.DataFrame,
    *,
    panel_label: str | None,
) -> None:
    spectra = case["spectra"]
    cst = case["cst"]
    request = requests[requests["request_id"].astype(str) == str(case["request_id"])].iloc[0]
    freq = spectra["frequency_thz"].to_numpy(dtype=float)
    cst_freq = cst["frequency_thz"].to_numpy(dtype=float)
    marker_step = max(1, len(cst_freq) // 16)

    ax.plot(freq, spectra["target_amplitude"], color=PALETTE["dark_gray"], linewidth=2.2, label="Target", zorder=3)
    ax.plot(
        freq,
        spectra["candidate_amplitude"],
        color=PALETTE["blue"],
        linestyle="--",
        linewidth=2.2,
        label="Offline candidate",
        zorder=4,
    )
    ax.plot(
        cst_freq,
        cst["cst_amplitude"],
        color=PALETTE["red"],
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        markevery=marker_step,
        label="CST",
        zorder=5,
    )
    ax.axvline(float(request["center_frequency_thz"]), color=PALETTE["gray"], linestyle=":", linewidth=1.1, zorder=1)
    ax.set_title(str(case["panel_title"]), fontsize=TITLE_FONTSIZE, fontweight="normal", pad=8)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Amplitude")
    style_axis(ax, y_lim=(-0.04, 1.08), x_grid=True)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper left")

    info = (
        f"{case['selector_label']} | rank {int(case['candidate_rank'])}\n"
        f"Offline MSE {format_mse(float(case['offline_target_mse']))}\n"
        f"CST MSE {format_mse(float(case['cst_target_mse']))}"
    )
    ax.text(
        0.045,
        0.06,
        info,
        transform=ax.transAxes,
        fontsize=VALUE_FONTSIZE,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.88},
        zorder=6,
    )
    draw_cylinder_structure_inset(ax, case["selected"])
    if panel_label:
        add_panel_label(ax, panel_label)


def export_plot_tables(
    requests: pd.DataFrame,
    selections: pd.DataFrame,
    exports: dict[str, dict[str, object]],
) -> None:
    requests.to_csv(OUTPUT_DIR / "phase6_metafilter_request_specs_plot.csv", index=False)
    selections.to_csv(OUTPUT_DIR / "phase6_metafilter_rank_audit_plot.csv", index=False)

    metric_rows: list[dict[str, object]] = []
    for case in exports.values():
        selected = case["selected"]
        metric_rows.append(
            {
                "case_key": case["key"],
                "request_id": case["request_id"],
                "selector_label": case["selector_label"],
                "candidate_rank": case["candidate_rank"],
                "offline_target_mse": case["offline_target_mse"],
                "cst_target_mse": case["cst_target_mse"],
                "cst_offline_mse": case["cst_offline_mse"],
                "offline_min_frequency_thz": case["offline_min_frequency_thz"],
                "offline_min_amplitude": case["offline_min_amplitude"],
                "cst_min_frequency_thz": case["cst_min_frequency_thz"],
                "cst_min_amplitude": case["cst_min_amplitude"],
                "epsilon_raw": float(selected["epsilon_raw"]),
                "gap_raw": float(selected["gap_raw"]),
                "thickness_raw": float(selected["thickness_raw"]),
                "radius_raw": float(selected["radius_raw"]),
                "period_um": 2.0 * float(selected["radius_raw"]) + float(selected["gap_raw"]),
                "spectra_csv": str(Path(case["spectra_csv"]).relative_to(PROJECT_ROOT)),
                "selected_csv": str(Path(case["selected_csv"]).relative_to(PROJECT_ROOT)),
                "cst_csv": str(Path(case["cst_csv"]).relative_to(PROJECT_ROOT)),
            }
        )
    pd.DataFrame(metric_rows).to_csv(OUTPUT_DIR / "phase6_metafilter_export_metrics_plot.csv", index=False)

    readme = """# Phase 6 meta-filter 绘图说明

## 图意

这张图把 meta-filter 结果组织成两层证据：
1. 在 `MF01`、`MF02`、`MF03` 三个保留的 on-demand notch 请求下，Manual、DeepSeek 和 GLM-5.1 的 selected rank 都可审计；
2. 四个保留的物理导出示例都回到 `Target / offline candidate / CST` 三谱线比较，并附带 cylinder top-view 与几何尺寸。

当前正文图不把求解器调试细节写进面板标题。物理验证面板只表达“保留示例已导出 CST spectrum”。原始 50 THz audit stress request 不作为当前正文正结果。

## 面板

- `a`: 三个保留的 Gaussian-notch 请求曲线。
- `b`: 三请求三 selector 的 selected-rank heatmap。
- `c`: 四个保留导出示例的 offline-vs-CST target MSE。
- `d`: `MF01` Manual / DeepSeek rank-0 深陷波候选。
- `e`: `MF01` GLM-5.1 rank-5 centre-aligned 候选。
- `f`: `MF02` lower-frequency shared rank-0 候选。
- `g`: `MF03` broader-notch shared rank-0 候选。

## 直接输入结果

- `outputs/phase6_metafilter_audit4_v1/phase6_metafilter_audit_requests.csv`
- `outputs/phase6_metafilter_audit4_v1/phase6_metafilter_audit_selection_summary.csv`
- `outputs/phase6_metafilter_deepseek_source_v1/phase6_metafilter_spectra.csv`
- `outputs/phase6_metafilter_glm51_source_v1/phase6_metafilter_spectra.csv`
- `outputs/phase6_metafilter_audit4_v1/requests/mf02_center42_bw5_d85/manual/phase6_metafilter_spectra.csv`
- `outputs/phase6_metafilter_audit4_v1/requests/mf04_center45_bw8_d75/manual/phase6_metafilter_spectra.csv`

图中 `MF03` 是当前保留结果的顺序编号，对应原始 audit request id
`mf04_center45_bw8_d75`；这一显示重编号只用于论文图面，原始结果文件名保持不变。

## 本目录导出的 source data

- `phase6_metafilter_request_specs_plot.csv`
- `phase6_metafilter_rank_audit_plot.csv`
- `phase6_metafilter_export_metrics_plot.csv`
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def draw_composite(
    requests: pd.DataFrame,
    selections: pd.DataFrame,
    exports: dict[str, dict[str, object]],
) -> plt.Figure:
    fig = plt.figure(figsize=(16.2, 12.0))
    gs = GridSpec(
        3,
        6,
        figure=fig,
        height_ratios=[0.92, 1.08, 1.08],
        hspace=0.52,
        wspace=0.80,
    )

    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])
    ax_f = fig.add_subplot(gs[2, 0:3])
    ax_g = fig.add_subplot(gs[2, 3:6])

    draw_request_panel(ax_a, requests, panel_label="a")
    draw_rank_heatmap(ax_b, requests, selections, panel_label="b")
    draw_export_mse_panel(ax_c, exports, panel_label="c")
    draw_spectrum_panel(ax_d, exports["mf01_rank0"], requests, panel_label="d")
    draw_spectrum_panel(ax_e, exports["mf01_rank5"], requests, panel_label="e")
    draw_spectrum_panel(ax_f, exports["mf02_rank0"], requests, panel_label="f")
    draw_spectrum_panel(ax_g, exports["mf04_rank0"], requests, panel_label="g")
    return fig


def save_standalone_panels(
    requests: pd.DataFrame,
    selections: pd.DataFrame,
    exports: dict[str, dict[str, object]],
) -> None:
    panel_specs = [
        ("phase6_metafilter_panel_a_requests", lambda ax: draw_request_panel(ax, requests, panel_label="a"), (5.8, 4.5)),
        ("phase6_metafilter_panel_b_rank_heatmap", lambda ax: draw_rank_heatmap(ax, requests, selections, panel_label="b"), (5.5, 4.5)),
        ("phase6_metafilter_panel_c_export_mse", lambda ax: draw_export_mse_panel(ax, exports, panel_label="c"), (5.8, 4.5)),
        ("phase6_metafilter_panel_d_mf01_rank0", lambda ax: draw_spectrum_panel(ax, exports["mf01_rank0"], requests, panel_label="d"), (7.3, 4.5)),
        ("phase6_metafilter_panel_e_mf01_rank5", lambda ax: draw_spectrum_panel(ax, exports["mf01_rank5"], requests, panel_label="e"), (7.3, 4.5)),
        ("phase6_metafilter_panel_f_mf02_rank0", lambda ax: draw_spectrum_panel(ax, exports["mf02_rank0"], requests, panel_label="f"), (7.3, 4.5)),
        ("phase6_metafilter_panel_g_mf04_rank0", lambda ax: draw_spectrum_panel(ax, exports["mf04_rank0"], requests, panel_label="g"), (7.3, 4.5)),
    ]
    for file_stem, draw_fn, figsize in panel_specs:
        panel_fig, panel_ax = plt.subplots(figsize=figsize)
        draw_fn(panel_ax)
        save_figure(panel_fig, file_stem)
        plt.close(panel_fig)


def main() -> None:
    setup_basic_matplotlib()
    ensure_output_dir()
    requests, selections, exports = load_phase6_data()
    export_plot_tables(requests, selections, exports)
    fig = draw_composite(requests, selections, exports)
    save_figure(fig, "phase6_metafilter_composite", tight=False)
    save_standalone_panels(requests, selections, exports)
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
