"""
Phase 6 four-request meta-filter audit figure.

The figure complements the single-request hero in plot_phase6_meta-filter.py.
It summarises four requested Gaussian-notch functions and the bounded selector
choices made by Manual, DeepSeek, and GLM-5.1. If CST spectra already exist for
the unique selected geometries, a CST target-MSE panel is filled automatically;
otherwise that panel records pending physical exports.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parent
AUDIT_DIR = PROJECT_ROOT / "outputs" / "phase6_metafilter_audit4_v1"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "phase6_meta_filter_audit"

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
SELECTOR_COLORS = {"manual": PALETTE["pale_blue"], "deepseek": PALETTE["blue"], "glm": PALETTE["red"]}
REQUEST_COLORS = [PALETTE["dark_gray"], PALETTE["blue"], PALETTE["red"], "#9D86A8"]


def setup_basic_matplotlib() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "Arial"


def save_figure(fig: plt.Figure, file_stem: str, *, tight: bool = True) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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


def style_axis(ax: plt.Axes, *, y_lim: tuple[float, float] | None = None, x_grid: bool = False) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.28, zorder=0)
    if x_grid:
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.7, alpha=0.20, zorder=0)
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
    ax.text(x, y, label, transform=ax.transAxes, fontsize=PANEL_FONTSIZE, fontweight="bold", ha="left", va="top")


def resolve_path(path_text: str) -> Path:
    path = Path(str(path_text))
    if path.exists():
        return path
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    return path


def read_audit_tables(audit_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_dir = AUDIT_DIR if audit_dir is None else audit_dir
    requests_csv = audit_dir / "phase6_metafilter_audit_requests.csv"
    selection_csv = audit_dir / "phase6_metafilter_audit_selection_summary.csv"
    if not requests_csv.exists() or not selection_csv.exists():
        raise FileNotFoundError(f"Run design_phase6_metafilter_audit first. Missing {requests_csv} or {selection_csv}")
    requests = pd.read_csv(requests_csv)
    selections = pd.read_csv(selection_csv)
    request_order = {request_id: idx for idx, request_id in enumerate(requests["request_id"].astype(str))}
    selections["request_order"] = selections["request_id"].astype(str).map(request_order)
    selections["selector_order"] = selections["selector"].map({selector: idx for idx, selector in enumerate(SELECTOR_ORDER)})
    selections = selections.sort_values(["request_order", "selector_order"], kind="stable").reset_index(drop=True)
    return requests, selections


def interp_to(x: np.ndarray, y: np.ndarray, ref_x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    return np.interp(ref_x, x[order], y[order])


def compute_cst_metrics(selections: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in selections.itertuples(index=False):
        spectra_path = resolve_path(row.selected_spectrum_csv)
        cst_path = resolve_path(row.unique_cst_expected_result_csv)
        metrics = {
            "audit_row_id": row.audit_row_id,
            "cst_csv": str(cst_path),
            "cst_exists": bool(cst_path.exists()),
            "cst_target_mse": np.nan,
            "cst_offline_mse": np.nan,
            "cst_min_amplitude": np.nan,
            "cst_min_frequency_thz": np.nan,
        }
        if spectra_path.exists() and cst_path.exists():
            spectra = pd.read_csv(spectra_path)
            cst = pd.read_csv(cst_path)
            cst_freq = cst["frequency_thz"].to_numpy(dtype=float)
            cst_amp = cst["cst_amplitude"].to_numpy(dtype=float)
            target = interp_to(
                spectra["frequency_thz"].to_numpy(dtype=float),
                spectra["target_amplitude"].to_numpy(dtype=float),
                cst_freq,
            )
            offline = interp_to(
                spectra["frequency_thz"].to_numpy(dtype=float),
                spectra["candidate_amplitude"].to_numpy(dtype=float),
                cst_freq,
            )
            min_idx = int(np.argmin(cst_amp))
            metrics.update(
                {
                    "cst_target_mse": float(np.mean((cst_amp - target) ** 2)),
                    "cst_offline_mse": float(np.mean((cst_amp - offline) ** 2)),
                    "cst_min_amplitude": float(cst_amp[min_idx]),
                    "cst_min_frequency_thz": float(cst_freq[min_idx]),
                }
            )
        rows.append(metrics)
    metrics_df = pd.DataFrame(rows)
    return selections.merge(metrics_df, on="audit_row_id", how="left")


def target_curve(center: float, bandwidth: float, depth: float, frequency: np.ndarray) -> np.ndarray:
    sigma = bandwidth / 2.355
    return 1.0 - depth * np.exp(-((frequency - center) ** 2) / (2.0 * sigma**2))


def draw_request_panel(ax: plt.Axes, requests: pd.DataFrame, *, panel_label: str = "a") -> None:
    frequency = np.linspace(30.0, 60.0, 301)
    for idx, row in enumerate(requests.itertuples(index=False)):
        label = f"{row.request_id.split('_')[0].upper()} | {row.center_frequency_thz:g} THz | BW {row.bandwidth_thz:g}"
        ax.plot(
            frequency,
            target_curve(float(row.center_frequency_thz), float(row.bandwidth_thz), float(row.depth), frequency),
            linewidth=2.2,
            color=REQUEST_COLORS[idx % len(REQUEST_COLORS)],
            label=label,
        )
    ax.set_title("Four on-demand notch requests", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=8)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Target amplitude")
    style_axis(ax, y_lim=(-0.03, 1.06))
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="lower right")
    add_panel_label(ax, panel_label)


def draw_rank_panel(ax: plt.Axes, requests: pd.DataFrame, selections: pd.DataFrame, *, panel_label: str = "b") -> None:
    matrix = np.full((len(requests), len(SELECTOR_ORDER)), np.nan)
    for row in selections.itertuples(index=False):
        if row.selector in SELECTOR_ORDER:
            matrix[int(row.request_order), SELECTOR_ORDER.index(row.selector)] = float(row.selected_rank)
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=max(1.0, np.nanmax(matrix)))
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            label = "-" if np.isnan(value) else str(int(value))
            ax.text(x, y, label, ha="center", va="center", fontsize=VALUE_FONTSIZE + 1, color=PALETTE["dark_gray"])
    ax.set_title("Selected candidate rank", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=8)
    ax.set_xticks(np.arange(len(SELECTOR_ORDER)))
    ax.set_xticklabels([SELECTOR_LABELS[selector] for selector in SELECTOR_ORDER], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(requests)))
    ax.set_yticklabels([request_id.split("_")[0].upper() for request_id in requests["request_id"].astype(str)])
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=0, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Rank", fontsize=LABEL_FONTSIZE - 1)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE - 1)
    add_panel_label(ax, panel_label)


def draw_grouped_metric_panel(
    ax: plt.Axes,
    requests: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    column: str,
    title: str,
    ylabel: str,
    panel_label: str,
    threshold: float | None = None,
) -> None:
    request_ids = list(requests["request_id"].astype(str))
    base_x = np.arange(len(request_ids), dtype=float)
    width = 0.23
    for idx, selector in enumerate(SELECTOR_ORDER):
        rows = selections[selections["selector"] == selector].set_index("request_id").reindex(request_ids)
        values = rows[column].to_numpy(dtype=float)
        x = base_x + (idx - 1) * width
        bars = ax.bar(
            x,
            values,
            width=width,
            color=SELECTOR_COLORS[selector],
            edgecolor="white",
            linewidth=0.8,
            label=SELECTOR_LABELS[selector],
            zorder=3,
        )
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=VALUE_FONTSIZE)
    if threshold is not None:
        ax.axhline(threshold, color=PALETTE["dark_gray"], linestyle=":", linewidth=1.1)
        ax.text(0.99, threshold, f"threshold={threshold:g}", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=11.0)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="normal", pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xticks(base_x)
    ax.set_xticklabels([request_id.split("_")[0].upper() for request_id in request_ids])
    max_value = np.nanmax(selections[column].to_numpy(dtype=float))
    y_top = max(max_value * 1.28, threshold * 1.12 if threshold is not None else 0.0, 0.001)
    style_axis(ax, y_lim=(0.0, y_top))
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper left", ncol=1)
    add_panel_label(ax, panel_label)


def draw_tradeoff_panel(ax: plt.Axes, selections: pd.DataFrame, *, panel_label: str = "d") -> None:
    for selector in SELECTOR_ORDER:
        rows = selections[selections["selector"] == selector]
        if rows.empty:
            continue
        ax.scatter(
            rows["offline_center_error_thz"],
            rows["offline_min_amplitude"],
            s=115,
            color=SELECTOR_COLORS[selector],
            edgecolor=PALETTE["dark_gray"],
            linewidth=0.7,
            label=SELECTOR_LABELS[selector],
            zorder=3,
        )
        for row in rows.itertuples(index=False):
            ax.text(
                float(row.offline_center_error_thz) + 0.035,
                float(row.offline_min_amplitude) + 0.003,
                str(row.request_id).split("_")[0].upper(),
                fontsize=VALUE_FONTSIZE,
                color=PALETTE["dark_gray"],
            )
    ax.set_title("Selected notch tradeoff", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=8)
    ax.set_xlabel("Offline notch-centre error (THz)")
    ax.set_ylabel("Offline minimum amplitude")
    style_axis(ax, x_grid=False)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper right")
    add_panel_label(ax, panel_label)


def draw_cst_panel(ax: plt.Axes, requests: pd.DataFrame, selections: pd.DataFrame, *, panel_label: str = "e") -> None:
    if selections["cst_target_mse"].notna().any():
        draw_grouped_metric_panel(
            ax,
            requests,
            selections,
            column="cst_target_mse",
            title="CST target-MSE audit",
            ylabel="CST amplitude MSE",
            panel_label=panel_label,
            threshold=0.03,
        )
        return
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0.08, 0.18), 0.84, 0.62, facecolor=PALETTE["pale_blue"], edgecolor=PALETTE["dark_gray"], alpha=0.74))
    ax.text(0.50, 0.60, "CST audit pending", ha="center", va="center", fontsize=TITLE_FONTSIZE + 1)
    ax.text(
        0.50,
        0.43,
        "Run the unique CST manifest\nthen re-run this plot.",
        ha="center",
        va="center",
        fontsize=LABEL_FONTSIZE - 1,
        linespacing=1.18,
    )
    ax.set_title("Physical export audit", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=8)
    add_panel_label(ax, panel_label)


def write_source_tables(selections: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selections.to_csv(OUTPUT_DIR / "phase6_metafilter_audit_plot_metrics.csv", index=False)
    readme = """# Phase 6 meta-filter four-request audit figure

## 面板说明

- `a`: 四个 Gaussian notch 功能请求。
- `b`: Manual、DeepSeek、GLM-5.1 在每个请求下选中的 candidate rank。
- `c`: 选中候选的 offline target MSE。
- `d`: 离线 notch centre error 与 notch minimum-amplitude tradeoff。
- `e`: 若 unique CST manifest 已经跑完，则显示 CST target MSE；否则提示 physical export 待补。

## 输入结果

- `outputs/phase6_metafilter_audit4_v1/phase6_metafilter_audit_requests.csv`
- `outputs/phase6_metafilter_audit4_v1/phase6_metafilter_audit_selection_summary.csv`
- `outputs/phase6_metafilter_audit4_v1/phase6_metafilter_audit_cst_manifest_unique.csv`

## 导出的 source data

- `phase6_metafilter_audit_plot_metrics.csv`
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def draw_composite(requests: pd.DataFrame, selections: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(15.7, 8.7))
    gs = GridSpec(2, 6, figure=fig, hspace=0.48, wspace=0.78)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])

    draw_request_panel(ax_a, requests)
    draw_rank_panel(ax_b, requests, selections)
    draw_grouped_metric_panel(
        ax_c,
        requests,
        selections,
        column="offline_target_mse",
        title="Offline selected quality",
        ylabel="Offline amplitude MSE",
        panel_label="c",
        threshold=0.03,
    )
    draw_tradeoff_panel(ax_d, selections)
    draw_cst_panel(ax_e, requests, selections)
    return fig


def main() -> None:
    global AUDIT_DIR
    global OUTPUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    AUDIT_DIR = args.audit_dir
    OUTPUT_DIR = args.output_dir

    setup_basic_matplotlib()
    requests, selections = read_audit_tables()
    selections = compute_cst_metrics(selections)
    write_source_tables(selections)
    fig = draw_composite(requests, selections)
    save_figure(fig, "phase6_metafilter_audit_composite", tight=False)
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
