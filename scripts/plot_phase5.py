"""
Phase 5 CST unit-cell validation figure.

This script follows the standalone plotting style of plot_phase01_retrieval_evidence.py,
plot_phase2.py, plot_phase3.py, and plot_phase4.py. It creates a dense main-text-ready
Phase 5 figure:

a. Closed-loop CST validation schematic and case map.
b-e. Target / offline candidate / CST spectrum overlays for four main unit-cell cases.
f. CST error audit across all available cases, including the SI Target-H case.

Each panel is saved as PNG/PDF, the composite figure is saved as PNG/PDF, and plotted
source data are exported as CSV. Running this file directly will also call plt.show().
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "phase5"

MAIN_CANDIDATES_CSV = PROJECT_ROOT / "outputs" / "phase5_cst_case_study_v1" / "phase5_cst_candidates.csv"
MAIN_METRICS_CSV = PROJECT_ROOT / "outputs" / "phase5_cst_case_study_v1" / "phase5_cst_case_metrics.csv"
SI_METRICS_CSV = PROJECT_ROOT / "outputs" / "phase5_cst_case_study_with_si_v1" / "phase5_cst_case_metrics.csv"

SAVE_PNG = True
SAVE_PDF = True
SAVE_DPI = 600
SHOW_FIGURES = True

TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
VALUE_FONTSIZE = 12
LEGEND_FONTSIZE = 14
PANEL_FONTSIZE = 18

# 新增：用于 compact composite 中的统一字号
COMPACT_TITLE_FONTSIZE = 16
COMPACT_LEGEND_FONTSIZE = 14
COMPACT_INFO_FONTSIZE = 12

PALETTE = {
    "blue": "#81AFD4",
    "pale_blue": "#CAE0E8",
    "red": "#DE7C7D",
    "peach": "#FBCFBB",
    "gray": "#CBCBCB",
    "dark_gray": "#4D4D4D",
}

CASE_ORDER_MAIN = [
    "case_001_target_h_q0028_domain_aware_fs100",
    "case_002_target_h_q0271_domain_aware_fs50",
    "case_003_target_self_q0323_raw_spectrum_fs20",
    "case_004_target_self_q0197_raw_spectrum_fs20",
]
CASE_ORDER_ALL = CASE_ORDER_MAIN + ["case_005_target_h_q0287_domain_aware_fs200"]

CASE_LABELS = {
    "case_001_target_h_q0028_domain_aware_fs100": "Case 1",
    "case_002_target_h_q0271_domain_aware_fs50": "Case 2",
    "case_003_target_self_q0323_raw_spectrum_fs20": "Case 3",
    "case_004_target_self_q0197_raw_spectrum_fs20": "Case 4",
    "case_005_target_h_q0287_domain_aware_fs200": "Case 5",
}

DOMAIN_LABELS = {
    "target_h": "Target-H",
    "target_self": "Target-Self",
}

STRUCTURE_LABELS = {
    "h_shape": "H-shape",
    "cylinder": "Cylinder",
}


def setup_basic_matplotlib() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "Arial"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, file_stem: str, *, tight: bool = True) -> None:
    if tight:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout.*")
            fig.tight_layout()
    if SAVE_PNG:
        fig.savefig(OUTPUT_DIR / f"{file_stem}.png", dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.08)
    if SAVE_PDF:
        fig.savefig(OUTPUT_DIR / f"{file_stem}.pdf", dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.08)


def save_panel(fig: plt.Figure, file_stem: str) -> None:
    save_figure(fig, file_stem, tight=True)


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Phase 5 result file: {path}")
    return pd.read_csv(path)


def resolve_path(path_text: str) -> Path:
    path = Path(str(path_text))
    if path.exists():
        return path
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    return path


def style_axis(
    ax: plt.Axes,
    *,
    y_lim: tuple[float, float] | None = None,
    x_grid: bool = False,
    y_grid: bool = True,
) -> None:
    if y_grid:
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
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_FONTSIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )


def sci_label(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value == 0:
        return "0"
    return f"{value:.2e}".replace("e-0", "e-").replace("e+0", "e+")


def mantissa_label(value: float) -> str:
    if not np.isfinite(value) or value <= 0:
        return ""
    exponent = np.floor(np.log10(value))
    mantissa = value / (10**exponent)
    return f"{mantissa:.1f}".rstrip("0").rstrip(".")


def interp_to_reference(x: np.ndarray, y: np.ndarray, ref_x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    return np.interp(ref_x, x[order], y[order])


def load_metrics_table() -> pd.DataFrame:
    main = read_csv_checked(MAIN_METRICS_CSV)
    frames = [main]
    if SI_METRICS_CSV.exists():
        si = read_csv_checked(SI_METRICS_CSV)
        si = si[(si["cst_exists"] == True) & (si["status"] == "ok")].copy()  # noqa: E712
        frames.append(si)

    metrics = pd.concat(frames, ignore_index=True)
    metrics = metrics[(metrics["cst_exists"] == True) & (metrics["status"] == "ok")].copy()  # noqa: E712
    metrics = metrics.drop_duplicates("case_id", keep="last")
    metrics["case_order"] = metrics["case_id"].map({case_id: i for i, case_id in enumerate(CASE_ORDER_ALL)})
    metrics = metrics.sort_values("case_order", kind="stable").reset_index(drop=True)
    return metrics


def load_candidates_table(metrics: pd.DataFrame) -> pd.DataFrame:
    candidates = read_csv_checked(MAIN_CANDIDATES_CSV)
    keep = [
        "case_id",
        "target_domain",
        "shift_type",
        "structure_type",
        "final_method",
        "final_fewshot_label",
        "final_fewshot_cost",
        "selected_rank",
        "selected_bank_domain",
        "selected_success_recomputed",
        "selected_amplitude_mse",
        "selected_amplitude_mae",
        "planner_stop_reason",
        "spectra_csv",
        "case_json",
        "thickness_raw",
        "refractive_index_raw",
        "Lx_raw",
        "Lx1_raw",
        "Ly_raw",
        "Ly1_raw",
        "epsilon_raw",
        "gap_raw",
        "radius_raw",
    ]
    available = [col for col in keep if col in candidates.columns]
    candidates = candidates[available].copy()
    merged = metrics.merge(candidates, on="case_id", how="left", suffixes=("", "_candidate"))
    for col in ["target_domain", "spectra_csv"]:
        alt_col = f"{col}_candidate"
        if alt_col in merged.columns:
            merged[col] = merged[col].fillna(merged[alt_col])
            merged = merged.drop(columns=[alt_col])
    return merged


def load_spectrum_pair(row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    spectra_path = resolve_path(str(row["spectra_csv"]))
    cst_path = resolve_path(str(row["cst_csv"]))
    spectra = read_csv_checked(spectra_path)
    cst = read_csv_checked(cst_path)
    return spectra, cst


def _row_float(row: pd.Series, key: str) -> float | None:
    value = row.get(key, np.nan)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _fmt_dim(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.2f}"


def draw_structure_inset(ax: plt.Axes, row: pd.Series, *, loc: str = "lower left", compact: bool = False) -> None:
    structure_type = str(row["structure_type"])
    if compact:
        width = 0.54    #控制 inset 占主坐标轴的宽度和高度。数值越大，结构示意图越大。
        height = 0.70
        y0 = 0.060
        title_fs = 12     # Top view
        period_fs = 12     # p=...
        dim_fs = 12      # 右侧参数文字
    else:
        width = 0.38
        height = 0.42
        y0 = 0.095
        title_fs = 8.8
        period_fs = 8.0
        dim_fs = 8.1
    if loc == "lower left":
        bounds = [0.045, y0, width, height]
    elif loc == "lower right":
        bounds = [1.0 - width - 0.045, y0, width, height]
    else:
        bounds = [0.045, 0.53, width, height]

    inset = ax.inset_axes(bounds)
    inset.set_xlim(-0.24, 2.35)
    inset.set_ylim(-0.20, 1.18)
    inset.set_aspect("equal")
    inset.axis("off")

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
    gold = "#C9A34E"
    if structure_type == "h_shape":
        thickness = _row_float(row, "thickness_raw")
        refractive_index = _row_float(row, "refractive_index_raw")
        lx = _row_float(row, "Lx_raw")
        lx1 = _row_float(row, "Lx1_raw")
        ly = _row_float(row, "Ly_raw")
        ly1 = _row_float(row, "Ly1_raw")
        period = max(3.0, (lx or 0.0) + 0.4, (ly or 0.0) + 0.4)
        inset.add_patch(Rectangle((0.19, 0.18), 0.18, 0.64, facecolor=gold, edgecolor="#7D6228", linewidth=0.7))
        inset.add_patch(Rectangle((0.63, 0.18), 0.18, 0.64, facecolor=gold, edgecolor="#7D6228", linewidth=0.7))
        inset.add_patch(Rectangle((0.34, 0.43), 0.34, 0.16, facecolor=gold, edgecolor="#7D6228", linewidth=0.7))
        dim_text = (
            f"p={period:.2f} um\n"
            f"t={_fmt_dim(thickness)} um\n"
            f"n={_fmt_dim(refractive_index)}\n"
            f"Lx={_fmt_dim(lx)}, Lx1={_fmt_dim(lx1)}\n"
            f"Ly={_fmt_dim(ly)}, Ly1={_fmt_dim(ly1)}"
        )
    else:
        epsilon = _row_float(row, "epsilon_raw")
        gap = _row_float(row, "gap_raw")
        thickness = _row_float(row, "thickness_raw")
        radius = _row_float(row, "radius_raw")
        period = max(2.0 * (radius or 0.0) + (gap or 0.0), 0.2)
        inset.add_patch(Circle((0.5, 0.5), 0.27, facecolor=gold, edgecolor="#7D6228", linewidth=0.7))
        dim_text = (
            f"p={period:.2f} um\n"
            f"r={_fmt_dim(radius)} um\n"
            f"g={_fmt_dim(gap)} um\n"
            f"t={_fmt_dim(thickness)} um\n"
            f"eps={_fmt_dim(epsilon)}"
        )

    inset.text(0.5, 1.08, "Top view", ha="center", va="bottom", fontsize=title_fs, color="#333333")
    inset.annotate("", xy=(-0.10, 1.0), xytext=(-0.10, 0.0), arrowprops=dict(arrowstyle="<|-|>", lw=0.55, color="#333333"))
    inset.annotate("", xy=(1.0, -0.10), xytext=(0.0, -0.10), arrowprops=dict(arrowstyle="<|-|>", lw=0.55, color="#333333"))
    inset.text(-0.17, 0.5, f"p={period:.2f}", ha="center", va="center", fontsize=period_fs, rotation=90)
    inset.text(0.5, -0.16, f"p={period:.2f}", ha="center", va="center", fontsize=period_fs)
    inset.text(1.10, 0.52, dim_text, ha="left", va="center", fontsize=dim_fs, color="#333333", linespacing=1.10)

def draw_workflow_overview(ax: plt.Axes, cases: pd.DataFrame, *, panel_label: str | None = "a") -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if panel_label:
        add_panel_label(ax, panel_label, x=-0.05, y=1.05)

    # 修改 1：标题 pad 从 8 降到 2，减少标题和流程图之间的空隙
    ax.set_title(
        "Closed-loop CST validation",
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
        pad=2,
    )

    # 修改 2：流程图整体上移，靠近标题；高度略增
    box_y = 0.735
    box_h = 0.145
    boxes = [
        (0.05, box_y, 0.24, box_h, "LLM-selected\ncandidate", PALETTE["pale_blue"]),
        (0.38, box_y, 0.24, box_h, "CST unit-cell\nsolver", PALETTE["peach"]),
        (0.71, box_y, 0.24, box_h, "Spectrum\nMSE audit", PALETTE["blue"]),
    ]

    for x, y, w, h, label, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=color,
            edgecolor=PALETTE["dark_gray"],
            linewidth=1.0,
            alpha=0.92,
        )
        ax.add_patch(patch)

        # 修改 3：流程图字体从 11.5 增大到 13.2
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=13.2,
        )

    arrow_y = box_y + box_h / 2
    for start_x, end_x in [(0.29, 0.38), (0.62, 0.71)]:
        ax.add_patch(
            FancyArrowPatch(
                (start_x, arrow_y),
                (end_x, arrow_y),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.2,
                color=PALETTE["dark_gray"],
            )
        )

    table_rows = cases[cases["case_id"].isin(CASE_ORDER_ALL)].copy()
    table_rows["display"] = table_rows["case_id"].map(CASE_LABELS)
    table_rows = table_rows.sort_values("case_order", kind="stable")

    # 修改 4：表格区域整体下移一些，给流程图和表格留出空隙
    header_y = 0.615
    first_row_y = 0.545
    row_gap = 0.102
    row_height = 0.082

    header_fs = 14.0
    row_fs = 14.0

    # 表头位置与第一行分开，避免重叠
    ax.text(0.05, header_y, "Case", fontsize=header_fs, color=PALETTE["dark_gray"], ha="left", va="center")
    ax.text(0.20, header_y, "Domain", fontsize=header_fs, color=PALETTE["dark_gray"], ha="left", va="center")
    ax.text(0.41, header_y, "Plan", fontsize=header_fs, color=PALETTE["dark_gray"], ha="left", va="center")
    ax.text(0.64, header_y, "n", fontsize=header_fs, color=PALETTE["dark_gray"], ha="left", va="center")
    ax.text(0.78, header_y, "CST MSE", fontsize=header_fs, color=PALETTE["dark_gray"], ha="left", va="center")

    for i, (_, row) in enumerate(table_rows.iterrows()):
        y = first_row_y - i * row_gap
        fill = PALETTE["pale_blue"] if i % 2 == 0 else "#F7F7F7"

        rect = FancyBboxPatch(
            (0.035, y - row_height / 2),
            0.93,
            row_height,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            facecolor=fill,
            edgecolor="none",
            alpha=0.70,
        )
        ax.add_patch(rect)

        domain = DOMAIN_LABELS.get(row["target_domain"], row["target_domain"])
        method_raw = str(row.get("final_method", ""))
        method = {"domain_aware": "DA", "raw_spectrum": "Raw"}.get(method_raw, method_raw.replace("_", "-"))
        fewshot = str(row.get("final_fewshot_label", row.get("final_fewshot_cost", "")))

        ax.text(0.05, y, str(row["display"]).replace("Case ", "C"), fontsize=row_fs, ha="left", va="center")
        ax.text(0.20, y, domain, fontsize=row_fs, ha="left", va="center")
        ax.text(0.41, y, method, fontsize=row_fs, ha="left", va="center")
        ax.text(0.64, y, fewshot, fontsize=row_fs, ha="left", va="center")
        ax.text(0.78, y, sci_label(float(row["cst_vs_target_amp_mse"])), fontsize=row_fs, ha="left", va="center")

    # 修改 5：根据最后一行位置自动放置注释，避免和 C5 重叠
    # 固定放到底部，避免被最后一行挤压
    note_y = 0.020

    ax.text(
        0.05,
        note_y,
        "Case 5 is retained as an extended Target-H CST check.",
        fontsize=11.2,
        color=PALETTE["dark_gray"],
        ha="left",
        va="bottom",
    )
def draw_spectrum_overlay(
    ax: plt.Axes,
    row: pd.Series,
    *,
    panel_label: str | None = None,
    show_legend: bool = False,
    compact: bool = False,
    legend_loc: str | None = None,
) -> None:
    spectra, cst = load_spectrum_pair(row)
    freq = spectra["frequency_thz"].to_numpy(dtype=float)
    target = spectra["target_amplitude"].to_numpy(dtype=float)
    candidate = spectra["candidate_amplitude"].to_numpy(dtype=float)
    cst_freq = cst["frequency_thz"].to_numpy(dtype=float)
    cst_amp = cst["cst_amplitude"].to_numpy(dtype=float)

    marker_step = max(1, len(cst_freq) // 28)

    ax.plot(freq, target, color=PALETTE["dark_gray"], linewidth=2.2, label="Target", zorder=3)
    ax.plot(freq, candidate, color=PALETTE["blue"], linewidth=2.2, linestyle="--", label="Offline", zorder=4)
    ax.plot(
        cst_freq,
        cst_amp,
        color=PALETTE["red"],
        linewidth=1.9,
        marker="o",
        markersize=3.2,
        markevery=marker_step,
        label="CST",
        zorder=5,
    )

    case_label = CASE_LABELS.get(row["case_id"], row["case_id"])
    domain = DOMAIN_LABELS.get(row["target_domain"], row["target_domain"])
    structure = STRUCTURE_LABELS.get(row["structure_type"], row["structure_type"])
    method_raw = str(row.get("final_method", ""))
    method = {"domain_aware": "DA", "raw_spectrum": "Raw"}.get(method_raw, method_raw.replace("_", "-"))
    fewshot = str(row.get("final_fewshot_label", ""))

    # 修改：compact 模式也使用统一标题字号
    title_size = COMPACT_TITLE_FONTSIZE if compact else TITLE_FONTSIZE

    ax.set_title(
        f"{case_label} | {domain} {structure} | {method}-{fewshot}",
        fontsize=title_size,
        fontweight="normal",
        pad=8,
    )
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Amplitude")

    info_text = (
        f"Offline MSE {sci_label(float(row['candidate_vs_target_amp_mse']))}\n"
        f"CST MSE {sci_label(float(row['cst_vs_target_amp_mse']))}"
    )

    inset_loc = "lower left"
    text_x = 0.48 if compact else 0.45

    draw_structure_inset(ax, row, loc=inset_loc, compact=compact)

    ax.text(
        text_x,
        0.065,
        info_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=COMPACT_INFO_FONTSIZE if compact else VALUE_FONTSIZE,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#D0D0D0",
            "alpha": 0.86,
        },
    )

    style_axis(ax, x_grid=False, y_grid=True, y_lim=(-0.04, 1.08))

    if panel_label:
        add_panel_label(ax, panel_label)

    if show_legend:
        if legend_loc is None:
            legend_loc = "center left" if row["target_domain"] == "target_h" else "lower right"

        ax.legend(
            frameon=False,
            loc=legend_loc,
            fontsize=COMPACT_LEGEND_FONTSIZE if compact else LEGEND_FONTSIZE,
        )
def draw_mse_audit(
    ax: plt.Axes,
    cases: pd.DataFrame,
    *,
    panel_label: str | None = "f",
    show_value_labels: bool = True,
    compact: bool = False,
) -> None:
    rows = cases[cases["case_id"].isin(CASE_ORDER_ALL)].copy()
    rows = rows.sort_values("case_order", kind="stable")
    x = np.arange(len(rows), dtype=float)
    width = 0.25

    bars1 = ax.bar(
        x - width,
        rows["candidate_vs_target_amp_mse"].astype(float),
        width=width,
        color=PALETTE["blue"],
        edgecolor="white",
        linewidth=0.8,
        label="Offline vs target",
        zorder=3,
    )
    bars2 = ax.bar(
        x,
        rows["cst_vs_target_amp_mse"].astype(float),
        width=width,
        color=PALETTE["red"],
        edgecolor="white",
        linewidth=0.8,
        label="CST vs target",
        zorder=3,
    )
    bars3 = ax.bar(
        x + width,
        rows["cst_vs_candidate_amp_mse"].astype(float),
        width=width,
        color=PALETTE["peach"],
        edgecolor="white",
        linewidth=0.8,
        label="CST vs offline",
        zorder=3,
    )

    all_values = np.concatenate(
        [
            rows["candidate_vs_target_amp_mse"].astype(float).to_numpy(),
            rows["cst_vs_target_amp_mse"].astype(float).to_numpy(),
            rows["cst_vs_candidate_amp_mse"].astype(float).to_numpy(),
        ]
    )
    positive_values = all_values[np.isfinite(all_values) & (all_values > 0)]

    if len(positive_values) > 0:
        ax.set_ylim(float(positive_values.min()) / 4.0, float(positive_values.max()) * 4.5)

    ax.set_yscale("log")

    # 修改：f 图标题字号与 b/c/d/e 统一
    title_size = COMPACT_TITLE_FONTSIZE if compact else TITLE_FONTSIZE
    ax.set_title("CST error audit", fontsize=title_size, fontweight="normal", pad=8)

    ax.set_ylabel("Amplitude MSE")
    ax.set_xlabel("Validation case", labelpad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CASE_LABELS.get(case_id, case_id) for case_id in rows["case_id"]],
        rotation=18,
        ha="right",
    )

    style_axis(ax, y_grid=True)

    # 修改：图例放在 f 图下方，不再和标题挤在一起
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        fontsize=COMPACT_LEGEND_FONTSIZE if compact else LEGEND_FONTSIZE,
        ncol=3,
        borderaxespad=0.0,
        handlelength=1.8,
        columnspacing=1.8,
    )

    if show_value_labels:
        for bars, values in [
            (bars1, rows["candidate_vs_target_amp_mse"].astype(float)),
            (bars2, rows["cst_vs_target_amp_mse"].astype(float)),
            (bars3, rows["cst_vs_candidate_amp_mse"].astype(float)),
        ]:
            for bar, value in zip(bars, values):
                if value <= 0 or not np.isfinite(value):
                    continue

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value * 1.35,
                    mantissa_label(float(value)),
                    ha="center",
                    va="bottom",
                    fontsize=11.6 if compact else 13.0,
                    rotation=0,
                    color="#222222",
                )

    if panel_label:
        add_panel_label(ax, panel_label, x=-0.08, y=1.08)

def write_source_data(cases: pd.DataFrame) -> None:
    columns = [
        "case_id",
        "target_domain",
        "structure_type",
        "final_method",
        "final_fewshot_label",
        "selected_bank_domain",
        "candidate_vs_target_amp_mse",
        "cst_vs_target_amp_mse",
        "cst_vs_candidate_amp_mse",
        "spectra_csv",
        "cst_csv",
    ]
    available = [col for col in columns if col in cases.columns]
    cases[available].to_csv(OUTPUT_DIR / "phase5_plot_source_metrics.csv", index=False)


def write_readme() -> None:
    readme = """# Phase 5 CST unit-cell validation figures

This folder contains the Phase 5 main-text composite and individual panels.

Source files:
- Main candidate table: `outputs/phase5_cst_case_study_v1/phase5_cst_candidates.csv`
- Main CST metrics: `outputs/phase5_cst_case_study_v1/phase5_cst_case_metrics.csv`
- SI CST metrics: `outputs/phase5_cst_case_study_with_si_v1/phase5_cst_case_metrics.csv`
- Offline spectra: `outputs/phase5_cst_case_study_v1/spectra/*_spectra.csv`
- CST spectra: `outputs/phase5_cst_case_study_v1/cst_results/*_cst_spectrum.csv`
- Extended SI CST spectrum: `outputs/phase5_cst_case_study_with_si_v1/cst_results/*_cst_spectrum.csv`

Main conclusion:
The selected unit-cell candidates are not only successful in the offline replay pool; they can be exported to CST, simulated, and audited against target spectra. The main figure includes the CST-vs-target and CST-vs-offline MSE audit because this quantifies the remaining simulation-domain discrepancy.
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")
def draw_composite(cases: pd.DataFrame) -> plt.Figure:
    main_rows = cases[cases["case_id"].isin(CASE_ORDER_MAIN)].copy()
    main_rows = main_rows.sort_values("case_order", kind="stable").reset_index(drop=True)

    fig = plt.figure(figsize=(18.5, 8.4))

    gs = GridSpec(
        2,
        4,
        figure=fig,
        wspace=0.30,
        hspace=0.52,
        height_ratios=[1.0, 1.0],
    )

    # b: workflow overview, top-left two columns
    ax_a = fig.add_subplot(gs[0, 0:2])
    draw_workflow_overview(ax_a, cases, panel_label="b")

    # c-d: first two spectrum cases, top-right
    labels_top = ["c", "d"]
    positions_top = [(0, 2), (0, 3)]
    for label, pos, (_, row) in zip(labels_top, positions_top, main_rows.iloc[:2].iterrows()):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        draw_spectrum_overlay(
            ax,
            row,
            panel_label=label,
            show_legend=False,
            compact=True,
        )

    # e-f: remaining two spectrum cases, bottom-left
    labels_bottom = ["e", "f"]
    positions_bottom = [(1, 0), (1, 1)]
    for label, pos, (_, row) in zip(labels_bottom, positions_bottom, main_rows.iloc[2:].iterrows()):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        draw_spectrum_overlay(
            ax,
            row,
            panel_label=label,
            show_legend=(label == "f"),
            compact=True,
            legend_loc="center right",
        )

    # g: MSE audit, bottom-right area.
    gs_f = gs[1, 2:4].subgridspec(
        2,
        20,
        height_ratios=[0.78, 0.14],
        hspace=0.0,
        wspace=0.0,
    )

    ax_f = fig.add_subplot(gs_f[0, 2:20])
    draw_mse_audit(
        ax_f,
        cases,
        panel_label="g",
        show_value_labels=True,
        compact=True,
    )

    fig.subplots_adjust(
        left=0.043,
        right=0.992,
        bottom=0.145,
        top=0.94,
    )

    save_figure(fig, "phase5_main_composite", tight=False)
    return fig

def draw_individual_panels(cases: pd.DataFrame) -> list[plt.Figure]:
    figures: list[plt.Figure] = []

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    draw_workflow_overview(ax, cases, panel_label=None)
    save_panel(fig, "panel_a_closed_loop_cst_workflow")
    figures.append(fig)

    main_rows = cases[cases["case_id"].isin(CASE_ORDER_MAIN)].copy()
    main_rows = main_rows.sort_values("case_order", kind="stable")
    for idx, (_, row) in enumerate(main_rows.iterrows(), start=1):
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        legend_loc = "center left" if row["target_domain"] == "target_h" else "lower right"
        draw_spectrum_overlay(ax, row, panel_label=None, show_legend=True, legend_loc=legend_loc)
        save_panel(fig, f"panel_{chr(96 + idx + 1)}_{row['case_id']}_spectrum_overlay")
        figures.append(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    draw_mse_audit(ax, cases, panel_label=None)
    save_panel(fig, "panel_f_cst_error_audit")
    figures.append(fig)

    si_rows = cases[~cases["case_id"].isin(CASE_ORDER_MAIN)].copy()
    if not si_rows.empty:
        for _, row in si_rows.iterrows():
            fig, ax = plt.subplots(figsize=(7.0, 4.5))

            draw_spectrum_overlay(
                ax,
                row,
                panel_label=None,
                show_legend=True,
                legend_loc="lower right",  # 图例/图注放到右下角
                compact=True,  # 使用放大版结构示意图和尺寸标注
            )

            save_panel(fig, f"panel_g_{row['case_id']}_si_spectrum_overlay")
            figures.append(fig)

    return figures


def main() -> None:
    setup_basic_matplotlib()
    ensure_output_dir()

    metrics = load_metrics_table()
    cases = load_candidates_table(metrics)
    write_source_data(cases)
    write_readme()

    figures = [draw_composite(cases)]
    figures.extend(draw_individual_panels(cases))

    print(f"Saved Phase 5 figures: {OUTPUT_DIR.resolve()}")
    print(cases[["case_id", "target_domain", "structure_type", "candidate_vs_target_amp_mse", "cst_vs_target_amp_mse"]].to_string(index=False))

    if SHOW_FIGURES:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main()
