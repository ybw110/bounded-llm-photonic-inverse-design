"""
Phase 4 main figure panels: few-shot OOD recovery and LLM-guided evidence allocation.

This script follows the standalone plotting style of plot_phase01_retrieval_evidence.py,
plot_phase2.py, and plot_phase3.py. It creates six main-text-ready Phase 4 panels:

a. Few-shot Success@50 recovery curves.
b. Few-shot target-domain Success@50 recovery curves.
c. Top-1 target-domain validity curves.
d. 500-query OOD planner policy comparison.
e. Sample-efficiency Pareto plot for Target-H.
f. Multi-model 500-query repeat robustness.

Each panel is saved as PNG/PDF and its plotted source data are exported as CSV.
The figures use Arial, large labels, and the same Morandi palette used in Phase 3.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "phase4"

SAVE_PNG = True
SAVE_PDF = True
SAVE_DPI = 600
SHOW_FIGURES = True

TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
VALUE_FONTSIZE = 13
LEGEND_FONTSIZE = 16

PALETTE = {
    "blue": "#81AFD4",
    "pale_blue": "#CAE0E8",
    "red": "#DE7C7D",
    "peach": "#FBCFBB",
    "gray": "#CBCBCB",
    "dark_gray": "#4D4D4D",
}

METHOD_ORDER = ["raw_spectrum", "physics_summary", "domain_aware"]
METHOD_LABELS = {
    "raw_spectrum": "Raw spectrum",
    "physics_summary": "Physics summary",
    "domain_aware": "Domain-aware",
}
METHOD_COLORS = {
    "raw_spectrum": PALETTE["blue"],
    "physics_summary": PALETTE["red"],
    "domain_aware": PALETTE["peach"],
}

TARGET_ORDER = ["target_h", "target_self"]
TARGET_LABELS = {
    "target_h": "Target-H",
    "target_self": "Target-Self",
}

POLICY_RUNS = [
    ("Source only", PROJECT_ROOT / "outputs" / "phase4b_ood_planner_source_only_500_v1"),
    ("Fixed rule", PROJECT_ROOT / "outputs" / "phase4b_ood_planner_shift_aware_500_v1"),
    ("DeepSeek", PROJECT_ROOT / "outputs" / "phase4b_ood_planner_deepseek_500_v1"),
    ("Oracle", PROJECT_ROOT / "outputs" / "phase4b_ood_planner_oracle_500_v1"),
]
POLICY_COLORS = {
    "Source only": PALETTE["gray"],
    "Fixed rule": PALETTE["blue"],
    "DeepSeek": PALETTE["peach"],
    "Oracle": PALETTE["red"],
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


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Phase 4 result file: {path}")
    return pd.read_csv(path)


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


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="left",
    )


def add_bar_labels(
    ax: plt.Axes,
    bars,
    *,
    fmt: str = "{:.2f}",
    offset: float = 0.015,
    fontsize: int = VALUE_FONTSIZE,
) -> None:
    for bar in bars:
        height = float(bar.get_height())
        if not np.isfinite(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#222222",
        )


def fewshot_sort_value(value: object) -> float:
    text = str(value).strip().lower()
    if text == "full":
        return 100000.0
    try:
        return float(text)
    except ValueError:
        return 99999.0


def add_fewshot_positions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    labels = (
        df["fewshot_label"]
        .astype(str)
        .drop_duplicates()
        .sort_values(key=lambda s: s.map(fewshot_sort_value), kind="stable")
        .tolist()
    )
    positions = {label: float(i) for i, label in enumerate(labels)}
    df["fewshot_plot_label"] = df["fewshot_label"].astype(str).str.replace("full", "Full", case=False, regex=False)
    df["fewshot_x"] = df["fewshot_label"].astype(str).map(positions).astype(float)
    return df


def xticks_for(df: pd.DataFrame) -> tuple[list[float], list[str]]:
    pairs = (
        df[["fewshot_x", "fewshot_plot_label"]]
        .drop_duplicates()
        .sort_values("fewshot_x", kind="stable")
        .itertuples(index=False, name=None)
    )
    xs, labels = zip(*pairs)
    return list(xs), list(labels)


def load_fewshot_aggregate() -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / "phase4_fewshot_retrieval_methods_full_with_upper_v1" / "fewshot_retrieval_summary_aggregate.csv"
    df = read_csv_checked(path)
    return add_fewshot_positions(df)


def load_planner_policy_summary() -> pd.DataFrame:
    frames = []
    for order, (label, run_dir) in enumerate(POLICY_RUNS):
        path = run_dir / "ood_recovery_planner_summary.csv"
        df = read_csv_checked(path)
        df["run_label"] = label
        df["run_order"] = order
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_multimodel_summary() -> pd.DataFrame:
    path = (
        PROJECT_ROOT
        / "outputs"
        / "phase4b_ood_planner_multimodel_500_repeats_summary_v1"
        / "phase4b_multimodel_repeat_summary.csv"
    )
    return read_csv_checked(path)


def load_target_h_extended() -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / "phase4_fewshot_retrieval_target_h_extended_v1" / "fewshot_retrieval_summary_aggregate.csv"
    return add_fewshot_positions(read_csv_checked(path))


def load_deepseek_action_summary() -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / "phase4b_ood_planner_deepseek_500_v1" / "ood_recovery_planner_action_summary.csv"
    return read_csv_checked(path)


def load_rich_planner_replay_table() -> pd.DataFrame:
    policy_df = load_planner_policy_summary()
    multi_df = load_multimodel_summary()
    rows = []

    for run_label in ["Source only", "Fixed rule"]:
        row = policy_df[(policy_df["run_label"] == run_label) & (policy_df["target_domain"] == "target_h")].iloc[0]
        rows.append(
            {
                "run_label": "Fixed H-200" if run_label == "Fixed rule" else run_label,
                "success_at_50": float(row["success_at_50"]),
                "target_domain_success_at_50": float(row["target_domain_success_at_50"]),
                "mean_fewshot_cost": float(row["mean_fewshot_cost"]),
            }
        )

    for run_label in ["DeepSeek", "GLM-5.1"]:
        row = multi_df[(multi_df["run_label"] == run_label) & (multi_df["target_domain"] == "target_h")].iloc[0]
        rows.append(
            {
                "run_label": run_label,
                "success_at_50": float(row["success_at_50_mean"]),
                "target_domain_success_at_50": float(row["target_domain_success_at_50_mean"]),
                "mean_fewshot_cost": float(row["mean_fewshot_cost_mean"]),
            }
        )

    row = policy_df[(policy_df["run_label"] == "Oracle") & (policy_df["target_domain"] == "target_h")].iloc[0]
    rows.append(
        {
            "run_label": "Oracle",
            "success_at_50": float(row["success_at_50"]),
            "target_domain_success_at_50": float(row["target_domain_success_at_50"]),
            "mean_fewshot_cost": float(row["mean_fewshot_cost"]),
        }
    )

    return pd.DataFrame(rows)


def draw_line_with_std(
    ax: plt.Axes,
    sub: pd.DataFrame,
    *,
    y_col: str,
    std_col: str,
    label: str,
    color: str,
    marker: str = "o",
    linestyle: str = "-",
) -> None:
    sub = sub.sort_values("fewshot_x", kind="stable")
    x = sub["fewshot_x"].to_numpy(dtype=float)
    y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)
    ax.plot(x, y, marker=marker, markersize=5.5, linewidth=2.2, label=label, color=color, linestyle=linestyle, zorder=3)
    if std_col in sub.columns:
        std = pd.to_numeric(sub[std_col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(std).any() and np.nanmax(std) > 0:
            ax.fill_between(x, y - std, y + std, color=color, alpha=0.16, linewidth=0, zorder=2)


def draw_recovery_curve(
    metric: str,
    std_metric: str,
    ylabel: str,
    title: str,
    output_stem: str,
    *,
    y_lim: tuple[float, float] = (-0.03, 1.04),
) -> pd.DataFrame:
    df = load_fewshot_aggregate()
    plot_methods = ["raw_spectrum", "domain_aware"]
    plot_df = df[df["method"].isin(plot_methods)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
    for ax, target, panel in zip(axes, TARGET_ORDER, ["a", "b"]):
        sub_target = plot_df[plot_df["target_domain"] == target]
        for method in plot_methods:
            sub = sub_target[sub_target["method"] == method]
            draw_line_with_std(
                ax,
                sub,
                y_col=metric,
                std_col=std_metric,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                marker="o" if method == "raw_spectrum" else "s",
            )
        xs, labels = xticks_for(sub_target)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_xlabel("Few-shot target samples")
        ax.set_title(TARGET_LABELS[target], fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10)
        style_axis(ax, y_lim=y_lim)
        add_panel_label(ax, panel)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="best")
    fig.suptitle(title, fontsize=TITLE_FONTSIZE + 1, fontweight="normal", y=1.02)
    save_figure(fig, output_stem)

    source = plot_df[
        ["target_domain", "method", "fewshot_label", "fewshot_x", metric, std_metric]
    ].copy()
    source.to_csv(OUTPUT_DIR / f"{output_stem}_source_data.csv", index=False)
    return source


def draw_panel_a_success_at50() -> None:
    draw_recovery_curve(
        "success_at_50_mean",
        "success_at_50_std",
        "Success@50",
        "Few-shot OOD recovery",
        "panel_a_success_at50_fewshot",
    )


def draw_panel_b_target_domain_success() -> None:
    draw_recovery_curve(
        "target_domain_success_at_50_mean",
        "target_domain_success_at_50_std",
        "Target-domain Success@50",
        "Target-domain valid recovery",
        "panel_b_target_domain_success_at50",
    )


def draw_panel_c_top1_target_domain_rate() -> None:
    draw_recovery_curve(
        "top1_target_domain_rate_mean",
        "top1_target_domain_rate_std",
        "Top-1 target-domain rate",
        "Target-domain validity of retrieved candidates",
        "panel_c_top1_target_domain_rate",
    )


def draw_panel_d_planner_policy_comparison() -> None:
    df = load_planner_policy_summary()
    labels = [label for label, _ in POLICY_RUNS]
    x = np.arange(len(labels))
    colors = [POLICY_COLORS[label] for label in labels]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    metric_specs = [
        ("target_h", "target_domain_success_at_50", "Target-H\nvalid Success@50", "Rate", (0.0, 1.04), "{:.2f}"),
        ("target_h", "mean_fewshot_cost", "Target-H\nsample cost", "Target samples", (0.0, 230.0), "{:.0f}"),
        ("target_self", "target_domain_success_at_50", "Target-Self\nvalid Success@50", "Rate", (0.0, 1.04), "{:.2f}"),
    ]
    for ax, panel, (target, metric, title, ylabel, ylim, fmt) in zip(axes, ["a", "b", "c"], metric_specs):
        sub = df[df["target_domain"] == target].set_index("run_label").reindex(labels).reset_index()
        values = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8, width=0.68, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10)
        style_axis(ax, y_lim=ylim)
        add_bar_labels(ax, bars, fmt=fmt, offset=(ylim[1] - ylim[0]) * 0.015)
        add_panel_label(ax, panel)
    save_figure(fig, "panel_d_ood_planner_policy_comparison")

    source = df[
        [
            "run_label",
            "target_domain",
            "success_at_50",
            "target_domain_success_at_50",
            "top1_target_domain_rate",
            "mean_fewshot_cost",
            "oracle_match_rate",
            "oracle_gap_target_domain_success_at_50",
        ]
    ].copy()
    source.to_csv(OUTPUT_DIR / "panel_d_ood_planner_policy_comparison_source_data.csv", index=False)


def draw_panel_e_efficiency_pareto() -> None:
    df = load_planner_policy_summary()
    sub = df[df["target_domain"] == "target_h"].set_index("run_label").reindex([label for label, _ in POLICY_RUNS]).reset_index()
    x = pd.to_numeric(sub["mean_fewshot_cost"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(sub["target_domain_success_at_50"], errors="coerce").to_numpy(dtype=float)
    labels = sub["run_label"].tolist()

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for xi, yi, label in zip(x, y, labels):
        ax.scatter(
            xi,
            yi,
            s=135,
            color=POLICY_COLORS[label],
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
            label=label,
        )
        ax.annotate(label, (xi, yi), xytext=(8, 6), textcoords="offset points", fontsize=VALUE_FONTSIZE, color="#222222")

    ax.set_xlabel("Mean target samples used")
    ax.set_ylabel("Target-H target-domain Success@50")
    ax.set_title("Sample-efficient OOD recovery", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10)
    ax.set_xlim(-10, 220)
    style_axis(ax, y_lim=(-0.03, 1.04), x_grid=True)
    save_figure(fig, "panel_e_target_h_sample_efficiency")

    sub[
        [
            "run_label",
            "target_domain",
            "mean_fewshot_cost",
            "target_domain_success_at_50",
            "oracle_match_rate",
            "oracle_gap_target_domain_success_at_50",
        ]
    ].to_csv(OUTPUT_DIR / "panel_e_target_h_sample_efficiency_source_data.csv", index=False)


def draw_panel_f_multimodel_repeat_robustness() -> None:
    df = load_multimodel_summary()
    target_h = df[df["target_domain"] == "target_h"].copy()
    target_h["run_label"] = pd.Categorical(target_h["run_label"], categories=["DeepSeek", "GLM-5.1"], ordered=True)
    target_h = target_h.sort_values("run_label")

    metrics = [
        ("target_domain_success_at_50", "Valid Success@50", (0.0, 1.04), "{:.2f}"),
        ("mean_fewshot_cost", "Sample cost", (0.0, 135.0), "{:.0f}"),
        ("oracle_match_rate", "Oracle match", (0.0, 1.04), "{:.2f}"),
    ]
    model_colors = {"DeepSeek": PALETTE["peach"], "GLM-5.1": PALETTE["blue"]}

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5))
    for ax, panel, (metric, title, ylim, fmt) in zip(axes, ["a", "b", "c"], metrics):
        labels = target_h["run_label"].astype(str).tolist()
        x = np.arange(len(labels))
        y = pd.to_numeric(target_h[f"{metric}_mean"], errors="coerce").to_numpy(dtype=float)
        yerr = pd.to_numeric(target_h[f"{metric}_std"], errors="coerce").fillna(0).to_numpy(dtype=float)
        bars = ax.bar(
            x,
            y,
            yerr=yerr,
            capsize=5,
            color=[model_colors[label] for label in labels],
            edgecolor="white",
            linewidth=0.8,
            width=0.62,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"Target-H {title}", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10)
        style_axis(ax, y_lim=ylim)
        add_bar_labels(ax, bars, fmt=fmt, offset=(ylim[1] - ylim[0]) * 0.018)
        add_panel_label(ax, panel)
    save_figure(fig, "panel_f_multimodel_repeat_robustness")

    target_h.to_csv(OUTPUT_DIR / "panel_f_multimodel_repeat_robustness_source_data.csv", index=False)


def draw_evidence_allocation_schematic(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_title("Few-shot OOD evidence allocation", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10, loc="left")
    add_panel_label(ax, "a")

    def box(x: float, y: float, w: float, h: float, text: str, *, fc: str = "#FFFFFF") -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            linewidth=1.2,
            edgecolor="#111111",
            facecolor=fc,
            transform=ax.transAxes,
            zorder=2,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12, transform=ax.transAxes)

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.1,
            color="#111111",
            transform=ax.transAxes,
            zorder=3,
        )
        ax.add_patch(patch)

    box(0.04, 0.72, 0.28, 0.13, "OOD query\nTarget-H / Target-Self", fc="#FFFFFF")
    box(0.04, 0.49, 0.28, 0.13, "Source-only\nmemory", fc="#FFFFFF")
    box(0.04, 0.26, 0.28, 0.13, "Few-shot target\nmemory: 20-200", fc="#FFFFFF")
    box(0.40, 0.53, 0.34, 0.17, "Structured evidence\nspectral match\n+ domain validity + cost", fc=PALETTE["pale_blue"])
    box(0.40, 0.24, 0.34, 0.14, "LLM / rule / oracle\nplanner", fc=PALETTE["peach"])
    box(0.80, 0.48, 0.18, 0.18, "Top-50\ncandidates\n+ recovery", fc="#FFFFFF")

    arrow((0.32, 0.785), (0.40, 0.63))
    arrow((0.32, 0.555), (0.40, 0.61))
    arrow((0.32, 0.325), (0.40, 0.57))
    arrow((0.57, 0.53), (0.57, 0.38))
    arrow((0.74, 0.31), (0.80, 0.53))
    arrow((0.74, 0.61), (0.80, 0.57))

    ax.text(
        0.06,
        0.11,
        "Key distinction: spectral success is not sufficient\nwhen topology/domain validity is required.",
        transform=ax.transAxes,
        fontsize=11,
        ha="left",
        va="center",
    )


def draw_combined_recovery_lines(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    target: str,
    title: str,
    show_overall: bool,
    panel: str,
) -> pd.DataFrame:
    sub_target = df[(df["target_domain"] == target) & (df["method"].isin(["raw_spectrum", "domain_aware"]))].copy()

    line_specs = []
    if show_overall:
        line_specs.extend(
            [
                ("raw_spectrum", "success_at_50_mean", "success_at_50_std", "Raw S@50", PALETTE["blue"], "o", "-"),
                ("domain_aware", "success_at_50_mean", "success_at_50_std", "Domain-aware S@50", PALETTE["peach"], "s", "-"),
            ]
        )
    line_specs.extend(
        [
            ("raw_spectrum", "target_domain_success_at_50_mean", "target_domain_success_at_50_std", "Raw target S@50", PALETTE["blue"], "o", "--"),
            (
                "domain_aware",
                "target_domain_success_at_50_mean",
                "target_domain_success_at_50_std",
                "Domain-aware target S@50",
                PALETTE["red"],
                "s",
                "--",
            ),
        ]
    )

    for method, metric, std, label, color, marker, linestyle in line_specs:
        draw_line_with_std(
            ax,
            sub_target[sub_target["method"] == method],
            y_col=metric,
            std_col=std,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
        )

    xs, labels = xticks_for(sub_target)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_xlabel("Target samples added")
    ax.set_ylabel("Success@50" if show_overall else "Target-domain Success@50")
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10, loc="left")
    style_axis(ax, y_lim=(-0.03, 1.04), x_grid=True)
    ax.legend(frameon=False, fontsize=11, loc="best")
    add_panel_label(ax, panel)
    return sub_target


def draw_extended_target_h_coverage(ax: plt.Axes) -> pd.DataFrame:
    df = load_target_h_extended()
    df = df[(df["target_domain"] == "target_h") & (df["method"].isin(["raw_spectrum", "domain_aware"]))].copy()
    keep_labels = ["0", "200", "500", "1000", "2000", "full"]
    df = df[df["fewshot_label"].astype(str).str.lower().isin(keep_labels)].copy()
    df = add_fewshot_positions(df)

    specs = [
        ("raw_spectrum", "success_at_50_mean", "success_at_50_std", "Raw S@50", PALETTE["blue"], "o", "-"),
        ("domain_aware", "success_at_50_mean", "success_at_50_std", "Domain-aware S@50", PALETTE["peach"], "s", "-"),
        ("raw_spectrum", "target_domain_success_at_50_mean", "target_domain_success_at_50_std", "Raw target S@50", PALETTE["blue"], "o", "--"),
        (
            "domain_aware",
            "target_domain_success_at_50_mean",
            "target_domain_success_at_50_std",
            "Domain-aware target S@50",
            PALETTE["red"],
            "s",
            "--",
        ),
    ]
    for method, metric, std, label, color, marker, linestyle in specs:
        draw_line_with_std(
            ax,
            df[df["method"] == method],
            y_col=metric,
            std_col=std,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
        )

    xs, labels = xticks_for(df)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("Target samples added")
    ax.set_ylabel("Success@50")
    ax.set_title("Extended Target-H coverage curve", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10, loc="left")
    style_axis(ax, y_lim=(-0.03, 1.04), x_grid=True)
    ax.legend(frameon=False, fontsize=11, loc="best")
    add_panel_label(ax, "d")
    return df


def draw_planner_replay_dual_axis(ax: plt.Axes) -> pd.DataFrame:
    df = load_rich_planner_replay_table()
    labels = df["run_label"].tolist()
    x = np.arange(len(labels))
    width = 0.34

    bars_overall = ax.bar(
        x - width / 2,
        df["success_at_50"],
        width=width,
        label="Overall S@50",
        color=PALETTE["blue"],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    bars_target = ax.bar(
        x + width / 2,
        df["target_domain_success_at_50"],
        width=width,
        label="Target S@50",
        color=PALETTE["peach"],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    ax2 = ax.twinx()
    ax2.plot(
        x,
        df["mean_fewshot_cost"],
        marker="D",
        markersize=6,
        linewidth=2.0,
        linestyle=":",
        color=PALETTE["red"],
        label="Target samples",
        zorder=4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Success@50")
    ax2.set_ylabel("Mean target samples")
    ax.set_title("500-query Target-H planner replay", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10, loc="left")
    style_axis(ax, y_lim=(-0.03, 1.04))
    ax2.set_ylim(0, 225)
    ax2.tick_params(axis="y", labelsize=TICK_FONTSIZE, width=1.0, length=4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_linewidth(1.0)

    lines, line_labels = ax.get_legend_handles_labels()
    lines2, line_labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, line_labels + line_labels2, frameon=False, fontsize=11, loc="upper left")
    add_panel_label(ax, "e")
    return df


def draw_deepseek_budget_distribution(ax: plt.Axes) -> pd.DataFrame:
    df = load_deepseek_action_summary()
    sub = df[(df["target_domain"] == "target_h") & (df["category"] == "fewshot")].copy()
    order = ["0", "20", "50", "100", "200"]
    sub["value"] = sub["value"].astype(str)
    sub = sub.set_index("value").reindex(order).reset_index()
    sub["proportion_percent"] = pd.to_numeric(sub["proportion"], errors="coerce").fillna(0) * 100.0

    x = np.arange(len(order))
    bars = ax.bar(
        x,
        sub["proportion_percent"],
        color=PALETTE["blue"],
        edgecolor="white",
        linewidth=0.8,
        width=0.72,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel("Selected target-memory size")
    ax.set_ylabel("Action proportion (%)")
    ax.set_title("DeepSeek Target-H evidence budget", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10, loc="left")
    style_axis(ax, y_lim=(0, 42))
    add_bar_labels(ax, bars, fmt="{:.1f}", offset=0.7)
    add_panel_label(ax, "f")
    return sub


def draw_main_individual_panels() -> None:
    """Save individual panels that exactly match the revised Phase 4 composite."""

    # Panel a: schematic.
    fig_a, ax_a = plt.subplots(figsize=(6.2, 4.5))
    panel_a_data = pd.DataFrame(
        {
            "panel": ["a"],
            "description": ["schematic: few-shot OOD evidence allocation workflow"],
        }
    )
    draw_evidence_allocation_schematic(ax_a)
    save_figure(fig_a, "main_panel_a_evidence_allocation_schematic")
    panel_a_data.to_csv(OUTPUT_DIR / "main_panel_a_evidence_allocation_schematic_source_data.csv", index=False)

    # Panels b/c: recovery curves.
    df = load_fewshot_aggregate()

    fig_b, ax_b = plt.subplots(figsize=(6.2, 4.5))
    panel_b_data = draw_combined_recovery_lines(
        ax_b,
        df,
        target="target_h",
        title="Target-H topology-shift recovery",
        show_overall=True,
        panel="b",
    )
    save_figure(fig_b, "main_panel_b_target_h_topology_shift_recovery")
    panel_b_data.to_csv(OUTPUT_DIR / "main_panel_b_target_h_topology_shift_recovery_source_data.csv", index=False)

    fig_c, ax_c = plt.subplots(figsize=(6.2, 4.5))
    panel_c_data = draw_combined_recovery_lines(
        ax_c,
        df,
        target="target_self",
        title="Target-Self calibration-like recovery",
        show_overall=False,
        panel="c",
    )
    save_figure(fig_c, "main_panel_c_target_self_calibration_like_recovery")
    panel_c_data.to_csv(OUTPUT_DIR / "main_panel_c_target_self_calibration_like_recovery_source_data.csv", index=False)

    # Panel d: extended coverage curve.
    fig_d, ax_d = plt.subplots(figsize=(6.2, 4.5))
    panel_d_data = draw_extended_target_h_coverage(ax_d)
    save_figure(fig_d, "main_panel_d_extended_target_h_coverage_curve")
    panel_d_data.to_csv(OUTPUT_DIR / "main_panel_d_extended_target_h_coverage_curve_source_data.csv", index=False)

    # Panel e: planner replay with twin y-axis.
    fig_e, ax_e = plt.subplots(figsize=(6.6, 4.5))
    panel_e_data = draw_planner_replay_dual_axis(ax_e)
    save_figure(fig_e, "main_panel_e_500query_target_h_planner_replay")
    panel_e_data.to_csv(OUTPUT_DIR / "main_panel_e_500query_target_h_planner_replay_source_data.csv", index=False)

    # Panel f: DeepSeek budget allocation.
    fig_f, ax_f = plt.subplots(figsize=(5.8, 4.5))
    panel_f_data = draw_deepseek_budget_distribution(ax_f)
    save_figure(fig_f, "main_panel_f_deepseek_target_h_evidence_budget")
    panel_f_data.to_csv(OUTPUT_DIR / "main_panel_f_deepseek_target_h_evidence_budget_source_data.csv", index=False)


def draw_composite() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18.2, 9.6))

    df = load_fewshot_aggregate()
    panel_a_data = pd.DataFrame(
        {
            "panel": ["a"],
            "description": ["schematic: few-shot OOD evidence allocation workflow"],
        }
    )
    draw_evidence_allocation_schematic(axes[0, 0])
    panel_b_data = draw_combined_recovery_lines(
        axes[0, 1],
        df,
        target="target_h",
        title="Target-H topology-shift recovery",
        show_overall=True,
        panel="b",
    )
    panel_c_data = draw_combined_recovery_lines(
        axes[0, 2],
        df,
        target="target_self",
        title="Target-Self calibration-like recovery",
        show_overall=False,
        panel="c",
    )
    panel_d_data = draw_extended_target_h_coverage(axes[1, 0])
    panel_e_data = draw_planner_replay_dual_axis(axes[1, 1])
    panel_f_data = draw_deepseek_budget_distribution(axes[1, 2])

    panel_a_data.to_csv(OUTPUT_DIR / "phase4_main_composite_panel_a_source_data.csv", index=False)
    panel_b_data.to_csv(OUTPUT_DIR / "phase4_main_composite_panel_b_source_data.csv", index=False)
    panel_c_data.to_csv(OUTPUT_DIR / "phase4_main_composite_panel_c_source_data.csv", index=False)
    panel_d_data.to_csv(OUTPUT_DIR / "phase4_main_composite_panel_d_source_data.csv", index=False)
    panel_e_data.to_csv(OUTPUT_DIR / "phase4_main_composite_panel_e_source_data.csv", index=False)
    panel_f_data.to_csv(OUTPUT_DIR / "phase4_main_composite_panel_f_source_data.csv", index=False)

    save_figure(fig, "phase4_main_composite")


def main() -> None:
    setup_basic_matplotlib()
    ensure_output_dir()

    draw_panel_a_success_at50()
    draw_panel_b_target_domain_success()
    draw_panel_c_top1_target_domain_rate()
    draw_panel_d_planner_policy_comparison()
    draw_panel_e_efficiency_pareto()
    draw_panel_f_multimodel_repeat_robustness()
    draw_main_individual_panels()
    draw_composite()

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")

    print(f"Saved Phase 4 main panels to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
