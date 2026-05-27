"""
Phase 4 supplementary figure panels.

This script complements plot_phase4.py with seven detailed SI panels:

S1. Success@B curves across budgets.
S2. Amplitude-MSE curves.
S3. Budget-to-success curves.
S4. Failure-recovery and OOD-gain summary.
S5. Extended Target-H few-shot scaling.
S6. LLM planner action distribution.
S7. Multi-model oracle-match and invalid/fallback rates.

The visual style matches plot_phase4.py and the previous Phase 1--3 plotting scripts.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "phase4_SI"

SAVE_PNG = True
SAVE_PDF = True
SAVE_DPI = 600
SHOW_FIGURES = True

TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
VALUE_FONTSIZE = 13
LEGEND_FONTSIZE = 14

PALETTE = {
    "blue": "#81AFD4",
    "pale_blue": "#CAE0E8",
    "red": "#DE7C7D",
    "peach": "#FBCFBB",
    "gray": "#CBCBCB",
    "dark_gray": "#4D4D4D",
}

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
METHOD_ORDER = ["raw_spectrum", "physics_summary", "domain_aware"]

BUDGET_COLORS = {
    1: PALETTE["blue"],
    5: PALETTE["pale_blue"],
    10: PALETTE["peach"],
    50: PALETTE["red"],
}

TARGET_LABELS = {
    "target_h": "Target-H",
    "target_self": "Target-Self",
}
TARGET_ORDER = ["target_h", "target_self"]

MODEL_COLORS = {
    "DeepSeek": PALETTE["peach"],
    "GLM-5.1": PALETTE["blue"],
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
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=18, fontweight="bold", va="top", ha="left")


def add_bar_labels(ax: plt.Axes, bars, *, fmt: str = "{:.2f}", offset: float = 0.015) -> None:
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
            fontsize=VALUE_FONTSIZE,
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
    return add_fewshot_positions(read_csv_checked(path))


def load_target_h_extended() -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / "phase4_fewshot_retrieval_target_h_extended_v1" / "fewshot_retrieval_summary_aggregate.csv"
    return add_fewshot_positions(read_csv_checked(path))


def load_multimodel_summary() -> pd.DataFrame:
    path = (
        PROJECT_ROOT
        / "outputs"
        / "phase4b_ood_planner_multimodel_500_repeats_summary_v1"
        / "phase4b_multimodel_repeat_summary.csv"
    )
    return read_csv_checked(path)


def load_ood_action_summary() -> pd.DataFrame:
    runs = [
        ("DeepSeek", PROJECT_ROOT / "outputs" / "phase4b_ood_planner_deepseek_500_v1"),
        ("GLM-5.1", PROJECT_ROOT / "outputs" / "phase4b_ood_planner_glm51_500_repeat1_v1"),
    ]
    frames = []
    for run_label, run_dir in runs:
        path = run_dir / "ood_recovery_planner_action_summary.csv"
        df = read_csv_checked(path)
        df["run_label"] = run_label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def draw_line_with_std(
    ax: plt.Axes,
    sub: pd.DataFrame,
    *,
    y_col: str,
    std_col: str | None,
    label: str,
    color: str,
    marker: str = "o",
    linestyle: str = "-",
) -> None:
    sub = sub.sort_values("fewshot_x", kind="stable")
    x = sub["fewshot_x"].to_numpy(dtype=float)
    y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)
    ax.plot(x, y, marker=marker, markersize=5.2, linewidth=2.1, label=label, color=color, linestyle=linestyle, zorder=3)
    if std_col and std_col in sub.columns:
        std = pd.to_numeric(sub[std_col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(std).any() and np.nanmax(std) > 0:
            ax.fill_between(x, y - std, y + std, color=color, alpha=0.15, linewidth=0, zorder=2)


def draw_si_a_success_at_b_curves() -> None:
    df = load_fewshot_aggregate()
    method = "domain_aware"
    budgets = [1, 5, 10, 50]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
    for ax, panel, target in zip(axes, ["a", "b"], TARGET_ORDER):
        sub_target = df[(df["target_domain"] == target) & (df["method"] == method)]
        for budget in budgets:
            draw_line_with_std(
                ax,
                sub_target,
                y_col=f"success_at_{budget}_mean",
                std_col=f"success_at_{budget}_std",
                label=f"Success@{budget}",
                color=BUDGET_COLORS[budget],
            )
        xs, labels = xticks_for(sub_target)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_xlabel("Few-shot target samples")
        ax.set_title(TARGET_LABELS[target], fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10)
        style_axis(ax, y_lim=(-0.03, 1.04))
        add_panel_label(ax, panel)
    axes[0].set_ylabel("Success rate")
    axes[0].legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="best")
    save_figure(fig, "si_panel_a_success_at_b_domain_aware")

    cols = ["target_domain", "method", "fewshot_label", "fewshot_x"]
    for budget in budgets:
        cols += [f"success_at_{budget}_mean", f"success_at_{budget}_std"]
    df[(df["method"] == method)][cols].to_csv(OUTPUT_DIR / "si_panel_a_success_at_b_domain_aware_source_data.csv", index=False)


def draw_si_b_mse_curves() -> None:
    df = load_fewshot_aggregate()
    methods = ["raw_spectrum", "domain_aware"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
    for ax, panel, target in zip(axes, ["a", "b"], TARGET_ORDER):
        sub_target = df[df["target_domain"] == target]
        for method in methods:
            draw_line_with_std(
                ax,
                sub_target[sub_target["method"] == method],
                y_col="mean_top1_amplitude_mse_mean",
                std_col="mean_top1_amplitude_mse_std",
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                marker="o" if method == "raw_spectrum" else "s",
            )
        xs, labels = xticks_for(sub_target)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_xlabel("Few-shot target samples")
        ax.set_yscale("log")
        ax.set_title(TARGET_LABELS[target], fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10)
        style_axis(ax, y_grid=True)
        add_panel_label(ax, panel)
    axes[0].set_ylabel("Mean top-1 amplitude MSE")
    axes[0].legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="best")
    save_figure(fig, "si_panel_b_top1_mse_curves")

    df[
        [
            "target_domain",
            "method",
            "fewshot_label",
            "fewshot_x",
            "mean_top1_amplitude_mse_mean",
            "mean_top1_amplitude_mse_std",
            "mean_best_amplitude_mse_mean",
            "mean_best_amplitude_mse_std",
        ]
    ].to_csv(OUTPUT_DIR / "si_panel_b_top1_mse_curves_source_data.csv", index=False)


def draw_si_c_budget_to_success() -> None:
    df = load_fewshot_aggregate()
    methods = ["raw_spectrum", "domain_aware"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=False)
    for ax, panel, target in zip(axes, ["a", "b"], TARGET_ORDER):
        sub_target = df[df["target_domain"] == target]
        for method in methods:
            draw_line_with_std(
                ax,
                sub_target[sub_target["method"] == method],
                y_col="mean_budget_to_success_mean",
                std_col="mean_budget_to_success_std",
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                marker="o" if method == "raw_spectrum" else "s",
            )
        xs, labels = xticks_for(sub_target)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_xlabel("Few-shot target samples")
        ax.set_ylabel("Budget to success")
        ax.set_title(TARGET_LABELS[target], fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10)
        style_axis(ax)
        add_panel_label(ax, panel)
    axes[0].legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="best")
    save_figure(fig, "si_panel_c_budget_to_success")

    df[
        [
            "target_domain",
            "method",
            "fewshot_label",
            "fewshot_x",
            "mean_budget_to_success_mean",
            "mean_budget_to_success_std",
            "mean_budget_to_target_domain_success_mean",
            "mean_budget_to_target_domain_success_std",
        ]
    ].to_csv(OUTPUT_DIR / "si_panel_c_budget_to_success_source_data.csv", index=False)


def draw_si_d_failure_recovery_gain() -> None:
    df = load_fewshot_aggregate()
    plot_df = df[(df["method"].isin(["raw_spectrum", "domain_aware"])) & (df["fewshot_label"].astype(str).isin(["20", "50", "100", "200"]))]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
    for ax, panel, target in zip(axes, ["a", "b"], TARGET_ORDER):
        sub_target = plot_df[plot_df["target_domain"] == target]
        labels = ["20", "50", "100", "200"]
        x = np.arange(len(labels))
        width = 0.36
        for offset, method in [(-width / 2, "raw_spectrum"), (width / 2, "domain_aware")]:
            sub = sub_target[sub_target["method"] == method].set_index(sub_target[sub_target["method"] == method]["fewshot_label"].astype(str)).reindex(labels)
            values = pd.to_numeric(sub["failure_recovery_rate_at_50_mean"], errors="coerce").to_numpy(dtype=float)
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Few-shot target samples")
        ax.set_title(TARGET_LABELS[target], fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10)
        style_axis(ax, y_lim=(0.0, 0.5))
        add_panel_label(ax, panel)
    axes[0].set_ylabel("Failure recovery rate@50")
    axes[0].legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="best")
    save_figure(fig, "si_panel_d_failure_recovery_rate")

    plot_df[
        [
            "target_domain",
            "method",
            "fewshot_label",
            "failure_recovery_rate_at_50_mean",
            "failure_recovery_rate_at_50_std",
            "ood_recovery_gain_at_50_mean",
            "ood_recovery_gain_at_50_std",
        ]
    ].to_csv(OUTPUT_DIR / "si_panel_d_failure_recovery_rate_source_data.csv", index=False)


def draw_si_e_target_h_extended_scaling() -> None:
    df = load_target_h_extended()
    df = df[(df["target_domain"] == "target_h") & (df["method"].isin(["raw_spectrum", "domain_aware"]))].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
    metric_specs = [
        ("target_domain_success_at_50_mean", "target_domain_success_at_50_std", "Target-domain Success@50"),
        ("top1_target_domain_rate_mean", "top1_target_domain_rate_std", "Top-1 target-domain rate"),
    ]
    for ax, panel, (metric, std, ylabel) in zip(axes, ["a", "b"], metric_specs):
        for method in ["raw_spectrum", "domain_aware"]:
            sub = df[df["method"] == method]
            draw_line_with_std(
                ax,
                sub,
                y_col=metric,
                std_col=std,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                marker="o" if method == "raw_spectrum" else "s",
            )
        xs, labels = xticks_for(df)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_xlabel("Few-shot target samples")
        ax.set_ylabel(ylabel)
        ax.set_title("Extended Target-H scaling", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10)
        style_axis(ax, y_lim=(-0.03, 1.04))
        add_panel_label(ax, panel)
    axes[0].legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="best")
    save_figure(fig, "si_panel_e_target_h_extended_scaling")

    df[
        [
            "target_domain",
            "method",
            "fewshot_label",
            "fewshot_x",
            "success_at_50_mean",
            "success_at_50_std",
            "target_domain_success_at_50_mean",
            "target_domain_success_at_50_std",
            "top1_target_domain_rate_mean",
            "top1_target_domain_rate_std",
        ]
    ].to_csv(OUTPUT_DIR / "si_panel_e_target_h_extended_scaling_source_data.csv", index=False)


def _stacked_bar_from_action_category(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    category: str,
    values_order: list[str],
    color_map: dict[str, str],
    title: str,
) -> pd.DataFrame:
    sub = df[(df["target_domain"] == "target_h") & (df["category"] == category)].copy()
    sub["value"] = sub["value"].astype(str)
    run_labels = ["DeepSeek", "GLM-5.1"]
    pivot = (
        sub.pivot_table(index="run_label", columns="value", values="proportion", aggfunc="sum", fill_value=0.0)
        .reindex(index=run_labels, columns=values_order, fill_value=0.0)
    )
    x = np.arange(len(run_labels))
    bottom = np.zeros(len(run_labels))
    for value in values_order:
        y = pivot[value].to_numpy(dtype=float)
        ax.bar(
            x,
            y,
            bottom=bottom,
            label=value,
            color=color_map.get(value, PALETTE["gray"]),
            edgecolor="white",
            linewidth=0.8,
            width=0.62,
            zorder=3,
        )
        bottom += y
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=15, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10)
    style_axis(ax, y_lim=(0.0, 1.04))
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper left", bbox_to_anchor=(1.02, 1.02))
    return pivot.reset_index()


def draw_si_f_planner_action_distribution() -> None:
    df = load_ood_action_summary()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
    method_source = _stacked_bar_from_action_category(
        axes[0],
        df,
        category="method",
        values_order=["raw_spectrum", "domain_aware", "physics_summary"],
        color_map={
            "raw_spectrum": PALETTE["blue"],
            "domain_aware": PALETTE["peach"],
            "physics_summary": PALETTE["red"],
        },
        title="Target-H method selection",
    )
    fewshot_source = _stacked_bar_from_action_category(
        axes[1],
        df,
        category="fewshot",
        values_order=["0", "20", "50", "100", "200"],
        color_map={
            "0": PALETTE["gray"],
            "20": PALETTE["pale_blue"],
            "50": PALETTE["blue"],
            "100": PALETTE["peach"],
            "200": PALETTE["red"],
        },
        title="Target-H few-shot selection",
    )
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    save_figure(fig, "si_panel_f_planner_action_distribution")

    df.to_csv(OUTPUT_DIR / "si_panel_f_planner_action_distribution_source_data.csv", index=False)
    method_source.to_csv(OUTPUT_DIR / "si_panel_f_method_distribution_pivot_source_data.csv", index=False)
    fewshot_source.to_csv(OUTPUT_DIR / "si_panel_f_fewshot_distribution_pivot_source_data.csv", index=False)


def draw_si_g_multimodel_stability() -> None:
    df = load_multimodel_summary()
    target_h = df[df["target_domain"] == "target_h"].copy()
    target_h["run_label"] = pd.Categorical(target_h["run_label"], categories=["DeepSeek", "GLM-5.1"], ordered=True)
    target_h = target_h.sort_values("run_label")

    metrics = [
        ("oracle_match_rate", "Oracle match rate", (0.0, 1.04), "{:.2f}"),
        ("fallback_or_invalid_action_rate", "Invalid/fallback rate", (0.0, 0.08), "{:.3f}"),
        ("top1_target_domain_rate", "Top-1 target-domain rate", (0.0, 1.04), "{:.2f}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    labels = target_h["run_label"].astype(str).tolist()
    x = np.arange(len(labels))
    for ax, panel, (metric, ylabel, ylim, fmt) in zip(axes, ["a", "b", "c"], metrics):
        y = pd.to_numeric(target_h[f"{metric}_mean"], errors="coerce").to_numpy(dtype=float)
        yerr = pd.to_numeric(target_h[f"{metric}_std"], errors="coerce").fillna(0).to_numpy(dtype=float)
        bars = ax.bar(
            x,
            y,
            yerr=yerr,
            capsize=5,
            color=[MODEL_COLORS[label] for label in labels],
            edgecolor="white",
            linewidth=0.8,
            width=0.62,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Target-H {ylabel}", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10)
        style_axis(ax, y_lim=ylim)
        add_bar_labels(ax, bars, fmt=fmt, offset=(ylim[1] - ylim[0]) * 0.018)
        add_panel_label(ax, panel)
    save_figure(fig, "si_panel_g_multimodel_oracle_invalid_rates")

    target_h.to_csv(OUTPUT_DIR / "si_panel_g_multimodel_oracle_invalid_rates_source_data.csv", index=False)


def draw_rich_panel_a_success_budget(ax: plt.Axes, *, target: str, panel: str) -> pd.DataFrame:
    df = load_fewshot_aggregate()
    sub_target = df[(df["target_domain"] == target) & (df["method"] == "domain_aware")].copy()
    for budget in [1, 5, 10, 50]:
        draw_line_with_std(
            ax,
            sub_target,
            y_col=f"success_at_{budget}_mean",
            std_col=f"success_at_{budget}_std",
            label=f"Success@{budget}",
            color=BUDGET_COLORS[budget],
            marker="o",
        )
    xs, labels = xticks_for(sub_target)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("Target samples added")
    ax.set_ylabel("Success rate")
    ax.set_title(f"{TARGET_LABELS[target]} domain-aware Success@B", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10, loc="left")
    style_axis(ax, y_lim=(-0.03, 1.04), x_grid=True)
    ax.legend(frameon=False, fontsize=11, loc="best")
    add_panel_label(ax, panel)
    return sub_target


def draw_rich_panel_c_mse(ax: plt.Axes) -> pd.DataFrame:
    df = load_fewshot_aggregate()
    plot_df = df[df["method"].isin(["raw_spectrum", "domain_aware"])].copy()
    specs = [
        ("target_h", "raw_spectrum", "Target-H raw", PALETTE["blue"], "o", "-"),
        ("target_h", "domain_aware", "Target-H domain-aware", PALETTE["peach"], "s", "-"),
        ("target_self", "raw_spectrum", "Target-Self raw", PALETTE["blue"], "o", "--"),
        ("target_self", "domain_aware", "Target-Self domain-aware", PALETTE["red"], "s", "--"),
    ]
    for target, method, label, color, marker, linestyle in specs:
        draw_line_with_std(
            ax,
            plot_df[(plot_df["target_domain"] == target) & (plot_df["method"] == method)],
            y_col="mean_top1_amplitude_mse_mean",
            std_col="mean_top1_amplitude_mse_std",
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
        )
    xs, labels = xticks_for(plot_df)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("Target samples added")
    ax.set_ylabel("Mean top-1 amplitude MSE")
    ax.set_yscale("log")
    ax.set_title("Spectral-error recovery", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10, loc="left")
    style_axis(ax, x_grid=True)
    ax.legend(frameon=False, fontsize=10.5, loc="best")
    add_panel_label(ax, "c")
    return plot_df


def draw_rich_panel_d_budget(ax: plt.Axes) -> pd.DataFrame:
    df = load_fewshot_aggregate()
    plot_df = df[df["method"].isin(["raw_spectrum", "domain_aware"])].copy()
    specs = [
        ("target_h", "raw_spectrum", "H raw", PALETTE["blue"], "o", "-"),
        ("target_h", "domain_aware", "H domain-aware", PALETTE["peach"], "s", "-"),
        ("target_self", "raw_spectrum", "Self raw", PALETTE["blue"], "o", "--"),
        ("target_self", "domain_aware", "Self domain-aware", PALETTE["red"], "s", "--"),
    ]
    for target, method, label, color, marker, linestyle in specs:
        draw_line_with_std(
            ax,
            plot_df[(plot_df["target_domain"] == target) & (plot_df["method"] == method)],
            y_col="mean_budget_to_target_domain_success_mean",
            std_col="mean_budget_to_target_domain_success_std",
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
        )
    xs, labels = xticks_for(plot_df)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("Target samples added")
    ax.set_ylabel("Budget to target success")
    ax.set_title("Budget-to-success under OOD recovery", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10, loc="left")
    style_axis(ax, x_grid=True)
    ax.legend(frameon=False, fontsize=10.5, loc="best")
    add_panel_label(ax, "d")
    return plot_df


def draw_rich_panel_e_extended(ax: plt.Axes) -> pd.DataFrame:
    df = load_target_h_extended()
    df = df[(df["target_domain"] == "target_h") & (df["method"].isin(["raw_spectrum", "domain_aware"]))].copy()
    keep_labels = ["0", "200", "500", "1000", "2000", "full"]
    df = df[df["fewshot_label"].astype(str).str.lower().isin(keep_labels)].copy()
    df = add_fewshot_positions(df)
    specs = [
        ("raw_spectrum", "target_domain_success_at_50_mean", "target_domain_success_at_50_std", "Raw target S@50", PALETTE["blue"], "o", "-"),
        ("domain_aware", "target_domain_success_at_50_mean", "target_domain_success_at_50_std", "Domain-aware target S@50", PALETTE["red"], "s", "-"),
        ("raw_spectrum", "top1_target_domain_rate_mean", "top1_target_domain_rate_std", "Raw top-1 target rate", PALETTE["blue"], "o", "--"),
        (
            "domain_aware",
            "top1_target_domain_rate_mean",
            "top1_target_domain_rate_std",
            "Domain-aware top-1 target rate",
            PALETTE["peach"],
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
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_xlabel("Target samples added")
    ax.set_ylabel("Rate")
    ax.set_title("Extended Target-H validity scaling", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10, loc="left")
    style_axis(ax, y_lim=(-0.03, 1.04), x_grid=True)
    ax.legend(frameon=False, fontsize=10.5, loc="best")
    add_panel_label(ax, "e")
    return df


def draw_rich_panel_f_action_distribution(ax: plt.Axes) -> pd.DataFrame:
    df = load_ood_action_summary()
    target_h = df[df["target_domain"] == "target_h"].copy()
    positions = {
        ("DeepSeek", "method"): 0.0,
        ("GLM-5.1", "method"): 0.85,
        ("DeepSeek", "fewshot"): 1.70,
        ("GLM-5.1", "fewshot"): 2.55,
    }
    method_order = ["raw_spectrum", "domain_aware"]
    fewshot_order = ["0", "20", "50", "100", "200"]
    color_map = {
        "raw_spectrum": PALETTE["blue"],
        "domain_aware": PALETTE["peach"],
        "0": PALETTE["gray"],
        "20": PALETTE["pale_blue"],
        "50": PALETTE["blue"],
        "100": PALETTE["peach"],
        "200": PALETTE["red"],
    }
    for run_label in ["DeepSeek", "GLM-5.1"]:
        for category, order in [("method", method_order), ("fewshot", fewshot_order)]:
            bottom = 0.0
            x = positions[(run_label, category)]
            for value in order:
                sub = target_h[
                    (target_h["run_label"] == run_label)
                    & (target_h["category"] == category)
                    & (target_h["value"].astype(str) == value)
                ]
                proportion = float(sub["proportion"].iloc[0]) if not sub.empty else 0.0
                ax.bar(
                    x,
                    proportion,
                    bottom=bottom,
                    width=0.62,
                    color=color_map[value],
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                )
                bottom += proportion
    ax.set_xticks([0.0, 0.85, 1.70, 2.55])
    ax.set_xticklabels(["DeepSeek\nmethod", "GLM-5.1\nmethod", "DeepSeek\nfew-shot", "GLM-5.1\nfew-shot"], rotation=15, ha="right")
    ax.set_ylabel("Action proportion")
    ax.set_title("Planner action distribution", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10, loc="left")
    style_axis(ax, y_lim=(0.0, 1.18))
    handles = [
        Patch(facecolor=color_map["raw_spectrum"], edgecolor="white", label="raw"),
        Patch(facecolor=color_map["domain_aware"], edgecolor="white", label="domain-aware"),
        Patch(facecolor=color_map["0"], edgecolor="white", label="0"),
        Patch(facecolor=color_map["20"], edgecolor="white", label="20"),
        Patch(facecolor=color_map["50"], edgecolor="white", label="50"),
        Patch(facecolor=color_map["100"], edgecolor="white", label="100"),
        Patch(facecolor=color_map["200"], edgecolor="white", label="200"),
    ]
    ax.legend(
        handles=handles,
        frameon=True,
        facecolor="white",
        edgecolor="#D0D0D0",
        framealpha=0.92,
        fontsize=12.0,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.99),
        ncol=4,
        columnspacing=0.9,
        handlelength=1.3,
        handletextpad=0.35,
        borderpad=0.35,
    )
    add_panel_label(ax, "f")
    return target_h


def draw_rich_panel_g_multimodel(ax: plt.Axes) -> pd.DataFrame:
    df = load_multimodel_summary()
    target_h = df[df["target_domain"] == "target_h"].copy()
    target_h["run_label"] = pd.Categorical(target_h["run_label"], categories=["DeepSeek", "GLM-5.1"], ordered=True)
    target_h = target_h.sort_values("run_label")
    metrics = [
        ("target_domain_success_at_50", "Valid\nS@50"),
        ("oracle_match_rate", "Oracle\nmatch"),
        ("fallback_or_invalid_action_rate", "Invalid /\nfallback"),
    ]
    x = np.arange(len(metrics))
    width = 0.34
    for offset, run_label in [(-width / 2, "DeepSeek"), (width / 2, "GLM-5.1")]:
        row = target_h[target_h["run_label"].astype(str) == run_label].iloc[0]
        values = [float(row[f"{metric}_mean"]) for metric, _ in metrics]
        errors = [float(row[f"{metric}_std"]) for metric, _ in metrics]
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            yerr=errors,
            capsize=4,
            label=run_label,
            color=MODEL_COLORS[run_label],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for bar in bars:
            height = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.018,
                f"{height:.2f}" if height >= 0.1 else f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                color="#222222",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Rate")
    ax.set_title("Multi-model planner robustness", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10, loc="left")
    style_axis(ax, y_lim=(0.0, 1.04))
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper right")
    add_panel_label(ax, "g")
    return target_h


def draw_rich_individual_panels() -> None:
    fig_a, ax_a = plt.subplots(figsize=(6.2, 4.5))
    data_a = draw_rich_panel_a_success_budget(ax_a, target="target_h", panel="a")
    save_figure(fig_a, "rich_panel_a_target_h_success_budget")
    data_a.to_csv(OUTPUT_DIR / "rich_panel_a_target_h_success_budget_source_data.csv", index=False)

    fig_b, ax_b = plt.subplots(figsize=(6.2, 4.5))
    data_b = draw_rich_panel_a_success_budget(ax_b, target="target_self", panel="b")
    save_figure(fig_b, "rich_panel_b_target_self_success_budget")
    data_b.to_csv(OUTPUT_DIR / "rich_panel_b_target_self_success_budget_source_data.csv", index=False)

    fig_c, ax_c = plt.subplots(figsize=(6.2, 4.5))
    data_c = draw_rich_panel_c_mse(ax_c)
    save_figure(fig_c, "rich_panel_c_spectral_error_recovery")
    data_c.to_csv(OUTPUT_DIR / "rich_panel_c_spectral_error_recovery_source_data.csv", index=False)

    fig_d, ax_d = plt.subplots(figsize=(6.2, 4.5))
    data_d = draw_rich_panel_d_budget(ax_d)
    save_figure(fig_d, "rich_panel_d_budget_to_target_success")
    data_d.to_csv(OUTPUT_DIR / "rich_panel_d_budget_to_target_success_source_data.csv", index=False)

    fig_e, ax_e = plt.subplots(figsize=(6.2, 4.5))
    data_e = draw_rich_panel_e_extended(ax_e)
    save_figure(fig_e, "rich_panel_e_extended_target_h_validity_scaling")
    data_e.to_csv(OUTPUT_DIR / "rich_panel_e_extended_target_h_validity_scaling_source_data.csv", index=False)

    fig_f, ax_f = plt.subplots(figsize=(7.2, 4.5))
    data_f = draw_rich_panel_f_action_distribution(ax_f)
    save_figure(fig_f, "rich_panel_f_planner_action_distribution")
    data_f.to_csv(OUTPUT_DIR / "rich_panel_f_planner_action_distribution_source_data.csv", index=False)

    fig_g, ax_g = plt.subplots(figsize=(6.2, 4.5))
    data_g = draw_rich_panel_g_multimodel(ax_g)
    save_figure(fig_g, "rich_panel_g_multimodel_planner_robustness")
    data_g.to_csv(OUTPUT_DIR / "rich_panel_g_multimodel_planner_robustness_source_data.csv", index=False)


def draw_composite() -> None:
    fig = plt.figure(figsize=(22.0, 9.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 1.0], wspace=0.35, hspace=0.48)
    axes = {
        "a": fig.add_subplot(gs[0, 0]),
        "b": fig.add_subplot(gs[0, 1]),
        "c": fig.add_subplot(gs[0, 2]),
        "d": fig.add_subplot(gs[0, 3]),
        "e": fig.add_subplot(gs[1, 0]),
        "f": fig.add_subplot(gs[1, 1:3]),
        "g": fig.add_subplot(gs[1, 3]),
    }

    data_a = draw_rich_panel_a_success_budget(axes["a"], target="target_h", panel="a")
    data_b = draw_rich_panel_a_success_budget(axes["b"], target="target_self", panel="b")
    data_c = draw_rich_panel_c_mse(axes["c"])
    data_d = draw_rich_panel_d_budget(axes["d"])
    data_e = draw_rich_panel_e_extended(axes["e"])
    data_f = draw_rich_panel_f_action_distribution(axes["f"])
    data_g = draw_rich_panel_g_multimodel(axes["g"])

    data_a.to_csv(OUTPUT_DIR / "phase4_SI_composite_panel_a_source_data.csv", index=False)
    data_b.to_csv(OUTPUT_DIR / "phase4_SI_composite_panel_b_source_data.csv", index=False)
    data_c.to_csv(OUTPUT_DIR / "phase4_SI_composite_panel_c_source_data.csv", index=False)
    data_d.to_csv(OUTPUT_DIR / "phase4_SI_composite_panel_d_source_data.csv", index=False)
    data_e.to_csv(OUTPUT_DIR / "phase4_SI_composite_panel_e_source_data.csv", index=False)
    data_f.to_csv(OUTPUT_DIR / "phase4_SI_composite_panel_f_source_data.csv", index=False)
    data_g.to_csv(OUTPUT_DIR / "phase4_SI_composite_panel_g_source_data.csv", index=False)

    save_figure(fig, "phase4_SI_composite")


def main() -> None:
    setup_basic_matplotlib()
    ensure_output_dir()

    draw_si_a_success_at_b_curves()
    draw_si_b_mse_curves()
    draw_si_c_budget_to_success()
    draw_si_d_failure_recovery_gain()
    draw_si_e_target_h_extended_scaling()
    draw_si_f_planner_action_distribution()
    draw_si_g_multimodel_stability()
    draw_rich_individual_panels()
    draw_composite()

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")

    print(f"Saved Phase 4 SI panels to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
