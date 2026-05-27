from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "raw_spectrum": "Raw spectrum",
    "physics_summary": "Physics summary",
    "domain_aware": "Domain-aware",
}

METHOD_COLORS = {
    "raw_spectrum": "#4C78A8",
    "physics_summary": "#F58518",
    "domain_aware": "#54A24B",
}

BUDGET_COLORS = {
    1: "#4C78A8",
    5: "#F58518",
    10: "#54A24B",
    50: "#B279A2",
}

TARGET_LABELS = {
    "target_h": "Target-H topology shift",
    "target_self": "Target-Self material/frequency shift",
}

TARGET_ORDER = ["target_h", "target_self"]
METHOD_ORDER = ["raw_spectrum", "physics_summary", "domain_aware"]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.9,
            "savefig.dpi": 300,
            "figure.dpi": 150,
        }
    )


def add_fewshot_positions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric = pd.to_numeric(df["fewshot_label"], errors="coerce")
    order_df = (
        pd.DataFrame({"fewshot_label": df["fewshot_label"].astype(str), "numeric": numeric})
        .drop_duplicates()
        .assign(sort_key=lambda x: x["numeric"].fillna(np.inf))
        .sort_values(["sort_key", "fewshot_label"], kind="stable")
        .reset_index(drop=True)
    )
    positions = {label: float(i) for i, label in enumerate(order_df["fewshot_label"])}
    df["fewshot_x"] = df["fewshot_label"].astype(str).map(positions).astype(float)
    df["fewshot_plot_label"] = df["fewshot_label"].astype(str)
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


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def ordered_values(values: pd.Series, preferred_order: list[str]) -> list[str]:
    value_set = set(values.dropna().astype(str))
    ordered = [value for value in preferred_order if value in value_set]
    ordered.extend(sorted(value_set - set(ordered)))
    return ordered


def draw_line_with_std(
    ax: plt.Axes,
    sub: pd.DataFrame,
    y_col: str,
    std_col: str | None,
    *,
    label: str,
    color: str,
    marker: str = "o",
    linestyle: str = "-",
) -> None:
    sub = sub.sort_values("fewshot_x", kind="stable")
    x = sub["fewshot_x"].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    ax.plot(x, y, marker=marker, linewidth=1.8, markersize=4.5, label=label, color=color, linestyle=linestyle)
    if std_col and std_col in sub.columns:
        std = sub[std_col].to_numpy(dtype=float)
        if np.isfinite(std).any():
            ax.fill_between(x, y - std, y + std, color=color, alpha=0.13, linewidth=0)


def style_axis(ax: plt.Axes, *, ylim: tuple[float, float] | None = None) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylim is not None:
        ax.set_ylim(*ylim)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_success_at_b(df: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    targets = ordered_values(df["target_domain"], TARGET_ORDER)
    methods = ordered_values(df["method"], METHOD_ORDER)
    budgets = [1, 5, 10, 50]

    fig, axes = plt.subplots(len(targets), len(methods), figsize=(12.4, 6.8), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(len(targets), len(methods))

    for r, target in enumerate(targets):
        for c, method in enumerate(methods):
            ax = axes[r, c]
            sub = df[(df["target_domain"] == target) & (df["method"] == method)]
            for budget in budgets:
                y_col = f"success_at_{budget}_mean"
                if y_col not in sub.columns:
                    continue
                draw_line_with_std(
                    ax,
                    sub,
                    y_col,
                    f"success_at_{budget}_std",
                    label=f"Success@{budget}",
                    color=BUDGET_COLORS[budget],
                )
            ax.set_title(f"{TARGET_LABELS.get(target, target)}\n{METHOD_LABELS.get(method, method)}")
            style_axis(ax, ylim=(-0.03, 1.04))
            if c == 0:
                ax.set_ylabel("Success rate")
            if r == len(targets) - 1:
                ax.set_xlabel("Few-shot target samples")
            xs, labels = xticks_for(sub)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels)
    axes[0, -1].legend(loc="lower right", frameon=False)
    save_figure(fig, output_dir, "phase4_success_at_b_curves", formats)


def plot_target_domain_validity(df: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    targets = ordered_values(df["target_domain"], TARGET_ORDER)
    fig, axes = plt.subplots(2, len(targets), figsize=(10.8, 6.6), sharex="col")
    axes = np.asarray(axes).reshape(2, len(targets))

    for c, target in enumerate(targets):
        sub_target = df[df["target_domain"] == target]
        for method, method_sub in sub_target.groupby("method", sort=False):
            color = METHOD_COLORS.get(method, "#555555")
            label = METHOD_LABELS.get(method, method)
            draw_line_with_std(
                axes[0, c],
                method_sub,
                "target_domain_success_at_50_mean",
                "target_domain_success_at_50_std",
                label=label,
                color=color,
            )
            draw_line_with_std(
                axes[1, c],
                method_sub,
                "top1_target_domain_rate_mean",
                "top1_target_domain_rate_std",
                label=label,
                color=color,
            )

        axes[0, c].set_title(TARGET_LABELS.get(target, target))
        axes[0, c].set_ylabel("Target-domain Success@50" if c == 0 else "")
        axes[1, c].set_ylabel("Top1 target-domain rate" if c == 0 else "")
        axes[1, c].set_xlabel("Few-shot target samples")
        for ax in axes[:, c]:
            style_axis(ax, ylim=(-0.03, 1.04))
            xs, labels = xticks_for(sub_target)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels)
    axes[0, -1].legend(loc="lower right", frameon=False)
    save_figure(fig, output_dir, "phase4_target_domain_validity_curves", formats)


def plot_mse_curves(df: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    targets = ordered_values(df["target_domain"], TARGET_ORDER)
    fig, axes = plt.subplots(1, len(targets), figsize=(max(5.4, 5.4 * len(targets)), 3.8), sharex=False)
    axes = np.asarray(axes).reshape(len(targets))

    for c, target in enumerate(targets):
        ax = axes[c]
        sub_target = df[df["target_domain"] == target]
        for method, method_sub in sub_target.groupby("method", sort=False):
            draw_line_with_std(
                ax,
                method_sub,
                "mean_top1_amplitude_mse_mean",
                "mean_top1_amplitude_mse_std",
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method, "#555555"),
            )
        ax.set_title(TARGET_LABELS.get(target, target))
        ax.set_ylabel("Mean top1 amplitude MSE" if c == 0 else "")
        ax.set_xlabel("Few-shot target samples")
        ax.set_yscale("log")
        style_axis(ax)
        xs, labels = xticks_for(sub_target)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
    axes[-1].legend(loc="upper right", frameon=False)
    save_figure(fig, output_dir, "phase4_mse_curves", formats)


def plot_main_composite(df: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), sharex=False)

    targets = set(df["target_domain"].dropna().astype(str))
    if {"target_h", "target_self"}.issubset(targets):
        panels = [
            ("target_h", "success_at_50_mean", "success_at_50_std", "Target-H Success@50", "Rate", (0.60, 1.01), False),
            (
                "target_h",
                "target_domain_success_at_50_mean",
                "target_domain_success_at_50_std",
                "Target-H target-domain Success@50",
                "Rate",
                (-0.03, 1.01),
                False,
            ),
            ("target_self", "success_at_50_mean", "success_at_50_std", "Target-Self Success@50", "Rate", (-0.03, 1.04), False),
            (
                "target_h",
                "top1_target_domain_rate_mean",
                "top1_target_domain_rate_std",
                "Target-H top1 target-domain rate",
                "Rate",
                (-0.03, 1.04),
                False,
            ),
        ]
    else:
        target = ordered_values(df["target_domain"], TARGET_ORDER)[0]
        target_label = TARGET_LABELS.get(target, target)
        panels = [
            (target, "success_at_50_mean", "success_at_50_std", f"{target_label} Success@50", "Rate", (-0.03, 1.04), False),
            (
                target,
                "target_domain_success_at_50_mean",
                "target_domain_success_at_50_std",
                f"{target_label} target-domain Success@50",
                "Rate",
                (-0.03, 1.04),
                False,
            ),
            (
                target,
                "top1_target_domain_rate_mean",
                "top1_target_domain_rate_std",
                f"{target_label} top1 target-domain rate",
                "Rate",
                (-0.03, 1.04),
                False,
            ),
            (
                target,
                "mean_top1_amplitude_mse_mean",
                "mean_top1_amplitude_mse_std",
                f"{target_label} top1 MSE",
                "Mean top1 amplitude MSE",
                None,
                True,
            ),
        ]

    for panel_label, ax, (target, y_col, std_col, title, ylabel, ylim, log_y) in zip("ABCD", axes.flat, panels):
        sub_target = df[df["target_domain"] == target]
        for method, method_sub in sub_target.groupby("method", sort=False):
            draw_line_with_std(
                ax,
                method_sub,
                y_col,
                std_col,
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method, "#555555"),
            )
        ax.set_title(title)
        ax.set_xlabel("Few-shot target samples")
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        style_axis(ax, ylim=ylim)
        xs, labels = xticks_for(sub_target)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        add_panel_label(ax, panel_label)

    axes[0, 0].legend(loc="lower right", frameon=False)
    save_figure(fig, output_dir, "phase4_main_composite", formats)


def write_plot_table(df: pd.DataFrame, output_dir: Path) -> None:
    keep = [
        "target_domain",
        "method",
        "fewshot_label",
        "success_at_1_mean",
        "success_at_5_mean",
        "success_at_10_mean",
        "success_at_50_mean",
        "target_domain_success_at_50_mean",
        "top1_target_domain_rate_mean",
        "mean_budget_to_success_mean",
        "mean_budget_to_target_domain_success_mean",
        "mean_top1_amplitude_mse_mean",
    ]
    existing = [col for col in keep if col in df.columns]
    output_dir.mkdir(parents=True, exist_ok=True)
    df[existing].to_csv(output_dir / "phase4_plot_key_metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="fewshot_retrieval_summary_aggregate.csv")
    parser.add_argument("--output-dir", required=True, help="figure output directory")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], help="figure formats, e.g. png pdf")
    args = parser.parse_args()

    configure_style()
    df = pd.read_csv(args.summary)
    df = add_fewshot_positions(df)
    output_dir = Path(args.output_dir)

    plot_success_at_b(df, output_dir, args.formats)
    plot_target_domain_validity(df, output_dir, args.formats)
    plot_mse_curves(df, output_dir, args.formats)
    plot_main_composite(df, output_dir, args.formats)
    write_plot_table(df, output_dir)

    print(f"Saved Phase 4 figures: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
