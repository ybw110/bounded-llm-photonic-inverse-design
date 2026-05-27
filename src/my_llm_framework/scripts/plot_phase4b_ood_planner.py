from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_LABELS = {
    "target_h": "Target-H topology shift",
    "target_self": "Target-Self material/frequency shift",
}

CATEGORY_COLORS = {
    "Source-only": "#9AA0A6",
    "Rule": "#4C78A8",
    "DeepSeek": "#54A24B",
    "Oracle": "#B279A2",
}

METHOD_COLORS = {
    "raw_spectrum": "#4C78A8",
    "domain_aware": "#54A24B",
    "physics_summary": "#F58518",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "axes.linewidth": 0.9,
            "savefig.dpi": 300,
            "figure.dpi": 150,
        }
    )


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.name, path
    label, path = spec.split("=", 1)
    return label.strip(), Path(path.strip())


def policy_category(label: str) -> str:
    lower = label.lower()
    if "source" in lower:
        return "Source-only"
    if "oracle" in lower:
        return "Oracle"
    if "deepseek" in lower:
        return "DeepSeek"
    return "Rule"


def load_runs(run_specs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_frames: list[pd.DataFrame] = []
    action_frames: list[pd.DataFrame] = []
    for index, spec in enumerate(run_specs):
        label, directory = parse_run_spec(spec)
        summary_path = directory / "ood_recovery_planner_summary.csv"
        action_path = directory / "ood_recovery_planner_action_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = pd.read_csv(summary_path)
        summary["run_label"] = label
        summary["run_order"] = index
        summary["policy_category"] = policy_category(label)
        summary_frames.append(summary)
        if action_path.exists():
            action = pd.read_csv(action_path)
            action["run_label"] = label
            action["run_order"] = index
            action["policy_category"] = policy_category(label)
            action_frames.append(action)

    summary_df = pd.concat(summary_frames, ignore_index=True)
    action_df = pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame()
    return summary_df, action_df


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes, *, ylim: tuple[float, float] | None = None) -> None:
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylim is not None:
        ax.set_ylim(*ylim)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", ha="left", va="top")


def ordered_labels(df: pd.DataFrame) -> list[str]:
    return list(df.sort_values("run_order", kind="stable")["run_label"].drop_duplicates())


def colors_for(labels: list[str]) -> list[str]:
    return [CATEGORY_COLORS.get(policy_category(label), "#555555") for label in labels]


def plot_main_comparison(summary_df: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    target_h = summary_df[summary_df["target_domain"] == "target_h"].copy()
    target_self = summary_df[summary_df["target_domain"] == "target_self"].copy()
    labels = ordered_labels(target_h)
    x = np.arange(len(labels))
    colors = colors_for(labels)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.0))
    metric_specs = [
        ("target_h", "target_domain_success_at_50", "Target-H target-domain Success@50", "Rate", (0.0, 1.04)),
        ("target_h", "mean_fewshot_cost", "Target-H mean few-shot cost", "Target samples", None),
        ("target_h", "oracle_gap_target_domain_success_at_50", "Target-H oracle gap", "Rate", (0.0, 1.04)),
        ("target_self", "target_domain_success_at_50", "Target-Self target-domain Success@50", "Rate", (0.0, 1.04)),
    ]

    for panel, ax, (target, metric, title, ylabel, ylim) in zip("ABCD", axes.flat, metric_specs):
        sub = target_h if target == "target_h" else target_self
        sub = sub.set_index("run_label").reindex(labels).reset_index()
        values = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
        ax.bar(x, values, color=colors, width=0.7)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        style_axis(ax, ylim=ylim)
        add_panel_label(ax, panel)
        for xi, value in zip(x, values):
            if np.isfinite(value):
                text = f"{value:.2f}" if value < 10 else f"{value:.0f}"
                ax.text(xi, value, text, ha="center", va="bottom", fontsize=7.5)

    save_figure(fig, output_dir, "phase4b_ood_planner_main_comparison", formats)


def plot_efficiency(summary_df: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    target_h = summary_df[summary_df["target_domain"] == "target_h"].copy()
    labels = ordered_labels(target_h)
    sub = target_h.set_index("run_label").reindex(labels).reset_index()
    x = pd.to_numeric(sub["mean_fewshot_cost"], errors="coerce")
    y = pd.to_numeric(sub["target_domain_success_at_50"], errors="coerce")
    colors = colors_for(labels)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.scatter(x, y, s=80, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean target samples used")
    ax.set_ylabel("Target-H target-domain Success@50")
    ax.set_title("Sample-efficient OOD recovery")
    style_axis(ax, ylim=(0.0, 1.04))
    save_figure(fig, output_dir, "phase4b_ood_planner_efficiency", formats)


def _stacked_bar(
    ax: plt.Axes,
    pivot: pd.DataFrame,
    *,
    color_map: dict[str, str] | None = None,
    title: str,
    legend_ncols: int = 3,
) -> None:
    labels = list(pivot.index)
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for col in pivot.columns:
        values = pivot[col].to_numpy(dtype=float)
        color = color_map.get(str(col), None) if color_map else None
        ax.bar(x, values, bottom=bottom, label=str(col), color=color, width=0.7)
        bottom += values
    ax.set_title(title)
    ax.set_ylabel("Proportion")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    style_axis(ax, ylim=(0.0, 1.04))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False, ncols=legend_ncols)


def plot_action_distribution(action_df: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    if action_df.empty:
        return
    target_h = action_df[action_df["target_domain"] == "target_h"].copy()
    if target_h.empty:
        return

    order = ordered_labels(target_h[["run_label", "run_order"]].drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))

    method = target_h[target_h["category"] == "method"].copy()
    if not method.empty:
        method_pivot = method.pivot_table(index="run_label", columns="value", values="proportion", aggfunc="sum").fillna(0.0)
        method_pivot = method_pivot.reindex(order)
        _stacked_bar(axes[0], method_pivot, color_map=METHOD_COLORS, title="Target-H method selection", legend_ncols=2)
        add_panel_label(axes[0], "A")

    fewshot = target_h[target_h["category"] == "fewshot"].copy()
    if not fewshot.empty:
        fewshot["sort_value"] = pd.to_numeric(fewshot["value"], errors="coerce").fillna(np.inf)
        fewshot_order = (
            fewshot[["value", "sort_value"]]
            .drop_duplicates()
            .sort_values(["sort_value", "value"], kind="stable")["value"]
            .astype(str)
            .tolist()
        )
        fewshot_pivot = fewshot.pivot_table(index="run_label", columns="value", values="proportion", aggfunc="sum").fillna(0.0)
        fewshot_pivot = fewshot_pivot.reindex(order)
        fewshot_pivot = fewshot_pivot[[col for col in fewshot_order if col in fewshot_pivot.columns]]
        _stacked_bar(axes[1], fewshot_pivot, color_map=None, title="Target-H few-shot action selection", legend_ncols=4)
        add_panel_label(axes[1], "B")

    save_figure(fig, output_dir, "phase4b_ood_planner_action_distribution", formats)


def write_tables(summary_df: pd.DataFrame, action_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "phase4b_ood_planner_plot_summary.csv", index=False)
    if not action_df.empty:
        action_df.to_csv(output_dir / "phase4b_ood_planner_plot_action_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="Label=output_dir containing ood planner CSV files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()

    configure_style()
    summary_df, action_df = load_runs(args.run)
    output_dir = Path(args.output_dir)
    plot_main_comparison(summary_df, output_dir, args.formats)
    plot_efficiency(summary_df, output_dir, args.formats)
    plot_action_distribution(action_df, output_dir, args.formats)
    write_tables(summary_df, action_df, output_dir)
    print(f"Saved Phase 4b OOD planner figures: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
