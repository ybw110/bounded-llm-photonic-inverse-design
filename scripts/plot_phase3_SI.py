"""
Phase 3 SI figure: DeepSeek action usage in offline planner replay.

This standalone script creates:

a. Full planner action-usage heatmap under 100-query and 500-query replay.
b. Expand action count under 100-query and 500-query replay.
c. Target-H strict-versus-expand success comparison.

The style intentionally matches plot_phase3.py: Arial, large labels, Morandi-like
palette, PDF/PNG export, source-data CSV export, and plt.show() enabled for
direct visual inspection.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "phase3_SI"

SAVE_PNG = True
SAVE_PDF = True
SAVE_DPI = 600
SHOW_FIGURES = True

TITLE_FONTSIZE = 21
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 15
VALUE_FONTSIZE = 14
LEGEND_FONTSIZE = 16

PALETTE = {
    "blue": "#81AFD4",
    "pale_blue": "#CAE0E8",
    "red": "#DE7C7D",
    "peach": "#FBCFBB",
}

ACTION_TYPES = [
    "stop_with_retrieval",
    "run_pso",
    "run_cmaes",
    "expand_top_seeds",
    "switch_domain_prior",
    "stop_with_best_mse",
    "stop_fail",
]

ACTION_LABELS = {
    "stop_with_retrieval": "stop\nretrieval",
    "run_pso": "run\nPSO",
    "run_cmaes": "run\nCMA-ES",
    "expand_top_seeds": "expand\nseeds",
    "switch_domain_prior": "switch\ndomain",
    "stop_with_best_mse": "stop\nbest MSE",
    "stop_fail": "stop\nfail",
}

REPLAY_RUNS = [
    {
        "replay_scale": "100-query",
        "seed_setting": "seed1->3",
        "run_dir": "phase3_deepseek_seed1_expand_to_seed3_100_v1",
    },
    {
        "replay_scale": "100-query",
        "seed_setting": "seed3->10",
        "run_dir": "phase3_deepseek_expand_100_v1",
    },
    {
        "replay_scale": "500-query",
        "seed_setting": "seed1->3",
        "run_dir": "phase3_deepseek_seed1_expand_to_seed3_500_v1",
    },
    {
        "replay_scale": "500-query",
        "seed_setting": "seed3->10",
        "run_dir": "phase3_deepseek_seed3_expand_to_seed10_500_v1",
    },
]


def setup_basic_matplotlib() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "Arial"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, file_stem: str) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout.*")
        fig.tight_layout()
    if SAVE_PNG:
        fig.savefig(OUTPUT_DIR / f"{file_stem}.png", dpi=SAVE_DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(OUTPUT_DIR / f"{file_stem}.pdf", dpi=SAVE_DPI, bbox_inches="tight")


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Phase 3 replay file: {path}")
    return pd.read_csv(path)


def read_summary(run_dir: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / run_dir / "planner_offline_summary.csv"
    return read_csv_checked(path)


def read_summary_value(run_dir: str, domain: str, column: str) -> float:
    df = read_summary(run_dir)
    row = df[df["domain"] == domain]
    if row.empty:
        raise ValueError(f"{run_dir} has no row for domain={domain}")
    if column not in row.columns:
        raise KeyError(f"{run_dir} has no column {column}")
    return float(pd.to_numeric(row[column], errors="coerce").iloc[0])


def style_axis(
    ax: plt.Axes,
    *,
    label_fontsize: int = LABEL_FONTSIZE,
    tick_fontsize: int = TICK_FONTSIZE,
    axis_linewidth: float = 1.0,
    grid_alpha: float = 0.28,
    y_lim: tuple[float, float] | None = None,
) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=grid_alpha, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(axis_linewidth)
    ax.spines["bottom"].set_linewidth(axis_linewidth)
    ax.tick_params(axis="both", labelsize=tick_fontsize, width=axis_linewidth, length=4)
    ax.xaxis.label.set_size(label_fontsize)
    ax.yaxis.label.set_size(label_fontsize)
    if y_lim is not None:
        ax.set_ylim(*y_lim)


def add_value_labels(ax: plt.Axes, bars, *, fontsize: int = VALUE_FONTSIZE, fmt: str = "{:.0f}", offset: float = 0.8) -> None:
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


def load_all_action_usage_data() -> pd.DataFrame:
    rows = []
    for run in REPLAY_RUNS:
        summary = read_summary(run["run_dir"])
        for _, row in summary.iterrows():
            action_counts = {
                action: int(pd.to_numeric(row.get(f"action_count_{action}", 0), errors="coerce"))
                for action in ACTION_TYPES
            }
            total_actions = sum(action_counts.values())
            num_queries = int(row["num_queries"])
            for action, count in action_counts.items():
                rows.append(
                    {
                        "replay_scale": run["replay_scale"],
                        "seed_setting": run["seed_setting"],
                        "setting": f"{run['replay_scale']} {run['seed_setting']}",
                        "domain": str(row["domain"]),
                        "action_type": action,
                        "action_label": ACTION_LABELS[action].replace("\n", " "),
                        "count": count,
                        "proportion_of_actions": count / total_actions if total_actions else 0.0,
                        "count_per_query": count / num_queries if num_queries else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def load_expand_count_data() -> pd.DataFrame:
    df = load_all_action_usage_data()
    df = df[df["action_type"] == "expand_top_seeds"].copy()
    df["domain"] = df["domain"].map({"source": "Source", "target_h": "Target-H", "target_self": "Target-Self"})
    df["setting"] = df["replay_scale"] + "\n" + df["seed_setting"]
    df = df.rename(columns={"count": "expand_actions"})
    return df[["setting", "domain", "expand_actions"]]


def load_expand_effectiveness_data() -> pd.DataFrame:
    rows = [
        {
            "setting": "100-query\nseed1->3",
            "strict": read_summary_value("phase3_deepseek_seed1_strict_100_v1", "target_h", "success_rate_final"),
            "expand": read_summary_value("phase3_deepseek_seed1_expand_to_seed3_100_v1", "target_h", "success_rate_final"),
        },
        {
            "setting": "100-query\nseed3->10",
            "strict": read_summary_value("phase3_deepseek_strict_100_v1", "target_h", "success_rate_final"),
            "expand": read_summary_value("phase3_deepseek_expand_100_v1", "target_h", "success_rate_final"),
        },
        {
            "setting": "500-query\nseed1->3",
            "strict": read_summary_value("phase3_deepseek_seed1_strict_500_v1", "target_h", "success_rate_final"),
            "expand": read_summary_value("phase3_deepseek_seed1_expand_to_seed3_500_v1", "target_h", "success_rate_final"),
        },
        {
            "setting": "500-query\nseed3->10",
            "strict": read_summary_value("phase3_deepseek_seed3_strict_500_v1", "target_h", "success_rate_final"),
            "expand": read_summary_value("phase3_deepseek_seed3_expand_to_seed10_500_v1", "target_h", "success_rate_final"),
        },
    ]
    df = pd.DataFrame(rows)
    df["net_gain"] = df["expand"] - df["strict"]
    return df


def draw_full_action_heatmap(ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(9.2, 6.2))
    else:
        fig = ax.figure

    df = load_all_action_usage_data()
    domain_labels = {"source": "Source", "target_h": "Target-H", "target_self": "Target-Self"}
    row_order = []
    y_labels = []
    for run in REPLAY_RUNS:
        for domain in ["source", "target_h", "target_self"]:
            row_order.append((run["replay_scale"], run["seed_setting"], domain))
            y_labels.append(f"{run['replay_scale']} {run['seed_setting']}  {domain_labels[domain]}")

    matrix = []
    for replay_scale, seed_setting, domain in row_order:
        sub = df[
            (df["replay_scale"] == replay_scale)
            & (df["seed_setting"] == seed_setting)
            & (df["domain"] == domain)
        ].set_index("action_type")
        matrix.append([float(sub.loc[action, "count"]) for action in ACTION_TYPES])
    values = np.asarray(matrix, dtype=float)

    cmap = LinearSegmentedColormap.from_list("phase3_action_usage", ["#FFFFFF", PALETTE["pale_blue"], PALETTE["blue"]])
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=max(1, float(values.max())))

    ax.set_xticks(np.arange(len(ACTION_TYPES)))
    ax.set_xticklabels([ACTION_LABELS[action] for action in ACTION_TYPES], fontsize=13, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=13)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            count = values[y, x]
            text_color = "white" if count > values.max() * 0.55 else "#222222"
            ax.text(x, y, f"{int(count)}", ha="center", va="center", fontsize=12, color=text_color)

    for boundary in [2.5, 5.5, 8.5]:
        ax.axhline(boundary, color="#BDBDBD", linewidth=1.0)

    ax.set_title("Full DeepSeek action usage", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=12, width=0.8, length=3)
    cbar.set_label("Action count", fontsize=14)
    return fig, ax


def draw_expand_count_panel(ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.2, 5.6))
    else:
        fig = ax.figure

    df = load_expand_count_data()
    settings = df["setting"].drop_duplicates().tolist()
    domains = ["Source", "Target-H", "Target-Self"]
    colors = {"Source": PALETTE["blue"], "Target-H": PALETTE["peach"], "Target-Self": PALETTE["red"]}

    x = np.arange(len(settings))
    width = 0.24
    offsets = [-width, 0.0, width]
    for domain, offset in zip(domains, offsets):
        sub = df[df["domain"] == domain].set_index("setting").loc[settings]
        bars = ax.bar(
            x + offset,
            sub["expand_actions"].astype(float).to_numpy(),
            width=width,
            label=domain,
            color=colors[domain],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        add_value_labels(ax, bars, fontsize=VALUE_FONTSIZE, fmt="{:.0f}", offset=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(settings, rotation=15, ha="right")
    ax.set_ylabel("Expand actions")
    ax.set_title("DeepSeek expand action usage", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=14)
    style_axis(ax, y_lim=(0, 82))
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper left", ncol=1)
    return fig, ax


def draw_expand_effectiveness_panel(ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.2, 5.6))
    else:
        fig = ax.figure

    df = load_expand_effectiveness_data()
    x = np.arange(len(df))
    width = 0.32

    bars_strict = ax.bar(
        x - width / 2,
        df["strict"],
        width=width,
        label="Strict",
        color=PALETTE["blue"],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    bars_expand = ax.bar(
        x + width / 2,
        df["expand"],
        width=width,
        label="Expand",
        color=PALETTE["peach"],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    for bars, x_shift, y_offset in ((bars_strict, -0.055, 0.006), (bars_expand, 0.055, 0.012)):
        for bar in bars:
            height = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2 + x_shift,
                height + y_offset,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=VALUE_FONTSIZE,
                color="#222222",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(df["setting"].tolist(), rotation=15, ha="right")
    ax.set_ylabel("Target-H success rate")
    ax.set_title("Expand action effectiveness", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=14)
    style_axis(ax, y_lim=(0.76, 0.95))
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper left", ncol=1)
    return fig, ax


def draw_composite() -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(1, 3, figsize=(24.0, 6.3), gridspec_kw={"width_ratios": [1.35, 1.0, 1.0], "wspace": 0.34})
    draw_full_action_heatmap(axes[0])
    draw_expand_count_panel(axes[1])
    draw_expand_effectiveness_panel(axes[2])
    return fig, axes


def write_source_data() -> None:
    load_all_action_usage_data().to_csv(OUTPUT_DIR / "panel_a_full_action_usage_source_data.csv", index=False)
    load_expand_count_data().to_csv(OUTPUT_DIR / "panel_b_expand_action_count_source_data.csv", index=False)
    load_expand_effectiveness_data().to_csv(OUTPUT_DIR / "panel_c_expand_effectiveness_source_data.csv", index=False)


def main() -> None:
    setup_basic_matplotlib()
    ensure_output_dir()

    fig_a, _ = draw_full_action_heatmap()
    save_figure(fig_a, "panel_a_full_action_usage_heatmap")

    fig_b, _ = draw_expand_count_panel()
    save_figure(fig_b, "panel_b_deepseek_expand_action_count")

    fig_c, _ = draw_expand_effectiveness_panel()
    save_figure(fig_c, "panel_c_expand_action_effectiveness")

    fig_all, _ = draw_composite()
    save_figure(fig_all, "figS_phase3_deepseek_action_usage")

    write_source_data()

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")

    print(f"Saved Phase 3 SI panels to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
