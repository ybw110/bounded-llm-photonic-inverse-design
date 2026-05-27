from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from my_llm_framework.plotting.publication_style import (
    DOMAIN_COLORS,
    DOMAIN_COLORS as _DOMAIN_COLORS,
    METHOD_COLORS,
    MORANDI,
    add_panel_label,
    bar_label,
    color_cycle,
    save_figure,
    set_publication_style,
    style_axis,
)


DOMAIN_LABELS = {
    "source": "Source",
    "target_h": "Target-H",
    "target_self": "Target-Self",
}

METHOD_LABELS = {
    "raw_spectrum": "Raw spectrum",
    "physics_summary": "Physics summary",
    "domain_aware": "Domain-aware",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _panel_label(ax: plt.Axes, label: str) -> None:
    if hasattr(ax, "text2D"):
        ax.text2D(-0.12, 1.06, label, transform=ax.transAxes, fontsize=plt.rcParams["figure.labelsize"], fontweight="bold")
    else:
        add_panel_label(ax, label)


PanelDraw = tuple[Callable[[plt.Axes], None], str | None]


def save_panel(
    output_dir: Path,
    stem: str,
    draw: Callable[[plt.Axes], None],
    *,
    formats: list[str],
    figsize: tuple[float, float] = (7, 5),
    projection: str | None = None,
) -> None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=projection)
    draw(ax)
    save_figure(fig, output_dir, stem, formats)


def save_composite(
    output_dir: Path,
    stem: str,
    draws: list[PanelDraw],
    *,
    formats: list[str],
    figsize: tuple[float, float] = (20, 14),
) -> None:
    rows = 2 if len(draws) > 2 else 1
    cols = int(np.ceil(len(draws) / rows))
    fig = plt.figure(figsize=figsize)
    for i, (draw, projection) in enumerate(draws, start=1):
        ax = fig.add_subplot(rows, cols, i, projection=projection)
        draw(ax)
        _panel_label(ax, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i - 1])
    save_figure(fig, output_dir, stem, formats)


def _memory_bank(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(root / "outputs/phase01_memory_bank_v1_train.npz", allow_pickle=True)
    return data["domains"], data["summary_features"], data["summary_names"]


def draw_memory_bar(ax: plt.Axes, root: Path) -> None:
    domains, _, _ = _memory_bank(root)
    labels, counts = np.unique(domains, return_counts=True)
    order = ["source", "target_h", "target_self"]
    count_map = dict(zip(labels, counts))
    values = [count_map.get(d, 0) for d in order]
    bars = ax.bar([DOMAIN_LABELS[d] for d in order], values, color=[DOMAIN_COLORS[d] for d in order], edgecolor="none")
    ax.set_title("Memory-bank composition")
    ax.set_ylabel("Training samples")
    style_axis(ax)
    bar_label(ax, bars, fmt="{:.0f}")


def draw_memory_donut(ax: plt.Axes, root: Path) -> None:
    domains, _, _ = _memory_bank(root)
    labels, counts = np.unique(domains, return_counts=True)
    order = ["source", "target_h", "target_self"]
    values = [dict(zip(labels, counts)).get(d, 0) for d in order]
    colors = [DOMAIN_COLORS[d] for d in order]
    wedges, _, autotexts = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct="%1.1f%%",
        pctdistance=0.78,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
    )
    for text in autotexts:
        text.set_fontsize(plt.rcParams["ytick.labelsize"])
    ax.legend(wedges, [DOMAIN_LABELS[d] for d in order], loc="center left", bbox_to_anchor=(0.96, 0.5), frameon=False)
    ax.set_title("Domain proportion")


def draw_summary_heatmap(ax: plt.Axes, root: Path) -> None:
    domains, features, names = _memory_bank(root)
    names = [str(n) for n in names]
    order = ["source", "target_h", "target_self"]
    means = []
    for domain in order:
        means.append(np.nanmean(features[domains == domain], axis=0))
    mat = np.vstack(means)
    mat = (mat - np.nanmean(mat, axis=0, keepdims=True)) / (np.nanstd(mat, axis=0, keepdims=True) + 1e-12)
    im = ax.imshow(mat, aspect="auto", cmap="PuOr", vmin=-1.5, vmax=1.5)
    ax.set_title("Physics-summary domain shift")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([DOMAIN_LABELS[d] for d in order])
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.grid(False)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="z-scored mean")


def draw_summary_pca_3d(ax: plt.Axes, root: Path) -> None:
    domains, features, _ = _memory_bank(root)
    rng = np.random.default_rng(13)
    idx = np.arange(len(domains))
    if len(idx) > 6000:
        idx = rng.choice(idx, size=6000, replace=False)
    x = features[idx].astype(float)
    x = (x - np.nanmean(x, axis=0, keepdims=True)) / (np.nanstd(x, axis=0, keepdims=True) + 1e-12)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    pcs = x @ vt[:3].T
    for domain in ["source", "target_h", "target_self"]:
        mask = domains[idx] == domain
        ax.scatter(
            pcs[mask, 0],
            pcs[mask, 1],
            pcs[mask, 2],
            s=8,
            alpha=0.42,
            color=_DOMAIN_COLORS[domain],
            label=DOMAIN_LABELS[domain],
            depthshade=False,
        )
    ax.set_title("3D PCA of physics summaries")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, alpha=0.22)


def plot_memory(root: Path, output_dir: Path, formats: list[str]) -> None:
    phase_dir = output_dir / "dataset_memory"
    draws: list[tuple[str, Callable[[plt.Axes], None], str | None]] = [
        ("panel_A_memory_count_bar", lambda ax: draw_memory_bar(ax, root), None),
        ("panel_B_memory_domain_donut", lambda ax: draw_memory_donut(ax, root), None),
        ("panel_C_summary_feature_heatmap", lambda ax: draw_summary_heatmap(ax, root), None),
        ("panel_D_summary_pca_3d", lambda ax: draw_summary_pca_3d(ax, root), "3d"),
    ]
    for name, draw, projection in draws:
        save_panel(phase_dir, name, draw, formats=formats, projection=projection)
    save_composite(phase_dir, "composite_dataset_memory_diagnostics", [(draw, projection) for _, draw, projection in draws], formats=formats)


def draw_pnn_curves(ax: plt.Axes, root: Path) -> None:
    for domain in ["source", "target_h", "target_self"]:
        hist = _read_csv(root / f"outputs/phase2_models/pnn_forward/{domain}/training_history.csv")
        color = DOMAIN_COLORS[domain]
        ax.plot(_num(hist["epoch"]), _num(hist["train_loss"]), color=color, linewidth=2.4, alpha=0.45, linestyle="--")
        ax.plot(_num(hist["epoch"]), _num(hist["val_loss"]), color=color, linewidth=3.0)
    ax.set_title("PNN forward training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    style_axis(ax)
    _add_training_curve_legend(ax, include_test=False)


def draw_inn_curves(ax: plt.Axes, root: Path) -> None:
    for domain in ["source", "target_h", "target_self"]:
        hist = _read_csv(root / f"outputs/phase2_models/inn_inverse/{domain}/training_history.csv")
        color = DOMAIN_COLORS[domain]
        ax.plot(_num(hist["epoch"]), _num(hist["train_param_mse"]), color=color, linewidth=2.4, alpha=0.4, linestyle="--")
        ax.plot(_num(hist["epoch"]), _num(hist["val_param_mse"]), color=color, linewidth=3.0)
        if "test_param_mse" in hist.columns:
            ax.plot(_num(hist["epoch"]), _num(hist["test_param_mse"]), color=color, linewidth=1.8, alpha=0.75, linestyle=":")
    ax.set_title("INN inverse training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Parameter MSE")
    ax.set_yscale("log")
    style_axis(ax)
    _add_training_curve_legend(ax, include_test=True)


def _add_training_curve_legend(ax: plt.Axes, *, include_test: bool) -> None:
    domain_handles = [
        Line2D([0], [0], color=DOMAIN_COLORS[domain], linewidth=4, label=DOMAIN_LABELS[domain])
        for domain in ["source", "target_h", "target_self"]
    ]
    split_handles = [
        Line2D([0], [0], color=MORANDI["charcoal"], linewidth=3.0, linestyle="--", alpha=0.55, label="Training"),
        Line2D([0], [0], color=MORANDI["charcoal"], linewidth=3.0, linestyle="-", label="Validation"),
    ]
    if include_test:
        split_handles.append(Line2D([0], [0], color=MORANDI["charcoal"], linewidth=2.2, linestyle=":", alpha=0.75, label="Test"))
    domain_legend = ax.legend(handles=domain_handles, frameon=False, loc="upper right", title="Domain")
    ax.add_artist(domain_legend)
    ax.legend(handles=split_handles, frameon=False, loc="lower left", title="Curve")


def draw_best_loss_bar(ax: plt.Axes, root: Path) -> None:
    pnn = _read_csv(root / "outputs/phase2_models/pnn_forward/training_summary_all.csv")
    inn = _read_csv(root / "outputs/phase2_models/inn_inverse/training_summary_all.csv")
    domains = ["source", "target_h", "target_self"]
    x = np.arange(len(domains))
    width = 0.34
    pnn_vals = [float(pnn[pnn["domain"] == d]["best_val_loss"].iloc[0]) for d in domains]
    inn_vals = [float(inn[inn["domain"] == d]["best_val_param_mse"].iloc[0]) for d in domains]
    ax.bar(x - width / 2, pnn_vals, width, color=MORANDI["deep_teal"], label="PNN forward")
    ax.bar(x + width / 2, inn_vals, width, color=MORANDI["lavender"], label="INN inverse")
    ax.set_title("Best validation loss")
    ax.set_ylabel("Validation MSE")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in domains])
    style_axis(ax)
    ax.legend(frameon=False)


def draw_difficulty_radar(ax: plt.Axes, root: Path) -> None:
    pnn = _read_csv(root / "outputs/phase2_models/pnn_forward/training_summary_all.csv")
    inn = _read_csv(root / "outputs/phase2_models/inn_inverse/training_summary_all.csv")
    domains = ["source", "target_h", "target_self"]
    labels = [DOMAIN_LABELS[d] for d in domains]
    pnn_vals = np.array([float(pnn[pnn["domain"] == d]["best_val_loss"].iloc[0]) for d in domains])
    inn_vals = np.array([float(inn[inn["domain"] == d]["best_val_param_mse"].iloc[0]) for d in domains])
    values = np.vstack([np.log10(pnn_vals), np.log10(inn_vals)])
    values = (values - values.min()) / (values.max() - values.min() + 1e-12)
    theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    theta = np.r_[theta, theta[0]]
    for vals, name, color in [
        (values[0], "PNN forward", MORANDI["deep_teal"]),
        (values[1], "INN inverse", MORANDI["lavender"]),
    ]:
        vals = np.r_[vals, vals[0]]
        ax.plot(theta, vals, color=color, linewidth=3.0, label=name)
        ax.fill(theta, vals, color=color, alpha=0.18)
    ax.set_title("Relative model difficulty")
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.28, 1.12))
    ax.grid(True, alpha=0.24)


def plot_training(root: Path, output_dir: Path, formats: list[str]) -> None:
    phase_dir = output_dir / "model_training"
    draws: list[tuple[str, Callable[[plt.Axes], None], str | None]] = [
        ("panel_A_pnn_training_loss", lambda ax: draw_pnn_curves(ax, root), None),
        ("panel_B_inn_training_loss", lambda ax: draw_inn_curves(ax, root), None),
        ("panel_C_best_validation_loss", lambda ax: draw_best_loss_bar(ax, root), None),
        ("panel_D_model_difficulty_radar", lambda ax: draw_difficulty_radar(ax, root), "polar"),
    ]
    for name, draw, projection in draws:
        save_panel(phase_dir, name, draw, formats=formats, projection=projection)
    save_composite(phase_dir, "composite_model_training_diagnostics", [(draw, projection) for _, draw, projection in draws], formats=formats)


def _fewshot_x(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    labels = sorted(df["fewshot_label"].astype(str).drop_duplicates(), key=lambda x: float("inf") if x == "full" else float(x))
    pos = {label: i for i, label in enumerate(labels)}
    df["fewshot_x"] = df["fewshot_label"].astype(str).map(pos)
    return df


def draw_uncertainty_curve(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str, ylabel: str, *, methods: list[str] | None = None, logy: bool = False) -> None:
    sub = df.copy()
    if methods is not None:
        sub = sub[sub["method"].isin(methods)].copy()
    sub = _fewshot_x(sub)
    std_col = metric.replace("_mean", "_std")
    for method, group in sub.groupby("method", sort=False):
        group = group.sort_values("fewshot_x")
        x = _num(group["fewshot_x"]).to_numpy(dtype=float)
        y = _num(group[metric]).to_numpy(dtype=float)
        std = _num(group[std_col]).fillna(0).to_numpy(dtype=float) if std_col in group.columns else np.zeros_like(y)
        color = METHOD_COLORS.get(method, MORANDI["warm_gray"])
        ax.plot(x, y, color=color, marker="o", linewidth=3.0, markersize=8, label=METHOD_LABELS.get(method, method))
        low = y - std
        high = y + std
        if "success" in metric or "rate" in metric:
            low = np.clip(low, 0, 1)
            high = np.clip(high, 0, 1)
        else:
            low = np.maximum(low, np.finfo(float).tiny)
        ax.fill_between(x, low, high, color=color, alpha=0.2, linewidth=0)
    ticks = sub[["fewshot_x", "fewshot_label"]].drop_duplicates().sort_values("fewshot_x")
    ax.set_xticks(ticks["fewshot_x"])
    ax.set_xticklabels(ticks["fewshot_label"].astype(str))
    ax.set_title(title)
    ax.set_xlabel("Target samples")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ylim = (0, 1.04) if "Success" in ylabel or "rate" in ylabel else None
    style_axis(ax, ylim=ylim)
    ax.legend(frameon=False)


def plot_phase4_uncertainty(root: Path, output_dir: Path, formats: list[str]) -> None:
    phase_dir = output_dir / "phase4_uncertainty"
    main = _read_csv(root / "outputs/phase4_fewshot_retrieval_methods_full_with_upper_v1/fewshot_retrieval_summary_aggregate.csv")
    h_ext = _read_csv(root / "outputs/phase4_fewshot_retrieval_target_h_extended_v1/fewshot_retrieval_summary_aggregate.csv")
    target_h = main[main["target_domain"] == "target_h"]
    target_self = main[main["target_domain"] == "target_self"]
    draws: list[tuple[str, Callable[[plt.Axes], None], str | None]] = [
        (
            "panel_A_target_h_success_mean_std",
            lambda ax: draw_uncertainty_curve(ax, target_h, "success_at_50_mean", "Target-H Success@50", "Success@50"),
            None,
        ),
        (
            "panel_B_target_h_topology_success_mean_std",
            lambda ax: draw_uncertainty_curve(
                ax,
                h_ext[h_ext["method"].isin(["raw_spectrum", "domain_aware"])],
                "target_domain_success_at_50_mean",
                "Target-H topology-valid Success@50",
                "Target-domain Success@50",
            ),
            None,
        ),
        (
            "panel_C_target_self_success_mean_std",
            lambda ax: draw_uncertainty_curve(ax, target_self, "target_domain_success_at_50_mean", "Target-Self recovery", "Target-domain Success@50"),
            None,
        ),
        (
            "panel_D_target_h_mse_mean_std",
            lambda ax: draw_uncertainty_curve(
                ax,
                h_ext[h_ext["method"].isin(["raw_spectrum", "domain_aware"])],
                "mean_top1_amplitude_mse_mean",
                "Target-H top1 spectral MSE",
                "Mean top1 amplitude MSE",
                logy=True,
            ),
            None,
        ),
    ]
    for name, draw, projection in draws:
        save_panel(phase_dir, name, draw, formats=formats, projection=projection)
    save_composite(phase_dir, "composite_phase4_uncertainty_mean_std", [(draw, projection) for _, draw, projection in draws], formats=formats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/figures/supplementary_style_v1")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--font-scale", choices=["large", "compact"], default="large")
    args = parser.parse_args()

    set_publication_style(args.font_scale)
    root = Path.cwd()
    output_dir = Path(args.output_dir)
    plot_memory(root, output_dir, args.formats)
    plot_training(root, output_dir, args.formats)
    plot_phase4_uncertainty(root, output_dir, args.formats)
    print(f"Saved supplementary publication-style figures to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
