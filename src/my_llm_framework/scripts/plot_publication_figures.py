from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from my_llm_framework.plotting.publication_style import (
    DOMAIN_COLORS,
    METHOD_COLORS,
    MORANDI,
    POLICY_COLORS,
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


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _read_csv(path: Path) -> pd.DataFrame:
    if not _exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def save_panel(
    output_dir: Path,
    stem: str,
    draw: Callable[[plt.Axes], None],
    *,
    formats: list[str],
    figsize: tuple[float, float] = (7, 5),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    draw(ax)
    save_figure(fig, output_dir, stem, formats)


def save_composite(
    output_dir: Path,
    stem: str,
    draws: list[Callable[[plt.Axes], None]],
    *,
    formats: list[str],
    figsize: tuple[float, float] = (20, 14),
) -> None:
    n = len(draws)
    rows = 2 if n > 2 else 1
    cols = int(np.ceil(n / rows))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_arr = np.asarray(axes).reshape(-1)
    for label, ax, draw in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", axes_arr, draws):
        draw(ax)
        add_panel_label(ax, label)
    for ax in axes_arr[len(draws) :]:
        ax.axis("off")
    save_figure(fig, output_dir, stem, formats)


def draw_memory_counts(ax: plt.Axes, root: Path) -> None:
    df = _read_csv(root / "outputs/figures/memory_bank_v1/domain_counts.csv")
    df["label"] = df["domain"].map(DOMAIN_LABELS).fillna(df["domain"])
    colors = [DOMAIN_COLORS.get(d, MORANDI["warm_gray"]) for d in df["domain"]]
    bars = ax.bar(df["label"], _num(df["count"]), color=colors, edgecolor="none", width=0.68)
    ax.set_title("Memory-bank composition")
    ax.set_ylabel("Training samples")
    ax.set_xlabel("")
    style_axis(ax)
    bar_label(ax, bars, fmt="{:.0f}")


def draw_retrieval_topk(ax: plt.Axes, root: Path, *, method: str = "raw_spectrum") -> None:
    df = _read_csv(root / "outputs/phase01_retrieval_baselines_v1/baseline_summary.csv")
    sub = df[df["method"] == method].copy()
    budgets = [1, 5, 10, 50]
    for domain, group in sub.groupby("domain", sort=False):
        y = [float(group.iloc[0][f"success_at_{b}"]) for b in budgets]
        ax.plot(
            budgets,
            y,
            marker="o",
            linewidth=3.0,
            markersize=8,
            color=DOMAIN_COLORS.get(domain, MORANDI["warm_gray"]),
            label=DOMAIN_LABELS.get(domain, domain),
        )
    ax.set_title(f"Retrieval coverage ({METHOD_LABELS.get(method, method)})")
    ax.set_xlabel("Retrieval budget B")
    ax.set_ylabel("Success@B")
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    style_axis(ax, ylim=(0, 1.04))
    ax.legend(frameon=False)


def draw_target_h_retrieval_methods(ax: plt.Axes, root: Path) -> None:
    df = _read_csv(root / "outputs/phase01_retrieval_baselines_v1/baseline_summary.csv")
    sub = df[df["domain"] == "target_h"].copy()
    methods = ["raw_spectrum", "physics_summary", "domain_aware"]
    x = np.arange(len(methods))
    width = 0.35
    bars1 = ax.bar(
        x - width / 2,
        [float(sub[sub["method"] == m]["success_at_1"].iloc[0]) for m in methods],
        width,
        color=MORANDI["dusty_blue"],
        label="Success@1",
    )
    bars2 = ax.bar(
        x + width / 2,
        [float(sub[sub["method"] == m]["success_at_50"].iloc[0]) for m in methods],
        width,
        color=MORANDI["salmon"],
        label="Success@50",
    )
    ax.set_title("Target-H retrieval gap")
    ax.set_ylabel("Success rate")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=25, ha="right")
    style_axis(ax, ylim=(0, 1.04))
    ax.legend(frameon=False)
    bar_label(ax, bars1)
    bar_label(ax, bars2)


def draw_pnn_training_curves(ax: plt.Axes, root: Path) -> None:
    for domain in ["source", "target_h", "target_self"]:
        path = root / f"outputs/phase2_models/pnn_forward/{domain}/training_history.csv"
        if not path.exists():
            continue
        hist = _read_csv(path)
        ax.plot(
            _num(hist["epoch"]),
            _num(hist["val_loss"]),
            linewidth=3.0,
            color=DOMAIN_COLORS.get(domain, MORANDI["warm_gray"]),
            label=DOMAIN_LABELS.get(domain, domain),
        )
    ax.set_title("PNN forward training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_yscale("log")
    style_axis(ax)
    ax.legend(frameon=False)


def plot_phase01(root: Path, output_dir: Path, formats: list[str]) -> None:
    phase_dir = output_dir / "phase01"
    draws = [
        lambda ax: draw_memory_counts(ax, root),
        lambda ax: draw_retrieval_topk(ax, root, method="raw_spectrum"),
        lambda ax: draw_target_h_retrieval_methods(ax, root),
        lambda ax: draw_pnn_training_curves(ax, root),
    ]
    names = [
        "panel_A_memory_bank_counts",
        "panel_B_raw_retrieval_topk",
        "panel_C_target_h_retrieval_gap",
        "panel_D_pnn_training_curves",
    ]
    for name, draw in zip(names, draws):
        save_panel(phase_dir, name, draw, formats=formats)
    save_composite(phase_dir, "composite_phase01_memory_retrieval", draws, formats=formats)


def _phase2_plot_df(root: Path) -> pd.DataFrame:
    df = _read_csv(root / "outputs/phase2_baseline_summary_final_with_inn_heuristic.csv")
    keep = [
        ("retrieval_only", "raw_spectrum", "Retrieval"),
        ("retrieval_plus_optimizer", "pnn_forward+pso", "Retrieval+PSO"),
        ("retrieval_plus_optimizer", "pnn_forward+cmaes", "Retrieval+CMA-ES"),
        ("direct_inverse", "inn_inverse+pnn_forward", "INN"),
        ("heuristic_planner", "retrieval_then_pso_then_cmaes", "Rule planner"),
    ]
    rows = []
    for family, method, label in keep:
        sub = df[(df["baseline_family"] == family) & (df["method"] == method)].copy()
        sub["plot_label"] = label
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def draw_phase2_success(ax: plt.Axes, root: Path) -> None:
    df = _phase2_plot_df(root)
    labels = list(df["plot_label"].drop_duplicates())
    domains = ["source", "target_h", "target_self"]
    x = np.arange(len(labels))
    width = 0.24
    for i, domain in enumerate(domains):
        values = []
        for label in labels:
            row = df[(df["plot_label"] == label) & (df["domain"] == domain)]
            values.append(float(row["success_rate_final"].iloc[0]) if not row.empty else np.nan)
        ax.bar(
            x + (i - 1) * width,
            values,
            width,
            color=DOMAIN_COLORS[domain],
            label=DOMAIN_LABELS[domain],
        )
    ax.set_title("Strong non-LLM baselines")
    ax.set_ylabel("Final success rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    style_axis(ax, ylim=(0, 1.04))
    ax.legend(frameon=False)


def draw_phase2_mse(ax: plt.Axes, root: Path) -> None:
    df = _phase2_plot_df(root)
    sub = df[df["domain"] == "target_h"].copy()
    labels = list(sub["plot_label"])
    values = _num(sub["mean_final_mse"]).to_numpy()
    bars = ax.bar(labels, values, color=color_cycle(len(labels)), edgecolor="none")
    ax.set_title("Target-H final MSE")
    ax.set_ylabel("Mean final amplitude MSE")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    style_axis(ax)
    bar_label(ax, bars, fmt="{:.3g}", fontsize=14)


def draw_phase2_acceptance(ax: plt.Axes, root: Path) -> None:
    df = _read_csv(root / "outputs/phase2_baseline_summary_final_with_inn_heuristic.csv")
    sub = df[df["baseline_family"] == "retrieval_plus_optimizer"].copy()
    sub = sub[sub["method"].isin(["pnn_forward+pso", "pnn_forward+cmaes"])]
    labels = ["PNN+PSO", "PNN+CMA-ES"]
    domains = ["source", "target_h", "target_self"]
    x = np.arange(len(labels))
    width = 0.24
    for i, domain in enumerate(domains):
        values = []
        for method in ["pnn_forward+pso", "pnn_forward+cmaes"]:
            row = sub[(sub["domain"] == domain) & (sub["method"] == method)]
            values.append(float(row["accepted_optimization_rate"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (i - 1) * width, values, width, color=DOMAIN_COLORS[domain], label=DOMAIN_LABELS[domain])
    ax.set_title("Conservative acceptance")
    ax.set_ylabel("Accepted optimization rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    style_axis(ax, ylim=(0, 1.04))
    ax.legend(frameon=False)


def draw_phase2_violin(ax: plt.Axes, root: Path) -> None:
    specs = [
        ("Source PSO", root / "outputs/phase2_retrieval_pnn_pso_source_full_strong30_nosr/retrieval_optimizer_query_results.csv"),
        ("Target-H PSO", root / "outputs/phase2_retrieval_pnn_pso_h_full_strong30_nosr/retrieval_optimizer_query_results.csv"),
        ("Target-H CMA-ES", root / "outputs/phase2_retrieval_pnn_cmaes_h_full_strong30_nosr/retrieval_optimizer_query_results.csv"),
        ("Target-Self PSO", root / "outputs/phase2_retrieval_pnn_pso_self_full_strong30_nosr/retrieval_optimizer_query_results.csv"),
    ]
    data = []
    labels = []
    for label, path in specs:
        if not path.exists():
            continue
        df = _read_csv(path)
        vals = _num(df["final_amplitude_mse"]).dropna().to_numpy()
        vals = vals[vals > 0]
        if len(vals) > 3000:
            rng = np.random.default_rng(7)
            vals = rng.choice(vals, size=3000, replace=False)
        data.append(np.log10(vals))
        labels.append(label)
    parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.75)
    for body, color in zip(parts["bodies"], color_cycle(len(data))):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.82)
    for key in ["cmedians", "cbars", "cmins", "cmaxes"]:
        if key in parts:
            parts[key].set_color(MORANDI["charcoal"])
            parts[key].set_linewidth(1.4)
    ax.set_title("Query-level MSE distribution")
    ax.set_ylabel("log10(final amplitude MSE)")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    style_axis(ax)


def plot_phase2(root: Path, output_dir: Path, formats: list[str]) -> None:
    phase_dir = output_dir / "phase2"
    draws = [
        lambda ax: draw_phase2_success(ax, root),
        lambda ax: draw_phase2_mse(ax, root),
        lambda ax: draw_phase2_acceptance(ax, root),
        lambda ax: draw_phase2_violin(ax, root),
    ]
    names = [
        "panel_A_baseline_success",
        "panel_B_target_h_mse",
        "panel_C_acceptance_rate",
        "panel_D_query_mse_violin",
    ]
    for name, draw in zip(names, draws):
        save_panel(phase_dir, name, draw, formats=formats)
    save_composite(phase_dir, "composite_phase2_strong_baselines", draws, formats=formats)


def _planner_summary(root: Path, rel: str, label: str) -> pd.DataFrame:
    df = _read_csv(root / rel)
    df["plot_label"] = label
    return df


def draw_phase3_success(ax: plt.Axes, root: Path) -> None:
    specs = [
        ("outputs/phase3_rule_seed1_strict_100_v1/planner_offline_summary.csv", "Rule strict"),
        ("outputs/phase3_deepseek_seed1_strict_100_v1/planner_offline_summary.csv", "DeepSeek strict"),
        ("outputs/phase3_rule_seed1_expand_to_seed3_100_v1/planner_offline_summary.csv", "Rule expand"),
        ("outputs/phase3_deepseek_seed1_expand_to_seed3_100_v1/planner_offline_summary.csv", "DeepSeek expand"),
    ]
    df = pd.concat([_planner_summary(root, rel, label) for rel, label in specs], ignore_index=True)
    sub = df[df["domain"] == "target_h"].copy()
    colors = [POLICY_COLORS["Rule"], POLICY_COLORS["DeepSeek"], POLICY_COLORS["Rule"], POLICY_COLORS["DeepSeek"]]
    bars = ax.bar(sub["plot_label"], _num(sub["success_rate_final"]), color=colors, edgecolor="none")
    ax.set_title("Target-H planner success")
    ax.set_ylabel("Final success rate")
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(sub["plot_label"], rotation=25, ha="right")
    style_axis(ax, ylim=(0, 1.04))
    bar_label(ax, bars)


def draw_phase3_mse(ax: plt.Axes, root: Path) -> None:
    specs = [
        ("outputs/phase3_rule_seed1_strict_100_v1/planner_offline_summary.csv", "Rule strict"),
        ("outputs/phase3_deepseek_seed1_strict_100_v1/planner_offline_summary.csv", "DeepSeek strict"),
        ("outputs/phase3_rule_seed1_expand_to_seed3_100_v1/planner_offline_summary.csv", "Rule expand"),
        ("outputs/phase3_deepseek_seed1_expand_to_seed3_100_v1/planner_offline_summary.csv", "DeepSeek expand"),
    ]
    df = pd.concat([_planner_summary(root, rel, label) for rel, label in specs], ignore_index=True)
    sub = df[df["domain"] == "target_h"].copy()
    colors = [POLICY_COLORS["Rule"], POLICY_COLORS["DeepSeek"], POLICY_COLORS["Rule"], POLICY_COLORS["DeepSeek"]]
    bars = ax.bar(sub["plot_label"], _num(sub["mean_final_amplitude_mse"]), color=colors, edgecolor="none")
    ax.set_title("Target-H planner MSE")
    ax.set_ylabel("Mean final amplitude MSE")
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(sub["plot_label"], rotation=25, ha="right")
    style_axis(ax)
    bar_label(ax, bars, fmt="{:.3g}", fontsize=14)


def draw_phase3_action_counts(ax: plt.Axes, root: Path) -> None:
    df = _read_csv(root / "outputs/phase3_deepseek_seed1_expand_to_seed3_500_v1/planner_offline_summary.csv")
    sub = df[df["domain"] == "target_h"].iloc[0]
    actions = ["stop_with_retrieval", "run_pso", "run_cmaes", "expand_top_seeds", "stop_with_best_mse"]
    values = [float(sub[f"action_count_{a}"]) / float(sub["num_queries"]) for a in actions]
    bars = ax.bar([a.replace("_", "\n") for a in actions], values, color=color_cycle(len(actions)), edgecolor="none")
    ax.set_title("DeepSeek action usage")
    ax.set_ylabel("Actions per query")
    style_axis(ax)
    bar_label(ax, bars)


def draw_phase3_oracle_gap(ax: plt.Axes, root: Path) -> None:
    df = _read_csv(root / "outputs/phase3_oracle_gap_full_v1/planner_oracle_gap_summary.csv")
    colors = [DOMAIN_COLORS.get(d, MORANDI["warm_gray"]) for d in df["domain"]]
    bars = ax.bar(df["domain"].map(DOMAIN_LABELS), _num(df["success_rate_gap"]), color=colors, edgecolor="none")
    ax.set_title("Oracle gap analysis")
    ax.set_ylabel("Success-rate gap")
    style_axis(ax)
    bar_label(ax, bars, fmt="{:.3f}")


def plot_phase3(root: Path, output_dir: Path, formats: list[str]) -> None:
    phase_dir = output_dir / "phase3"
    draws = [
        lambda ax: draw_phase3_success(ax, root),
        lambda ax: draw_phase3_mse(ax, root),
        lambda ax: draw_phase3_action_counts(ax, root),
        lambda ax: draw_phase3_oracle_gap(ax, root),
    ]
    names = [
        "panel_A_planner_success",
        "panel_B_planner_mse",
        "panel_C_deepseek_action_usage",
        "panel_D_oracle_gap",
    ]
    for name, draw in zip(names, draws):
        save_panel(phase_dir, name, draw, formats=formats)
    save_composite(phase_dir, "composite_phase3_planner_analysis", draws, formats=formats)


def _fewshot_x(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    labels = list(df["fewshot_label"].astype(str).drop_duplicates())
    labels = sorted(labels, key=lambda x: float("inf") if x == "full" else float(x))
    pos = {label: i for i, label in enumerate(labels)}
    df["fewshot_x"] = df["fewshot_label"].astype(str).map(pos)
    return df


def _line_metric(ax: plt.Axes, sub: pd.DataFrame, metric: str, *, label_col: str = "method") -> None:
    sub = _fewshot_x(sub)
    std_col = metric.replace("_mean", "_std") if metric.endswith("_mean") else f"{metric}_std"
    for key, group in sub.groupby(label_col, sort=False):
        group = group.sort_values("fewshot_x")
        color = METHOD_COLORS.get(str(key), MORANDI["warm_gray"])
        x = _num(group["fewshot_x"]).to_numpy(dtype=float)
        y = _num(group[metric]).to_numpy(dtype=float)
        ax.plot(x, y, marker="o", markersize=8, linewidth=3, color=color, label=METHOD_LABELS.get(str(key), str(key)))
        if std_col in group.columns:
            std = _num(group[std_col]).fillna(0).to_numpy(dtype=float)
            low = y - std
            high = y + std
            if "success" in metric or "rate" in metric:
                low = np.clip(low, 0, 1)
                high = np.clip(high, 0, 1)
            else:
                low = np.maximum(low, np.finfo(float).tiny)
            ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
    ticks = sub[["fewshot_x", "fewshot_label"]].drop_duplicates().sort_values("fewshot_x")
    ax.set_xticks(ticks["fewshot_x"])
    ax.set_xticklabels(ticks["fewshot_label"].astype(str))


def draw_phase4_success(ax: plt.Axes, root: Path) -> None:
    df = _read_csv(root / "outputs/phase4_fewshot_retrieval_methods_full_with_upper_v1/fewshot_retrieval_summary_aggregate.csv")
    sub = df[(df["target_domain"] == "target_h") & (df["method"].isin(["raw_spectrum", "domain_aware"]))]
    _line_metric(ax, sub, "success_at_50_mean")
    ax.set_title("Target-H few-shot recovery")
    ax.set_xlabel("Target samples")
    ax.set_ylabel("Success@50")
    style_axis(ax, ylim=(0, 1.04))
    ax.legend(frameon=False)


def draw_phase4_target_valid(ax: plt.Axes, root: Path) -> None:
    df = _read_csv(root / "outputs/phase4_fewshot_retrieval_target_h_extended_v1/fewshot_retrieval_summary_aggregate.csv")
    sub = df[df["method"].isin(["raw_spectrum", "domain_aware"])]
    _line_metric(ax, sub, "target_domain_success_at_50_mean")
    ax.set_title("Target-H topology-valid recovery")
    ax.set_xlabel("Target samples")
    ax.set_ylabel("Target-domain Success@50")
    style_axis(ax, ylim=(0, 1.04))
    ax.legend(frameon=False)


def draw_phase4_self(ax: plt.Axes, root: Path) -> None:
    df = _read_csv(root / "outputs/phase4_fewshot_retrieval_methods_full_with_upper_v1/fewshot_retrieval_summary_aggregate.csv")
    sub = df[df["target_domain"] == "target_self"]
    _line_metric(ax, sub, "target_domain_success_at_50_mean")
    ax.set_title("Target-Self fast calibration")
    ax.set_xlabel("Target samples")
    ax.set_ylabel("Target-domain Success@50")
    style_axis(ax, ylim=(0, 1.04))
    ax.legend(frameon=False)


def draw_phase4_mse(ax: plt.Axes, root: Path) -> None:
    df = _read_csv(root / "outputs/phase4_fewshot_retrieval_target_h_extended_v1/fewshot_retrieval_summary_aggregate.csv")
    sub = df[df["method"].isin(["raw_spectrum", "domain_aware"])]
    _line_metric(ax, sub, "mean_top1_amplitude_mse_mean")
    ax.set_title("Target-H spectral quality")
    ax.set_xlabel("Target samples")
    ax.set_ylabel("Mean top1 amplitude MSE")
    ax.set_yscale("log")
    style_axis(ax)
    ax.legend(frameon=False)


def plot_phase4(root: Path, output_dir: Path, formats: list[str]) -> None:
    phase_dir = output_dir / "phase4"
    draws = [
        lambda ax: draw_phase4_success(ax, root),
        lambda ax: draw_phase4_target_valid(ax, root),
        lambda ax: draw_phase4_self(ax, root),
        lambda ax: draw_phase4_mse(ax, root),
    ]
    names = [
        "panel_A_target_h_success",
        "panel_B_target_h_validity",
        "panel_C_target_self_recovery",
        "panel_D_target_h_mse",
    ]
    for name, draw in zip(names, draws):
        save_panel(phase_dir, name, draw, formats=formats)
    save_composite(phase_dir, "composite_phase4_fewshot_ood", draws, formats=formats)


def _policy_category(label: str) -> str:
    lower = label.lower()
    if "source" in lower:
        return "Source-only"
    if "oracle" in lower:
        return "Oracle"
    if "deepseek" in lower:
        return "DeepSeek"
    return "Rule"


def _load_phase4b_runs(root: Path, use_500: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    suffix = "500" if use_500 and (root / "outputs/phase4b_ood_planner_deepseek_500_v1/ood_recovery_planner_summary.csv").exists() else "100"
    specs = [
        ("Source-only", f"outputs/phase4b_ood_planner_source_only_{suffix}_v1"),
        ("Rule h200", f"outputs/phase4b_ood_planner_shift_aware_{suffix}_v1"),
        ("DeepSeek", f"outputs/phase4b_ood_planner_deepseek_{suffix}_v2" if suffix == "100" else f"outputs/phase4b_ood_planner_deepseek_{suffix}_v1"),
        ("Oracle", f"outputs/phase4b_ood_planner_oracle_{suffix}_v1"),
    ]
    s_frames = []
    a_frames = []
    for order, (label, rel) in enumerate(specs):
        summary = root / rel / "ood_recovery_planner_summary.csv"
        action = root / rel / "ood_recovery_planner_action_summary.csv"
        if not summary.exists():
            continue
        s = _read_csv(summary)
        s["plot_label"] = label
        s["plot_order"] = order
        s["policy_category"] = _policy_category(label)
        s_frames.append(s)
        if action.exists():
            a = _read_csv(action)
            a["plot_label"] = label
            a["plot_order"] = order
            a["policy_category"] = _policy_category(label)
            a_frames.append(a)
    return pd.concat(s_frames, ignore_index=True), pd.concat(a_frames, ignore_index=True)


def draw_phase4b_success(ax: plt.Axes, root: Path, *, use_500: bool) -> None:
    df, _ = _load_phase4b_runs(root, use_500)
    sub = df[df["target_domain"] == "target_h"].sort_values("plot_order")
    colors = [POLICY_COLORS.get(c, MORANDI["warm_gray"]) for c in sub["policy_category"]]
    bars = ax.bar(sub["plot_label"], _num(sub["target_domain_success_at_50"]), color=colors, edgecolor="none")
    ax.set_title("LLM OOD recovery")
    ax.set_ylabel("Target-H target-domain Success@50")
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(sub["plot_label"], rotation=25, ha="right")
    style_axis(ax, ylim=(0, 1.04))
    bar_label(ax, bars)


def draw_phase4b_cost(ax: plt.Axes, root: Path, *, use_500: bool) -> None:
    df, _ = _load_phase4b_runs(root, use_500)
    sub = df[df["target_domain"] == "target_h"].sort_values("plot_order")
    colors = [POLICY_COLORS.get(c, MORANDI["warm_gray"]) for c in sub["policy_category"]]
    bars = ax.bar(sub["plot_label"], _num(sub["mean_fewshot_cost"]), color=colors, edgecolor="none")
    ax.set_title("Sample cost")
    ax.set_ylabel("Mean target samples used")
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(sub["plot_label"], rotation=25, ha="right")
    style_axis(ax)
    bar_label(ax, bars, fmt="{:.0f}")


def draw_phase4b_efficiency(ax: plt.Axes, root: Path, *, use_500: bool) -> None:
    df, _ = _load_phase4b_runs(root, use_500)
    sub = df[df["target_domain"] == "target_h"].sort_values("plot_order")
    for _, row in sub.iterrows():
        color = POLICY_COLORS.get(row["policy_category"], MORANDI["warm_gray"])
        ax.scatter(float(row["mean_fewshot_cost"]), float(row["target_domain_success_at_50"]), s=180, color=color, edgecolor="white", linewidth=1.2)
        ax.annotate(row["plot_label"], (float(row["mean_fewshot_cost"]), float(row["target_domain_success_at_50"])), xytext=(8, 8), textcoords="offset points", fontsize=14)
    ax.set_title("Sample-efficient planning")
    ax.set_xlabel("Mean target samples used")
    ax.set_ylabel("Target-H target-domain Success@50")
    style_axis(ax, ylim=(0, 1.04))


def draw_phase4b_action(ax: plt.Axes, root: Path, *, use_500: bool) -> None:
    _, action = _load_phase4b_runs(root, use_500)
    sub = action[(action["target_domain"] == "target_h") & (action["category"] == "fewshot")].copy()
    if sub.empty:
        ax.axis("off")
        return
    order = list(sub[["plot_label", "plot_order"]].drop_duplicates().sort_values("plot_order")["plot_label"])
    sub["value_num"] = _num(sub["value"]).fillna(np.inf)
    values = list(sub[["value", "value_num"]].drop_duplicates().sort_values(["value_num", "value"])["value"].astype(str))
    pivot = sub.pivot_table(index="plot_label", columns="value", values="proportion", aggfunc="sum").fillna(0)
    pivot = pivot.reindex(order)
    pivot = pivot[[v for v in values if v in pivot.columns]]
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for value, color in zip(pivot.columns, color_cycle(len(pivot.columns))):
        ax.bar(x, pivot[value].to_numpy(), bottom=bottom, color=color, label=str(value), edgecolor="none")
        bottom += pivot[value].to_numpy()
    ax.set_title("Target-H few-shot actions")
    ax.set_ylabel("Proportion")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right")
    style_axis(ax, ylim=(0, 1.04))
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))


def plot_phase4b(root: Path, output_dir: Path, formats: list[str], *, use_500: bool) -> None:
    phase_dir = output_dir / ("phase4b_500" if use_500 else "phase4b_100")
    draws = [
        lambda ax: draw_phase4b_success(ax, root, use_500=use_500),
        lambda ax: draw_phase4b_cost(ax, root, use_500=use_500),
        lambda ax: draw_phase4b_efficiency(ax, root, use_500=use_500),
        lambda ax: draw_phase4b_action(ax, root, use_500=use_500),
    ]
    names = [
        "panel_A_ood_planner_success",
        "panel_B_ood_planner_sample_cost",
        "panel_C_ood_planner_efficiency",
        "panel_D_ood_planner_actions",
    ]
    for name, draw in zip(names, draws):
        save_panel(phase_dir, name, draw, formats=formats)
    save_composite(phase_dir, "composite_phase4b_llm_ood_planner", draws, formats=formats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/figures/publication_style_v1")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--font-scale", choices=["large", "compact"], default="large")
    parser.add_argument("--use-500-phase4b", action="store_true", help="Use Phase 4b 500-query outputs if available")
    args = parser.parse_args()

    root = Path.cwd()
    set_publication_style(args.font_scale)
    output_dir = Path(args.output_dir)

    plot_phase01(root, output_dir, args.formats)
    plot_phase2(root, output_dir, args.formats)
    plot_phase3(root, output_dir, args.formats)
    plot_phase4(root, output_dir, args.formats)
    plot_phase4b(root, output_dir, args.formats, use_500=args.use_500_phase4b)
    print(f"Saved publication-style figures to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
