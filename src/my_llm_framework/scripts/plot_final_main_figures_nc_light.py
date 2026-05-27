from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from my_llm_framework.plotting.publication_style import (
    DOMAIN_COLORS,
    MORANDI,
    add_panel_label,
    bar_label,
    save_figure,
    set_publication_style,
    style_axis,
)
from my_llm_framework.scripts.plot_phase5_cst_case_study import _load_available_cases, _plot_case
from my_llm_framework.scripts.plot_phase5_hero_field import _read_field_ascii
from my_llm_framework.scripts.plot_publication_figures import (
    draw_phase2_acceptance,
    draw_phase2_mse,
    draw_phase2_success,
    draw_phase2_violin,
    draw_phase3_action_counts,
    draw_phase3_mse,
    draw_phase3_oracle_gap,
    draw_phase3_success,
    draw_phase4_self,
    draw_phase4_success,
    draw_phase4_target_valid,
    draw_phase4b_efficiency,
    draw_phase4b_success,
)


PanelDraw = Callable[[plt.Axes], None]


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _save_panel(output_dir: Path, stem: str, draw: PanelDraw, formats: list[str], figsize: tuple[float, float] = (7, 5)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    draw(ax)
    save_figure(fig, output_dir, stem, formats=formats)


def _save_grid(
    output_dir: Path,
    stem: str,
    panels: list[tuple[str, PanelDraw]],
    formats: list[str],
    *,
    ncols: int,
    figsize: tuple[float, float],
) -> None:
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = axes.ravel()
    for idx, (label, draw) in enumerate(panels):
        ax = flat[idx]
        draw(ax)
        add_panel_label(ax, label.lower())
    for ax in flat[len(panels) :]:
        ax.axis("off")
    save_figure(fig, output_dir, stem, formats=formats)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#555555") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=2.0,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def _rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    *,
    fc: str,
    ec: str = "#FFFFFF",
    fontsize: int = 13,
    weight: str = "bold",
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            wh[0],
            wh[1],
            boxstyle="round,pad=0.03,rounding_size=0.04",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.5,
        )
    )
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight)


def _setup_schematic(ax: plt.Axes, title: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, pad=8)


def draw_fig1_domains(ax: plt.Axes) -> None:
    _setup_schematic(ax, "OOD metasurface domains")
    xs = [0.17, 0.50, 0.83]
    labels = ["Source\ncylinder", "Target-H\ntopology", "Target-Self\nfrequency/material"]
    colors = [DOMAIN_COLORS["source"], DOMAIN_COLORS["target_h"], DOMAIN_COLORS["target_self"]]
    for x, label, color in zip(xs, labels, colors):
        ax.add_patch(Rectangle((x - 0.12, 0.36), 0.24, 0.22, facecolor="#F7F7F5", edgecolor="#B0B0B0", linewidth=1.5))
        if "Target-H" in label:
            ax.add_patch(Rectangle((x - 0.08, 0.40), 0.16, 0.04, facecolor=color, edgecolor="none"))
            ax.add_patch(Rectangle((x - 0.08, 0.50), 0.16, 0.04, facecolor=color, edgecolor="none"))
            ax.add_patch(Rectangle((x - 0.02, 0.40), 0.04, 0.14, facecolor=color, edgecolor="none"))
        else:
            ax.add_patch(Circle((x, 0.47), 0.075, facecolor=color, edgecolor="#555555", linewidth=1.2))
        ax.text(x, 0.22, label, ha="center", va="center", fontsize=12, fontweight="bold")
    _arrow(ax, (0.31, 0.47), (0.39, 0.47), MORANDI["charcoal"])
    _arrow(ax, (0.63, 0.47), (0.71, 0.47), MORANDI["charcoal"])
    ax.text(0.50, 0.78, "Interpolate -> topology shift -> material/frequency shift", ha="center", fontsize=12)


def draw_fig1_evidence(ax: plt.Axes) -> None:
    _setup_schematic(ax, "Structured evidence, not free text")
    t = np.linspace(0, 1, 120)
    for i, c in enumerate([MORANDI["pale_blue"], MORANDI["seafoam"], MORANDI["peach"]]):
        y = 0.60 - 0.10 * i - 0.20 * np.exp(-((t - (0.35 + 0.12 * i)) / 0.10) ** 2)
        ax.plot(0.08 + 0.35 * t, y, color=c, linewidth=2.5)
    _rounded_box(ax, (0.55, 0.68), (0.34, 0.12), "retrieval top-k", fc=MORANDI["pale_blue"])
    _rounded_box(ax, (0.55, 0.50), (0.34, 0.12), "physics summary", fc=MORANDI["peach"])
    _rounded_box(ax, (0.55, 0.32), (0.34, 0.12), "domain evidence", fc=MORANDI["mint"])
    _arrow(ax, (0.44, 0.50), (0.55, 0.56), MORANDI["charcoal"])
    ax.text(0.25, 0.23, "Query spectrum", ha="center", fontsize=13, fontweight="bold")


def draw_fig1_optimizer(ax: plt.Axes) -> None:
    _setup_schematic(ax, "Physics-constrained optimizer")
    boxes = [
        ((0.08, 0.52), (0.22, 0.16), "retrieval\nseeds", MORANDI["pale_blue"]),
        ((0.39, 0.52), (0.22, 0.16), "PNN\nforward", MORANDI["mint"]),
        ((0.70, 0.62), (0.22, 0.12), "PSO", MORANDI["dusty_blue"]),
        ((0.70, 0.40), (0.22, 0.12), "CMA-ES", MORANDI["lavender"]),
    ]
    for xy, wh, text, color in boxes:
        _rounded_box(ax, xy, wh, text, fc=color)
    _arrow(ax, (0.30, 0.60), (0.39, 0.60))
    _arrow(ax, (0.61, 0.60), (0.70, 0.68))
    _arrow(ax, (0.61, 0.56), (0.70, 0.46))
    ax.text(0.50, 0.22, "continuous amplitude MSE objective\nconservative acceptance", ha="center", fontsize=12)


def draw_fig1_critic(ax: plt.Axes) -> None:
    _setup_schematic(ax, "Numeric critic")
    reasons = [
        ("already_successful", MORANDI["mint"]),
        ("amplitude_mse_too_high", MORANDI["salmon"]),
        ("center_frequency_shifted", MORANDI["peach"]),
        ("bandwidth_mismatch", MORANDI["lavender"]),
        ("optimizer_no_gain", MORANDI["warm_gray"]),
    ]
    y = 0.78
    for reason, color in reasons:
        _rounded_box(ax, (0.12, y - 0.055), (0.76, 0.09), reason, fc=color, fontsize=11, weight="normal")
        y -= 0.13
    ax.text(0.50, 0.10, "failure reason -> planner observation", ha="center", fontsize=13, fontweight="bold")


def draw_fig1_planner(ax: plt.Axes) -> None:
    _setup_schematic(ax, "LLM planner action JSON")
    _rounded_box(ax, (0.05, 0.60), (0.32, 0.18), "Observation\nMSE, success,\nbudget, critic", fc=MORANDI["pale_blue"], fontsize=11)
    ax.text(
        0.66,
        0.54,
        '{\n  "action": "expand_top_seeds",\n  "fewshot_size": 100,\n  "reason": "target evidence"\n}',
        family="monospace",
        fontsize=9.5,
        va="center",
        ha="center",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8F8F8", "edgecolor": MORANDI["charcoal"]},
    )
    _arrow(ax, (0.37, 0.66), (0.46, 0.58), MORANDI["charcoal"])
    ax.text(0.50, 0.16, "constrained actions only", ha="center", fontsize=13, fontweight="bold")


def draw_fig1_validation(ax: plt.Axes) -> None:
    _setup_schematic(ax, "Closed-loop validation")
    _rounded_box(ax, (0.07, 0.56), (0.22, 0.13), "offline\nreplay", fc=MORANDI["pale_blue"], fontsize=12)
    _rounded_box(ax, (0.39, 0.56), (0.22, 0.13), "unit-cell\nCST", fc=MORANDI["peach"], fontsize=12)
    _rounded_box(ax, (0.71, 0.56), (0.22, 0.13), "metalens\nhero demo", fc=MORANDI["mint"], fontsize=12)
    _arrow(ax, (0.29, 0.62), (0.39, 0.62))
    _arrow(ax, (0.61, 0.62), (0.71, 0.62))
    ax.text(0.50, 0.25, "Phase 4 = LLM evidence allocation\nPhase 5 = physical validation\nPhase 6 = qualitative device extension", ha="center", fontsize=12)


def plot_fig1(root: Path, output_dir: Path, formats: list[str]) -> None:
    panels = [
        ("a", draw_fig1_domains),
        ("b", draw_fig1_evidence),
        ("c", draw_fig1_optimizer),
        ("d", draw_fig1_critic),
        ("e", draw_fig1_planner),
        ("f", draw_fig1_validation),
    ]
    panel_dir = output_dir / "panels"
    for label, draw in panels:
        _save_panel(panel_dir, f"fig1_panel_{label}", draw, formats, figsize=(6, 5))
    _save_grid(output_dir, "fig1_workflow_ood_taxonomy", panels, formats, ncols=3, figsize=(18, 10.8))


def plot_fig2(root: Path, output_dir: Path, formats: list[str]) -> None:
    panels: list[tuple[str, PanelDraw]] = [
        ("a", lambda ax: draw_phase2_success(ax, root)),
        ("b", lambda ax: draw_phase2_mse(ax, root)),
        ("c", lambda ax: draw_phase2_acceptance(ax, root)),
        ("d", lambda ax: draw_phase2_violin(ax, root)),
    ]
    panel_dir = output_dir / "panels"
    names = ["baseline_success", "target_h_mse", "acceptance_rate", "query_mse_violin"]
    for (label, draw), name in zip(panels, names):
        _save_panel(panel_dir, f"fig2_panel_{label}_{name}", draw, formats)
    _save_grid(output_dir, "fig2_strong_non_llm_baselines", panels, formats, ncols=2, figsize=(15, 10.8))


def _draw_phase3_oracle_annotated(ax: plt.Axes, root: Path) -> None:
    draw_phase3_oracle_gap(ax, root)
    ax.text(
        0.52,
        0.95,
        "near-oracle\nsaturated baseline",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": MORANDI["salmon"], "alpha": 0.9},
    )


def plot_fig3(root: Path, output_dir: Path, formats: list[str]) -> None:
    panels: list[tuple[str, PanelDraw]] = [
        ("a", lambda ax: draw_phase3_success(ax, root)),
        ("b", lambda ax: draw_phase3_mse(ax, root)),
        ("c", lambda ax: draw_phase3_action_counts(ax, root)),
        ("d", lambda ax: _draw_phase3_oracle_annotated(ax, root)),
    ]
    panel_dir = output_dir / "panels"
    names = ["planner_success", "planner_mse", "deepseek_actions", "oracle_gap"]
    for (label, draw), name in zip(panels, names):
        _save_panel(panel_dir, f"fig3_panel_{label}_{name}", draw, formats)
    _save_grid(output_dir, "fig3_planner_oracle_gap", panels, formats, ncols=2, figsize=(15, 10.8))


def _load_phase4b_summaries(root: Path) -> pd.DataFrame:
    specs = [
        ("Source-only", "outputs/phase4b_ood_planner_source_only_500_v1/ood_recovery_planner_summary.csv", MORANDI["warm_gray"]),
        ("Fixed H-200", "outputs/phase4b_ood_planner_shift_aware_500_v1/ood_recovery_planner_summary.csv", MORANDI["dusty_blue"]),
        ("Oracle", "outputs/phase4b_ood_planner_oracle_500_v1/ood_recovery_planner_summary.csv", MORANDI["deep_teal"]),
    ]
    rows = []
    for order, (label, rel, color) in enumerate(specs):
        df = pd.read_csv(root / rel)
        df["plot_label"] = label
        df["plot_order"] = order
        df["plot_color"] = color
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def _phase4b_repeat_summary_path(root: Path) -> Path:
    return root / "outputs/phase4b_ood_planner_multimodel_500_repeats_summary_v1/phase4b_multimodel_repeat_summary.csv"


def _load_phase4b_repeat_summary(root: Path) -> pd.DataFrame | None:
    path = _phase4b_repeat_summary_path(root)
    if not path.exists():
        return None
    return pd.read_csv(path)


def _phase4b_target_h_plot_rows(root: Path) -> pd.DataFrame:
    base = _load_phase4b_summaries(root)
    target_h = base[base["target_domain"] == "target_h"].copy()
    rows = []
    for _, row in target_h.iterrows():
        label = row["plot_label"]
        rows.append(
            {
                "plot_label": label,
                "plot_order": {"Source-only": 0, "Fixed H-200": 1, "Oracle": 4}.get(label, 99),
                "plot_color": row["plot_color"],
                "success_mean": float(row["success_at_50"]),
                "success_std": 0.0,
                "target_success_mean": float(row["target_domain_success_at_50"]),
                "target_success_std": 0.0,
                "top1_target_mean": float(row["top1_target_domain_rate"]),
                "best_mse_mean": float(row["mean_final_best_amplitude_mse_at_budget"]),
                "best_mse_std": 0.0,
                "sample_mean": float(row["mean_fewshot_cost"]),
                "sample_std": 0.0,
                "fallback_mean": float(row.get("fallback_or_invalid_action_rate", 0.0)),
                "fallback_std": 0.0,
            }
        )

    repeats = _load_phase4b_repeat_summary(root)
    if repeats is not None:
        llm_specs = {
            "DeepSeek": (2, MORANDI["salmon"]),
            "GLM-5.1": (3, MORANDI["mint"]),
        }
        for label, (order, color) in llm_specs.items():
            sub = repeats[(repeats["run_label"] == label) & (repeats["target_domain"] == "target_h")]
            if sub.empty:
                continue
            row = sub.iloc[0]
            rows.append(
                {
                    "plot_label": f"{label}\n(n=3)",
                    "plot_order": order,
                    "plot_color": color,
                    "success_mean": float(row["success_at_50_mean"]),
                    "success_std": float(row["success_at_50_std"]),
                    "target_success_mean": float(row["target_domain_success_at_50_mean"]),
                    "target_success_std": float(row["target_domain_success_at_50_std"]),
                    "top1_target_mean": float(row["top1_target_domain_rate_mean"]),
                    "best_mse_mean": float(row["mean_final_best_amplitude_mse_at_budget_mean"]),
                    "best_mse_std": float(row["mean_final_best_amplitude_mse_at_budget_std"]),
                    "sample_mean": float(row["mean_fewshot_cost_mean"]),
                    "sample_std": float(row["mean_fewshot_cost_std"]),
                    "fallback_mean": float(row["fallback_or_invalid_action_rate_mean"]),
                    "fallback_std": float(row["fallback_or_invalid_action_rate_std"]),
                }
            )
    else:
        deep_path = root / "outputs/phase4b_ood_planner_deepseek_500_v1/ood_recovery_planner_summary.csv"
        if deep_path.exists():
            deep = pd.read_csv(deep_path)
            deep = deep[deep["target_domain"] == "target_h"].iloc[0]
            rows.append(
                {
                    "plot_label": "DeepSeek",
                    "plot_order": 2,
                    "plot_color": MORANDI["salmon"],
                    "success_mean": float(deep["success_at_50"]),
                    "success_std": 0.0,
                    "target_success_mean": float(deep["target_domain_success_at_50"]),
                    "target_success_std": 0.0,
                    "top1_target_mean": float(deep["top1_target_domain_rate"]),
                    "best_mse_mean": float(deep["mean_final_best_amplitude_mse_at_budget"]),
                    "best_mse_std": 0.0,
                    "sample_mean": float(deep["mean_fewshot_cost"]),
                    "sample_std": 0.0,
                    "fallback_mean": float(deep.get("fallback_or_invalid_action_rate", 0.0)),
                    "fallback_std": 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values("plot_order").reset_index(drop=True)


def _draw_phase4b_core_success(ax: plt.Axes, root: Path) -> None:
    sub = _phase4b_target_h_plot_rows(root)
    x = np.arange(len(sub))
    colors = list(sub["plot_color"])
    bars = ax.bar(
        x,
        sub["success_mean"],
        yerr=sub["success_std"],
        width=0.50,
        color=colors,
        edgecolor="none",
        capsize=4,
        label="Success@50",
    )
    ax.errorbar(
        x,
        sub["target_success_mean"],
        yerr=sub["target_success_std"],
        fmt="D",
        markersize=8,
        color=MORANDI["charcoal"],
        capsize=4,
        label="Target-domain S@50",
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sub["plot_label"], rotation=20, ha="right")
    ax.set_ylabel("Target-H success rate")
    ax.set_title("Multimodel LLM recovery approaches oracle")
    style_axis(ax, ylim=(0, 1.04))
    bar_label(ax, bars, fontsize=13)
    ax.legend(frameon=False, loc="upper left")
    deep = sub[sub["plot_label"].str.contains("DeepSeek")].iloc[0]
    oracle = sub[sub["plot_label"] == "Oracle"].iloc[0]
    ax.annotate(
        f"DeepSeek target S@50\n{deep['target_success_mean']:.3f} ± {deep['target_success_std']:.3f}",
        xy=(int(deep.name), float(deep["target_success_mean"])),
        xytext=(1.45, 0.55),
        arrowprops={"arrowstyle": "->", "color": MORANDI["salmon"], "lw": 1.8},
        fontsize=13,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": MORANDI["salmon"], "alpha": 0.9},
    )


def _draw_phase4b_sample_efficiency(ax: plt.Axes, root: Path) -> None:
    sub = _phase4b_target_h_plot_rows(root)
    for _, row in sub.iterrows():
        ax.errorbar(
            row["sample_mean"],
            row["target_success_mean"],
            xerr=row["sample_std"],
            yerr=row["target_success_std"],
            fmt="o",
            markersize=12,
            color=row["plot_color"],
            markeredgecolor="white",
            markeredgewidth=1.5,
            capsize=4,
        )
        ax.text(row["sample_mean"] + 4, row["target_success_mean"] + 0.012, row["plot_label"].replace("\n", " "), fontsize=13, fontweight="bold")
    ax.axvline(200, color=MORANDI["dusty_blue"], linestyle="--", linewidth=2.0, alpha=0.8)
    ax.text(202, 0.18, "fixed\n200 shots", color=MORANDI["dusty_blue"], fontsize=12)
    ax.set_xlabel("Mean target samples used")
    ax.set_ylabel("Target-H target-domain S@50")
    ax.set_title("Sample-efficient evidence allocation")
    style_axis(ax, ylim=(0, 1.04))


def _mean_action_proportions(root: Path, dirs: list[str]) -> pd.Series:
    rows = []
    for rel in dirs:
        path = root / rel / "ood_recovery_planner_action_summary.csv"
        if not path.exists():
            continue
        action = pd.read_csv(path)
        sub = action[(action["target_domain"] == "target_h") & (action["category"] == "fewshot")].copy()
        sub["shot"] = pd.to_numeric(sub["value"], errors="coerce")
        rows.append(sub.set_index("shot")["proportion"])
    if not rows:
        return pd.Series(dtype=float)
    return pd.concat(rows, axis=1).mean(axis=1).sort_index()


def _draw_phase4b_llm_actions(ax: plt.Axes, root: Path) -> None:
    specs = {
        "DeepSeek": [
            "outputs/phase4b_ood_planner_deepseek_500_repeat1_v1",
            "outputs/phase4b_ood_planner_deepseek_500_repeat2_v1",
            "outputs/phase4b_ood_planner_deepseek_500_repeat3_v1",
        ],
        "GLM-5.1": [
            "outputs/phase4b_ood_planner_glm51_500_repeat1_v1",
            "outputs/phase4b_ood_planner_glm51_500_repeat2_v1",
            "outputs/phase4b_ood_planner_glm51_500_repeat3_v1",
        ],
    }
    shot_values = [0, 20, 50, 100, 200]
    shot_colors = {
        0: MORANDI["warm_gray"],
        20: MORANDI["pale_blue"],
        50: MORANDI["mint"],
        100: MORANDI["peach"],
        200: MORANDI["salmon"],
    }
    labels = list(specs)
    bottom = np.zeros(len(labels))
    x = np.arange(len(labels))
    for shot in shot_values:
        vals = []
        for dirs in specs.values():
            props = _mean_action_proportions(root, dirs)
            vals.append(float(props.get(shot, 0.0)))
        ax.bar(x, vals, bottom=bottom, color=shot_colors[shot], edgecolor="white", linewidth=0.8, label=str(shot))
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("LLM target-sample allocation")
    ax.set_ylabel("Mean action proportion")
    style_axis(ax, ylim=(0, 1.0))
    ax.legend(title="Few-shot samples", frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.20))
    ax.text(
        0.02,
        0.08,
        "GLM is more sample-thrifty;\nDeepSeek is more conservative",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": MORANDI["mint"], "alpha": 0.9},
    )


def plot_fig4(root: Path, output_dir: Path, formats: list[str]) -> None:
    panels: list[tuple[str, PanelDraw]] = [
        ("a", lambda ax: draw_phase4_success(ax, root)),
        ("b", lambda ax: draw_phase4_target_valid(ax, root)),
        ("c", lambda ax: draw_phase4_self(ax, root)),
        ("d", lambda ax: _draw_phase4b_core_success(ax, root)),
        ("e", lambda ax: _draw_phase4b_sample_efficiency(ax, root)),
        ("f", lambda ax: _draw_phase4b_llm_actions(ax, root)),
    ]
    panel_dir = output_dir / "panels"
    names = ["target_h_fewshot_success", "target_h_domain_validity", "target_self_recovery", "planner_success", "sample_efficiency", "llm_allocation"]
    for (label, draw), name in zip(panels, names):
        _save_panel(panel_dir, f"fig4_panel_{label}_{name}", draw, formats)
    _save_grid(output_dir, "fig4_llm_ood_recovery_core", panels, formats, ncols=3, figsize=(21, 11.2))


def _draw_phase5_mse_bars(ax: plt.Axes, cases: pd.DataFrame) -> None:
    rows = []
    for _, row in cases.iterrows():
        spectra = pd.read_csv(row["spectra_csv"])
        cst = pd.read_csv(row["cst_csv"])
        freq = spectra["frequency_thz"].to_numpy(dtype=float)
        target = spectra["target_amplitude"].to_numpy(dtype=float)
        candidate = spectra["candidate_amplitude"].to_numpy(dtype=float)
        cst_interp = np.interp(freq, cst["frequency_thz"].to_numpy(dtype=float), cst["cst_amplitude"].to_numpy(dtype=float))
        rows.append(
            {
                "label": row["case_id"].replace("case_", "").split("_q")[0],
                "Offline": float(np.mean((candidate - target) ** 2)),
                "CST": float(np.mean((cst_interp - target) ** 2)),
            }
        )
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    width = 0.36
    ax.bar(x - width / 2, df["Offline"], width=width, color=MORANDI["dusty_blue"], label="Offline")
    ax.bar(x + width / 2, df["CST"], width=width, color=MORANDI["salmon"], label="CST")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=18, ha="right")
    ax.set_ylabel("Amplitude MSE")
    ax.set_title("Offline-to-CST error")
    style_axis(ax)
    ax.legend(frameon=False)


def plot_fig5(root: Path, output_dir: Path, formats: list[str]) -> None:
    main = _load_available_cases(root / "outputs/phase5_cst_case_study_v1/cst_batch_manifest.csv")
    si = _load_available_cases(root / "outputs/phase5_cst_case_study_with_si_v1/cst_batch_manifest.csv")
    main = main.head(4).reset_index(drop=True)
    si = si.head(1).reset_index(drop=True)
    combined = pd.concat([main, si], ignore_index=True)

    panels: list[tuple[str, PanelDraw]] = []
    for idx, row in main.iterrows():
        panels.append((chr(ord("a") + idx), lambda ax, r=row: _plot_case(ax, r)))
    panels.append(("e", lambda ax: _draw_phase5_mse_bars(ax, combined)))
    if not si.empty:
        panels.append(("f", lambda ax, r=si.iloc[0]: _plot_case(ax, r)))
    panel_dir = output_dir / "panels"
    for label, draw in panels:
        _save_panel(panel_dir, f"fig5_panel_{label}", draw, formats, figsize=(7.5, 5.2))
    _save_grid(output_dir, "fig5_cst_unit_cell_validation", panels, formats, ncols=3, figsize=(21, 11.2))


def _wrap_phase(y: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    return (arr + np.pi) % (2 * np.pi) - np.pi


def _phase6_units(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "outputs/figures/phase6_device_design_comparison_v1/phase6_device_design_comparison_units.csv")


def _phase6_metrics(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "outputs/figures/phase6_device_field_comparison_fast_compact_v1/phase6_device_field_comparison_metrics.csv")


def _draw_phase6_phase_profile(ax: plt.Axes, root: Path) -> None:
    df = _phase6_units(root)
    for label, color, marker in [("Manual", MORANDI["dusty_blue"], "o"), ("DeepSeek", MORANDI["salmon"], "s")]:
        sub = df[df["run_label"] == label].sort_values("x_um")
        ax.plot(sub["x_um"], _wrap_phase(sub["selected_phase_rad"]), marker=marker, linewidth=2.4, color=color, label=label)
    ref = df[df["run_label"] == "Manual"].sort_values("x_um")
    ax.plot(ref["x_um"], _wrap_phase(ref["target_phase_rad"]), color=MORANDI["charcoal"], linewidth=2.8, linestyle="--", label="Target")
    ax.set_title("Metalens phase profile")
    ax.set_xlabel("Position x (um)")
    ax.set_ylabel("Wrapped phase (rad)")
    style_axis(ax, grid_axis="both")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))


def _draw_phase6_phase_error(ax: plt.Axes, root: Path) -> None:
    df = _phase6_units(root)
    for label, color in [("Manual", MORANDI["dusty_blue"]), ("DeepSeek", MORANDI["salmon"])]:
        sub = df[df["run_label"] == label].sort_values("x_um")
        ax.plot(sub["x_um"], np.abs(sub["phase_error_rad"]), marker="o", linewidth=2.5, color=color, label=label)
    ax.set_title("Phase error along aperture")
    ax.set_xlabel("Position x (um)")
    ax.set_ylabel("|phase error| (rad)")
    style_axis(ax)
    ax.legend(frameon=False, loc="upper right")


def _draw_phase6_amplitude(ax: plt.Axes, root: Path) -> None:
    df = _phase6_units(root)
    for label, color in [("Manual", MORANDI["dusty_blue"]), ("DeepSeek", MORANDI["salmon"])]:
        sub = df[df["run_label"] == label].sort_values("x_um")
        ax.plot(sub["x_um"], sub["selected_amplitude"], marker="o", linewidth=2.5, color=color, label=label)
    ax.set_title("Selected unit amplitude")
    ax.set_xlabel("Position x (um)")
    ax.set_ylabel("Amplitude")
    style_axis(ax, ylim=(0.45, 1.03))
    ax.legend(frameon=False, loc="lower right")


def _field_pivot(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = _read_field_ascii(path)
    pivot = df.pivot_table(index="z", columns="x", values="field_abs", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    return pivot.columns.to_numpy(float), pivot.index.to_numpy(float), pivot.to_numpy(float)


def _draw_phase6_field(ax: plt.Axes, root: Path, label: str, vmax: float) -> None:
    metrics = _phase6_metrics(root)
    row = metrics[metrics["run_label"] == label].iloc[0]
    x, z, values = _field_pivot(root / str(row["field_path"]))
    values = values / (vmax + 1e-12)
    im = ax.imshow(values, origin="lower", aspect="auto", extent=[x.min(), x.max(), z.min(), z.max()], cmap="mako" if False else "viridis", vmin=0, vmax=1)
    ax.axhline(float(row["design_focal_length_um"]), color="white", linewidth=1.8, linestyle="--")
    ax.axvline(0, color="white", linewidth=1.2, linestyle=":")
    ax.scatter([float(row["focal_plane_peak_x_um"])], [float(row["focal_plane_z_um"])], s=65, color=MORANDI["salmon"], edgecolor="white", linewidth=1.0)
    ax.set_title(f"{label} fast-mode CST field")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("z (um)")
    return im


def _draw_phase6_manual_field(ax: plt.Axes, root: Path) -> None:
    metrics = _phase6_metrics(root)
    vmax = 0.0
    for _, row in metrics.iterrows():
        _, _, values = _field_pivot(root / str(row["field_path"]))
        vmax = max(vmax, float(np.nanmax(values)))
    im = _draw_phase6_field(ax, root, "Manual", vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Normalized |E|")


def _draw_phase6_deepseek_field(ax: plt.Axes, root: Path) -> None:
    metrics = _phase6_metrics(root)
    vmax = 0.0
    for _, row in metrics.iterrows():
        _, _, values = _field_pivot(root / str(row["field_path"]))
        vmax = max(vmax, float(np.nanmax(values)))
    im = _draw_phase6_field(ax, root, "DeepSeek", vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Normalized |E|")


def _draw_phase6_metrics(ax: plt.Axes, root: Path) -> None:
    metrics = _phase6_metrics(root)
    labels = list(metrics["run_label"])
    x = np.arange(len(labels))
    width = 0.32
    ax.bar(x - width / 2, metrics["focal_plane_peak_field"], width, color=MORANDI["dusty_blue"], label="Focal-plane peak")
    ax.bar(x + width / 2, metrics["focal_window_peak_field"], width, color=MORANDI["salmon"], label="Focal-window peak")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Field magnitude")
    ax.set_title("Qualitative device-level metrics")
    style_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    manual = metrics[metrics["run_label"] == "Manual"].iloc[0]
    deepseek = metrics[metrics["run_label"] == "DeepSeek"].iloc[0]
    ax.text(
        0.98,
        0.95,
        "fast-mode qualitative CST\n"
        f"peak offset: {manual['focal_plane_peak_offset_um']:.1f} -> {deepseek['focal_plane_peak_offset_um']:.1f} um",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": MORANDI["charcoal"], "alpha": 0.9},
    )


def plot_fig6(root: Path, output_dir: Path, formats: list[str]) -> None:
    panels: list[tuple[str, PanelDraw]] = [
        ("a", lambda ax: _draw_phase6_phase_profile(ax, root)),
        ("b", lambda ax: _draw_phase6_phase_error(ax, root)),
        ("c", lambda ax: _draw_phase6_amplitude(ax, root)),
        ("d", lambda ax: _draw_phase6_manual_field(ax, root)),
        ("e", lambda ax: _draw_phase6_deepseek_field(ax, root)),
        ("f", lambda ax: _draw_phase6_metrics(ax, root)),
    ]
    panel_dir = output_dir / "panels"
    names = ["phase_profile", "phase_error", "amplitude", "manual_field", "deepseek_field", "metrics"]
    for (label, draw), name in zip(panels, names):
        _save_panel(panel_dir, f"fig6_panel_{label}_{name}", draw, formats, figsize=(7, 5))
    _save_grid(output_dir, "fig6_metalens_hero_demo", panels, formats, ncols=3, figsize=(21, 11.2))


def write_readme(output_dir: Path) -> None:
    text = """# Final Main Figures for NC / Light Style Draft

This folder contains regenerated main figures with a unified publication style.

Backend: Python / matplotlib only. Editable text is preserved in SVG and PDF exports.

## Files

- `fig1_workflow_ood_taxonomy.*`: framework schematic and OOD taxonomy.
- `fig2_strong_non_llm_baselines.*`: strong non-LLM baseline evidence.
- `fig3_planner_oracle_gap.*`: planner analysis and near-oracle saturation.
- `fig4_llm_ood_recovery_core.*`: main LLM contribution, OOD evidence allocation.
- `fig5_cst_unit_cell_validation.*`: CST unit-cell validation and SI cautionary case.
- `fig6_metalens_hero_demo.*`: qualitative fast-mode metalens hero demo.
- `panels/`: standalone subpanels for each main figure.

## Recommended manuscript use

Use Fig. 4 as the central LLM claim. Fig. 3 should be read as mechanistic evidence that the strong fixed workflow is already close to the oracle ceiling. Fig. 6 should be described as qualitative fast-mode CST, not as final device-level metrology.

## Figure contracts

1. Fig. 1 core conclusion: the LLM is a constrained evidence-guided planner embedded in a physics workflow, not a free-form geometry generator. Archetype: schematic-led composite.
2. Fig. 2 core conclusion: retrieval-seeded PNN optimisation is already a strong non-LLM design engine. Archetype: quantitative grid.
3. Fig. 3 core conclusion: LLM-rule similarity under strong baselines reflects near-oracle saturation rather than planner failure. Archetype: quantitative grid.
4. Fig. 4 core conclusion: the LLM's primary contribution is few-shot OOD evidence allocation, where it approaches oracle allocation with lower target-sample cost than a fixed rule. Archetype: quantitative grid with hero evidence.
5. Fig. 5 core conclusion: selected offline candidates remain physically meaningful in CST unit-cell validation. Archetype: spectra plate plus quantification.
6. Fig. 6 core conclusion: the workflow can be extended to device-level unit selection, but current CST is qualitative fast-mode validation. Archetype: mixed modality hero demo.

## Key data sources

- `outputs/phase01_retrieval_baselines_v1`
- `outputs/phase2_baseline_summary_final_with_inn_heuristic.csv`
- `outputs/phase3_oracle_gap_full_v1`
- `outputs/phase4_fewshot_retrieval_methods_full_with_upper_v1`
- `outputs/phase4_fewshot_retrieval_target_h_extended_v1`
- `outputs/phase4b_ood_planner_*_500_v1`
- `outputs/phase4b_ood_planner_multimodel_500_repeats_summary_v1`
- `outputs/phase5_cst_case_study_v1`
- `outputs/phase5_cst_case_study_with_si_v1`
- `outputs/figures/phase6_device_design_comparison_v1`
- `outputs/figures/phase6_device_field_comparison_fast_compact_v1`
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/final_main_figures_nc_light_v2"))
    parser.add_argument("--formats", nargs="+", default=["png", "pdf", "svg", "tiff"])
    parser.add_argument("--font-scale", choices=["large", "compact"], default="large")
    args = parser.parse_args()

    root = Path.cwd()
    set_publication_style(args.font_scale)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_fig1(root, args.output_dir, args.formats)
    plot_fig2(root, args.output_dir, args.formats)
    plot_fig3(root, args.output_dir, args.formats)
    plot_fig4(root, args.output_dir, args.formats)
    plot_fig5(root, args.output_dir, args.formats)
    plot_fig6(root, args.output_dir, args.formats)
    write_readme(args.output_dir)
    print(f"Saved final main figures: {args.output_dir}")


if __name__ == "__main__":
    main()
