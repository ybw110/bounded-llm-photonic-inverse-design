from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from my_llm_framework.plotting.publication_style import (
    MORANDI,
    add_panel_label,
    save_figure,
    set_publication_style,
    style_axis,
)
from my_llm_framework.scripts.plot_phase5_cst_case_study import _interp, _load_available_cases, _plot_case


FORMATS = ["png", "pdf"]

METHOD_LABELS = {
    "raw_spectrum": "Raw spectrum",
    "physics_summary": "Physics summary",
    "domain_aware": "Domain-aware",
}

METHOD_COLORS = {
    "raw_spectrum": MORANDI["dusty_blue"],
    "physics_summary": MORANDI["peach"],
    "domain_aware": MORANDI["mint"],
}

POLICY_COLORS = {
    "Source-only": MORANDI["warm_gray"],
    "Fixed rule": MORANDI["dusty_blue"],
    "DeepSeek": MORANDI["salmon"],
    "DeepSeek (n=3)": MORANDI["salmon"],
    "GLM-5.1 (n=3)": MORANDI["mint"],
    "Oracle": MORANDI["lavender"],
}


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    save_figure(fig, output_dir, stem, formats=FORMATS)


def _fewshot_positions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric = pd.to_numeric(df["fewshot_label"], errors="coerce")
    order = (
        pd.DataFrame({"fewshot_label": df["fewshot_label"].astype(str), "numeric": numeric})
        .drop_duplicates()
        .assign(sort_key=lambda x: x["numeric"].fillna(np.inf))
        .sort_values(["sort_key", "fewshot_label"], kind="stable")
        .reset_index(drop=True)
    )
    positions = {label: idx for idx, label in enumerate(order["fewshot_label"])}
    df["fewshot_x"] = df["fewshot_label"].astype(str).map(positions).astype(float)
    return df


def _xticks(df: pd.DataFrame) -> tuple[list[float], list[str]]:
    pairs = (
        df[["fewshot_x", "fewshot_label"]]
        .drop_duplicates()
        .sort_values("fewshot_x", kind="stable")
        .itertuples(index=False, name=None)
    )
    xs, labels = zip(*pairs)
    return list(xs), [str(label) for label in labels]


def _line_with_std(ax: plt.Axes, sub: pd.DataFrame, y_col: str, *, label: str, color: str, log_y: bool = False) -> None:
    sub = sub.sort_values("fewshot_x", kind="stable")
    x = sub["fewshot_x"].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    ax.plot(x, y, color=color, linewidth=3.0, marker="o", markersize=7, label=label)
    std_col = y_col.replace("_mean", "_std")
    if std_col in sub.columns:
        std = sub[std_col].to_numpy(dtype=float)
        if np.isfinite(std).any():
            lower = y - std
            upper = y + std
            if log_y:
                lower = np.maximum(lower, 1e-12)
            ax.fill_between(x, lower, upper, color=color, alpha=0.18, linewidth=0)


def _plot_phase4_line_panel(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    target: str,
    y_col: str,
    title: str,
    ylabel: str,
    stem: str,
    panel: str,
    methods: list[str] = ["raw_spectrum", "physics_summary", "domain_aware"],
    ylim: tuple[float, float] | None = None,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    sub_target = df[df["target_domain"] == target]
    for method in methods:
        sub = sub_target[sub_target["method"] == method]
        if sub.empty or y_col not in sub.columns:
            continue
        _line_with_std(ax, sub, y_col, label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method, MORANDI["charcoal"]), log_y=log_y)
    xs, labels = _xticks(sub_target)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Target-domain samples")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="both", ylim=ylim)
    add_panel_label(ax, panel)
    _save(fig, output_dir, stem)


def generate_phase4_panels(root: Path) -> None:
    summary = root / "outputs/phase4_fewshot_retrieval_methods_full_with_upper_v1/fewshot_retrieval_summary_aggregate.csv"
    extended = root / "outputs/phase4_fewshot_retrieval_target_h_extended_v1/fewshot_retrieval_summary_aggregate.csv"
    if summary.exists():
        df = _fewshot_positions(pd.read_csv(summary))
        out = root / "outputs/figures/phase4_fewshot_recovery_v2/panels"
        _plot_phase4_line_panel(df, out, target="target_h", y_col="success_at_50_mean", title="Target-H Success@50", ylabel="Success rate", stem="panel_A_target_h_success50", panel="A", ylim=(-0.03, 1.04))
        _plot_phase4_line_panel(df, out, target="target_h", y_col="target_domain_success_at_50_mean", title="Target-H Target-Domain Success@50", ylabel="Success rate", stem="panel_B_target_h_target_domain_success50", panel="B", ylim=(-0.03, 1.04))
        _plot_phase4_line_panel(df, out, target="target_self", y_col="target_domain_success_at_50_mean", title="Target-Self Target-Domain Success@50", ylabel="Success rate", stem="panel_C_target_self_target_domain_success50", panel="C", ylim=(-0.03, 1.04))
        _plot_phase4_line_panel(df, out, target="target_h", y_col="top1_target_domain_rate_mean", title="Target-H Top1 Target-Domain Rate", ylabel="Rate", stem="panel_D_target_h_top1_target_domain_rate", panel="D", ylim=(-0.03, 1.04))
        _plot_phase4_line_panel(df, out, target="target_h", y_col="mean_top1_amplitude_mse_mean", title="Target-H Top1 Amplitude MSE", ylabel="Mean top1 amplitude MSE", stem="panel_E_target_h_top1_mse", panel="E", log_y=True)
        _plot_phase4_line_panel(df, out, target="target_self", y_col="mean_top1_amplitude_mse_mean", title="Target-Self Top1 Amplitude MSE", ylabel="Mean top1 amplitude MSE", stem="panel_F_target_self_top1_mse", panel="F", log_y=True)
        (out / "README.md").write_text(
            "\n".join(
                [
                    "# Phase 4 standalone panels",
                    "",
                    "数据来源：",
                    f"- `{summary}`",
                    "",
                    "这些子图对应 Phase 4 few-shot recovery 大图中的独立 panel。所有曲线带 mean +/- std 阴影，使用 Arial 字体、Morandi 配色，并保存 PNG/PDF。",
                ]
            ),
            encoding="utf-8",
        )

    if extended.exists():
        df_ext = _fewshot_positions(pd.read_csv(extended))
        out_ext = root / "outputs/figures/phase4_target_h_extended_v1/panels"
        _plot_phase4_line_panel(df_ext, out_ext, target="target_h", y_col="target_domain_success_at_50_mean", title="Extended Target-H Target-Domain Success@50", ylabel="Success rate", stem="panel_A_extended_target_h_target_domain_success50", panel="A", methods=["raw_spectrum", "domain_aware"], ylim=(-0.03, 1.04))
        _plot_phase4_line_panel(df_ext, out_ext, target="target_h", y_col="top1_target_domain_rate_mean", title="Extended Target-H Top1 Target-Domain Rate", ylabel="Rate", stem="panel_B_extended_target_h_top1_target_domain_rate", panel="B", methods=["raw_spectrum", "domain_aware"], ylim=(-0.03, 1.04))
        _plot_phase4_line_panel(df_ext, out_ext, target="target_h", y_col="mean_top1_amplitude_mse_mean", title="Extended Target-H Top1 MSE", ylabel="Mean top1 amplitude MSE", stem="panel_C_extended_target_h_top1_mse", panel="C", methods=["raw_spectrum", "domain_aware"], log_y=True)
        (out_ext / "README.md").write_text(
            "\n".join(
                [
                    "# Phase 4 Target-H extended standalone panels",
                    "",
                    "数据来源：",
                    f"- `{extended}`",
                    "",
                    "这些子图用于展示 Target-H topology shift 的 coverage-limited recovery。",
                ]
            ),
            encoding="utf-8",
        )


def _load_planner_summaries(root: Path) -> pd.DataFrame:
    specs = [
        ("Source-only", 0, root / "outputs/phase4b_ood_planner_source_only_500_v1/ood_recovery_planner_summary.csv"),
        ("Fixed rule", 1, root / "outputs/phase4b_ood_planner_shift_aware_500_v1/ood_recovery_planner_summary.csv"),
        ("Oracle", 4, root / "outputs/phase4b_ood_planner_oracle_500_v1/ood_recovery_planner_summary.csv"),
    ]
    frames = []
    for label, order, path in specs:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["run_label"] = label
        df["run_order"] = order
        frames.append(df)
    repeat_path = root / "outputs/phase4b_ood_planner_multimodel_500_repeats_summary_v1/phase4b_multimodel_repeat_summary.csv"
    if repeat_path.exists():
        repeat = pd.read_csv(repeat_path)
        order_map = {"DeepSeek": 2, "GLM-5.1": 3}
        for _, row in repeat[repeat["target_domain"] == "target_h"].iterrows():
            frames.append(
                pd.DataFrame(
                    [
                        {
                            "target_domain": "target_h",
                            "success_at_50": row["success_at_50_mean"],
                            "target_domain_success_at_50": row["target_domain_success_at_50_mean"],
                            "mean_fewshot_cost": row["mean_fewshot_cost_mean"],
                            "run_label": f"{row['run_label']} (n=3)",
                            "run_order": order_map.get(row["run_label"], 99),
                        }
                    ]
                )
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _bar_panel(ax: plt.Axes, labels: list[str], values: list[float], title: str, ylabel: str, *, ylim: tuple[float, float] | None = None) -> None:
    colors = [POLICY_COLORS.get(label, MORANDI["warm_gray"]) for label in labels]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.68)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axis(ax, grid_axis="y", ylim=ylim)
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            text = f"{value:.3f}" if value <= 1.2 else f"{value:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, value, text, ha="center", va="bottom", fontsize=14)


def generate_phase4b_panels(root: Path) -> None:
    summary = _load_planner_summaries(root)
    if summary.empty:
        return
    out = root / "outputs/figures/publication_style_500_v1/phase4b_500/panels"
    out.mkdir(parents=True, exist_ok=True)
    target_h = summary[summary["target_domain"] == "target_h"].sort_values("run_order")
    labels = target_h["run_label"].tolist()

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    _bar_panel(ax, labels, target_h["target_domain_success_at_50"].astype(float).tolist(), "Target-H OOD Planner Recovery", "Target-domain Success@50", ylim=(0, 1.04))
    add_panel_label(ax, "A")
    _save(fig, out, "panel_A_target_h_planner_success")

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    _bar_panel(ax, labels, target_h["mean_fewshot_cost"].astype(float).tolist(), "Target-H Target-Sample Cost", "Mean target samples")
    add_panel_label(ax, "B")
    _save(fig, out, "panel_B_target_h_sample_cost")

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    x = target_h["mean_fewshot_cost"].astype(float).to_numpy()
    y = target_h["target_domain_success_at_50"].astype(float).to_numpy()
    colors = [POLICY_COLORS.get(label, MORANDI["warm_gray"]) for label in labels]
    ax.scatter(x, y, s=160, color=colors, edgecolor="white", linewidth=1.2, zorder=3)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), xytext=(6, 6), textcoords="offset points", fontsize=16)
    ax.set_xlabel("Mean target samples")
    ax.set_ylabel("Target-domain Success@50")
    ax.set_title("Sample-Efficient OOD Recovery")
    style_axis(ax, grid_axis="both", ylim=(0, 1.04))
    add_panel_label(ax, "C")
    _save(fig, out, "panel_C_target_h_efficiency")

    action_dirs = {
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
    shot_colors = [MORANDI["warm_gray"], MORANDI["pale_blue"], MORANDI["mint"], MORANDI["peach"], MORANDI["salmon"]]
    action_props: dict[str, pd.Series] = {}
    for label, dirs in action_dirs.items():
        series = []
        for rel in dirs:
            path = root / rel / "ood_recovery_planner_action_summary.csv"
            if not path.exists():
                continue
            action = pd.read_csv(path)
            fewshot = action[(action["target_domain"] == "target_h") & (action["category"] == "fewshot")].copy()
            fewshot["shot"] = pd.to_numeric(fewshot["value"], errors="coerce")
            series.append(fewshot.set_index("shot")["proportion"])
        if series:
            action_props[label] = pd.concat(series, axis=1).mean(axis=1)
    if action_props:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        labels_actions = list(action_props)
        x_actions = np.arange(len(labels_actions))
        bottom = np.zeros(len(labels_actions))
        for shot, color in zip(shot_values, shot_colors):
            values = [float(action_props[label].get(shot, 0.0)) for label in labels_actions]
            ax.bar(x_actions, values, bottom=bottom, color=color, edgecolor="white", linewidth=0.8, label=str(shot))
            bottom += np.array(values)
        ax.set_xticks(x_actions)
        ax.set_xticklabels(labels_actions)
        ax.set_ylabel("Mean action proportion")
        ax.set_title("Target-H Few-Shot Actions")
        style_axis(ax, grid_axis="y", ylim=(0, 1.0))
        ax.legend(title="Target samples", frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.24))
        add_panel_label(ax, "D")
        _save(fig, out, "panel_D_llm_target_h_fewshot_actions")

    action_path = root / "outputs/phase4b_ood_planner_deepseek_500_v1/ood_recovery_planner_action_summary.csv"
    if action_path.exists():
        action = pd.read_csv(action_path)
        deep_h = action[action["target_domain"] == "target_h"]

        critic = deep_h[deep_h["category"] == "critic"].copy()
        if not critic.empty:
            critic = critic.sort_values("proportion", ascending=True)
            fig, ax = plt.subplots(figsize=(8.0, 5.2))
            ax.barh(critic["value"].astype(str), critic["proportion"], color=MORANDI["lavender"])
            ax.set_xlabel("Proportion")
            ax.set_ylabel("Critic reason")
            ax.set_title("DeepSeek Target-H Critic Outcomes")
            style_axis(ax, grid_axis="x")
            add_panel_label(ax, "E")
            _save(fig, out, "panel_E_deepseek_target_h_critic_outcomes")

    (out / "README.md").write_text(
        "\n".join(
            [
                "# Phase 4B standalone planner panels",
                "",
                "数据来源：",
                "- `outputs/phase4b_ood_planner_source_only_500_v1/ood_recovery_planner_summary.csv`",
                "- `outputs/phase4b_ood_planner_shift_aware_500_v1/ood_recovery_planner_summary.csv`",
                "- `outputs/phase4b_ood_planner_deepseek_500_repeat{1,2,3}_v1/ood_recovery_planner_summary.csv`",
                "- `outputs/phase4b_ood_planner_glm51_500_repeat{1,2,3}_v1/ood_recovery_planner_summary.csv`",
                "- `outputs/phase4b_ood_planner_oracle_500_v1/ood_recovery_planner_summary.csv`",
                "- `outputs/phase4b_ood_planner_multimodel_500_repeats_summary_v1/phase4b_multimodel_repeat_summary.csv`",
                "",
                "结论：DeepSeek 和 GLM-5.1 在 Target-H 500-query 三次重复中都接近 oracle target-domain recovery；DeepSeek 更稳健，GLM-5.1 更节省样本。",
            ]
        ),
        encoding="utf-8",
    )


def generate_phase5_panels(root: Path) -> None:
    main_manifest = root / "outputs/phase5_cst_case_study_v1/cst_batch_manifest.csv"
    si_manifest = root / "outputs/phase5_cst_case_study_with_si_v1/cst_batch_manifest.csv"
    if not main_manifest.exists():
        return
    out = root / "outputs/figures/phase5_cst_case_study_v1/panels"
    out.mkdir(parents=True, exist_ok=True)
    cases = _load_available_cases(main_manifest).reset_index(drop=True)
    for idx, (_, row) in enumerate(cases.iterrows()):
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        _plot_case(ax, row)
        ax.legend(frameon=False, loc="upper right")
        label = chr(ord("A") + idx)
        add_panel_label(ax, label)
        _save(fig, out, f"panel_{label}_{row['case_id']}_overlay")

    if not cases.empty:
        rows = []
        for _, row in cases.iterrows():
            spectra = pd.read_csv(row["spectra_csv"])
            cst = pd.read_csv(row["cst_csv"])
            freq = spectra["frequency_thz"].to_numpy(dtype=float)
            target = spectra["target_amplitude"].to_numpy(dtype=float)
            candidate = spectra["candidate_amplitude"].to_numpy(dtype=float)
            cst_interp = _interp(cst["frequency_thz"].to_numpy(dtype=float), cst["cst_amplitude"].to_numpy(dtype=float), freq)
            rows.append({"case": row["case_id"].replace("case_", "").split("_q")[0], "offline": float(np.mean((candidate - target) ** 2)), "cst": float(np.mean((cst_interp - target) ** 2))})
        metric = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(7.4, 5.2))
        x = np.arange(len(metric))
        width = 0.36
        ax.bar(x - width / 2, metric["offline"], width=width, color=MORANDI["dusty_blue"], label="Offline")
        ax.bar(x + width / 2, metric["cst"], width=width, color=MORANDI["salmon"], label="CST")
        ax.set_xticks(x)
        ax.set_xticklabels(metric["case"], rotation=18, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("Amplitude MSE")
        ax.set_xlabel("Case")
        ax.set_title("Offline vs CST MSE")
        ax.legend(frameon=False)
        style_axis(ax, grid_axis="y")
        add_panel_label(ax, "E")
        _save(fig, out, "panel_E_phase5_cst_mse_bars")

    if si_manifest.exists():
        si_cases = _load_available_cases(si_manifest)
        si_cases = si_cases[~si_cases["case_id"].isin(cases["case_id"].tolist())]
        for idx, (_, row) in enumerate(si_cases.iterrows()):
            fig, ax = plt.subplots(figsize=(7.2, 5.2))
            _plot_case(ax, row)
            ax.legend(frameon=False, loc="upper right")
            label = chr(ord("F") + idx)
            add_panel_label(ax, label)
            _save(fig, out, f"panel_{label}_{row['case_id']}_si_overlay")

    (out / "README.md").write_text(
        "\n".join(
            [
                "# Phase 5 standalone CST panels",
                "",
                "数据来源：",
                f"- `{main_manifest}`",
                f"- `{si_manifest}`",
                "- `outputs/phase5_cst_case_study_v1/spectra/*_spectra.csv`",
                "- `outputs/phase5_cst_case_study_v1/cst_results/*_cst_spectrum.csv`",
                "- `outputs/phase5_cst_case_study_with_si_v1/cst_results/*_cst_spectrum.csv`",
                "",
                "结论：A-D 为四个主文 CST validation case，E 为 MSE 汇总，F 为 SI topology/domain-validity caution case。",
            ]
        ),
        encoding="utf-8",
    )


def _load_phase6_design_runs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = [("Manual", root / "outputs/phase6_metalens_manual_source_v1"), ("DeepSeek", root / "outputs/phase6_metalens_deepseek_source_v1")]
    units = []
    metrics = []
    for label, run_dir in runs:
        design = pd.read_json(run_dir / "phase6_metalens_design.json", typ="series")
        unit = pd.read_csv(run_dir / "phase6_metalens_units.csv")
        unit["run_label"] = label
        units.append(unit)
        metrics.append(
            {
                "run_label": label,
                "mean_phase_error_rad": float(design["mean_phase_error_rad"]),
                "max_phase_error_rad": float(design["max_phase_error_rad"]),
                "mean_selected_amplitude": float(design["mean_selected_amplitude"]),
                "min_selected_amplitude": float(design["min_selected_amplitude"]),
                "mirror_phase_mismatch_rad": float(design.get("mirror_phase_mismatch_rad", 0.0)),
            }
        )
    return pd.concat(units, ignore_index=True), pd.DataFrame(metrics)


def generate_phase6_design_panels(root: Path) -> None:
    manual = root / "outputs/phase6_metalens_manual_source_v1/phase6_metalens_units.csv"
    deep = root / "outputs/phase6_metalens_deepseek_source_v1/phase6_metalens_units.csv"
    if not manual.exists() or not deep.exists():
        return
    all_units, metrics = _load_phase6_design_runs(root)
    out = root / "outputs/figures/phase6_device_design_comparison_v1/panels"
    out.mkdir(parents=True, exist_ok=True)
    colors = {"Manual": MORANDI["dusty_blue"], "DeepSeek": MORANDI["salmon"]}

    first = all_units[all_units["run_label"] == "Manual"]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.plot(first["x_um"], first["target_phase_rad"], color=MORANDI["charcoal"], linewidth=3.0, label="Ideal phase")
    for label, group in all_units.groupby("run_label", sort=False):
        ax.scatter(group["x_um"], group["selected_phase_rad"], s=70, color=colors[label], label=label, alpha=0.9)
    ax.set_xlabel("Position x (um)")
    ax.set_ylabel("Phase (rad)")
    ax.set_title("Device Phase Profile")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="both")
    add_panel_label(ax, "A")
    _save(fig, out, "panel_A_device_phase_profile")

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for label, group in all_units.groupby("run_label", sort=False):
        ax.plot(group["x_um"], group["phase_error_rad"], color=colors[label], linewidth=3.0, marker="o", markersize=6, label=label)
    ax.set_xlabel("Position x (um)")
    ax.set_ylabel("Phase error (rad)")
    ax.set_title("Phase Error Along Aperture")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="both")
    add_panel_label(ax, "B")
    _save(fig, out, "panel_B_phase_error_along_aperture")

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for label, group in all_units.groupby("run_label", sort=False):
        ax.plot(group["x_um"], group["selected_amplitude"], color=colors[label], linewidth=3.0, marker="o", markersize=6, label=label)
    ax.set_xlabel("Position x (um)")
    ax.set_ylabel("Selected amplitude")
    ax.set_title("Transmission-Amplitude Tradeoff")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="both", ylim=(0, 1.08))
    add_panel_label(ax, "C")
    _save(fig, out, "panel_C_selected_amplitude")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    labels = metrics["run_label"].tolist()
    x = np.arange(3)
    width = 0.36
    metric_cols = ["mean_phase_error_rad", "max_phase_error_rad", "mirror_phase_mismatch_rad"]
    for idx, label in enumerate(labels):
        row = metrics[metrics["run_label"] == label].iloc[0]
        ax.bar(x - width / 2 + idx * width, [1000 * float(row[col]) for col in metric_cols], width=width, color=colors[label], label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(["Mean phase\nerror", "Max phase\nerror", "Mirror phase\nmismatch"])
    ax.set_ylabel("Phase metric (mrad)")
    ax.set_title("Phase-Quality Metrics")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="y")
    add_panel_label(ax, "D")
    _save(fig, out, "panel_D_phase_quality_metrics")

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    x = np.arange(2)
    amp_cols = ["mean_selected_amplitude", "min_selected_amplitude"]
    for idx, label in enumerate(labels):
        row = metrics[metrics["run_label"] == label].iloc[0]
        ax.bar(x - width / 2 + idx * width, [float(row[col]) for col in amp_cols], width=width, color=colors[label], label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(["Mean\namplitude", "Minimum\namplitude"])
    ax.set_ylabel("Amplitude")
    ax.set_title("Amplitude Tradeoff Metrics")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="y", ylim=(0, 1.08))
    add_panel_label(ax, "E")
    _save(fig, out, "panel_E_amplitude_metrics")

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    rank_counts = all_units.groupby(["run_label", "candidate_rank"]).size().reset_index(name="count")
    ranks = sorted(rank_counts["candidate_rank"].astype(int).unique())
    x = np.arange(len(ranks))
    for idx, label in enumerate(labels):
        counts = rank_counts[rank_counts["run_label"] == label].set_index("candidate_rank")["count"].to_dict()
        ax.bar(x - width / 2 + idx * width, [counts.get(rank, 0) for rank in ranks], width=width, color=colors[label], label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(rank) for rank in ranks])
    ax.set_xlabel("Candidate rank")
    ax.set_ylabel("Number of units")
    ax.set_title("Planner Selection Pattern")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="y")
    add_panel_label(ax, "F")
    _save(fig, out, "panel_F_candidate_rank_usage")

    (out / "README.md").write_text(
        "\n".join(
            [
                "# Phase 6 design standalone panels",
                "",
                "数据来源：",
                "- `outputs/phase6_metalens_manual_source_v1/phase6_metalens_design.json`",
                "- `outputs/phase6_metalens_manual_source_v1/phase6_metalens_units.csv`",
                "- `outputs/phase6_metalens_deepseek_source_v1/phase6_metalens_design.json`",
                "- `outputs/phase6_metalens_deepseek_source_v1/phase6_metalens_units.csv`",
                "",
                "结论：DeepSeek 选择了大量 non-rank-0 candidate，显著降低相位误差，同时牺牲部分 amplitude。",
            ]
        ),
        encoding="utf-8",
    )


def _pivot_field(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = df.pivot_table(index="z", columns="x", values="field_abs", aggfunc="mean").sort_index().sort_index(axis=1)
    return pivot.columns.to_numpy(dtype=float), pivot.index.to_numpy(dtype=float), pivot.to_numpy(dtype=float)


def generate_phase6_field_panels(root: Path) -> None:
    field_dir = root / "outputs/figures/phase6_device_field_comparison_fast_compact_v1"
    manual_path = field_dir / "phase6_manual_field_parsed.csv"
    deep_path = field_dir / "phase6_deepseek_field_parsed.csv"
    metrics_path = field_dir / "phase6_device_field_comparison_metrics.csv"
    if not manual_path.exists() or not deep_path.exists() or not metrics_path.exists():
        return
    manual = pd.read_csv(manual_path)
    deep = pd.read_csv(deep_path)
    metrics = pd.read_csv(metrics_path)
    loaded = [("Manual", manual), ("DeepSeek", deep)]
    pivots = {label: _pivot_field(df) for label, df in loaded}
    common_max = max(float(np.nanmax(pivots[label][2])) for label, _ in loaded) + 1e-12
    focal_length = float(metrics["design_focal_length_um"].iloc[0])
    colors = {"Manual": MORANDI["dusty_blue"], "DeepSeek": MORANDI["salmon"]}
    out = field_dir / "panels"
    out.mkdir(parents=True, exist_ok=True)

    for panel, label in [("A", "Manual"), ("B", "DeepSeek")]:
        x, z, values = pivots[label]
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
        im = ax.imshow(values / common_max, origin="lower", aspect="auto", extent=[float(x.min()), float(x.max()), float(z.min()), float(z.max())], cmap="viridis", vmin=0, vmax=1)
        ax.axhline(focal_length, color="white", linewidth=2.4, linestyle="--", label="Design focus")
        ax.axvline(0.0, color="white", linewidth=1.8, linestyle=":", alpha=0.9)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("z (um)")
        ax.set_title(f"{label} Field Map")
        ax.legend(frameon=False, loc="upper right")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Common-normalized |E|")
        add_panel_label(ax, panel)
        _save(fig, out, f"panel_{panel}_{label.lower()}_field_map")

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for label, _df in loaded:
        x, z, values = pivots[label]
        center_idx = int(np.nanargmin(np.abs(x - 0.0)))
        ax.plot(z, values[:, center_idx] / common_max, color=colors[label], linewidth=3, label=label)
    ax.axvline(focal_length, color=MORANDI["charcoal"], linewidth=2.4, linestyle="--", label="Design focus")
    ax.set_xlabel("z (um)")
    ax.set_ylabel("Common-normalized |E|")
    ax.set_title("On-Axis Profile")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="both")
    add_panel_label(ax, "C")
    _save(fig, out, "panel_C_on_axis_profile")

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    focal_z = float(metrics["focal_plane_z_um"].iloc[0])
    for label, _df in loaded:
        x, z, values = pivots[label]
        focal_idx = int(np.nanargmin(np.abs(z - focal_length)))
        ax.plot(x, values[focal_idx, :] / common_max, color=colors[label], linewidth=3, label=label)
    ax.axvline(0.0, color=MORANDI["charcoal"], linewidth=2.0, linestyle=":")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("Common-normalized |E|")
    ax.set_title(f"Focal-Plane Profile, z={focal_z:.1f} um")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="both")
    add_panel_label(ax, "D")
    _save(fig, out, "panel_D_focal_plane_profile")

    labels = metrics["run_label"].tolist()
    x_pos = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    window_norm = metrics["focal_window_peak_field"].to_numpy(dtype=float) / common_max
    center_norm = metrics["focal_plane_center_field"].to_numpy(dtype=float) / common_max
    ax.bar(x_pos - width / 2, window_norm, width=width, color=[colors[label] for label in labels], label="Focal-window peak")
    ax.bar(x_pos + width / 2, center_norm, width=width, color=[colors[label] for label in labels], alpha=0.45, hatch="//", label="Focal-plane center")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Common-normalized |E|")
    ax.set_title("Focusing Strength")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="y", ylim=(0, 1.05))
    add_panel_label(ax, "E")
    _save(fig, out, "panel_E_focusing_strength")

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    metric_names = ["on_axis_focus_distance_um", "focal_plane_peak_offset_um"]
    metric_labels = ["On-axis peak\ndistance to focus", "Focal-plane\npeak offset"]
    x_metric = np.arange(len(metric_names))
    for idx, label in enumerate(labels):
        row = metrics[metrics["run_label"] == label].iloc[0]
        ax.bar(x_metric - width / 2 + idx * width, [float(row[name]) for name in metric_names], width=width, color=colors[label], label=label)
    ax.set_xticks(x_metric)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Distance (um)")
    ax.set_title("Focus-Location Error")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="y")
    add_panel_label(ax, "F")
    _save(fig, out, "panel_F_focus_location_error")

    (out / "README.md").write_text(
        "\n".join(
            [
                "# Phase 6 field standalone panels",
                "",
                "数据来源：",
                f"- `{manual_path}`",
                f"- `{deep_path}`",
                f"- `{metrics_path}`",
                "",
                "结论：这些子图与 `phase6_device_field_comparison.png/pdf` 对应，但每个 panel 可单独用于论文、PPT 或补充材料。场图使用共同归一化色标。",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--font-scale", choices=["large", "compact"], default="compact")
    args = parser.parse_args()

    root = args.root.resolve()
    set_publication_style(args.font_scale)
    generate_phase4_panels(root)
    generate_phase4b_panels(root)
    generate_phase5_panels(root)
    generate_phase6_design_panels(root)
    generate_phase6_field_panels(root)
    print(f"Saved standalone Phase 4/5/6 panels under: {root / 'outputs/figures'}")


if __name__ == "__main__":
    main()
