from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from my_llm_framework.plotting.publication_style import (
    MORANDI,
    add_panel_label,
    save_figure,
    set_publication_style,
    style_axis,
)


def _load_design(run_dir: Path, label: str) -> tuple[dict, pd.DataFrame]:
    design_json = run_dir / "phase6_metalens_design.json"
    with design_json.open("r", encoding="utf-8") as f:
        design = json.load(f)
    units = pd.read_csv(design["units_csv"])
    units["run_label"] = label
    return design, units


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    label, path = value.split("=", 1)
    return label, Path(path)


def plot_comparison(runs: list[str], output_dir: Path, formats: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    designs: list[dict] = []
    units_list: list[pd.DataFrame] = []
    for run in runs:
        label, path = _parse_run(run)
        design, units = _load_design(path, label)
        design["run_label"] = label
        designs.append(design)
        units_list.append(units)
    all_units = pd.concat(units_list, ignore_index=True)
    metrics = pd.DataFrame(
        [
            {
                "run_label": design["run_label"],
                "planner_policy": design["planner_policy"],
                "mean_phase_error_rad": design["mean_phase_error_rad"],
                "max_phase_error_rad": design["max_phase_error_rad"],
                "mean_selected_amplitude": design["mean_selected_amplitude"],
                "min_selected_amplitude": design["min_selected_amplitude"],
                "mirror_phase_mismatch_rad": design.get("mirror_phase_mismatch_rad", 0.0),
            }
            for design in designs
        ]
    )
    metrics.to_csv(output_dir / "phase6_device_design_comparison_metrics.csv", index=False)
    all_units.to_csv(output_dir / "phase6_device_design_comparison_units.csv", index=False)

    colors = [MORANDI["dusty_blue"], MORANDI["salmon"], MORANDI["mint"], MORANDI["lavender"]]
    color_map = {label: colors[idx % len(colors)] for idx, label in enumerate(metrics["run_label"])}

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    ax = axes[0, 0]
    first_units = units_list[0]
    ax.plot(first_units["x_um"], first_units["target_phase_rad"], color=MORANDI["charcoal"], linewidth=3, label="Ideal phase")
    for label, group in all_units.groupby("run_label", sort=False):
        ax.scatter(group["x_um"], group["selected_phase_rad"], s=50, color=color_map[label], label=label, alpha=0.9)
    ax.set_xlabel("Position x (um)")
    ax.set_ylabel("Phase (rad)")
    ax.set_title("Device Phase Profile")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="both")
    add_panel_label(ax, "A")

    ax = axes[0, 1]
    for label, group in all_units.groupby("run_label", sort=False):
        ax.plot(group["x_um"], group["phase_error_rad"], color=color_map[label], linewidth=2.8, marker="o", markersize=4, label=label)
    ax.set_xlabel("Position x (um)")
    ax.set_ylabel("Phase error (rad)")
    ax.set_title("Phase Error Along Aperture")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="both")
    add_panel_label(ax, "B")

    ax = axes[0, 2]
    for label, group in all_units.groupby("run_label", sort=False):
        ax.plot(group["x_um"], group["selected_amplitude"], color=color_map[label], linewidth=2.8, marker="o", markersize=4, label=label)
    ax.set_xlabel("Position x (um)")
    ax.set_ylabel("Selected amplitude")
    ax.set_title("Transmission-Amplitude Tradeoff")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="both", ylim=(0.0, 1.08))
    add_panel_label(ax, "C")

    ax = axes[1, 0]
    metric_labels = ["mean_phase_error_rad", "max_phase_error_rad", "mirror_phase_mismatch_rad"]
    x = range(len(metric_labels))
    width = 0.8 / max(1, len(metrics))
    for idx, row in metrics.reset_index(drop=True).iterrows():
        values = [1000.0 * float(row[m]) for m in metric_labels]
        offsets = [pos - 0.4 + width / 2 + idx * width for pos in x]
        ax.bar(offsets, values, width=width, color=color_map[row["run_label"]], label=row["run_label"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Mean phase\nerror", "Max phase\nerror", "Mirror phase\nmismatch"])
    ax.set_ylabel("Phase metric (mrad)")
    ax.set_title("Phase-Quality Metrics")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="y")
    add_panel_label(ax, "D")

    ax = axes[1, 1]
    amp_metrics = ["mean_selected_amplitude", "min_selected_amplitude"]
    x = range(len(amp_metrics))
    for idx, row in metrics.reset_index(drop=True).iterrows():
        values = [float(row[m]) for m in amp_metrics]
        offsets = [pos - 0.4 + width / 2 + idx * width for pos in x]
        ax.bar(offsets, values, width=width, color=color_map[row["run_label"]], label=row["run_label"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Mean\namplitude", "Minimum\namplitude"])
    ax.set_ylabel("Amplitude")
    ax.set_title("Amplitude Tradeoff Metrics")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="y", ylim=(0.0, 1.08))
    add_panel_label(ax, "E")

    ax = axes[1, 2]
    rank_counts = all_units.groupby(["run_label", "candidate_rank"]).size().reset_index(name="count")
    ranks = sorted(rank_counts["candidate_rank"].astype(int).unique())
    x = range(len(ranks))
    for idx, label in enumerate(metrics["run_label"]):
        row_counts = rank_counts[rank_counts["run_label"] == label].set_index("candidate_rank")["count"].to_dict()
        values = [int(row_counts.get(rank, 0)) for rank in ranks]
        offsets = [pos - 0.4 + width / 2 + idx * width for pos in x]
        ax.bar(offsets, values, width=width, color=color_map[label], label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(rank) for rank in ranks])
    ax.set_xlabel("Candidate rank")
    ax.set_ylabel("Number of units")
    ax.set_title("Planner Selection Pattern")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="y")
    add_panel_label(ax, "F")

    save_figure(fig, output_dir, "phase6_device_design_comparison", formats=formats)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Phase 6 Device Design Comparison Figures",
                "",
                "这个目录保存 Phase 6 器件级设计对比图。",
                "",
                "## 数据来源",
                "",
                *[f"- `{_parse_run(run)[1]}\\phase6_metalens_design.json`" for run in runs],
                "",
                "## 图说明",
                "",
                "Panel A 比较理想聚焦相位与不同 planner 选择的 unit-cell 相位。",
                "Panel B 比较 aperture 上的逐点相位误差。",
                "Panel C 比较透射幅值 tradeoff。",
                "Panel D 汇总 phase-quality metrics，单位为 mrad。",
                "Panel E 比较 mean/minimum amplitude tradeoff。",
                "Panel F 展示 planner 对 candidate_rank 的使用比例，用于说明 LLM 是否真的做了不同选择。",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="Label=path to a Phase 6 design directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/phase6_device_design_comparison_v1"))
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--font-scale", choices=["large", "compact"], default="large")
    args = parser.parse_args()

    set_publication_style(args.font_scale)
    plot_comparison(args.run, args.output_dir, args.formats)
    print(f"Saved Phase 6 device design comparison figures: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
