from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_llm_framework.plotting.publication_style import (
    MORANDI,
    add_panel_label,
    save_figure,
    set_publication_style,
    style_axis,
)


def _load_design(design_json: Path) -> tuple[dict, pd.DataFrame]:
    with design_json.open("r", encoding="utf-8") as f:
        design = json.load(f)
    units = pd.read_csv(design["units_csv"])
    return design, units


def plot_design(design_json: Path, output_dir: Path, formats: list[str]) -> None:
    design, units = _load_design(design_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    ax.plot(units["x_um"], units["target_phase_rad"], color=MORANDI["charcoal"], linewidth=3, label="Ideal phase")
    ax.scatter(units["x_um"], units["selected_phase_rad"], color=MORANDI["salmon"], s=48, label="Selected unit")
    ax.set_xlabel("Position x (um)")
    ax.set_ylabel("Phase (rad)")
    ax.set_title("Phase Profile")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="both")
    add_panel_label(ax, "A")

    ax = axes[0, 1]
    ax.bar(units["x_um"], units["phase_error_rad"], width=float(design["lattice_size_um"]) * 0.75, color=MORANDI["dusty_blue"])
    ax.set_xlabel("Position x (um)")
    ax.set_ylabel("Absolute phase error (rad)")
    ax.set_title("Phase Matching Error")
    style_axis(ax, grid_axis="y")
    add_panel_label(ax, "B")

    ax = axes[1, 0]
    ax.plot(units["x_um"], units["selected_amplitude"], color=MORANDI["mint"], linewidth=2.8, marker="o", markersize=4)
    ax.set_xlabel("Position x (um)")
    ax.set_ylabel("Unit-cell amplitude")
    ax.set_title("Selected Unit Amplitude")
    style_axis(ax, grid_axis="both", ylim=(0.0, 1.08))
    add_panel_label(ax, "C")

    ax = axes[1, 1]
    for col, color, label in [
        ("epsilon_raw", MORANDI["salmon"], "epsilon"),
        ("thickness_raw", MORANDI["dusty_blue"], "thickness"),
        ("radius_raw", MORANDI["mint"], "radius"),
        ("gap_raw", MORANDI["lavender"], "gap"),
    ]:
        if col in units.columns:
            values = units[col].to_numpy(dtype=float)
            norm = (values - values.min()) / (values.max() - values.min() + 1e-12)
            ax.plot(units["x_um"], norm, linewidth=2.5, marker="o", markersize=3, color=color, label=label)
    ax.set_xlabel("Position x (um)")
    ax.set_ylabel("Normalized parameter")
    ax.set_title("Geometry Distribution")
    ax.legend(frameon=False, ncol=2)
    style_axis(ax, grid_axis="both", ylim=(-0.05, 1.05))
    add_panel_label(ax, "D")

    summary = (
        f"f={design['design_frequency_thz_actual']:.2f} THz, "
        f"N={design['num_units']}, aperture={design['aperture_um']:.1f} um, "
        f"focal length={design['focal_length_um']:.1f} um, "
        f"mean phase error={design['mean_phase_error_rad']:.3f} rad"
    )
    fig.suptitle(summary, fontsize=18)
    save_figure(fig, output_dir, "phase5_hero_metalens_design", formats=formats)

    metrics = pd.DataFrame(
        [
            {
                "design_frequency_thz": design["design_frequency_thz_actual"],
                "num_units": design["num_units"],
                "aperture_um": design["aperture_um"],
                "focal_length_um": design["focal_length_um"],
                "mean_phase_error_rad": design["mean_phase_error_rad"],
                "max_phase_error_rad": design["max_phase_error_rad"],
                "mean_selected_amplitude": design["mean_selected_amplitude"],
                "min_selected_amplitude": design["min_selected_amplitude"],
            }
        ]
    )
    metrics.to_csv(output_dir / "phase5_hero_metalens_design_metrics.csv", index=False)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Phase 5 Hero Metalens Figures",
                "",
                f"Source design: `{design_json}`",
                "",
                "This figure shows the array-level phase-profile synthesis from a cylinder unit-cell library.",
                "It is an optional hero demonstration, not part of the main statistical benchmark.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-json", type=Path, default=Path("outputs/phase5_hero_metalens_source_cylinder_v1/phase5_hero_metalens_design.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/phase5_hero_metalens_source_cylinder_v1"))
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--font-scale", choices=["large", "compact"], default="large")
    args = parser.parse_args()

    set_publication_style(args.font_scale)
    plot_design(args.design_json, args.output_dir, args.formats)
    print(f"Saved Phase 5 hero metalens figures: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
