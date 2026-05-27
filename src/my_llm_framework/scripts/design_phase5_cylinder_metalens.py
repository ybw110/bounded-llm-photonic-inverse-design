from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from my_llm_framework.config import load_config
from my_llm_framework.data.processed import load_processed_split
from my_llm_framework.features.physics_summary import amplitude


C_UM_PER_PS = 299.792458


def _wrap_phase(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _phase_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(_wrap_phase(a - b))


def _nearest_frequency_index(freq: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(freq - float(value))))


def design_metalens(
    cfg: dict[str, Any],
    *,
    domain: str,
    split: str,
    design_frequency_thz: float,
    num_units: int,
    lattice_size_um: float,
    focal_length_um: float,
    min_amplitude: float,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if num_units % 2 == 0:
        raise ValueError("num_units must be odd so the design has a central unit.")

    processed_root = Path(cfg["_processed_root_abs"])
    data = load_processed_split(processed_root, domain, split)
    freq = data["frequency"]
    freq_idx = _nearest_frequency_index(freq, design_frequency_thz)
    actual_frequency = float(freq[freq_idx])
    spectra_at_f = data["spectra"][:, freq_idx]
    unit_amp = amplitude(spectra_at_f)
    unit_phase = np.angle(spectra_at_f)

    valid = unit_amp >= float(min_amplitude)
    if not np.any(valid):
        raise ValueError(f"No unit cells satisfy min_amplitude={min_amplitude}")

    valid_indices = np.where(valid)[0]
    valid_phase = unit_phase[valid_indices]
    valid_amp = unit_amp[valid_indices]

    positions = (np.arange(num_units) - (num_units - 1) / 2.0) * float(lattice_size_um)
    wavelength_um = C_UM_PER_PS / actual_frequency
    k0 = 2.0 * np.pi / wavelength_um
    ideal_phase = _wrap_phase(-k0 * (np.sqrt(positions**2 + focal_length_um**2) - focal_length_um))

    rows: list[dict[str, Any]] = []
    param_names = cfg["domains"][domain]["param_names"][: int(cfg["domains"][domain]["active_param_dim"])]
    for unit_id, (x_um, target_phase) in enumerate(zip(positions, ideal_phase)):
        phase_err = _phase_error(valid_phase, target_phase)
        # Prefer correct phase first, then high amplitude.
        score = phase_err + 0.15 * (1.0 - np.clip(valid_amp, 0.0, 1.5))
        best_local = int(np.argmin(score))
        dataset_idx = int(valid_indices[best_local])
        row: dict[str, Any] = {
            "unit_id": unit_id,
            "x_um": float(x_um),
            "target_phase_rad": float(target_phase),
            "selected_phase_rad": float(unit_phase[dataset_idx]),
            "phase_error_rad": float(_phase_error(np.asarray([unit_phase[dataset_idx]]), np.asarray([target_phase]))[0]),
            "selected_amplitude": float(unit_amp[dataset_idx]),
            "dataset_index": dataset_idx,
            "source_index": int(data["source_indices"][dataset_idx]),
        }
        for idx, name in enumerate(param_names):
            row[f"{name}_raw"] = float(data["params_raw"][dataset_idx, idx])
            row[f"{name}_norm"] = float(data["params"][dataset_idx, idx])
        rows.append(row)

    design_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    units_csv = output_dir / "phase5_hero_metalens_units.csv"
    design_df.to_csv(units_csv, index=False)

    summary = {
        "design_name": f"phase5_{domain}_cylinder_metalens_hero",
        "domain": domain,
        "split": split,
        "design_frequency_thz_requested": float(design_frequency_thz),
        "design_frequency_thz_actual": actual_frequency,
        "num_units": int(num_units),
        "lattice_size_um": float(lattice_size_um),
        "aperture_um": float((num_units - 1) * lattice_size_um),
        "focal_length_um": float(focal_length_um),
        "wavelength_um": float(wavelength_um),
        "min_amplitude": float(min_amplitude),
        "mean_phase_error_rad": float(design_df["phase_error_rad"].mean()),
        "max_phase_error_rad": float(design_df["phase_error_rad"].max()),
        "mean_selected_amplitude": float(design_df["selected_amplitude"].mean()),
        "min_selected_amplitude": float(design_df["selected_amplitude"].min()),
        "units_csv": str(units_csv),
        "expected_cst_field_csv": str(output_dir / "phase5_hero_metalens_field.csv"),
        "expected_cst_project_dir": str(output_dir / "cst_projects"),
        "param_names": list(param_names),
    }
    (output_dir / "phase5_hero_metalens_design.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    phase_csv = output_dir / "phase5_hero_metalens_phase_profile.csv"
    design_df[
        [
            "unit_id",
            "x_um",
            "target_phase_rad",
            "selected_phase_rad",
            "phase_error_rad",
            "selected_amplitude",
        ]
    ].to_csv(phase_csv, index=False)

    write_readme(output_dir, summary)
    return design_df, summary


def write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase 5 Hero Metalens Design",
        "",
        "This folder contains a small optional array-level / metalens hero design based on Target-Self/source-cylinder unit cells.",
        "",
        "The design converts unit-cell evidence into a device-level layout by matching the ideal focusing phase profile at the design frequency.",
        "",
        "## Key settings",
        "",
        f"- Domain: `{summary['domain']}`",
        f"- Design frequency: `{summary['design_frequency_thz_actual']:.4f}` THz",
        f"- Number of units: `{summary['num_units']}`",
        f"- Lattice size: `{summary['lattice_size_um']}` um",
        f"- Aperture: `{summary['aperture_um']}` um",
        f"- Focal length: `{summary['focal_length_um']}` um",
        f"- Mean phase error: `{summary['mean_phase_error_rad']:.4f}` rad",
        f"- Mean selected amplitude: `{summary['mean_selected_amplitude']:.4f}`",
        "",
        "## Files",
        "",
        "- `phase5_hero_metalens_design.json`: design metadata for CST.",
        "- `phase5_hero_metalens_units.csv`: one selected unit cell per array position.",
        "- `phase5_hero_metalens_phase_profile.csv`: target and selected phase profile.",
        "",
        "## Interpretation",
        "",
        "This is an optional hero demonstration. It is not needed for the main statistical claim, but it can visually strengthen the closed-loop workflow by showing how unit-cell candidates become an array-level device.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_v1.json")
    parser.add_argument("--domain", default="target_self", choices=["target_self", "source"])
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--design-frequency-thz", type=float, default=0.75)
    parser.add_argument("--num-units", type=int, default=41)
    parser.add_argument("--lattice-size-um", type=float, default=60.0)
    parser.add_argument("--focal-length-um", type=float, default=2000.0)
    parser.add_argument("--min-amplitude", type=float, default=0.90)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase5_hero_metalens_target_self_v1"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    design_df, summary = design_metalens(
        cfg,
        domain=args.domain,
        split=args.split,
        design_frequency_thz=args.design_frequency_thz,
        num_units=args.num_units,
        lattice_size_um=args.lattice_size_um,
        focal_length_um=args.focal_length_um,
        min_amplitude=args.min_amplitude,
        output_dir=args.output_dir,
    )
    print(f"Saved Phase 5 hero metalens design: {args.output_dir.resolve()}")
    print(json.dumps(summary, indent=2))
    print(design_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
