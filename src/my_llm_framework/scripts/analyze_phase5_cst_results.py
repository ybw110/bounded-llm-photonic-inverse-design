from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _interp(values_x: np.ndarray, values_y: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    order = np.argsort(values_x)
    return np.interp(target_x, values_x[order], values_y[order])


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def analyze_case(row: pd.Series) -> dict[str, object]:
    case_id = str(row["case_id"])
    spectra_path = Path(str(row["case_json"])).parent.parent / "spectra" / f"{case_id}_spectra.csv"
    cst_path = Path(str(row["expected_result_csv"]))
    out: dict[str, object] = {
        "case_id": case_id,
        "target_domain": row.get("target_domain"),
        "structure_type": row.get("structure_type"),
        "spectra_csv": str(spectra_path),
        "cst_csv": str(cst_path),
        "cst_exists": cst_path.exists(),
    }
    if not spectra_path.exists() or not cst_path.exists():
        out["status"] = "missing_file"
        return out

    spectra = pd.read_csv(spectra_path)
    cst = pd.read_csv(cst_path)
    target_freq = spectra["frequency_thz"].to_numpy(dtype=float)
    target_amp = spectra["target_amplitude"].to_numpy(dtype=float)
    candidate_amp = spectra["candidate_amplitude"].to_numpy(dtype=float)
    cst_freq = cst["frequency_thz"].to_numpy(dtype=float)
    cst_amp = cst["cst_amplitude"].to_numpy(dtype=float)

    cst_on_target = _interp(cst_freq, cst_amp, target_freq)
    freq_on_cst = cst_freq[: len(cst_amp)]
    looks_like_frequency = bool(
        len(freq_on_cst) == len(cst_amp)
        and np.allclose(cst_amp, freq_on_cst, rtol=1e-4, atol=1e-8)
    )

    out.update(
        {
            "status": "suspicious_frequency_column" if looks_like_frequency else "ok",
            "num_cst_points": int(len(cst)),
            "cst_freq_min": float(np.nanmin(cst_freq)),
            "cst_freq_max": float(np.nanmax(cst_freq)),
            "cst_amp_min": float(np.nanmin(cst_amp)),
            "cst_amp_max": float(np.nanmax(cst_amp)),
            "candidate_vs_target_amp_mse": _mse(candidate_amp, target_amp),
            "cst_vs_target_amp_mse": _mse(cst_on_target, target_amp),
            "cst_vs_candidate_amp_mse": _mse(cst_on_target, candidate_amp),
            "cst_amp_looks_like_frequency": looks_like_frequency,
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("outputs/phase5_cst_case_study_v1/cst_batch_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase5_cst_case_study_v1"))
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    rows = [analyze_case(row) for _, row in manifest.iterrows()]
    metrics = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "phase5_cst_case_metrics.csv"
    metrics.to_csv(output_path, index=False)
    print(f"Saved Phase 5 CST metrics: {output_path.resolve()}")
    display_cols = [
        "case_id",
        "status",
        "num_cst_points",
        "candidate_vs_target_amp_mse",
        "cst_vs_target_amp_mse",
        "cst_amp_looks_like_frequency",
    ]
    existing = [col for col in display_cols if col in metrics.columns]
    print(metrics[existing].to_string(index=False))


if __name__ == "__main__":
    main()
