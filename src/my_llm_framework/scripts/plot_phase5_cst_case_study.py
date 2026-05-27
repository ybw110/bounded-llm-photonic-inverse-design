from __future__ import annotations

import argparse
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


def _resolve_existing(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    candidate = base / path
    return candidate


def _case_spectra_path(case_json: str) -> Path:
    case_path = Path(case_json)
    case_id = case_path.stem
    return case_path.parent.parent / "spectra" / f"{case_id}_spectra.csv"


def _load_available_cases(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    rows = []
    for _, row in manifest.iterrows():
        cst_path = Path(str(row["expected_result_csv"]))
        spectra_path = _case_spectra_path(str(row["case_json"]))
        if not cst_path.exists() or not spectra_path.exists():
            continue
        item = row.to_dict()
        item["cst_csv"] = str(cst_path)
        item["spectra_csv"] = str(spectra_path)
        rows.append(item)
    return pd.DataFrame(rows)


def _interp(x: np.ndarray, y: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    return np.interp(target_x, x[order], y[order])


def _plot_case(ax: plt.Axes, row: pd.Series) -> dict[str, float]:
    spectra = pd.read_csv(row["spectra_csv"])
    cst = pd.read_csv(row["cst_csv"])

    freq = spectra["frequency_thz"].to_numpy(dtype=float)
    target = spectra["target_amplitude"].to_numpy(dtype=float)
    candidate = spectra["candidate_amplitude"].to_numpy(dtype=float)
    cst_freq = cst["frequency_thz"].to_numpy(dtype=float)
    cst_amp = cst["cst_amplitude"].to_numpy(dtype=float)
    cst_interp = _interp(cst_freq, cst_amp, freq)

    ax.plot(freq, target, color=MORANDI["charcoal"], linewidth=3.0, label="Target")
    ax.plot(freq, candidate, color=MORANDI["dusty_blue"], linewidth=2.8, linestyle="--", label="Offline candidate")
    ax.plot(cst_freq, cst_amp, color=MORANDI["salmon"], linewidth=2.8, marker="o", markersize=4, label="CST")

    cst_mse = float(np.mean((cst_interp - target) ** 2))
    offline_mse = float(np.mean((candidate - target) ** 2))
    domain_label = "Target-H" if row["target_domain"] == "target_h" else "Target-Self"
    subtitle = f"{domain_label}, {row['structure_type']}, {int(float(row['frequency_min_thz'])) if float(row['frequency_min_thz']) >= 2 else row['frequency_min_thz']}-{int(float(row['frequency_max_thz'])) if float(row['frequency_max_thz']) >= 2 else row['frequency_max_thz']} THz"
    ax.set_title(subtitle)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Amplitude")
    ax.text(
        0.03,
        0.05,
        f"CST MSE={cst_mse:.2e}\nOffline MSE={offline_mse:.2e}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.88},
    )
    style_axis(ax, grid_axis="both", ylim=(-0.04, 1.08))
    return {"cst_mse": cst_mse, "offline_mse": offline_mse}


def plot_composite(cases: pd.DataFrame, output_dir: Path, stem: str, *, max_cases: int, formats: list[str]) -> None:
    if cases.empty:
        raise ValueError("No available CST cases to plot.")
    cases = cases.head(max_cases).reset_index(drop=True)
    n = len(cases)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5.6 * nrows), squeeze=False)
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for idx, (_, row) in enumerate(cases.iterrows()):
        ax = axes[idx // ncols][idx % ncols]
        _plot_case(ax, row)
        add_panel_label(ax, labels[idx])
        if idx == 0:
            ax.legend(frameon=False, loc="upper right")
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    save_figure(fig, output_dir, stem, formats=formats)


def plot_metric_bars(cases: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    rows = []
    for _, row in cases.iterrows():
        spectra = pd.read_csv(row["spectra_csv"])
        cst = pd.read_csv(row["cst_csv"])
        freq = spectra["frequency_thz"].to_numpy(dtype=float)
        target = spectra["target_amplitude"].to_numpy(dtype=float)
        candidate = spectra["candidate_amplitude"].to_numpy(dtype=float)
        cst_interp = _interp(
            cst["frequency_thz"].to_numpy(dtype=float),
            cst["cst_amplitude"].to_numpy(dtype=float),
            freq,
        )
        rows.append(
            {
                "case_id": row["case_id"],
                "short_label": row["case_id"].replace("case_", "").split("_q")[0],
                "offline_mse": float(np.mean((candidate - target) ** 2)),
                "cst_mse": float(np.mean((cst_interp - target) ** 2)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "phase5_cst_plot_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(df))
    width = 0.36
    ax.bar(x - width / 2, df["offline_mse"], width=width, color=MORANDI["dusty_blue"], label="Offline")
    ax.bar(x + width / 2, df["cst_mse"], width=width, color=MORANDI["salmon"], label="CST")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(df["short_label"], rotation=18, ha="right")
    ax.set_ylabel("Amplitude MSE")
    ax.set_xlabel("Case")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="y")
    save_figure(fig, output_dir, "phase5_cst_mse_bars", formats=formats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-manifest", type=Path, default=Path("outputs/phase5_cst_case_study_v1/cst_batch_manifest.csv"))
    parser.add_argument("--si-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/phase5_cst_case_study_v1"))
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--font-scale", choices=["large", "compact"], default="large")
    args = parser.parse_args()

    set_publication_style(args.font_scale)
    main_cases = _load_available_cases(args.main_manifest)
    plot_composite(main_cases, args.output_dir, "phase5_cst_main_overlay", max_cases=args.max_cases, formats=args.formats)
    plot_metric_bars(main_cases, args.output_dir, args.formats)

    if args.si_manifest is not None:
        si_cases = _load_available_cases(args.si_manifest)
        combined = pd.concat([main_cases, si_cases], ignore_index=True).drop_duplicates("case_id", keep="last")
        plot_composite(combined, args.output_dir, "phase5_cst_main_plus_si_overlay", max_cases=args.max_cases + 1, formats=args.formats)

    readme = [
        "# Phase 5 CST Figures",
        "",
        "This folder contains target/offline-candidate/CST amplitude overlays for Phase 5 closed-loop validation.",
        "",
        "Source files:",
        f"- Main manifest: `{args.main_manifest}`",
    ]
    if args.si_manifest is not None:
        readme.append(f"- SI manifest: `{args.si_manifest}`")
    readme.extend(
        [
            "- Spectra: `outputs/phase5_cst_case_study_v1/spectra/*_spectra.csv`",
            "- CST results: `outputs/phase5_cst_case_study_v1/cst_results/*_cst_spectrum.csv` and optional SI result folder",
            "",
            "Conclusion: available cases are plotted only when both the offline spectra CSV and CST spectrum CSV exist.",
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Saved Phase 5 CST figures: {args.output_dir.resolve()}")
    print(main_cases[["case_id", "target_domain", "structure_type", "cst_csv"]].to_string(index=False))


if __name__ == "__main__":
    main()
