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


def _read_field_ascii(path: Path) -> pd.DataFrame:
    lines = path.read_text(errors="ignore").splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        tokens = line.strip().replace(",", " ").split()
        lowered = [tok.lower() for tok in tokens]
        if "x" in lowered and ("z" in lowered or "y" in lowered):
            header_idx = idx
            break
    if header_idx is None:
        # Fallback: try generic whitespace table without header.
        df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, engine="python")
        if df.shape[1] < 3:
            raise ValueError(f"Cannot parse field ASCII file: {path}")
        df = df.iloc[:, :3]
        df.columns = ["x", "z", "field_abs"]
        return df.apply(pd.to_numeric, errors="coerce").dropna()

    header_line = lines[header_idx]
    is_cst_vector_field = all(token in header_line for token in ["x [um]", "y [um]", "z [um]", "ExRe", "ExIm", "EzIm"])
    if is_cst_vector_field:
        header = ["x", "y", "z", "ExRe", "ExIm", "EyRe", "EyIm", "EzRe", "EzIm"]
    else:
        header = header_line.replace("[", "_").replace("]", "").replace("/", "_").replace(",", " ").split()
    data_start = header_idx + 1
    while data_start < len(lines):
        stripped = lines[data_start].strip()
        if stripped and not set(stripped) <= {"-"}:
            break
        data_start += 1
    data_lines = "\n".join(lines[data_start:])
    from io import StringIO

    df = pd.read_csv(StringIO(data_lines), sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, : len(header)]
    df.columns = header[: df.shape[1]]
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")

    lower_map = {col.lower(): col for col in df.columns}
    x_col = lower_map.get("x", df.columns[0])
    y_col = lower_map.get("y", None)
    z_col = lower_map.get("z", lower_map.get("y", df.columns[1]))

    def col(name: str) -> pd.Series | None:
        key = name.lower()
        if key not in lower_map:
            return None
        return pd.to_numeric(df[lower_map[key]], errors="coerce")

    ex_re, ex_im = col("ExRe"), col("ExIm")
    ey_re, ey_im = col("EyRe"), col("EyIm")
    ez_re, ez_im = col("EzRe"), col("EzIm")
    if ex_re is not None and ex_im is not None and ey_re is not None and ey_im is not None and ez_re is not None and ez_im is not None:
        field_abs = np.sqrt(ex_re**2 + ex_im**2 + ey_re**2 + ey_im**2 + ez_re**2 + ez_im**2)
    else:
        value_candidates = [col_name for col_name in df.columns if col_name not in {x_col, z_col, y_col}]
        value_col = value_candidates[-1] if value_candidates else df.columns[-1]
        field_abs = pd.to_numeric(df[value_col], errors="coerce").abs()
    out = pd.DataFrame(
        {
            "x": pd.to_numeric(df[x_col], errors="coerce"),
            "y": pd.to_numeric(df[y_col], errors="coerce") if y_col is not None else 0.0,
            "z": pd.to_numeric(df[z_col], errors="coerce"),
            "field_abs": field_abs,
        }
    ).dropna()
    return out


def _load_design(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_field(field_path: Path, design_json: Path, output_dir: Path, formats: list[str]) -> None:
    design = _load_design(design_json)
    field = _read_field_ascii(field_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed_csv = output_dir / "phase5_hero_field_parsed.csv"
    field.to_csv(parsed_csv, index=False)

    pivot = field.pivot_table(index="z", columns="x", values="field_abs", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    x = pivot.columns.to_numpy(dtype=float)
    z = pivot.index.to_numpy(dtype=float)
    values = pivot.to_numpy(dtype=float)
    values = values / (np.nanmax(values) + 1e-12)

    focal_length = float(design["focal_length_um"])
    focal_idx = int(np.nanargmin(np.abs(z - focal_length)))
    center_idx = int(np.nanargmin(np.abs(x - 0.0)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax = axes[0]
    im = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=[float(x.min()), float(x.max()), float(z.min()), float(z.max())],
        cmap="viridis",
    )
    ax.axhline(focal_length, color="white", linewidth=2, linestyle="--")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("z (um)")
    ax.set_title("Normalized E-field")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    add_panel_label(ax, "A")

    ax = axes[1]
    ax.plot(x, values[focal_idx, :], color=MORANDI["salmon"], linewidth=3)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("Normalized field")
    ax.set_title(f"Focal-plane profile, z={z[focal_idx]:.1f} um")
    style_axis(ax, grid_axis="both")
    add_panel_label(ax, "B")

    ax = axes[2]
    ax.plot(z, values[:, center_idx], color=MORANDI["dusty_blue"], linewidth=3)
    ax.axvline(focal_length, color=MORANDI["salmon"], linewidth=2.5, linestyle="--", label="Design focus")
    ax.set_xlabel("z (um)")
    ax.set_ylabel("Normalized field")
    ax.set_title("On-axis profile")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="both")
    add_panel_label(ax, "C")

    save_figure(fig, output_dir, "phase5_hero_metalens_field", formats=formats)

    metrics = {
        "field_path": str(field_path),
        "design_json": str(design_json),
        "num_points": int(len(field)),
        "x_min": float(np.nanmin(x)),
        "x_max": float(np.nanmax(x)),
        "z_min": float(np.nanmin(z)),
        "z_max": float(np.nanmax(z)),
        "design_focal_length_um": focal_length,
        "peak_z_um": float(z[int(np.nanargmax(values[:, center_idx]))]),
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "phase5_hero_field_metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field-path", type=Path, required=True)
    parser.add_argument("--design-json", type=Path, default=Path("outputs/phase5_hero_metalens_source_cylinder_v1/phase5_hero_metalens_design.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/phase5_hero_metalens_source_cylinder_v1"))
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--font-scale", choices=["large", "compact"], default="large")
    args = parser.parse_args()

    set_publication_style(args.font_scale)
    plot_field(args.field_path, args.design_json, args.output_dir, args.formats)
    print(f"Saved Phase 5 hero field figures: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
