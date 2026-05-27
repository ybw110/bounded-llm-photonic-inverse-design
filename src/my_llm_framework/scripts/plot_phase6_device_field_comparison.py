"""
Phase 6 metalens fast-mode CST field comparison figure.

This script follows the manuscript plotting style used by plot_phase3.py,
plot_phase5.py, and plot_phase6_meta-filter.py:

1. Arial typography with editable PDF text.
2. Morandi-style method colors.
3. Compact multi-panel layout.
4. Composite PNG/PDF plus standalone panel PNG/PDF exports.
5. Source-data CSVs and a short README in the figure output folder.

The expected input is one or more CST ASCII field exports, each paired with the
corresponding Phase 6 metalens design JSON:

    --run Label=field_ascii_path=design_json_path

For the current paper figure, use three runs:

    Manual, DeepSeek, and GLM-5.1.

Important interpretation note:
These are fast-mode CST field maps. They are useful for qualitative device-level
checking, but they should not be described as fully mesh-converged final CST
benchmarks.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from my_llm_framework.scripts.plot_phase5_hero_field import _read_field_ascii


# -----------------------------------------------------------------------------
# Global plotting configuration.
# These constants deliberately mirror the "metasurface-paper-figures" style used
# by the standalone Phase 3/4/5/6 plotting scripts.
# -----------------------------------------------------------------------------

SAVE_DPI = 600

TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
VALUE_FONTSIZE = 12
LEGEND_FONTSIZE = 13
PANEL_FONTSIZE = 18

PALETTE = {
    "blue": "#81AFD4",
    "pale_blue": "#CAE0E8",
    "red": "#DE7C7D",
    "peach": "#FBCFBB",
    "gray": "#CBCBCB",
    "dark_gray": "#4D4D4D",
}

METHOD_COLORS = {
    "Manual": PALETTE["blue"],
    "DeepSeek": PALETTE["red"],
    "GLM-5.1": PALETTE["peach"],
    "GLM": PALETTE["peach"],
}

FIELD_CMAP = "magma"


def setup_basic_matplotlib(font_family: str = "Arial") -> None:
    """Apply the paper-wide matplotlib style."""

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_family, "DejaVu Sans"]
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def style_axis(
    ax: plt.Axes,
    *,
    x_grid: bool = False,
    y_grid: bool = True,
    y_lim: tuple[float, float] | None = None,
) -> None:
    """Use the same clean axis styling as the other manuscript figures."""

    if y_grid:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.28, zorder=0)
    if x_grid:
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.7, alpha=0.22, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=1.0, length=4)
    ax.xaxis.label.set_size(LABEL_FONTSIZE)
    ax.yaxis.label.set_size(LABEL_FONTSIZE)
    if y_lim is not None:
        ax.set_ylim(*y_lim)


def add_panel_label(ax: plt.Axes, label: str, *, x: float = -0.12, y: float = 1.08) -> None:
    """Add a bold lowercase panel label."""

    if not label:
        return
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_FONTSIZE,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=30,
    )


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    file_stem: str,
    *,
    formats: list[str],
    tight: bool = True,
) -> None:
    """Save a figure with paper-ready DPI and editable PDF text."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if tight:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout.*",
            )
            fig.tight_layout()
    for fmt in formats:
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.08}
        if fmt.lower() in {"png", "tif", "tiff", "jpg", "jpeg"}:
            save_kwargs["dpi"] = SAVE_DPI
        fig.savefig(output_dir / f"{file_stem}.{fmt}", **save_kwargs)


def _resolve_path(path: Path) -> Path:
    """Resolve a path passed from the command line while keeping relative paths usable."""

    if path.exists():
        return path
    project_root = Path.cwd()
    candidate = project_root / path
    if candidate.exists():
        return candidate
    return path


def _safe_filename(label: str) -> str:
    """Convert a plot label into a filename-safe lowercase token."""

    return (
        label.lower()
        .replace(" ", "_")
        .replace(".", "")
        .replace("-", "")
        .replace("/", "_")
        .replace("\\", "_")
    )


def _method_color(label: str) -> str:
    """Return the preferred color for a method label."""

    if label in METHOD_COLORS:
        return METHOD_COLORS[label]
    lower = label.lower()
    if "manual" in lower:
        return PALETTE["blue"]
    if "deep" in lower:
        return PALETTE["red"]
    if "glm" in lower:
        return PALETTE["peach"]
    return PALETTE["gray"]


def _parse_run(value: str) -> tuple[str, Path, Path]:
    """Parse one --run argument."""

    parts = value.split("=", 2)
    if len(parts) != 3:
        raise ValueError("--run must have the form Label=field_ascii_path=design_json_path")
    label, field_path, design_json = parts
    return label, _resolve_path(Path(field_path)), _resolve_path(Path(design_json))


def _load_design(path: Path) -> dict:
    """Read a Phase 6 metalens design JSON."""

    if not path.exists():
        raise FileNotFoundError(f"Missing design JSON: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_design_csv_path(design: dict, design_json: Path, key: str) -> Path:
    """Resolve a CSV path stored inside a design JSON.

    The design files usually store paths relative to the MY-LLM-Framework
    project root, for example outputs/phase6_.../phase6_metalens_units.csv.
    This helper keeps that convention working while also supporting absolute
    paths and design-folder-relative paths.
    """

    if key not in design:
        raise KeyError(f"Design JSON does not contain `{key}`: {design_json}")

    raw_path = Path(str(design[key]))
    candidates = [
        raw_path,
        Path.cwd() / raw_path,
        design_json.parent / raw_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve `{key}` from {design_json}: {raw_path}")


def _load_phase_profile(design: dict, design_json: Path) -> pd.DataFrame:
    """Load the selected metalens unit profile used for design-side panels."""

    profile_path = _resolve_design_csv_path(design, design_json, "phase_profile_csv")
    profile = pd.read_csv(profile_path)
    required_columns = {
        "x_um",
        "target_phase_rad",
        "selected_phase_rad",
        "phase_error_rad",
        "selected_amplitude",
    }
    missing = sorted(required_columns - set(profile.columns))
    if missing:
        raise ValueError(f"Missing columns in {profile_path}: {missing}")
    return profile.sort_values("x_um").reset_index(drop=True)


def _pivot_field(field: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert long-form CST field points into an x-z image matrix."""

    pivot = field.pivot_table(index="z", columns="x", values="field_abs", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    x = pivot.columns.to_numpy(dtype=float)
    z = pivot.index.to_numpy(dtype=float)
    values = pivot.to_numpy(dtype=float)
    return x, z, values


def _run_metrics(
    label: str,
    field_path: Path,
    design_json: Path,
    field: pd.DataFrame,
    x: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
) -> dict:
    """Calculate focus-related metrics for one metalens CST export."""

    design = _load_design(design_json)
    focal_length = float(design["focal_length_um"])
    center_idx = int(np.nanargmin(np.abs(x - 0.0)))
    focal_idx = int(np.nanargmin(np.abs(z - focal_length)))

    global_flat_idx = int(np.nanargmax(values))
    global_z_idx, global_x_idx = np.unravel_index(global_flat_idx, values.shape)

    on_axis = values[:, center_idx]
    on_axis_peak_idx = int(np.nanargmax(on_axis))

    focal_profile = values[focal_idx, :]
    focal_peak_x_idx = int(np.nanargmax(focal_profile))

    # Focal-window metric: restrict the search to a z-window near the designed
    # focus so a strong near-source hot spot does not dominate the comparison.
    window_mask = (z >= 0.75 * focal_length) & (z <= 1.3125 * focal_length)
    if np.any(window_mask):
        window_values = values[window_mask, :]
        window_z = z[window_mask]
        window_flat_idx = int(np.nanargmax(window_values))
        window_z_idx, window_x_idx = np.unravel_index(window_flat_idx, window_values.shape)
        window_peak_z = float(window_z[window_z_idx])
        window_peak_x = float(x[window_x_idx])
        window_peak_field = float(window_values[window_z_idx, window_x_idx])
    else:
        window_peak_z = np.nan
        window_peak_x = np.nan
        window_peak_field = np.nan

    return {
        "run_label": label,
        "field_path": str(field_path),
        "design_json": str(design_json),
        "num_points": int(len(field)),
        "x_min_um": float(np.nanmin(x)),
        "x_max_um": float(np.nanmax(x)),
        "z_min_um": float(np.nanmin(z)),
        "z_max_um": float(np.nanmax(z)),
        "design_focal_length_um": focal_length,
        "global_peak_x_um": float(x[global_x_idx]),
        "global_peak_z_um": float(z[global_z_idx]),
        "global_peak_field": float(values[global_z_idx, global_x_idx]),
        "on_axis_peak_z_um": float(z[on_axis_peak_idx]),
        "on_axis_peak_field": float(on_axis[on_axis_peak_idx]),
        "on_axis_focus_distance_um": float(abs(z[on_axis_peak_idx] - focal_length)),
        "focal_plane_z_um": float(z[focal_idx]),
        "focal_plane_peak_x_um": float(x[focal_peak_x_idx]),
        "focal_plane_peak_field": float(focal_profile[focal_peak_x_idx]),
        "focal_plane_center_field": float(focal_profile[center_idx]),
        "focal_plane_peak_offset_um": float(abs(x[focal_peak_x_idx])),
        "focal_window_peak_x_um": window_peak_x,
        "focal_window_peak_z_um": window_peak_z,
        "focal_window_peak_field": window_peak_field,
        "mean_phase_error_rad": float(design.get("mean_phase_error_rad", np.nan)),
        "max_phase_error_rad": float(design.get("max_phase_error_rad", np.nan)),
        "mean_selected_amplitude": float(design.get("mean_selected_amplitude", np.nan)),
        "min_selected_amplitude": float(design.get("min_selected_amplitude", np.nan)),
    }


def _load_runs(runs: list[str]) -> tuple[list[dict], pd.DataFrame, float]:
    """Load all requested field exports and return parsed data plus metrics."""

    if len(runs) < 2:
        raise ValueError("Provide at least two --run entries.")
    if len(runs) > 3:
        raise ValueError("This paper layout is designed for two or three runs. Use three for Manual/DeepSeek/GLM-5.1.")

    loaded: list[dict] = []
    metrics_rows: list[dict] = []
    global_max = 0.0

    for run in runs:
        label, field_path, design_json = _parse_run(run)
        if not field_path.exists():
            raise FileNotFoundError(
                f"Missing CST field export for {label}: {field_path}\n"
                "Run the fast-mode CST command first, then re-run this plotting script."
            )
        design = _load_design(design_json)
        phase_profile = _load_phase_profile(design, design_json)
        field = _read_field_ascii(field_path)
        x, z, values = _pivot_field(field)
        global_max = max(global_max, float(np.nanmax(values)))
        metrics_rows.append(_run_metrics(label, field_path, design_json, field, x, z, values))
        loaded.append(
            {
                "label": label,
                "field_path": field_path,
                "design_json": design_json,
                "design": design,
                "phase_profile": phase_profile,
                "field": field,
                "x": x,
                "z": z,
                "values": values,
            }
        )

    return loaded, pd.DataFrame(metrics_rows), global_max + 1e-12


def draw_field_map(
    ax: plt.Axes,
    item: dict,
    *,
    common_max: float,
    focal_length_um: float,
    panel_label: str | None = None,
) -> object:
    """Draw one common-normalized x-z field map."""

    x = item["x"]
    z = item["z"]
    values_norm = item["values"] / common_max

    image = ax.imshow(
        values_norm,
        origin="lower",
        aspect="auto",
        extent=[float(x.min()), float(x.max()), float(z.min()), float(z.max())],
        cmap=FIELD_CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.axhline(focal_length_um, color="white", linewidth=2.0, linestyle="--")
    ax.axvline(0.0, color="white", linewidth=1.5, linestyle=":", alpha=0.88)
    ax.set_xlabel("x (um)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("z (um)", fontsize=LABEL_FONTSIZE)
    ax.set_title(f"{item['label']} field map", fontsize=TITLE_FONTSIZE, fontweight="normal")
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=1.0, length=4)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_linewidth(1.0)
    if panel_label:
        add_panel_label(ax, panel_label)
    return image


def draw_on_axis_profile(
    ax: plt.Axes,
    loaded: list[dict],
    *,
    common_max: float,
    focal_length_um: float,
    panel_label: str | None = None,
) -> None:
    """Draw |E| along the optical axis x=0."""

    for item in loaded:
        x = item["x"]
        z = item["z"]
        values_norm = item["values"] / common_max
        center_idx = int(np.nanargmin(np.abs(x - 0.0)))
        ax.plot(
            z,
            values_norm[:, center_idx],
            color=_method_color(item["label"]),
            linewidth=2.8,
            label=item["label"],
        )
    ax.axvline(focal_length_um, color=PALETTE["dark_gray"], linewidth=2.0, linestyle="--", label="Design focus")
    ax.set_xlabel("z (um)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Common-normalized |E|", fontsize=LABEL_FONTSIZE)
    ax.set_title("On-axis profile", fontsize=TITLE_FONTSIZE, fontweight="normal")
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    style_axis(ax, x_grid=True, y_grid=True)
    if panel_label:
        add_panel_label(ax, panel_label)


def draw_focal_plane_profile(
    ax: plt.Axes,
    loaded: list[dict],
    *,
    common_max: float,
    focal_length_um: float,
    panel_label: str | None = None,
) -> None:
    """Draw |E| across x at the sampled focal plane closest to the target focus."""

    focal_plane_z = np.nan
    for item in loaded:
        x = item["x"]
        z = item["z"]
        values_norm = item["values"] / common_max
        focal_idx = int(np.nanargmin(np.abs(z - focal_length_um)))
        focal_plane_z = float(z[focal_idx])
        ax.plot(
            x,
            values_norm[focal_idx, :],
            color=_method_color(item["label"]),
            linewidth=2.8,
            label=item["label"],
        )
    ax.axvline(0.0, color=PALETTE["dark_gray"], linewidth=1.8, linestyle=":")
    ax.set_xlabel("x (um)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Common-normalized |E|", fontsize=LABEL_FONTSIZE)
    ax.set_title(f"Focal-plane profile, z={focal_plane_z:.1f} um", fontsize=TITLE_FONTSIZE, fontweight="normal")
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    style_axis(ax, x_grid=True, y_grid=True)
    if panel_label:
        add_panel_label(ax, panel_label)


def _add_bar_values(ax: plt.Axes, bars, *, fmt: str = "{:.2f}") -> None:
    """Add small value labels above bars."""

    y_min, y_max = ax.get_ylim()
    y_span = max(y_max - y_min, 1e-9)
    for bar in bars:
        height = float(bar.get_height())
        if not np.isfinite(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.018 * y_span,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=VALUE_FONTSIZE,
            color="#222222",
            rotation=0,
            zorder=20,
        )


def draw_strength_metrics(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    *,
    common_max: float,
    panel_label: str | None = None,
) -> None:
    """Draw field-strength metrics near the designed focus."""

    labels = metrics["run_label"].tolist()
    x_pos = np.arange(len(labels))
    width = 0.34
    window_norm = metrics["focal_window_peak_field"].to_numpy(dtype=float) / common_max
    center_norm = metrics["focal_plane_center_field"].to_numpy(dtype=float) / common_max

    bars_window = ax.bar(
        x_pos - width / 2,
        window_norm,
        width=width,
        color=[_method_color(label) for label in labels],
        alpha=0.92,
        label="Focal-window peak",
        zorder=5,
    )
    bars_center = ax.bar(
        x_pos + width / 2,
        center_norm,
        width=width,
        color=PALETTE["pale_blue"],
        edgecolor=[_method_color(label) for label in labels],
        linewidth=1.2,
        alpha=0.70,
        hatch="//",
        label="Focal-plane center",
        zorder=5,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=TICK_FONTSIZE)
    ax.set_ylabel("Common-normalized |E|", fontsize=LABEL_FONTSIZE)
    ax.set_title("Focusing strength", fontsize=TITLE_FONTSIZE, fontweight="normal")
    y_max = max(1.05, float(np.nanmax([np.nanmax(window_norm), np.nanmax(center_norm)])) * 1.22)
    style_axis(ax, y_grid=True, y_lim=(0.0, y_max))
    _add_bar_values(ax, bars_window, fmt="{:.2f}")
    _add_bar_values(ax, bars_center, fmt="{:.2f}")
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper left")
    if panel_label:
        add_panel_label(ax, panel_label)


def draw_location_metrics(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    *,
    panel_label: str | None = None,
) -> None:
    """Draw distance-to-focus metrics. Lower is better."""

    metric_names = ["on_axis_focus_distance_um", "focal_plane_peak_offset_um"]
    metric_labels = ["On-axis peak\ndistance to focus", "Focal-plane\npeak offset"]
    labels = metrics["run_label"].tolist()
    x_metric = np.arange(len(metric_names))
    bar_width = 0.78 / len(labels)

    for idx, label in enumerate(labels):
        row = metrics[metrics["run_label"] == label].iloc[0]
        offsets = x_metric - 0.39 + bar_width / 2 + idx * bar_width
        values = [float(row[name]) for name in metric_names]
        bars = ax.bar(
            offsets,
            values,
            width=bar_width,
            color=_method_color(label),
            label=label,
            alpha=0.94,
            zorder=5,
        )
        _add_bar_values(ax, bars, fmt="{:.1f}")

    ax.set_xticks(x_metric)
    ax.set_xticklabels(metric_labels, fontsize=TICK_FONTSIZE)
    ax.set_ylabel("Distance (um)", fontsize=LABEL_FONTSIZE)
    ax.set_title("Focus-location error", fontsize=TITLE_FONTSIZE, fontweight="normal")
    style_axis(ax, y_grid=True)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper right")
    if panel_label:
        add_panel_label(ax, panel_label)


def _write_source_data(output_dir: Path, loaded: list[dict], metrics: pd.DataFrame) -> None:
    """Save parsed field source data and metrics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "phase6_device_field_comparison_metrics.csv", index=False)
    for item in loaded:
        parsed = item["field"].copy()
        parsed["run_label"] = item["label"]
        parsed.to_csv(output_dir / f"phase6_{_safe_filename(item['label'])}_field_parsed.csv", index=False)
        profile = item["phase_profile"].copy()
        profile["run_label"] = item["label"]
        profile.to_csv(output_dir / f"phase6_{_safe_filename(item['label'])}_phase_profile.csv", index=False)


def draw_phase_profile(
    ax: plt.Axes,
    loaded: list[dict],
    *,
    panel_label: str | None = None,
) -> None:
    """Draw the target and selected wrapped phase profiles along the aperture."""

    target_profile = loaded[0]["phase_profile"]
    ax.plot(
        target_profile["x_um"],
        target_profile["target_phase_rad"],
        color=PALETTE["dark_gray"],
        linewidth=2.2,
        linestyle="--",
        label="Target",
        zorder=4,
    )

    for item in loaded:
        profile = item["phase_profile"]
        ax.plot(
            profile["x_um"],
            profile["selected_phase_rad"],
            color=_method_color(item["label"]),
            linewidth=2.0,
            marker="o",
            markersize=4.2,
            markeredgewidth=0.0,
            label=item["label"],
            zorder=5,
        )

    ax.set_xlabel("Position x (um)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Wrapped phase (rad)", fontsize=LABEL_FONTSIZE)
    ax.set_title("Metalens phase profile", fontsize=TITLE_FONTSIZE, fontweight="normal")
    ax.set_ylim(-3.35, 3.35)
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    style_axis(ax, x_grid=True, y_grid=True)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, ncol=2, loc="upper center", bbox_to_anchor=(0.50, 1.03))
    if panel_label:
        add_panel_label(ax, panel_label)


def draw_phase_error_profile(
    ax: plt.Axes,
    loaded: list[dict],
    *,
    panel_label: str | None = None,
) -> None:
    """Draw absolute selected-unit phase error along the aperture."""

    max_error = 0.0
    for item in loaded:
        profile = item["phase_profile"]
        error = np.abs(profile["phase_error_rad"].to_numpy(dtype=float))
        max_error = max(max_error, float(np.nanmax(error)))
        ax.plot(
            profile["x_um"],
            error,
            color=_method_color(item["label"]),
            linewidth=2.0,
            marker="o",
            markersize=4.2,
            markeredgewidth=0.0,
            label=item["label"],
            zorder=5,
        )

    ax.set_xlabel("Position x (um)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("|Phase error| (rad)", fontsize=LABEL_FONTSIZE)
    ax.set_title("Phase error along aperture", fontsize=TITLE_FONTSIZE, fontweight="normal")
    style_axis(ax, x_grid=True, y_grid=True, y_lim=(0.0, max(0.006, max_error * 1.18)))
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper right")
    if panel_label:
        add_panel_label(ax, panel_label)


def draw_selected_amplitude(
    ax: plt.Axes,
    loaded: list[dict],
    *,
    panel_label: str | None = None,
) -> None:
    """Draw selected unit amplitude along the metalens aperture."""

    for item in loaded:
        profile = item["phase_profile"]
        ax.plot(
            profile["x_um"],
            profile["selected_amplitude"],
            color=_method_color(item["label"]),
            linewidth=2.0,
            marker="o",
            markersize=4.2,
            markeredgewidth=0.0,
            label=item["label"],
            zorder=5,
        )

    ax.set_xlabel("Position x (um)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Amplitude", fontsize=LABEL_FONTSIZE)
    ax.set_title("Selected unit amplitude", fontsize=TITLE_FONTSIZE, fontweight="normal")
    style_axis(ax, x_grid=True, y_grid=True, y_lim=(0.45, 1.04))
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="lower right")
    if panel_label:
        add_panel_label(ax, panel_label)


def draw_field_magnitude_metrics(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    *,
    panel_label: str | None = None,
) -> None:
    """Draw the old-panel-f style raw field magnitude metrics."""

    labels = metrics["run_label"].tolist()
    x_pos = np.arange(len(labels))
    width = 0.34
    focal_plane_peak = metrics["focal_plane_peak_field"].to_numpy(dtype=float)
    focal_window_peak = metrics["focal_window_peak_field"].to_numpy(dtype=float)

    bars_plane = ax.bar(
        x_pos - width / 2,
        focal_plane_peak,
        width=width,
        color=PALETTE["blue"],
        alpha=0.92,
        label="Focal-plane peak",
        zorder=5,
    )
    bars_window = ax.bar(
        x_pos + width / 2,
        focal_window_peak,
        width=width,
        color=PALETTE["red"],
        alpha=0.88,
        label="Focal-window peak",
        zorder=5,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=TICK_FONTSIZE)
    ax.set_ylabel("Field magnitude", fontsize=LABEL_FONTSIZE)
    ax.set_title("Qualitative device-level metrics", fontsize=TITLE_FONTSIZE, fontweight="normal")
    y_max = max(1.0, float(np.nanmax([focal_plane_peak, focal_window_peak])) * 1.20)
    style_axis(ax, y_grid=True, y_lim=(0.0, y_max))
    _add_bar_values(ax, bars_plane, fmt="{:.2f}")
    _add_bar_values(ax, bars_window, fmt="{:.2f}")
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper left")
    if panel_label:
        add_panel_label(ax, panel_label)


def _write_readme(output_dir: Path, loaded: list[dict]) -> None:
    """Write a short provenance note beside the exported figures."""

    lines = [
        "# Phase 6 Metalens Fast-Mode CST Field Comparison",
        "",
        "This folder contains the manuscript-style field comparison figure for the Phase 6 finite metalens designs.",
        "",
        "## Inputs",
        "",
    ]
    for item in loaded:
        lines.append(f"- {item['label']} field export: `{item['field_path']}`")
        lines.append(f"- {item['label']} design JSON: `{item['design_json']}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `phase6_device_field_comparison.png/pdf`: one dense composite figure containing design-side phase evidence, three fast-mode CST field maps, field-profile diagnostics, and device-level metric panels.",
            "- `panel_*.png/pdf`: standalone panels for manual layout editing; the composite figure is the main paper-ready output.",
            "- `phase6_device_field_comparison_metrics.csv`: focus and field-strength metrics.",
            "- `phase6_<method>_field_parsed.csv`: parsed CST field source data.",
            "- `phase6_<method>_phase_profile.csv`: selected unit phase/amplitude source data.",
            "",
            "## Interpretation",
            "",
            "All panels use a common normalization across loaded field maps. These are fast-mode CST outputs and should be presented as qualitative device-level evidence rather than fully mesh-converged final simulations.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def plot_comparison(runs: list[str], output_dir: Path, formats: list[str]) -> None:
    """Create the composite and standalone Phase 6 field comparison panels."""

    loaded, metrics, common_max = _load_runs(runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_source_data(output_dir, loaded, metrics)

    focal_length = float(metrics["design_focal_length_um"].iloc[0])
    # Composite layout:
    # row 1: old a/b/c/f information, namely phase profile, phase error,
    # selected amplitude, and device-level field magnitude metrics;
    # row 2: one common-normalized CST field map per selector plus focus-location
    # error;
    # row 3: two detailed field profiles, each spanning two columns.
    fig = plt.figure(figsize=(22.0, 13.4))
    gs = fig.add_gridspec(
        3,
        4,
        left=0.050,
        right=0.970,
        bottom=0.060,
        top=0.955,
        wspace=0.36,
        hspace=0.52,
        height_ratios=[1.00, 1.05, 0.92],
    )

    ax_phase = fig.add_subplot(gs[0, 0])
    ax_phase_error = fig.add_subplot(gs[0, 1])
    ax_amplitude = fig.add_subplot(gs[0, 2])
    ax_field_magnitude = fig.add_subplot(gs[0, 3])
    ax_maps = [fig.add_subplot(gs[1, col]) for col in range(3)]
    ax_location = fig.add_subplot(gs[1, 3])
    ax_on_axis = fig.add_subplot(gs[2, 0:2])
    ax_focal_plane = fig.add_subplot(gs[2, 2:4])

    draw_phase_profile(ax_phase, loaded, panel_label="a")
    draw_phase_error_profile(ax_phase_error, loaded, panel_label="b")
    draw_selected_amplitude(ax_amplitude, loaded, panel_label="c")
    draw_field_magnitude_metrics(ax_field_magnitude, metrics, panel_label="d")

    last_image = None
    for idx, ax_map in enumerate(ax_maps):
        if idx < len(loaded):
            last_image = draw_field_map(
                ax_map,
                loaded[idx],
                common_max=common_max,
                focal_length_um=focal_length,
                panel_label=chr(ord("e") + idx),
            )
        else:
            ax_map.axis("off")
            ax_map.text(
                0.02,
                0.92,
                "Third selector field\nnot supplied",
                transform=ax_map.transAxes,
                fontsize=TITLE_FONTSIZE,
                color=PALETTE["dark_gray"],
                va="top",
                ha="left",
            )

    if last_image is not None:
        cbar = fig.colorbar(
            last_image,
            ax=ax_maps,
            orientation="horizontal",
            fraction=0.050,
            pad=0.090,
            aspect=35,
        )
        cbar.set_label("Common-normalized |E|", fontsize=LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    draw_location_metrics(ax_location, metrics, panel_label="h")
    draw_on_axis_profile(
        ax_on_axis,
        loaded,
        common_max=common_max,
        focal_length_um=focal_length,
        panel_label="i",
    )
    draw_focal_plane_profile(
        ax_focal_plane,
        loaded,
        common_max=common_max,
        focal_length_um=focal_length,
        panel_label="j",
    )

    save_figure(fig, output_dir, "phase6_device_field_comparison", formats=formats, tight=False)
    plt.close(fig)

    # Standalone panel exports. These are useful for PowerPoint/Illustrator
    # assembly and match the workflow of the other Phase plotting scripts.
    standalone_specs: list[tuple[str, plt.Figure]] = []

    fig_phase, ax_phase_single = plt.subplots(figsize=(5.8, 4.5))
    draw_phase_profile(ax_phase_single, loaded, panel_label=None)
    standalone_specs.append(("panel_a_phase_profile", fig_phase))

    fig_error, ax_error_single = plt.subplots(figsize=(5.8, 4.5))
    draw_phase_error_profile(ax_error_single, loaded, panel_label=None)
    standalone_specs.append(("panel_b_phase_error", fig_error))

    fig_amp, ax_amp_single = plt.subplots(figsize=(5.8, 4.5))
    draw_selected_amplitude(ax_amp_single, loaded, panel_label=None)
    standalone_specs.append(("panel_c_selected_amplitude", fig_amp))

    fig_magnitude, ax_magnitude = plt.subplots(figsize=(5.8, 4.5))
    draw_field_magnitude_metrics(ax_magnitude, metrics, panel_label=None)
    standalone_specs.append(("panel_d_field_magnitude_metrics", fig_magnitude))

    for idx, item in enumerate(loaded):
        panel_fig, panel_ax = plt.subplots(figsize=(5.6, 4.6))
        image = draw_field_map(
            panel_ax,
            item,
            common_max=common_max,
            focal_length_um=focal_length,
            panel_label=None,
        )
        cbar = panel_fig.colorbar(image, ax=panel_ax, fraction=0.045, pad=0.025)
        cbar.set_label("Common-normalized |E|", fontsize=LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
        standalone_specs.append((f"panel_{chr(ord('e') + idx)}_{_safe_filename(item['label'])}_field_map", panel_fig))

    fig_on, ax_on = plt.subplots(figsize=(6.4, 4.8))
    draw_on_axis_profile(ax_on, loaded, common_max=common_max, focal_length_um=focal_length, panel_label=None)
    standalone_specs.append(("panel_i_on_axis_profile", fig_on))

    fig_fp, ax_fp = plt.subplots(figsize=(6.4, 4.8))
    draw_focal_plane_profile(ax_fp, loaded, common_max=common_max, focal_length_um=focal_length, panel_label=None)
    standalone_specs.append(("panel_j_focal_plane_profile", fig_fp))

    fig_strength, ax_strength = plt.subplots(figsize=(6.2, 4.8))
    draw_strength_metrics(ax_strength, metrics, common_max=common_max, panel_label=None)
    standalone_specs.append(("panel_extra_normalized_focusing_strength", fig_strength))

    fig_location, ax_location = plt.subplots(figsize=(6.2, 4.8))
    draw_location_metrics(ax_location, metrics, panel_label=None)
    standalone_specs.append(("panel_h_focus_location_error", fig_location))

    for stem, panel_fig in standalone_specs:
        save_figure(panel_fig, output_dir, stem, formats=formats, tight=True)
        plt.close(panel_fig)

    _write_readme(output_dir, loaded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="Label=field_ascii_path=design_json_path")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/phase6_device_field_comparison_fast_v2"))
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()

    setup_basic_matplotlib()
    plot_comparison(args.run, args.output_dir, args.formats)
    print(f"Saved Phase 6 device field comparison figures: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
