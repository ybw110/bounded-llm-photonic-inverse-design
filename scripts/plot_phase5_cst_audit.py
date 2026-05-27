"""
Phase 5 extended CST audit figure.

This script plots the statistical CST audit that complements the four-case Phase 5
unit-cell validation figure. It expects an audit directory containing:

- phase5_cst_candidates.csv
- cst_batch_manifest.csv
- phase5_cst_case_metrics.csv

The figure is designed for the 8 + 8 + 4 audit:
Target-H domain-valid H-shape cases, Target-Self domain-valid cylinder cases,
and Target-H topology-invalid source/cylinder near-misses.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

SAVE_DPI = 600
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13
VALUE_FONTSIZE = 12
LEGEND_FONTSIZE = 12
PANEL_FONTSIZE = 18

PALETTE = {
    "blue": "#81AFD4",
    "pale_blue": "#CAE0E8",
    "red": "#DE7C7D",
    "peach": "#FBCFBB",
    "gray": "#CBCBCB",
    "dark_gray": "#4D4D4D",
}

GROUP_ORDER = ["Target-H valid", "Target-Self valid", "Target-H topology-invalid"]
GROUP_COLORS = {
    "Target-H valid": PALETTE["red"],
    "Target-Self valid": PALETTE["blue"],
    "Target-H topology-invalid": PALETTE["peach"],
    "Other": PALETTE["gray"],
}
GROUP_MARKERS = {
    "Target-H valid": "o",
    "Target-Self valid": "s",
    "Target-H topology-invalid": "X",
    "Other": "D",
}
GROUP_SHORT_LABELS = {
    "Target-H valid": "H valid",
    "Target-Self valid": "Self valid",
    "Target-H topology-invalid": "H invalid",
    "Other": "Other",
}


def setup_basic_matplotlib() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "Arial"


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: tuple[str, ...]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout.*")
        fig.tight_layout()
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.08)


def style_axis(
    ax: plt.Axes,
    *,
    x_grid: bool = False,
    y_grid: bool = True,
    y_lim: tuple[float, float] | None = None,
) -> None:
    if y_grid:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.30, zorder=0)
    if x_grid:
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.7, alpha=0.24, zorder=0)
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


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, low_memory=False)


def _coalesce_columns(df: pd.DataFrame, column: str) -> pd.DataFrame:
    variants = [column, f"{column}_candidate", f"{column}_manifest", f"{column}_metric"]
    existing = [col for col in variants if col in df.columns]
    if not existing:
        return df
    base = df[existing[0]].copy()
    for col in existing[1:]:
        base = base.fillna(df[col])
    df[column] = base
    drop_cols = [col for col in existing[1:] if col in df.columns]
    return df.drop(columns=drop_cols)


def load_audit_table(audit_dir: Path) -> pd.DataFrame:
    metrics = _read_csv_checked(audit_dir / "phase5_cst_case_metrics.csv")
    candidates = _read_csv_checked(audit_dir / "phase5_cst_candidates.csv")
    manifest_path = audit_dir / "cst_batch_manifest.csv"
    manifest = _read_csv_checked(manifest_path) if manifest_path.exists() else pd.DataFrame({"case_id": []})

    df = metrics.merge(candidates, on="case_id", how="left", suffixes=("_metric", "_candidate"))
    if not manifest.empty:
        df = df.merge(
            manifest[["case_id", "selected_bank_domain", "structure_type", "expected_result_csv"]],
            on="case_id",
            how="left",
            suffixes=("", "_manifest"),
        )

    for column in ["target_domain", "selected_bank_domain", "structure_type", "phase5_selection_group"]:
        df = _coalesce_columns(df, column)

    if "domain_valid" not in df.columns:
        df["domain_valid"] = df["target_domain"].astype(str).eq(df["selected_bank_domain"].astype(str))
    else:
        df["domain_valid"] = df["domain_valid"].map(_to_bool)

    df["status"] = df["status"].fillna("missing")
    if "cst_exists" in df.columns:
        df["cst_exists"] = df["cst_exists"].map(_to_bool)
    else:
        df["cst_exists"] = False

    numeric_cols = [
        "candidate_vs_target_amp_mse",
        "cst_vs_target_amp_mse",
        "cst_vs_candidate_amp_mse",
        "selected_rank",
        "final_fewshot_cost",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["audit_group"] = "Other"
    df.loc[
        (df["target_domain"].astype(str) == "target_h") & df["domain_valid"],
        "audit_group",
    ] = "Target-H valid"
    df.loc[
        (df["target_domain"].astype(str) == "target_self") & df["domain_valid"],
        "audit_group",
    ] = "Target-Self valid"
    df.loc[
        (df["target_domain"].astype(str) == "target_h") & (~df["domain_valid"]),
        "audit_group",
    ] = "Target-H topology-invalid"

    ok = df[(df["cst_exists"]) & (df["status"].astype(str) == "ok")].copy()
    if ok.empty:
        raise ValueError(
            "No completed CST spectra were found. Run the CST batch and analyze_phase5_cst_results first."
        )

    eps = 1e-14
    ok["offline_mse_plot"] = ok["candidate_vs_target_amp_mse"].clip(lower=eps)
    ok["cst_mse_plot"] = ok["cst_vs_target_amp_mse"].clip(lower=eps)
    ok["cst_vs_offline_ratio"] = ok["cst_mse_plot"] / ok["offline_mse_plot"]
    ok["cst_minus_offline_log10_mse"] = np.log10(ok["cst_mse_plot"]) - np.log10(ok["offline_mse_plot"])
    return ok.reset_index(drop=True)


def ordered_groups(df: pd.DataFrame) -> list[str]:
    groups = [group for group in GROUP_ORDER if group in set(df["audit_group"])]
    extra = [group for group in df["audit_group"].dropna().unique().tolist() if group not in groups]
    return groups + extra


def draw_scatter_offline_vs_cst(ax: plt.Axes, df: pd.DataFrame, threshold: float, panel: str = "a") -> None:
    for group in ordered_groups(df):
        sub = df[df["audit_group"] == group]
        ax.scatter(
            sub["offline_mse_plot"],
            sub["cst_mse_plot"],
            s=90,
            marker=GROUP_MARKERS.get(group, "o"),
            color=GROUP_COLORS.get(group, PALETTE["gray"]),
            edgecolor="white",
            linewidth=0.9,
            alpha=0.92,
            label=f"{GROUP_SHORT_LABELS.get(group, group)} (n={len(sub)})",
            zorder=4,
        )

    values = np.concatenate([df["offline_mse_plot"].to_numpy(), df["cst_mse_plot"].to_numpy(), [threshold]])
    vmin = 10 ** np.floor(np.log10(np.nanmin(values[values > 0])) - 0.2)
    vmax = 10 ** np.ceil(np.log10(np.nanmax(values)) + 0.2)
    ax.plot([vmin, vmax], [vmin, vmax], color=PALETTE["dark_gray"], linestyle="--", linewidth=1.2, label="y = x")
    ax.axhline(threshold, color=PALETTE["red"], linestyle=":", linewidth=1.2, zorder=9)
    ax.axvline(threshold, color=PALETTE["red"], linestyle=":", linewidth=1.2, zorder=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel("Offline candidate MSE")
    ax.set_ylabel("CST vs target MSE")
    ax.set_title("Offline-to-CST error transfer", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10)
    style_axis(ax, x_grid=True, y_grid=True)
    legend = ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="lower left", handletextpad=0.4, borderaxespad=0.2)
    legend.set_zorder(20)
    add_panel_label(ax, panel)


def _draw_box_strip(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    threshold: float | None = None,
    panel: str = "b",
    log_scale: bool = True,
) -> None:
    groups = ordered_groups(df)
    rng = np.random.default_rng(48)
    positions = np.arange(len(groups), dtype=float)
    data = [df[df["audit_group"] == group][metric].dropna().to_numpy(dtype=float) for group in groups]

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.2},
        boxprops={"linewidth": 1.0, "edgecolor": PALETTE["dark_gray"]},
        whiskerprops={"linewidth": 1.0, "color": PALETTE["dark_gray"]},
        capprops={"linewidth": 1.0, "color": PALETTE["dark_gray"]},
    )
    for patch, group in zip(box["boxes"], groups):
        patch.set_facecolor(GROUP_COLORS.get(group, PALETTE["gray"]))
        patch.set_alpha(0.58)

    label_specs: list[tuple[float, str, np.ndarray, bool]] = []
    for x, group, values in zip(positions, groups, data):
        jitter = rng.normal(0, 0.055, size=len(values))
        ax.scatter(
            np.full(len(values), x) + jitter,
            values,
            s=52,
            marker=GROUP_MARKERS.get(group, "o"),
            color=GROUP_COLORS.get(group, PALETTE["gray"]),
            edgecolor="white",
            linewidth=0.7,
            alpha=0.95,
            zorder=4,
        )
        label_specs.append((x, group, values, bool(x == positions[-1])))

    if threshold is not None:
        ax.axhline(
            threshold,
            color=PALETTE["red"],
            linestyle=":",
            linewidth=1.2,
            label=f"threshold={threshold:g}",
            zorder=9,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([GROUP_SHORT_LABELS.get(group, group) for group in groups], rotation=25, ha="right")
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10)
    style_axis(ax, y_grid=True)

    if label_specs:
        y_min, y_max = ax.get_ylim()
        if log_scale:
            log_min = float(np.log10(max(y_min, 1e-300)))
            log_max = float(np.log10(max(y_max, 1e-300)))
            log_span = max(log_max - log_min, 1e-6)
            log_mid = log_min + 0.50 * log_span
            y_floor = 10 ** (log_min + 0.05 * log_span)
            y_ceil = 10 ** (log_max - 0.05 * log_span)
        else:
            y_span = max(y_max - y_min, 1e-6)
            y_mid = y_min + 0.50 * y_span

        for x, _group, values, is_rightmost in label_specs:
            if len(values) == 0:
                continue
            clean = values[np.isfinite(values)]
            if len(clean) == 0:
                continue
            if log_scale:
                clean = clean[clean > 0]
            if len(clean) == 0:
                continue

            q1, q3 = np.nanpercentile(clean, [25, 75])
            if log_scale:
                q1_log = float(np.log10(max(q1, 1e-300)))
                q3_log = float(np.log10(max(q3, 1e-300)))
                center_log = 0.5 * (q1_log + q3_log)
                box_is_high = center_log > log_mid
                label_gap = 0.045 * log_span
                if box_is_high:
                    y_text = max(10 ** (q1_log - label_gap), y_floor)
                    va = "top"
                else:
                    y_text = min(10 ** (q3_log + label_gap), y_ceil)
                    va = "bottom"
            else:
                center = 0.5 * (q1 + q3)
                box_is_high = center > y_mid
                label_gap = 0.045 * y_span
                if box_is_high:
                    y_text = max(q1 - label_gap, y_min + 0.05 * y_span)
                    va = "top"
                else:
                    y_text = min(q3 + label_gap, y_max - 0.05 * y_span)
                    va = "bottom"

            x_text = x - 0.24 if is_rightmost else x + 0.24
            ha = "right" if is_rightmost else "left"
            ax.text(
                x_text,
                y_text,
                f"n={len(clean)}",
                ha=ha,
                va=va,
                fontsize=VALUE_FONTSIZE,
                zorder=30,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
            )
    if threshold is not None:
        legend = ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="best")
        legend.set_zorder(20)
    add_panel_label(ax, panel)


def draw_success_bar(ax: plt.Axes, df: pd.DataFrame, threshold: float, panel: str = "d") -> pd.DataFrame:
    rows = []
    for group in ordered_groups(df):
        sub = df[df["audit_group"] == group]
        success = sub["cst_vs_target_amp_mse"] <= threshold
        rows.append({"audit_group": group, "n": len(sub), "cst_success_rate": float(success.mean())})
    summary = pd.DataFrame(rows)

    x = np.arange(len(summary), dtype=float)
    bars = ax.bar(
        x,
        summary["cst_success_rate"],
        width=0.62,
        color=[GROUP_COLORS.get(group, PALETTE["gray"]) for group in summary["audit_group"]],
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_SHORT_LABELS.get(group, group) for group in summary["audit_group"]], rotation=25, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(f"CST success rate\n(MSE <= {threshold:g})")
    ax.set_title("CST success after physical export", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=10)
    style_axis(ax, y_grid=True)
    for bar, (_, row) in zip(bars, summary.iterrows()):
        label_y = max(float(row["cst_success_rate"]) - 0.055, 0.05)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{row['cst_success_rate']:.2f}\nn={int(row['n'])}",
            ha="center",
            va="top",
            fontsize=VALUE_FONTSIZE,
            color="#222222",
            zorder=20,
        )
    add_panel_label(ax, panel)
    return summary


def draw_composite(df: pd.DataFrame, output_dir: Path, formats: tuple[str, ...], threshold: float) -> plt.Figure:
    fig = plt.figure(figsize=(18.5, 4.5))
    axes = [
        fig.add_axes([0.055, 0.28, 0.205, 0.55]),
        fig.add_axes([0.305, 0.28, 0.185, 0.55]),
        fig.add_axes([0.545, 0.28, 0.185, 0.55]),
        fig.add_axes([0.790, 0.28, 0.185, 0.55]),
    ]
    draw_scatter_offline_vs_cst(axes[0], df, threshold, panel="h")
    _draw_box_strip(
        axes[1],
        df,
        metric="cst_vs_target_amp_mse",
        ylabel="CST vs target MSE",
        title="CST error by audit group",
        threshold=threshold,
        panel="i",
    )
    _draw_box_strip(
        axes[2],
        df,
        metric="cst_minus_offline_log10_mse",
        ylabel=r"$\Delta\log_{10}$ MSE" + "\n(CST - offline)",
        title="CST error increase after export",
        threshold=None,
        panel="j",
        log_scale=False,
    )
    success_summary = draw_success_bar(axes[3], df, threshold, panel="k")

    df.to_csv(output_dir / "phase5_cst_audit_plot_source_data.csv", index=False)
    success_summary.to_csv(output_dir / "phase5_cst_audit_success_summary.csv", index=False)
    for fmt in formats:
        fig.savefig(output_dir / f"phase5_cst_audit_composite.{fmt}", dpi=SAVE_DPI)
        fig.savefig(
            output_dir / f"phase5_cst_audit_composite_hijk.{fmt}",
            dpi=SAVE_DPI,
        )
        fig.savefig(
            output_dir / f"phase5_cst_audit_composite_hijk_plus2.{fmt}",
            dpi=SAVE_DPI,
        )
        fig.savefig(
            output_dir / f"phase5_cst_audit_composite_hijk_delta_log10.{fmt}",
            dpi=SAVE_DPI,
        )
    return fig


def draw_individual_panels(df: pd.DataFrame, output_dir: Path, formats: tuple[str, ...], threshold: float) -> list[plt.Figure]:
    figures: list[plt.Figure] = []

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    draw_scatter_offline_vs_cst(ax, df, threshold, panel="")
    save_figure(fig, output_dir, "panel_a_offline_vs_cst_mse_scatter", formats)
    figures.append(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    _draw_box_strip(
        ax,
        df,
        metric="cst_vs_target_amp_mse",
        ylabel="CST vs target MSE",
        title="CST error by audit group",
        threshold=threshold,
        panel="",
    )
    save_figure(fig, output_dir, "panel_b_cst_mse_by_group", formats)
    figures.append(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    _draw_box_strip(
        ax,
        df,
        metric="cst_minus_offline_log10_mse",
        ylabel=r"$\Delta\log_{10}$ MSE" + "\n(CST - offline)",
        title="CST error increase after export",
        threshold=None,
        panel="",
        log_scale=False,
    )
    save_figure(fig, output_dir, "panel_c_delta_log10_mse_by_group", formats)
    figures.append(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    draw_success_bar(ax, df, threshold, panel="")
    save_figure(fig, output_dir, "panel_d_cst_success_rate_by_group", formats)
    figures.append(fig)
    return figures


def write_readme(output_dir: Path, audit_dir: Path, df: pd.DataFrame, threshold: float) -> None:
    group_summary = (
        df.groupby("audit_group", dropna=False)
        .agg(
            n=("case_id", "count"),
            median_offline_mse=("candidate_vs_target_amp_mse", "median"),
            median_cst_mse=("cst_vs_target_amp_mse", "median"),
            median_delta_log10_mse=("cst_minus_offline_log10_mse", "median"),
            cst_success_rate=("cst_vs_target_amp_mse", lambda s: float((s <= threshold).mean())),
        )
        .reset_index()
    )
    lines = [
        "# Phase 5 extended CST audit figures",
        "",
        "本文件夹保存 Phase 5 扩展 CST audit 的统计图，用于回应“4 个 CST case 数量偏少”的审稿风险。",
        "",
        "## Source files",
        "",
        f"- Candidate table: `{audit_dir / 'phase5_cst_candidates.csv'}`",
        f"- CST manifest: `{audit_dir / 'cst_batch_manifest.csv'}`",
        f"- CST metrics: `{audit_dir / 'phase5_cst_case_metrics.csv'}`",
        "",
        "## Figure panels",
        "",
        "- `phase5_cst_audit_composite`: 主统计图，包含 offline MSE vs CST MSE scatter、CST MSE 分组箱线图、CST/offline MSE ratio、CST success rate。",
        "- `panel_a_offline_vs_cst_mse_scatter`: 离线误差与 CST 误差的对应关系。",
        "- `panel_b_cst_mse_by_group`: Target-H、Target-Self、topology-invalid 的 CST 误差分布。",
        "- `panel_c_cst_offline_ratio_by_group`: 从离线候选到 CST 仿真的误差放大倍数。",
        "- `panel_d_cst_success_rate_by_group`: 以 amplitude MSE 阈值判断的 CST 成功率。",
        "",
        "## Conclusion",
        "",
        "这组图用于说明：代表性 planner-selected candidates 可以被系统导出到 CST 并进行统计审计；Target-H topology-shift 通常比 Target-Self calibration-like shift 更难；而 topology-invalid source/cylinder near-miss 即使谱线接近，也应作为 domain-invalid control 单独报告。",
        "",
        "## Group summary",
        "",
        "```text",
        group_summary.to_string(index=False),
        "```",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, default=Path("outputs/phase5_cst_audit20_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/phase5_cst_audit20_v1"))
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_basic_matplotlib()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    audit_dir = args.audit_dir if args.audit_dir.is_absolute() else PROJECT_ROOT / args.audit_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_audit_table(audit_dir)
    df.to_csv(output_dir / "phase5_cst_audit_clean_table.csv", index=False)
    write_readme(output_dir, audit_dir, df, args.threshold)

    figures = [draw_composite(df, output_dir, tuple(args.formats), args.threshold)]
    figures.extend(draw_individual_panels(df, output_dir, tuple(args.formats), args.threshold))

    print(f"Saved Phase 5 CST audit figures: {output_dir.resolve()}")
    print(
        df[
            [
                "case_id",
                "audit_group",
                "candidate_vs_target_amp_mse",
                "cst_vs_target_amp_mse",
                "cst_vs_offline_ratio",
                "cst_minus_offline_log10_mse",
            ]
        ].to_string(index=False)
    )

    if args.no_show:
        for fig in figures:
            plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
