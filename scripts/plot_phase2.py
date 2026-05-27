"""
Phase 2 non-LLM baseline figure 绘图脚本。

这个脚本用于绘制你示例图里的 c/d/e 三个 panel：

c. Target-H 上不同非 LLM 候选生成方法的 final amplitude-MSE 分布。
   数据来源：
   - retrieval-only top-1: outputs/phase01_retrieval_baselines_v1/query_level_results.csv
   - retrieval + PNN + PSO: outputs/phase2_retrieval_pnn_pso_h_full_strong30_nosr/retrieval_optimizer_query_results.csv
   - retrieval + PNN + CMA-ES: outputs/phase2_retrieval_pnn_cmaes_h_full_strong30_nosr/retrieval_optimizer_query_results.csv
   - random/LHS seed + PNN + PSO: outputs/phase2_seed_pnn_pso_h_full_strong30_nosr/seed_optimizer_query_results.csv
   - INN direct inverse + PNN: outputs/phase2_inn_inverse_baseline_v1/inn_inverse_query_results.csv
   - offline heuristic planner: outputs/phase2_heuristic_planner_offline_v1/heuristic_planner_query_results.csv

d. PSO 路径下 conservative acceptance 的诊断柱状图。
   数据来源：
   - Source / Target-H / Target-Self 三个 PSO strong30 目录下的 retrieval_optimizer_summary.csv

e. offline heuristic planner 的停止统计。
   数据来源：
   - outputs/phase2_heuristic_planner_offline_v1/heuristic_planner_summary.csv

运行方式：
直接在 IDE 里打开这个文件并点击运行即可；不需要命令行参数。

输出位置：
outputs/figures/phase2

重要说明：
为了方便你后期手动微调，每一个 draw_panel_* 函数内部都单独写了尺寸、字体、
颜色、坐标轴范围、线宽、透明度等参数；没有做全局统一样式。
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


# ============================================================
# 0. 只保留运行相关设置：路径、保存格式、是否显示图片
# ============================================================

# 项目根目录。这个脚本放在 MY-LLM-Framework 根目录下时，不需要修改。
PROJECT_ROOT = Path(__file__).resolve().parent

# 图片输出目录。
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "phase2"

# 是否保存 png/pdf。论文排版建议同时保存 PDF 和 PNG。
SAVE_PNG = True
SAVE_PDF = True
SAVE_DPI = 600

# 是否运行后直接弹出 matplotlib 图窗。
# True：适合你在 IDE 里点击运行并检查图片。
# False：适合只批量保存、不弹窗。
SHOW_FIGURES = True


# ============================================================
# 1. 数据文件路径
# ============================================================

# retrieval-only baseline 的 query-level 结果和 summary 结果。
RETRIEVAL_QUERY_CSV = PROJECT_ROOT / "outputs" / "phase01_retrieval_baselines_v1" / "query_level_results.csv"
RETRIEVAL_SUMMARY_CSV = PROJECT_ROOT / "outputs" / "phase01_retrieval_baselines_v1" / "baseline_summary.csv"

# Target-H 上 retrieval + PNN + PSO / CMA-ES 的 query-level 结果。
TARGET_H_PSO_QUERY_CSV = (
    PROJECT_ROOT / "outputs" / "phase2_retrieval_pnn_pso_h_full_strong30_nosr" / "retrieval_optimizer_query_results.csv"
)
TARGET_H_PSO_SUMMARY_CSV = (
    PROJECT_ROOT / "outputs" / "phase2_retrieval_pnn_pso_h_full_strong30_nosr" / "retrieval_optimizer_summary.csv"
)
TARGET_H_CMAES_QUERY_CSV = (
    PROJECT_ROOT / "outputs" / "phase2_retrieval_pnn_cmaes_h_full_strong30_nosr" / "retrieval_optimizer_query_results.csv"
)
TARGET_H_CMAES_SUMMARY_CSV = (
    PROJECT_ROOT / "outputs" / "phase2_retrieval_pnn_cmaes_h_full_strong30_nosr" / "retrieval_optimizer_summary.csv"
)

# Target-H 上 random / LHS seed + PNN + PSO 的 query-level 和 summary 结果。
TARGET_H_SEED_PSO_QUERY_CSV = (
    PROJECT_ROOT / "outputs" / "phase2_seed_pnn_pso_h_full_strong30_nosr" / "seed_optimizer_query_results.csv"
)
TARGET_H_SEED_PSO_SUMMARY_CSV = (
    PROJECT_ROOT / "outputs" / "phase2_seed_pnn_pso_h_full_strong30_nosr" / "seed_optimizer_summary.csv"
)

# INN direct inverse + PNN forward 结果。
INN_QUERY_CSV = PROJECT_ROOT / "outputs" / "phase2_inn_inverse_baseline_v1" / "inn_inverse_query_results.csv"
INN_SUMMARY_CSV = PROJECT_ROOT / "outputs" / "phase2_inn_inverse_baseline_v1" / "inn_inverse_summary.csv"

# Source / Target-H / Target-Self 三个 PSO strong30 summary，用于 panel d。
PSO_SUMMARY_CSV_BY_DOMAIN = {
    "source": PROJECT_ROOT
    / "outputs"
    / "phase2_retrieval_pnn_pso_source_full_strong30_nosr"
    / "retrieval_optimizer_summary.csv",
    "target_h": TARGET_H_PSO_SUMMARY_CSV,
    "target_self": PROJECT_ROOT
    / "outputs"
    / "phase2_retrieval_pnn_pso_self_full_strong30_nosr"
    / "retrieval_optimizer_summary.csv",
}

# offline heuristic planner 结果，用于 panel c/e。
HEURISTIC_QUERY_CSV = PROJECT_ROOT / "outputs" / "phase2_heuristic_planner_offline_v1" / "heuristic_planner_query_results.csv"
HEURISTIC_SUMMARY_CSV = PROJECT_ROOT / "outputs" / "phase2_heuristic_planner_offline_v1" / "heuristic_planner_summary.csv"


# ============================================================
# 2. 基础工具函数：只处理保存、读表、数值转换，不统一控制具体绘图风格
# ============================================================


def setup_basic_matplotlib() -> None:
    """设置 matplotlib 的基础字体嵌入方式。

    这里不设置具体字号、颜色、线宽，因为你希望每个 panel 都在自己的函数里单独调整。
    """

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.unicode_minus"] = False


def ensure_output_dir() -> None:
    """如果输出目录不存在，就自动创建。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, file_stem: str) -> None:
    """保存单张图。

    参数
    ----
    fig:
        matplotlib 的 figure 对象。
    file_stem:
        不带扩展名的文件名。例如传入 "panel_c"，会保存成 panel_c.png/pdf。
    """

    # 自动调整边距，尽量避免标题、坐标轴标签、图例、数值标注被裁掉。
    # 如果个别复杂布局触发 tight_layout warning，不影响图片保存，这里统一忽略。
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout.*")
        fig.tight_layout()

    if SAVE_PNG:
        fig.savefig(OUTPUT_DIR / f"{file_stem}.png", dpi=SAVE_DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(OUTPUT_DIR / f"{file_stem}.pdf", dpi=SAVE_DPI, bbox_inches="tight")


def add_panel_letter(
    ax: plt.Axes,
    letter: str,
    fontsize: int,
    x: float = -0.10,
    y: float = 1.08,
) -> None:
    """给子图添加 c/d/e 这种 panel 字母。"""

    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="left",
        zorder=30,
    )


def read_csv_checked(path: Path) -> pd.DataFrame:
    """读取 CSV，并在文件缺失时给出清楚的报错。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到绘图所需结果文件: {path}")
    return pd.read_csv(path)


def to_float_array(series: pd.Series) -> np.ndarray:
    """把 pandas 列转换成 float 数组，并自动丢掉空值。"""

    return pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)


def get_single_value(df: pd.DataFrame, column: str) -> float:
    """从只有一行的 summary 表中读取一个数值。"""

    if column not in df.columns:
        raise KeyError(f"表中缺少列: {column}")
    return float(pd.to_numeric(df[column], errors="coerce").iloc[0])


def get_retrieval_success_rate(domain: str, method: str, success_column: str) -> float:
    """从 retrieval baseline summary 里读取 Success@K。

    这里 panel c 用 Target-H raw-spectrum 的 success_at_1。
    """

    summary = read_csv_checked(RETRIEVAL_SUMMARY_CSV)
    row = summary[(summary["domain"] == domain) & (summary["method"] == method)]
    if row.empty:
        raise ValueError(f"找不到 retrieval summary 行: domain={domain}, method={method}")
    return float(pd.to_numeric(row[success_column], errors="coerce").iloc[0])


def get_heuristic_success_rate(domain: str) -> float:
    """从 heuristic planner summary 里读取最终成功率。"""

    summary = read_csv_checked(HEURISTIC_SUMMARY_CSV)
    row = summary[summary["domain"] == domain]
    if row.empty:
        raise ValueError(f"找不到 heuristic summary 行: domain={domain}")
    return float(pd.to_numeric(row["success_rate_final"], errors="coerce").iloc[0])


def get_seed_optimizer_success_rate(domain: str, seed_strategy: str) -> float:
    """读取 random/LHS seed + PNN + PSO 的最终成功率。"""

    summary = read_csv_checked(TARGET_H_SEED_PSO_SUMMARY_CSV)
    row = summary[(summary["domain"] == domain) & (summary["seed_strategy"] == seed_strategy)]
    if row.empty:
        raise ValueError(f"找不到 seed optimizer summary 行: domain={domain}, seed_strategy={seed_strategy}")
    return float(pd.to_numeric(row["success_rate_final"], errors="coerce").iloc[0])


def get_inn_success_rate(domain: str) -> float:
    """读取 INN direct inverse + PNN 的最终成功率。"""

    summary = read_csv_checked(INN_SUMMARY_CSV)
    row = summary[summary["domain"] == domain]
    if row.empty:
        raise ValueError(f"找不到 INN summary 行: domain={domain}")
    return float(pd.to_numeric(row["success_rate_final"], errors="coerce").iloc[0])


# ============================================================
# 3. Panel c：Target-H final amplitude-MSE 分布
# ============================================================


def load_panel_c_target_h_mse_data() -> list[dict[str, object]]:
    """整理 panel c 所需的四组 Target-H query-level MSE。

    返回值是一个 list。每个元素对应一种方法，包含：
    - label: 图中显示的方法名
    - values: 每个 query 的 final/top1 amplitude MSE
    - success_rate: 该方法在 Target-H 上的成功率
    """

    # 读取 retrieval-only query-level 结果。
    retrieval = read_csv_checked(RETRIEVAL_QUERY_CSV)
    retrieval_h = retrieval[(retrieval["domain"] == "target_h") & (retrieval["method"] == "raw_spectrum")].copy()

    # 读取 PSO / CMA-ES query-level 结果。
    pso_h = read_csv_checked(TARGET_H_PSO_QUERY_CSV)
    cmaes_h = read_csv_checked(TARGET_H_CMAES_QUERY_CSV)

    # 读取 random/LHS seed + PNN + PSO query-level 结果。
    seed_pso = read_csv_checked(TARGET_H_SEED_PSO_QUERY_CSV)
    seed_pso_h = seed_pso[seed_pso["domain"] == "target_h"].copy()

    # 读取 INN direct inverse + PNN query-level 结果。
    inn = read_csv_checked(INN_QUERY_CSV)
    inn_h = inn[inn["domain"] == "target_h"].copy()

    # 读取 heuristic planner query-level 结果。
    heuristic = read_csv_checked(HEURISTIC_QUERY_CSV)
    heuristic_h = heuristic[heuristic["domain"] == "target_h"].copy()

    # 读取 summary 结果，用于图下方 success rate 标注。
    pso_summary = read_csv_checked(TARGET_H_PSO_SUMMARY_CSV)
    cmaes_summary = read_csv_checked(TARGET_H_CMAES_SUMMARY_CSV)

    return [
        {
            "label": "Retrieval-only\ntop-1",
            "values": to_float_array(retrieval_h["top1_amplitude_mse"]),
            "success_rate": get_retrieval_success_rate("target_h", "raw_spectrum", "success_at_1"),
        },
        {
            "label": "Retrieval +\nPNN + PSO",
            "values": to_float_array(pso_h["final_amplitude_mse"]),
            "success_rate": get_single_value(pso_summary, "success_rate_final"),
        },
        {
            "label": "Retrieval +\nPNN + CMA-ES",
            "values": to_float_array(cmaes_h["final_amplitude_mse"]),
            "success_rate": get_single_value(cmaes_summary, "success_rate_final"),
        },
        {
            "label": "Random seed +\nPNN + PSO",
            "values": to_float_array(seed_pso_h[seed_pso_h["seed_strategy"] == "random"]["final_amplitude_mse"]),
            "success_rate": get_seed_optimizer_success_rate("target_h", "random"),
        },
        {
            "label": "LHS seed +\nPNN + PSO",
            "values": to_float_array(seed_pso_h[seed_pso_h["seed_strategy"] == "lhs"]["final_amplitude_mse"]),
            "success_rate": get_seed_optimizer_success_rate("target_h", "lhs"),
        },
        {
            "label": "INN direct\ninverse + PNN",
            "values": to_float_array(inn_h["amplitude_mse"]),
            "success_rate": get_inn_success_rate("target_h"),
        },
        {
            "label": "Offline heuristic\nplanner",
            "values": to_float_array(heuristic_h["planner_final_amplitude_mse"]),
            "success_rate": get_heuristic_success_rate("target_h"),
        },
    ]


def draw_panel_c(ax: plt.Axes | None = None, show_letter: bool = True) -> tuple[plt.Figure, plt.Axes]:
    """绘制 panel c：Target-H final amplitude-MSE 分布。

    这张图要表达：
    - retrieval-only top-1 是基础；
    - 加 PSO / CMA-ES 后，Target-H 的 final MSE 分布整体向低误差移动；
    - heuristic planner 进一步整合 retrieval、PSO、CMA-ES，给出最高 final success。
    """

    # ---------- 本 panel 的可调参数：单图尺寸 ----------
    panel_figsize = (15.5, 4.5)
    if ax is None:
        fig, ax = plt.subplots(figsize=panel_figsize)
    else:
        fig = ax.figure

    # ---------- 本 panel 的可调参数：字体 ----------
    font_family = "Arial"
    title_fontsize = 16
    label_fontsize = 16
    tick_fontsize = 16
    legend_fontsize = 16
    value_fontsize = 14
    panel_letter_fontsize = 16
    note_fontsize = 14

    # ---------- 本 panel 的可调参数：颜色 ----------
    # panel c 有 7 种方法，所以这里单独使用你给的参考图里的柔和配色挑选 7 个颜色。
    # 顺序对应下面 x 轴上的 7 种方法：
    # 1 Retrieval-only top-1
    # 2 Retrieval + PNN + PSO
    # 3 Retrieval + PNN + CMA-ES
    # 4 Random seed + PNN + PSO
    # 5 LHS seed + PNN + PSO
    # 6 INN direct inverse + PNN
    # 7 Offline heuristic planner
    method_colors = ["#CED6F6", "#8D9ECF", "#9DDFE4", "#FDE4E4", "#F7DFE5", "#F5CED0", "#FEF1E8"]
    violin_alpha = 0.58
    violin_edge_alpha = 0.95
    point_alpha = 0.22
    point_size = 8
    quartile_line_width = 3.2
    median_marker_size = 35

    # ---------- 本 panel 的可调参数：中位数趋势线 ----------
    # 连接 7 个方法的小提琴中位数点，辅助观察不同 baseline 的整体 MSE 水平。
    # 颜色从你给的参考图配色中选，#DEE3EF 是浅灰蓝，不会压过小提琴主体。
    draw_median_trend_line = True
    median_trend_line_color = "#DEE3EF"
    median_trend_line_width = 2.2
    median_trend_line_style = "-"
    median_trend_marker = "o"
    median_trend_marker_size = 5.5
    median_trend_zorder = 9

    # ---------- 本 panel 的可调参数：MSE log 坐标 ----------
    # 这里使用项目配置里的 amplitude_mse success threshold = 0.03。
    # 如果你想完全按示例图改成 1e-2，就把这里改成 0.01。
    success_threshold_mse = 0.03
    mse_clip_min = 1e-8
    y_lim_log10 = (-4.2, 0.05)
    y_ticks_log10 = [-4, -3, -2, -1, 0]

    # ---------- 本 panel 的可调参数：散点采样 ----------
    # 每组 query 数量几千个，全画会太黑；这里每组最多抽样这么多个点。
    max_scatter_points_per_method = 650
    random_seed = 7
    jitter_width = 0.11

    # ---------- 本 panel 的可调参数：坐标轴和阈值线 ----------
    # 阈值线颜色从参考图配色中选，#9DDFE4 和中位数趋势线形成区分。
    threshold_line_color = "#9DDFE4"
    threshold_line_width = 1.25
    threshold_line_dash = (6, 4)
    threshold_text_y_offset = 0.25
    grid_linewidth = 0.7
    grid_alpha = 0.30
    axis_linewidth = 0.9

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_family, "DejaVu Sans"]

    panel_data = load_panel_c_target_h_mse_data()
    positions = np.arange(1, len(panel_data) + 1)

    # 把每组 MSE 转成 log10(MSE) 后绘图；这样极小误差和较大误差都能看清。
    log_data: list[np.ndarray] = []
    for item in panel_data:
        values = np.asarray(item["values"], dtype=float)
        values = np.clip(values, mse_clip_min, None)
        log_data.append(np.log10(values))

    # 画小提琴图，显示每种方法在所有 query 上的 MSE 分布形状。
    violin = ax.violinplot(
        log_data,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    # 逐个设置小提琴主体颜色。
    for body, color in zip(violin["bodies"], method_colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(violin_alpha)
        body.set_linewidth(1.0)

    # 叠加 query-level 散点。每个点代表一个 query。
    rng = np.random.default_rng(random_seed)
    for pos, values, color in zip(positions, log_data, method_colors):
        if len(values) > max_scatter_points_per_method:
            values_to_plot = rng.choice(values, size=max_scatter_points_per_method, replace=False)
        else:
            values_to_plot = values
        jitter = rng.uniform(-jitter_width, jitter_width, size=len(values_to_plot))
        ax.scatter(
            pos + jitter,
            values_to_plot,
            s=point_size,
            color="white",
            edgecolor=color,
            linewidth=0.45,
            alpha=point_alpha,
            zorder=4,
        )

    # 保存每个方法的中位数位置和值，后面把中位数点连成趋势线。
    median_positions = []
    median_values = []

    # 手动画四分位线和中位数点。
    for pos, values, color in zip(positions, log_data, method_colors):
        q1, q2, q3 = np.percentile(values, [25, 50, 75])
        median_positions.append(pos)
        median_values.append(q2)
        ax.plot(
            [pos, pos],
            [q1, q3],
            color=color,
            linewidth=quartile_line_width,
            solid_capstyle="butt",
            alpha=violin_edge_alpha,
            zorder=6,
        )
        ax.scatter(
            [pos],
            [q2],
            s=median_marker_size,
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
            zorder=7,
        )

    # 连接 7 个方法的中位数点，形成趋势线。
    # 注意：这里连接的是 log10(MSE) 坐标下的中位数，所以线越低表示 MSE 越小。
    if draw_median_trend_line:
        ax.plot(
            median_positions,
            median_values,
            color=median_trend_line_color,
            linewidth=median_trend_line_width,
            linestyle=median_trend_line_style,
            marker=median_trend_marker,
            markersize=median_trend_marker_size,
            markerfacecolor="white",
            markeredgecolor=median_trend_line_color,
            zorder=median_trend_zorder,
        )

    # 画 amplitude-MSE 成功阈值线。
    threshold_log10 = np.log10(success_threshold_mse)
    ax.axhline(
        threshold_log10,
        color=threshold_line_color,
        linestyle="--",
        linewidth=threshold_line_width,
        dashes=threshold_line_dash,
        zorder=3,
    )
    # 把阈值说明放在图内部上方，而不是放到右侧外面。
    # threshold_text_y_offset 控制文字相对阈值线向上移动多少 log10 坐标单位。
    threshold_text_x = np.mean(positions)
    threshold_text_y = threshold_log10 + threshold_text_y_offset
    ax.text(
        threshold_text_x,
        threshold_text_y,
        f"Amplitude-MSE threshold\n(MSE = {success_threshold_mse:g})",
        color=threshold_line_color,
        fontsize=note_fontsize,
        fontweight="bold",
        va="bottom",
        ha="center",
    )

    # 设置标题、坐标轴标签和刻度。
    # ax.set_title("Final amplitude-MSE distributions on Target-H", fontsize=title_fontsize)
    ax.set_ylabel("Final amplitude MSE\n(log scale)", fontsize=label_fontsize)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(item["label"]) for item in panel_data], fontsize=tick_fontsize, rotation=12, ha="right")
    ax.set_ylim(*y_lim_log10)
    ax.set_yticks(y_ticks_log10)
    ax.set_yticklabels([rf"$10^{{{tick}}}$" for tick in y_ticks_log10], fontsize=tick_fontsize)
    ax.grid(True, axis="y", linestyle="--", linewidth=grid_linewidth, alpha=grid_alpha)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, width=axis_linewidth)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(axis_linewidth)
    ax.spines["bottom"].set_linewidth(axis_linewidth)

    if show_letter:
        add_panel_letter(ax, "c", fontsize=panel_letter_fontsize, x=-0.10, y=1.08)

    return fig, ax


# ============================================================
# 4. Panel d：Conservative acceptance diagnostics
# ============================================================


def load_panel_d_acceptance_data() -> pd.DataFrame:
    """读取 Source / Target-H / Target-Self 三个 PSO summary，并整理成绘图表。"""

    rows = []
    for domain, path in PSO_SUMMARY_CSV_BY_DOMAIN.items():
        summary = read_csv_checked(path)
        rows.append(
            {
                "domain": domain,
                "initial_success": get_single_value(summary, "success_rate_initial_top1"),
                "optimized_success": get_single_value(summary, "success_rate_optimized"),
                "final_success": get_single_value(summary, "success_rate_final"),
                "accepted_rate": get_single_value(summary, "accepted_optimization_rate"),
            }
        )
    return pd.DataFrame(rows)


def draw_panel_d(ax: plt.Axes | None = None, show_letter: bool = True) -> tuple[plt.Figure, plt.Axes]:
    """绘制 panel d：PSO 路径的 conservative acceptance 诊断。

    这张图要表达：
    - initial retrieval 是优化前的 top-1 成功率；
    - optimized raw 是 PSO 原始输出成功率；
    - final accepted 是经过 no-success-regression gate 后的最终成功率；
    - Target-Self 中 raw PSO 会造成明显退化，但 gate 把 final 保持在 1.0000。
    """

    # ---------- 本 panel 的可调参数：单图尺寸 ----------
    panel_figsize = (7, 4.5)
    if ax is None:
        fig, ax = plt.subplots(figsize=panel_figsize)
    else:
        fig = ax.figure

    # ---------- 本 panel 的可调参数：字体 ----------
    font_family = "Arial"
    title_fontsize = 16
    label_fontsize = 16
    tick_fontsize = 16
    legend_fontsize = 16
    value_fontsize = 14
    group_fontsize = 16
    panel_letter_fontsize = 16

    # ---------- 本 panel 的可调参数：颜色 ----------
    domain_labels = {
        "source": "Source",
        "target_h": "Target-H",
        "target_self": "Target-Self",
    }
    domain_text_colors = {
        "source": "#81AFD4",
        "target_h": "#DE7C7D",
        "target_self": "#FBCFBB",
    }
    stage_labels = [
        "Initial\nretrieval",
        "PSO\nraw",
        "Final\naccepted",
    ]
    stage_colors = ["#81AFD4", "#CAE0E8", "#FBCFBB"]
    bar_width = 0.2
    stage_spacing = 0.25
    bar_alpha = 0.95
    bar_edge_color = "white"
    bar_edge_width = 0.5
    value_label_y_offsets = [0.035, 0.065, 0.035]

    # bar_width：柱子宽度
    # stage_spacing：同一组内三根柱子的中心距离，越大越不挤
    # group_gap：不同domain组之间的间隔，越小组间越紧凑

    # ---------- 本 panel 的可调参数：坐标轴和分隔线 ----------
    y_lim = (0.0, 1.22)
    grid_linewidth = 0.7
    grid_alpha = 0.28
    axis_linewidth = 0.9
    group_gap = 0.4
    separator_color = "#B8B8B8"
    separator_linewidth = 0.9

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_family, "DejaVu Sans"]

    df = load_panel_d_acceptance_data()
    domain_order = ["source", "target_h", "target_self"]
    stage_keys = ["initial_success", "optimized_success", "final_success"]

    # 每个 domain 是一组，每组有三根柱子。
    group_step = (len(stage_keys) - 1) * stage_spacing + group_gap
    group_centers = np.arange(len(domain_order), dtype=float) * group_step
    offsets = np.array([-stage_spacing, 0.0, stage_spacing])

    for group_i, domain in enumerate(domain_order):
        row = df[df["domain"] == domain].iloc[0]
        center = group_centers[group_i]
        values = [float(row[key]) for key in stage_keys]

        for offset, value, color, stage_label, label_y_offset in zip(
            offsets,
            values,
            stage_colors,
            stage_labels,
            value_label_y_offsets,
        ):
            x = center + offset
            bar = ax.bar(
                x,
                value,
                width=bar_width,
                color=color,
                edgecolor=bar_edge_color,
                linewidth=bar_edge_width,
                alpha=bar_alpha,
                zorder=5,
            )
            ax.text(
                x,
                value + label_y_offset,
                f"{value:.4f}",
                fontsize=value_fontsize,
                color="black",
                ha="center",
                va="bottom",
                zorder=10,
            )

        # 在每组上方写 domain 名。
        ax.text(
            center,
            1.16,
            domain_labels[domain],
            fontsize=group_fontsize,
            fontweight="bold",
            color=domain_text_colors[domain],
            ha="center",
            va="bottom",
        )

        # 在 domain 之间画竖向分隔虚线。
        if group_i < len(domain_order) - 1:
            separator_x = (group_centers[group_i] + group_centers[group_i + 1]) / 2
            ax.axvline(
                separator_x,
                color=separator_color,
                linestyle=":",
                linewidth=separator_linewidth,
                zorder=1,
            )

    # 设置 x 轴：每根柱子保留自己的阶段标签。
    all_x = []
    all_labels = []
    for center in group_centers:
        for offset, label in zip(offsets, stage_labels):
            all_x.append(center + offset)
            all_labels.append(label)
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, fontsize=tick_fontsize, rotation=25, ha="right")

    # ax.set_title("Conservative acceptance diagnostics (PSO path)", fontsize=title_fontsize)
    ax.set_ylabel("Success rate", fontsize=label_fontsize)
    ax.set_ylim(*y_lim)
    ax.grid(True, axis="y", linestyle="--", linewidth=grid_linewidth, alpha=grid_alpha)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, width=axis_linewidth)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(axis_linewidth)
    ax.spines["bottom"].set_linewidth(axis_linewidth)

    if show_letter:
        add_panel_letter(ax, "d", fontsize=panel_letter_fontsize, x=-0.10, y=1.08)

    return fig, ax


# ============================================================
# 5. Panel e：Offline heuristic planner stopping statistics
# ============================================================


def draw_panel_e(ax: plt.Axes | None = None, show_letter: bool = True) -> tuple[plt.Figure, plt.Axes]:
    """绘制 panel e：heuristic planner 的停止统计。

    这张图要表达：
    - 有多少 query 在 retrieval 阶段就成功并停止；
    - 有多少 query 需要 PSO 才成功；
    - 有多少 query 需要 CMA-ES 才成功；
    - 最后仍然失败的 query 占多少。
    """

    # ---------- 本 panel 的可调参数：单图尺寸 ----------
    panel_figsize = (7, 4.5)
    if ax is None:
        fig, ax = plt.subplots(figsize=panel_figsize)
    else:
        fig = ax.figure

    # ---------- 本 panel 的可调参数：字体 ----------
    font_family = "Arial"
    title_fontsize = 16
    label_fontsize = 16
    tick_fontsize = 16
    legend_fontsize = 16
    value_fontsize = 14
    panel_letter_fontsize = 16

    # ---------- 本 panel 的可调参数：颜色 ----------
    domain_order = ["source", "target_h", "target_self"]
    domain_labels = {
        "source": "Source",
        "target_h": "Target-H",
        "target_self": "Target-Self",
    }
    domain_success_colors = {
        "source": "#81AFD4",
        "target_h": "#DE7C7D",
        "target_self": "#FBCFBB",
    }
    step_columns = ["retrieval_stop_count", "pso_stop_count", "cmaes_stop_count", "failed_count"]
    step_labels = ["Retrieval stop", "PSO stop", "CMA-ES stop", "Failed"]
    step_colors = ["#81AFD4", "#CAE0E8", "#DE7C7D", "#FBCFBB"]
    bar_height = 0.58       #控制每条横向柱子的粗细。越大，条越厚；越小，条越细。
    bar_alpha = 0.95

    # ---------- 本 panel 的可调参数：百分比标注 ----------
    # True：在每一段堆叠条上标出该部分 query 的比例。
    # False：不显示百分比，图会更干净。
    show_percentage_labels = True

    # 大于这个比例的段，百分比放在条形内部居中。
    # 小于这个比例的段太窄，文字放在该段右侧一点，避免挤在细条里面。
    inside_label_min_fraction = 0.075

    # 小段百分比文字向右偏移多少。
    small_label_x_offset = 0.012

    # 小段百分比文字向下偏移多少。
    # 这里 y 轴后面会 invert_yaxis，所以 y 加正数会在视觉上向下放。
    small_label_y_offset = 0.22

    # 百分比文字的背景框。小段文字放在其它颜色条上时，加白底更容易读。
    percentage_label_bbox = {
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.72,
        "pad": 0.35,
    }

    # ---------- 本 panel 的可调参数：坐标轴 ----------
    x_lim = (0.0, 1.05)
    grid_linewidth = 0.7
    grid_alpha = 0.28
    axis_linewidth = 0.9

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_family, "DejaVu Sans"]

    summary = read_csv_checked(HEURISTIC_SUMMARY_CSV)
    summary = summary.set_index("domain").loc[domain_order].reset_index()

    y_positions = np.arange(len(domain_order))

    # 逐个 domain 画横向堆叠条。
    for y, (_, row) in zip(y_positions, summary.iterrows()):
        domain = str(row["domain"])
        num_queries = float(row["num_queries"])
        left = 0.001
        for column, label, color in zip(step_columns, step_labels, step_colors):
            count = float(row[column])
            fraction = count / num_queries if num_queries > 0 else 0.0
            ax.barh(
                y,
                fraction,
                left=left,
                height=bar_height,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                alpha=bar_alpha,
                label=label if y == 0 else None,
                zorder=5,
            )

            if show_percentage_labels and fraction > 0:
                percentage_text = f"{fraction * 100:.1f}%"
                if fraction >= inside_label_min_fraction:
                    text_x = left + fraction / 2
                    text_y = y
                    text_ha = "center"
                    text_bbox = None
                else:
                    text_x = left + fraction + small_label_x_offset
                    text_y = y + small_label_y_offset
                    text_ha = "left"
                    text_bbox = percentage_label_bbox

                ax.text(
                    text_x,
                    text_y,
                    percentage_text,
                    fontsize=value_fontsize,
                    color="black",
                    ha=text_ha,
                    va="center",
                    bbox=text_bbox,
                    zorder=10,
                    clip_on=False,
                )

            left += fraction

    # y 轴标签写 domain 和样本量。
    y_labels = [
        f"{domain_labels[d]}\n(N={int(float(summary[summary['domain'] == d]['num_queries'].iloc[0])):,})"
        for d in domain_order
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=tick_fontsize)
    ax.invert_yaxis()

    # ax.set_title("Offline heuristic planner stopping statistics", fontsize=title_fontsize)
    ax.set_xlabel("Proportion of queries", fontsize=label_fontsize)
    ax.set_xlim(*x_lim)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(True, axis="x", linestyle="--", linewidth=grid_linewidth, alpha=grid_alpha)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, width=axis_linewidth)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(axis_linewidth)
    ax.spines["bottom"].set_linewidth(axis_linewidth)

    # 图例放在坐标轴下方外侧，并排排列。
    # loc="upper center" 表示用图例的上方中心作为锚点。
    # bbox_to_anchor=(0.5, -0.24) 表示锚点放在坐标轴宽度 50%、高度下方 24% 的位置。
    # ncol=4 表示四个图例并排成一行。
    # 这样图例不占用绘图区内部空间；保存图片时 bbox_inches="tight" 会把外部图例一起包含进去。
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=4,
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=1.0,
        handletextpad=0.45,
        columnspacing=1.2,
        labelspacing=0.35,
        borderaxespad=0.0,
    )

    if show_letter:
        add_panel_letter(ax, "e", fontsize=panel_letter_fontsize, x=-0.10, y=1.08)

    return fig, ax


# ============================================================
# 6. 绘制单图和组合图
# ============================================================


def draw_single_panels() -> None:
    """分别保存 panel c/d/e 单图，方便你逐张放进论文或 PPT。"""

    fig_c, _ = draw_panel_c(ax=None, show_letter=False)
    save_figure(fig_c, "panel_c_target_h_final_mse_distribution")

    fig_d, _ = draw_panel_d(ax=None, show_letter=False)
    save_figure(fig_d, "panel_d_conservative_acceptance_pso")

    fig_e, _ = draw_panel_e(ax=None, show_letter=False)
    save_figure(fig_e, "panel_e_heuristic_planner_stopping")


def draw_composite_figure() -> plt.Figure:
    """绘制 c/d/e 组合图。

    布局：
    - 上方整行：panel c，因为 MSE 分布图信息最多，需要更宽；
    - 下方左侧：panel d；
    - 下方右侧：panel e。
    """

    fig = plt.figure(figsize=(15.0, 9.0))
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.02, 1.00],
        width_ratios=[1.0, 1.0],
        hspace=0.58,
        wspace=0.28,
    )

    ax_c = fig.add_subplot(grid[0, :])
    ax_d = fig.add_subplot(grid[1, 0])
    ax_e = fig.add_subplot(grid[1, 1])

    draw_panel_c(ax=ax_c, show_letter=False)
    draw_panel_d(ax=ax_d, show_letter=False)
    draw_panel_e(ax=ax_e, show_letter=False)

    return fig


def main() -> None:
    """脚本入口：创建输出目录，绘制单图和组合图。"""

    setup_basic_matplotlib()
    ensure_output_dir()

    print("[Phase2 Figure] 绘制 panel c/d/e 单图...")
    draw_single_panels()

    print("[Phase2 Figure] 绘制 c-d-e 组合图...")
    composite = draw_composite_figure()
    save_figure(composite, "phase2_panels_cde_composite")

    print(f"[Phase2 Figure] 图片已保存到: {OUTPUT_DIR}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
