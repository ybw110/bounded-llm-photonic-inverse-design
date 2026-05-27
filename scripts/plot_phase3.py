"""
Phase 3 planner / critic figure panels.

This script follows the standalone plotting style of plot_phase01_retrieval_evidence.py
and plot_phase2.py. It creates only the three figure panels requested:

c. Target-H seed-budget replay success rates across 100-query, 500-query, and full-set scales.
d. Oracle-bounded headroom: rule planner versus oracle, shown as success and final MSE.
e. MSE-aware selector: baseline versus selector on the held-out/evaluation split.

No explanatory callout boxes or bottom note banners are added. The saved figures are
plain plotting panels that can be combined later in PowerPoint/Illustrator.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "phase3"

SAVE_PNG = True
SAVE_PDF = True
SAVE_DPI = 600
SHOW_FIGURES = True

PALETTE = {
    "blue": "#81AFD4",
    "pale_blue": "#CAE0E8",
    "red": "#DE7C7D",
    "peach": "#FBCFBB",
}


def setup_basic_matplotlib() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "Arial"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_figure_c(fig: plt.Figure, file_stem: str) -> None:
    """
    Save Panel C figure.

    Panel C 使用原来的 tight_layout 保存逻辑。
    适合 draw_panel_c 这种单图宽幅柱状图。
    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout.*",
        )
        fig.tight_layout()

    if SAVE_PNG:
        fig.savefig(
            OUTPUT_DIR / f"{file_stem}.png",
            dpi=SAVE_DPI,
            bbox_inches="tight",
        )

    if SAVE_PDF:
        fig.savefig(
            OUTPUT_DIR / f"{file_stem}.pdf",
            dpi=SAVE_DPI,
            bbox_inches="tight",
        )


def save_figure_de(fig: plt.Figure, file_stem: str) -> None:
    """
    Save Panel D/E figures.

    Panel D/E 不使用 tight_layout。
    因为 draw_success_mse_pair() 里面已经使用 fig.subplots_adjust()
    手动控制了左右子图间距、边距和标题空间。

    如果这里再调用 fig.tight_layout()，
    会打乱已经手动设置好的布局，导致两个子图间距变大或出现 warning。
    """

    if SAVE_PNG:
        fig.savefig(
            OUTPUT_DIR / f"{file_stem}.png",
            dpi=SAVE_DPI,
            bbox_inches="tight",
            pad_inches=0.08,
        )

    if SAVE_PDF:
        fig.savefig(
            OUTPUT_DIR / f"{file_stem}.pdf",
            dpi=SAVE_DPI,
            bbox_inches="tight",
            pad_inches=0.08,
        )

def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Phase 3 result file: {path}")
    return pd.read_csv(path)


def read_domain_value(path: Path, domain: str, column: str) -> float:
    df = read_csv_checked(path)
    if "domain" not in df.columns:
        raise KeyError(f"{path} has no 'domain' column")
    row = df[df["domain"] == domain]
    if row.empty:
        raise ValueError(f"{path} has no row for domain={domain}")
    if column not in row.columns:
        raise KeyError(f"{path} has no column {column}")
    return float(pd.to_numeric(row[column], errors="coerce").iloc[0])


def style_axis(
    ax: plt.Axes,
    *,
    label_fontsize: int,
    tick_fontsize: int,
    axis_linewidth: float = 1.0,
    grid_alpha: float = 0.25,
    y_lim: tuple[float, float] | None = None,
) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=grid_alpha, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(axis_linewidth)
    ax.spines["bottom"].set_linewidth(axis_linewidth)
    ax.tick_params(axis="both", labelsize=tick_fontsize, width=axis_linewidth, length=4)
    ax.xaxis.label.set_size(label_fontsize)
    ax.yaxis.label.set_size(label_fontsize)
    if y_lim is not None:
        ax.set_ylim(*y_lim)


def add_value_labels(
    ax: plt.Axes,
    bars,
    *,
    fontsize: int,
    fmt: str = "{:.3f}",
    log_scale: bool = False,
) -> None:
    for bar in bars:
        height = float(bar.get_height())
        if not np.isfinite(height):
            continue
        if log_scale:
            y = height * 1.35
            va = "bottom"
        else:
            y = height + 0.006
            va = "bottom"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(height),
            ha="center",
            va=va,
            fontsize=fontsize,
            color="#222222",
        )


def sci_label(value: float) -> str:
    return f"{value:.3g}".replace("e-0", "e-").replace("e+0", "e+")

def mse_mantissa_label(value: float, digits: int = 2) -> str:
    """
    Convert an MSE value to a mantissa-only label for log-scale bar plots.

    Example:
        0.00137   -> "1.37"
        2.45e-10  -> "2.45"
        9.8e-6    -> "9.80"

    设计逻辑：
    MSE 子图本身已经是 log-scale y-axis，
    y 轴刻度会显示 10^-3、10^-6、10^-10 等数量级。
    因此柱子顶部不需要再写完整科学计数法，
    只写 mantissa 可以显著减少标签长度和重叠。
    """
    if not np.isfinite(value) or value <= 0:
        return ""

    exponent = np.floor(np.log10(value))
    mantissa = value / (10 ** exponent)

    return f"{mantissa:.{digits}f}"

def load_panel_c_data() -> pd.DataFrame:
    """Target-H seed-budget replay success values."""

    summary_file = "planner_offline_summary.csv"
    paths = [
        ("100-query pilot replay (LLM)", "seed1\n(strict)", PROJECT_ROOT / "outputs/phase3_deepseek_seed1_strict_100_v1" / summary_file),
        ("100-query pilot replay (LLM)", "seed1→3\n(expand)", PROJECT_ROOT / "outputs/phase3_deepseek_seed1_expand_to_seed3_100_v1" / summary_file),
        ("100-query pilot replay (LLM)", "seed3\n(strict)", PROJECT_ROOT / "outputs/phase3_deepseek_strict_100_v1" / summary_file),
        ("100-query pilot replay (LLM)", "seed3→10\n(expand)", PROJECT_ROOT / "outputs/phase3_deepseek_expand_100_v1" / summary_file),
        ("500-query DeepSeek replay (LLM)", "seed1\n(strict)", PROJECT_ROOT / "outputs/phase3_deepseek_seed1_strict_500_v1" / summary_file),
        ("500-query DeepSeek replay (LLM)", "seed1→3\n(expand)", PROJECT_ROOT / "outputs/phase3_deepseek_seed1_expand_to_seed3_500_v1" / summary_file),
        ("500-query DeepSeek replay (LLM)", "seed3\n(strict)", PROJECT_ROOT / "outputs/phase3_deepseek_seed3_strict_500_v1" / summary_file),
        ("500-query DeepSeek replay (LLM)", "seed3→10\n(expand)", PROJECT_ROOT / "outputs/phase3_deepseek_seed3_expand_to_seed10_500_v1" / summary_file),
        ("Full-set deterministic rule replay", "seed1\n(strict)", PROJECT_ROOT / "outputs/phase3_rule_seed1_strict_full_v1" / summary_file),
        ("Full-set deterministic rule replay", "seed1→3\n(expand)", PROJECT_ROOT / "outputs/phase3_rule_seed1_expand_to_seed3_full_v1" / summary_file),
        ("Full-set deterministic rule replay", "seed3\n(strict)", PROJECT_ROOT / "outputs/phase3_oracle_gap_full_v1" / "planner_oracle_gap_summary.csv"),
        ("Full-set deterministic rule replay", "seed3→10\n(expand)", PROJECT_ROOT / "outputs/phase3_rule_seed3_expand_to_seed10_full_v1" / summary_file),
    ]

    rows = []
    for scale, label, path in paths:
        if path.name == "planner_oracle_gap_summary.csv":
            value = read_domain_value(path, "target_h", "baseline_success_rate")
        else:
            value = read_domain_value(path, "target_h", "success_rate_final")
        rows.append({"scale": scale, "label": label, "success_rate": value})
    return pd.DataFrame(rows)

def draw_panel_c(ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """
    Draw Panel C: Target-H seed-budget replay success rates.

    图的核心逻辑：
    1. 从 load_panel_c_data() 读取不同实验规模下的 Target-H success rate；
    2. 横轴分成 3 个大组：
       - 100-query pilot replay (LLM)
       - 500-query DeepSeek replay (LLM)
       - Full-set deterministic rule replay
    3. 每个大组内部有 4 根柱子：
       - seed1 strict
       - seed1→3 expand
       - seed3 strict
       - seed3→10 expand
    4. 用不同颜色区分不同大组；
    5. 用竖向虚线分隔三个大组。
    """

    # ------------------------------------------------------------------
    # 1. 创建画布
    # ------------------------------------------------------------------
    # 如果外部没有传入 ax，就单独创建一张图；
    # 如果外部传入了 ax，说明这个 panel 要画到已有的多子图布局里。
    if ax is None:
        fig, ax = plt.subplots(figsize=(13.5, 4.5))
    else:
        fig = ax.figure

    # ------------------------------------------------------------------
    # 2. 读取数据
    # ------------------------------------------------------------------
    # load_panel_c_data() 会返回一个 DataFrame，
    # 至少包含三列：
    #   - scale: 实验规模，例如 100-query / 500-query / full-set
    #   - label: 每根柱子的标签，例如 seed1 strict
    #   - success_rate: 柱子的高度，即成功率
    df = load_panel_c_data()

    # ------------------------------------------------------------------
    # 3. 基础字体和尺寸参数
    # ------------------------------------------------------------------
    font_family = "Arial"

    label_fontsize = 16   # y 轴标签字号
    tick_fontsize = 16    # x/y 坐标刻度字号
    value_fontsize = 16   # 柱子上方数值标签字号

    # 柱子本身的宽度。
    # 注意：bar_width 控制的是柱子的“粗细”，不是柱子之间的距离。
    bar_width = 0.62

    # ------------------------------------------------------------------
    # 4. 控制柱子间距
    # ------------------------------------------------------------------
    # inner_gap 控制同一大组内部，相邻柱子中心点之间的距离。
    # 数值越小，同组柱子越紧凑；
    # 数值越大，同组柱子越分散。
    inner_gap = 0.95

    # group_gap 控制不同大组之间额外增加的空白距离。
    # 数值越大，不同大组之间间隔越明显；
    # 数值越小，不同大组之间越靠近。
    group_gap = 1.10

    # ------------------------------------------------------------------
    # 5. 不同大组对应的柱子颜色
    # ------------------------------------------------------------------
    # 这里的 key 必须和 df["scale"] 里的文字完全一致，
    # 否则 bar_colors.append(colors[scale]) 会报 KeyError。
    colors = {
        "100-query pilot replay (LLM)": PALETTE["blue"],
        "500-query DeepSeek replay (LLM)": PALETTE["pale_blue"],
        "Full-set deterministic rule replay": PALETTE["peach"],
    }

    # ------------------------------------------------------------------
    # 6. 准备绘图所需的列表
    # ------------------------------------------------------------------
    x_positions = []   # 每根柱子的 x 坐标
    x_labels = []      # 每根柱子的 x 轴标签
    values = []        # 每根柱子的高度，也就是 success rate
    bar_colors = []    # 每根柱子的颜色

    # group_ranges 用来记录每个大组的 x 轴范围。
    # 后面画竖向虚线时，会根据相邻大组的边界自动计算分隔线位置。
    group_ranges = []

    # current_x 是当前正在分配的 x 坐标。
    # 每放置一根柱子，就增加 inner_gap；
    # 每结束一个大组，就额外增加 group_gap。
    current_x = 0.0

    # ------------------------------------------------------------------
    # 7. 计算每根柱子的 x 坐标、标签、高度和颜色
    # ------------------------------------------------------------------
    # df["scale"].drop_duplicates() 会按照 df 中出现的顺序取出三个大组。
    for scale in df["scale"].drop_duplicates():

        # 取出当前大组的数据。
        # 例如 scale == "100-query pilot replay (LLM)" 时，
        # sub 就包含这个大组下面的 4 行 seed-budget 结果。
        sub = df[df["scale"] == scale]

        # 记录当前大组第一根柱子的 x 坐标。
        # 后面用于计算这个大组的范围。
        group_start = current_x

        # 遍历当前大组中的每一个 seed-budget 设置。
        for _, row in sub.iterrows():

            # 当前柱子的 x 坐标
            x_positions.append(current_x)

            # 当前柱子的横轴标签，例如：
            # "seed1\n(strict)"
            # "seed1→3\n(expand)"
            x_labels.append(row["label"])

            # 当前柱子的高度，即 success rate
            values.append(float(row["success_rate"]))

            # 当前柱子的颜色，由所属大组 scale 决定
            bar_colors.append(colors[scale])

            # 移动到下一根柱子的位置。
            # 这里控制的是同一大组内部柱子的间距。
            current_x += inner_gap

        # 当前大组最后一根柱子的 x 坐标。
        # 因为循环结束后 current_x 已经多加了一次 inner_gap，
        # 所以这里需要减回 inner_gap。
        group_end = current_x - inner_gap

        # 保存当前大组的起点和终点。
        # 例如第一个大组可能是：
        # group_start = 0.0
        # group_end   = 2.1
        group_ranges.append((group_start, group_end))

        # 当前大组结束后，额外增加组间距。
        # 这里控制的是不同大组之间的距离。
        current_x += group_gap

    # ------------------------------------------------------------------
    # 8. 绘制柱状图
    # ------------------------------------------------------------------
    bars = ax.bar(
        x_positions,          # 每根柱子的 x 坐标
        values,               # 每根柱子的高度
        width=bar_width,      # 柱子宽度
        color=bar_colors,     # 柱子颜色
        edgecolor="white",    # 柱子边框颜色
        linewidth=1.0,        # 柱子边框线宽
        zorder=3,             # 图层顺序，保证柱子压在网格线上方
    )

    # ------------------------------------------------------------------
    # 9. 给柱子顶部添加数值标签
    # ------------------------------------------------------------------
    # fmt="{:.3f}" 表示保留三位小数。
    # 例如 0.87654 会显示为 0.877。
    add_value_labels(
        ax,
        bars,
        fontsize=value_fontsize,
        fmt="{:.3f}",
    )

    # ------------------------------------------------------------------
    # 10. 设置 x 轴刻度和标签
    # ------------------------------------------------------------------
    ax.set_xticks(x_positions)

    # rotation=15 表示 x 轴标签旋转 15 度；
    # ha="right" 表示右对齐，旋转后更不容易重叠。
    ax.set_xticklabels(
        x_labels,
        fontfamily=font_family,
        rotation=30,
        ha="right",
    )

    # ------------------------------------------------------------------
    # 11. 设置 y 轴标签
    # ------------------------------------------------------------------
    ax.set_ylabel(
        "Success rate",
        fontfamily=font_family,
    )

    # 如果之后想加标题，可以取消下面这行注释。
    # 目前你要求去掉大组标题，所以这里也保持不显示主标题。
    # ax.set_title(
    #     "Target-H seed-budget replay",
    #     fontsize=16,
    #     fontweight="bold",
    #     fontfamily=font_family,
    #     pad=12,
    # )

    # ------------------------------------------------------------------
    # 12. 统一坐标轴风格，并控制 y 轴范围
    # ------------------------------------------------------------------
    # y_lim=(0.74, 0.95) 控制 y 轴显示范围。
    # 如果想显示更完整，可以改成：
    #   y_lim=(0.70, 1.00)
    # 如果想从 0 开始，可以改成：
    #   y_lim=(0.00, 1.05)
    # 如果想自动范围，可以改成：
    #   y_lim=None
    style_axis(
        ax,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        y_lim=(0.74, 0.95),
    )

    # ------------------------------------------------------------------
    # 13. 绘制三个大组之间的竖向分隔线
    # ------------------------------------------------------------------
    # group_ranges 记录了每个大组的起止位置。
    # 这里在相邻两个大组之间的中点画虚线。
    #
    # 好处：
    # 即使你之后修改 inner_gap 或 group_gap，
    # 分隔线也会自动重新计算，不会错位。
    for i in range(len(group_ranges) - 1):

        # 当前大组的最后一根柱子位置
        left_group_end = group_ranges[i][1]

        # 下一个大组的第一根柱子位置
        right_group_start = group_ranges[i + 1][0]

        # 两个大组之间的中点，就是分隔线位置
        boundary = (left_group_end + right_group_start) / 2.0

        ax.axvline(
            boundary,
            color="#9A9A9A",   # 分隔线颜色
            linestyle="--",    # 虚线
            linewidth=1.0,     # 线宽
            alpha=0.75,        # 透明度
            zorder=1,          # 放在柱子后面
        )

    # ------------------------------------------------------------------
    # 14. 设置 x 轴显示范围，避免最左/最右柱子贴边
    # ------------------------------------------------------------------
    ax.set_xlim(
        min(x_positions) - 0.75,
        max(x_positions) + 0.75,
    )

    # ------------------------------------------------------------------
    # 15. 返回 Figure 和 Axes，方便外部继续保存或组合
    # ------------------------------------------------------------------
    return fig, ax


def load_panel_d_data() -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs/phase3_oracle_gap_full_v1" / "planner_oracle_gap_summary.csv"
    df = read_csv_checked(path)
    rows = []
    domain_labels = {"source": "Source", "target_h": "Target-H", "target_self": "Target-Self"}
    for _, row in df.iterrows():
        domain = row["domain"]
        rows.append(
            {
                "domain": domain,
                "domain_label": domain_labels.get(domain, domain),
                "Rule_success": float(row["baseline_success_rate"]),
                "Oracle_success": float(row["oracle_success_rate"]),
                "Rule_mse": float(row["baseline_mean_mse"]),
                "Oracle_mse": float(row["oracle_mean_mse"]),
            }
        )
    return pd.DataFrame(rows)


def load_panel_e_data() -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs/phase3_mse_selector_full_v1" / "mse_aware_selector_gap_summary.csv"
    df = read_csv_checked(path)
    df = df[df["split"] == "evaluation"].copy()
    rows = []
    domain_labels = {"source": "Source", "target_h": "Target-H", "target_self": "Target-Self"}
    for _, row in df.iterrows():
        domain = row["domain"]
        rows.append(
            {
                "domain": domain,
                "domain_label": domain_labels.get(domain, domain),
                "Baseline_success": float(row["baseline_success_rate"]),
                "MSE-aware selector_success": float(row["selector_success_rate"]),
                "Baseline_mse": float(row["baseline_mean_mse"]),
                "MSE-aware selector_mse": float(row["selector_mean_mse"]),
            }
        )
    return pd.DataFrame(rows)

def add_bar_value_labels_offset(
    ax: plt.Axes,
    bars,
    *,
    fontsize: int,
    fmt: str = "{:.4f}",
    x_offset: float = 0.0,
    y_offset: float = 0.006,
) -> None:
    """
    给柱状图添加数值标签，并允许手动设置 x / y 方向偏移。

    参数
    ----
    ax : plt.Axes
        当前绘图坐标轴。

    bars :
        ax.bar() 返回的柱子对象。

    fontsize : int
        标签字号。

    fmt : str
        数值格式。
        这里默认 "{:.4f}"，表示 success rate 保留 4 位小数。

    x_offset : float
        标签在 x 方向上的偏移量。
        用来避免左右两组柱子的数字重叠。

    y_offset : float
        标签在 y 方向上的偏移量。
        适用于线性坐标的 success rate 图。
    """
    for bar in bars:
        height = float(bar.get_height())

        if not np.isfinite(height):
            continue

        ax.text(
            bar.get_x() + bar.get_width() / 2 + x_offset,
            height + y_offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#222222",
        )

def draw_success_mse_pair(
    *,
    df: pd.DataFrame,
    left_methods: tuple[str, str],
    right_methods: tuple[str, str],
    colors: tuple[str, str],
    title: str,
    file_panel: str,
) -> tuple[plt.Figure, np.ndarray]:
    """
    通用绘图函数：同时绘制 success rate 和 final amplitude MSE 两张并排柱状图。

    当前版本：
    1. 不设置整张图总标题；
    2. 子图标题不加粗；
    3. 只在左侧 success rate 图中保留图例；
    4. success rate 数值保留 4 位小数；
    5. success rate 数值左右、上下错开；
    6. MSE 使用 log-scale y-axis；
    7. MSE 柱子顶部只显示 mantissa，不显示 e-3 / e-10；
    8. MSE 数值上下交错标注，减少重叠；
    9. 使用 constrained_layout=True 自适应边距。
    """

    # ==========================================================
    # 1. 创建画布
    # ==========================================================
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8, 4.5),
        # constrained_layout=True,
        gridspec_kw={"wspace": 0.42},
    )
    ax_success, ax_mse = axes

    # ==========================================================
    # 2. 基础参数设置
    # ==========================================================
    font_family = "Arial"

    subtitle_fontsize = 16   # 子图标题字号
    label_fontsize = 16      # 坐标轴标题字号
    tick_fontsize = 16       # 坐标刻度字号
    legend_fontsize = 16     # 图例字号
    value_fontsize = 14      # 柱子顶部数值字号

    # 每组并排柱子的宽度
    width = 0.30

    # 横轴 domain
    domains = df["domain_label"].tolist()
    x = np.arange(len(domains))

    # ==========================================================
    # 3. 绘制左图：Success rate
    # ==========================================================
    method_a, method_b = left_methods

    bars_a = ax_success.bar(
        x - width / 2,
        df[f"{method_a}_success"],
        width,
        label=method_a,
        color=colors[0],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    bars_b = ax_success.bar(
        x + width / 2,
        df[f"{method_b}_success"],
        width,
        label=method_b,
        color=colors[1],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    ax_success.set_xticks(x)
    ax_success.set_xticklabels(
        domains,
        rotation=15,
        ha="right",
        fontfamily=font_family,
    )

    ax_success.set_ylabel(
        "Success rate",
        fontfamily=font_family,
    )

    # 子标题保留，但不加粗
    ax_success.set_title(
        "Success rate",
        fontsize=subtitle_fontsize,
        fontweight="normal",
        fontfamily=font_family,
        pad=14,
    )

    # y 轴范围：上限留出数值标签空间
    style_axis(
        ax_success,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        y_lim=(0.78, 1.10),
    )

    # ----------------------------------------------------------
    # Success rate 数值标签
    # ----------------------------------------------------------
    # 第一组：稍左、稍低
    add_bar_value_labels_offset(
        ax_success,
        bars_a,
        fontsize=value_fontsize,
        fmt="{:.4f}",
        x_offset=-0.04,
        y_offset=0.0005,
    )

    # 第二组：稍右、稍高
    add_bar_value_labels_offset(
        ax_success,
        bars_b,
        fontsize=value_fontsize,
        fmt="{:.4f}",
        x_offset=0.035,
        y_offset=0.015,
    )

    # 只在左图保留图例
    ax_success.legend(
        frameon=False,
        fontsize=legend_fontsize,
        loc="upper left",
    )

    # ==========================================================
    # 4. 绘制右图：Final amplitude MSE
    # ==========================================================
    mse_a, mse_b = right_methods

    bars_mse_a = ax_mse.bar(
        x - width / 2,
        df[f"{mse_a}_mse"],
        width,
        label=mse_a,
        color=colors[0],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    bars_mse_b = ax_mse.bar(
        x + width / 2,
        df[f"{mse_b}_mse"],
        width,
        label=mse_b,
        color=colors[1],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    ax_mse.set_xticks(x)
    ax_mse.set_xticklabels(
        domains,
        rotation=15,
        ha="right",
        fontfamily=font_family,
    )

    # MSE 跨多个数量级，因此使用 log 轴
    ax_mse.set_yscale("log")
    ax_mse.set_ylim(1e-13, 1e-1)

    # 这里明确写 log scale，让读者知道柱子顶部的 1.37 需要结合 y 轴数量级理解
    ax_mse.set_ylabel(
        "Final amplitude MSE",
        fontfamily=font_family,
    )

    # 子标题中保留 log scale 说明
    ax_mse.set_title(
        "Final amplitude MSE (log scale)",
        fontsize=subtitle_fontsize,
        fontweight="normal",
        fontfamily=font_family,
        pad=14,
    )

    style_axis(
        ax_mse,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
    )

    ax_mse.grid(
        axis="y",
        which="both",
        color="#D9D9D9",
        linewidth=0.75,
        alpha=0.30,
    )

    # ----------------------------------------------------------
    # MSE 数值标签：只显示 mantissa
    # ----------------------------------------------------------
    # 例如：
    #   0.00137  -> 1.37
    #   2.45e-10 -> 2.45
    #
    # 数量级由 log-scale y 轴刻度表达，
    # 柱子顶部不再显示 e-3 / e-10。
    # ----------------------------------------------------------

    # 第一组 MSE 标签：稍左、稍低
    for bar in bars_mse_a:
        height = float(bar.get_height())

        if not np.isfinite(height) or height <= 0:
            continue

        ax_mse.text(
            bar.get_x() + bar.get_width() / 2 - 0.04,
            height * 0.95,
            mse_mantissa_label(height, digits=2),
            ha="center",
            va="bottom",
            fontsize=value_fontsize,
            fontfamily=font_family,
            color="#222222",
        )

    # 第二组 MSE 标签：稍右、明显更高
    for bar in bars_mse_b:
        height = float(bar.get_height())

        if not np.isfinite(height) or height <= 0:
            continue

        ax_mse.text(
            bar.get_x() + bar.get_width() / 2 + 0.035,
            height * 3.20,
            mse_mantissa_label(height, digits=2),
            ha="center",
            va="bottom",
            fontsize=value_fontsize,
            fontfamily=font_family,
            color="#222222",
        )

    # 右图不再添加图例，避免重复
    # ax_mse.legend(...)
    fig.subplots_adjust(
        left=0.11,  # 左边距，防止左图 y 轴标题被裁切
        right=0.98,  # 右边距，尽量利用右侧空间
        bottom=0.15,  # 底部边距，给旋转 15 度的 x 轴标签留空间
        top=0.88,  # 顶部边距，给两个子图标题留空间
        wspace=0.35,  # 两个子图之间的水平距离，越小越紧凑
    )

    return fig, axes
def draw_panel_d() -> tuple[plt.Figure, np.ndarray]:
    """
    绘制 Panel d：
    Oracle-bounded headroom

    比较对象：
    - Rule
    - Oracle
    """
    df = load_panel_d_data()

    return draw_success_mse_pair(
        df=df,
        left_methods=("Rule", "Oracle"),
        right_methods=("Rule", "Oracle"),
        colors=("#CED6F6", "#FDE4E4"),
        title="Oracle-bounded headroom",
        file_panel="d",
    )


def draw_panel_e() -> tuple[plt.Figure, np.ndarray]:
    """
    绘制 Panel e：
    MSE-aware selector

    比较对象：
    - Baseline
    - MSE-aware selector
    """
    df = load_panel_e_data()

    return draw_success_mse_pair(
        df=df,
        left_methods=("Baseline", "MSE-aware selector"),
        right_methods=("Baseline", "MSE-aware selector"),
        colors=("#CED6F6", "#FDE4E4"),
        title="MSE-aware selector",
        file_panel="e",
    )




def write_source_data() -> None:
    load_panel_c_data().to_csv(OUTPUT_DIR / "panel_c_source_data.csv", index=False)
    load_panel_d_data().to_csv(OUTPUT_DIR / "panel_d_source_data.csv", index=False)
    load_panel_e_data().to_csv(OUTPUT_DIR / "panel_e_source_data.csv", index=False)


def main() -> None:
    setup_basic_matplotlib()
    ensure_output_dir()

    # Panel C 使用原来的 tight_layout 保存方式
    fig_c, _ = draw_panel_c()
    save_figure_c(fig_c, "panel_c_seed_budget_replay")

    # Panel D/E 使用不调用 tight_layout 的保存方式
    fig_d, _ = draw_panel_d()
    save_figure_de(fig_d, "panel_d_oracle_headroom")

    fig_e, _ = draw_panel_e()
    save_figure_de(fig_e, "panel_e_mse_aware_selector")

    write_source_data()

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")

    print(f"Saved Phase 3 panels to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
