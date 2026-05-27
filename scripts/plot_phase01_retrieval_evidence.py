"""
Phase 0/1 retrieval-only evidence 绘图脚本。

这个脚本用于绘制论文中 Phase 0/1 对应的 retrieval-only 证据图。
你可以在 IDE 中直接点击运行，不需要输入命令行参数。

本脚本输出五个 panel：

a. Raw-spectrum Success@K curves across Source, Target-H, and Target-Self.
   中文理解：raw-spectrum 检索在三个 domain 上的 Success@K 曲线。

b. Success@K grouped comparison for the raw-spectrum baseline.
   中文理解：raw-spectrum 检索在不同 K 下的分组柱状图。

c. Top-1 amplitude MSE.
   中文理解：raw-spectrum top1 候选的平均振幅 MSE，用来突出 Target-H 误差更大，
   Target-Self 已经几乎饱和。

d. Target-H best-MSE distribution within top-K retrieved seeds.
   中文理解：Target-H 中，如果允许看 top1/top5/top10/top50 个检索种子，
   每个 query 能拿到的最低 MSE 分布如何变化。

e. Retrieval representation ablation in terms of Success@50.
   中文理解：比较 raw-spectrum、physics-summary、domain-aware 三种检索表征在
   Success@50 指标上的差异。

重要说明：
- 你要求“不使用全局统一绘图参数”，所以每个 draw_panel_* 函数内部都单独写了
  该 panel 的颜色、字号、尺寸、坐标范围、标注开关等参数。
- 顶部只保留路径、保存格式、是否弹窗等运行参数。
- 字体统一建议使用 Arial，但每个 panel 内部都可以单独改。
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 0. 只保留运行相关设置：路径、保存格式、是否显示图片
# ============================================================

# 项目根目录。这个脚本放在 MY-LLM-Framework 根目录下时，不需要修改。
PROJECT_ROOT = Path(__file__).resolve().parent

# retrieval-only baseline 的汇总表。
# panel a/b/c/e 主要从这个文件读取数据。
SUMMARY_CSV = PROJECT_ROOT / "outputs" / "phase01_retrieval_baselines_v1" / "baseline_summary.csv"

# panel d 需要重新计算 Target-H 的 top-K best-MSE 分布，因此需要读 memory bank 和 processed test split。
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_v1.json"
MEMORY_BANK_PATH = PROJECT_ROOT / "outputs" / "phase01_memory_bank_v1_train.npz"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "v1"

# 图片输出目录。
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "phase01"

# 是否保存 png/pdf。论文排版一般建议同时保存 PDF 和 PNG。
SAVE_PNG = True
SAVE_PDF = True
SAVE_DPI = 600

# 是否运行后弹出 matplotlib 图窗。
# True：绘制完成后直接展示图片窗口，适合你在 IDE 里手动调图。
# False：只保存图片，不弹窗，适合批量生成最终图片。
SHOW_FIGURES = True


# ============================================================
# 1. 让双击运行时也能导入 src/my_llm_framework
# ============================================================

# 因为项目采用 src/ 包结构，直接点击运行这个脚本时，Python 可能找不到 my_llm_framework。
# 所以这里手动把 src 加入模块搜索路径。
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from my_llm_framework.config import load_config  # noqa: E402
from my_llm_framework.data.processed import load_processed_split  # noqa: E402
from my_llm_framework.evaluation.metrics import amplitude_mse  # noqa: E402
from my_llm_framework.retrieval.methods import retrieve_raw_spectrum  # noqa: E402
from my_llm_framework.scripts.evaluate_retrieval_baselines import load_memory_bank  # noqa: E402


# ============================================================
# 2. 基础工具函数：只处理数据和保存，不放统一绘图风格
# ============================================================


def setup_basic_matplotlib() -> None:
    """设置 matplotlib 的基础字体嵌入方式。

    这里不设置字号、颜色、线宽等具体绘图风格。
    那些内容都放到每个 draw_panel_* 函数内部，方便你逐个 panel 修改。
    """

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
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
        不带扩展名的文件名。例如传入 "panel_a"，会保存为 panel_a.png/pdf。
    """

    # 自动调整子图边距，尽量防止标题、坐标轴标签、图例、数值标注被裁掉。
    # 但是组合图里包含 polar 雷达图时，matplotlib 可能提示：
    # "Axes that are not compatible with tight_layout"
    # 这个提示不是致命错误，所以这里把它作为 warning 忽略掉。
    # 真正保存时下面还有 bbox_inches="tight" 兜底，仍然会尽量避免裁剪。
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
    x: float = -0.14,
    y: float = 1.08,
) -> None:
    """给子图添加 a/b/c/d/e 标号。

    这个函数只负责放 panel 字母。字体大小由每个 panel 自己传进来。
    """

    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="left",
    )


def load_summary_table() -> pd.DataFrame:
    """读取 retrieval-only baseline 的汇总表。

    这个表由下面的 Phase 1 命令生成：
    python -m my_llm_framework.scripts.evaluate_retrieval_baselines ...

    表中每一行对应一个 domain + retrieval method。
    """

    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"找不到汇总表: {SUMMARY_CSV}")

    summary = pd.read_csv(SUMMARY_CSV)

    # 检查绘图需要的列是否存在，避免后面报难懂的 KeyError。
    required_columns = {
        "domain",
        "method",
        "mean_top1_amplitude_mse",
        "success_at_1",
        "success_at_5",
        "success_at_10",
        "success_at_50",
    }
    missing_columns = required_columns - set(summary.columns)
    if missing_columns:
        raise ValueError(f"baseline_summary.csv 缺少列: {sorted(missing_columns)}")

    return summary


def get_summary_row(summary: pd.DataFrame, domain: str, method: str) -> pd.Series:
    """取出某个 domain + method 对应的一行结果。"""

    selected = summary[(summary["domain"] == domain) & (summary["method"] == method)]
    if selected.empty:
        raise ValueError(f"summary 中找不到 domain={domain}, method={method}")
    return selected.iloc[0]


def read_success_at_k(summary: pd.DataFrame, domain: str, method: str, budgets: list[int]) -> list[float]:
    """读取指定 domain/method 的 Success@K 数组。"""

    row = get_summary_row(summary, domain, method)
    return [float(row[f"success_at_{k}"]) for k in budgets]


# ============================================================
# 3. Panel a：raw-spectrum Success@K 曲线
# ============================================================


def draw_panel_a(ax: plt.Axes | None, summary: pd.DataFrame, show_letter: bool = True) -> tuple[plt.Figure, plt.Axes]:
    """绘制 panel a。

    对应 caption：
    Raw-spectrum Success@K curves across Source, Target-H, and Target-Self.

    数据来源：
    outputs/phase01_retrieval_baselines_v1/baseline_summary.csv 中 method == raw_spectrum 的
    success_at_1 / success_at_5 / success_at_10 / success_at_50。

    这张图想表达：
    - Source 和 Target-H 在 K 变大时成功率明显上升；
    - Target-Self 从 top1 开始就已经 100% 成功，说明该任务在 retrieval-only 下饱和。
    """

    # ---------- 本 panel 的可调参数：单图尺寸 ----------
    # 这个尺寸只在单独绘制 panel a 时生效。
    # 如果 panel a 被放进组合图，组合图会提前创建 ax，这里的 figsize 不会改变组合图布局。
    # 格式是 (宽, 高)，单位是 inch。你可以改成 (6, 4)、(8, 5) 这类形式。
    panel_figsize = (4.5,4.5)

    # 如果外部没有传入 ax，说明现在是在单独画 panel a，就在这里创建画布。
    # 如果外部传入了 ax，说明现在是在组合图里画 panel a，就继续使用已有坐标轴。
    if ax is None:
        fig, ax = plt.subplots(figsize=panel_figsize)
    else:
        fig = ax.figure

    # ---------- 本 panel 的可调参数：字体 ----------
    font_family = "Arial"
    title_fontsize = 16
    label_fontsize = 16         #控制坐标轴标题大小。比如 x 轴的 K (top-K)，y 轴的 Success@K：
    tick_fontsize = 16         #控制坐标轴刻度数字大小。比如 x 轴上的 1, 5, 10, 50，y 轴上的 0.2, 0.4, 0.6：
    legend_fontsize = 16       #控制图例文字大小。比如 Source (in-domain)、Target-H 这些：
    value_fontsize = 14         #控制图中数值标注大小。比如曲线点旁边的 0.7167、柱子上方的 0.9377：
    panel_letter_fontsize = 12   #控制子图左上角的 panel 字母大小。比如 a、b、c、d、e：

    # ---------- 本 panel 的可调参数：颜色、线型、点型 ----------
    budgets = [1, 5, 10, 50]
    domain_order = ["source", "target_h", "target_self"]
    domain_labels = {
        "source": "Source",
        "target_h": "Target-H",
        "target_self": "Target-Self",
    }
    domain_colors = {
        "source": "#60A8D4",
        "target_h": "#FF9192",
        "target_self": "#8CC1AF",
        # "source": "#cce4c2",
        # "target_h": "#efbcb8",
        # "target_self": "#aecce4",
    }
    domain_markers = {
        "source": "o",
        "target_h": "s",
        "target_self": "^",
    }
    line_width = 1.8
    marker_size = 5.5

    # ---------- 本 panel 的可调参数：坐标轴、网格、标注 ----------
    y_lim = (0.6, 1.05)                   #控制 y 轴显示范围。
    y_ticks = [0.6, 0.7, 0.8, 0.9, 1.0]   #控制 y 轴显示哪些刻度数字。

    grid_linewidth = 0.7                 #控制网格线粗细。越大越粗，越小越细。
    grid_alpha = 0.35                    #控制网格线透明度。0 是完全透明，1 是完全不透明。
    axis_linewidth = 0.9                #控制坐标轴边框和刻度线粗细。比如左边 y 轴、下面 x 轴的线条粗细。
    annotate_values = True              #控制是否在曲线点旁边显示具体数值。显示 0.7167 这些数字。


    # 不同 domain 的标注偏移，避免数字和线条重叠。控制每条曲线的数值标注上下偏移量，避免数字压在线上或者互相重叠。
    label_y_offsets = {
        "source": 0.01,
        "target_h": -0.02,
        "target_self": 0.02,
    }

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_family, "DejaVu Sans"]

    for domain in domain_order:
        # 从 summary 表中取出当前 domain 的 raw-spectrum Success@K。
        # values 的顺序与 budgets 一一对应，例如 [Success@1, Success@5, Success@10, Success@50]。
        values = read_success_at_k(summary, domain, "raw_spectrum", budgets)

        # 画一条 Success@K 曲线。
        # budgets 控制横坐标，即 K 的取值；values 控制纵坐标，即成功率。
        # 如果你想把折线改成虚线，可以加 linestyle="--"。
        # 如果你想把点去掉，可以把 marker=None。
        # 如果你想让线更粗或更细，改 linewidth=line_width 对应的 line_width。
        ax.plot(
            budgets,  # 横坐标：top-K 中的 K。
            values,  # 纵坐标：对应的 Success@K。
            marker=domain_markers[domain],  # 当前 domain 的点型，例如圆点、方块、三角形。
            markersize=marker_size,  # 点的大小。
            linewidth=line_width,  # 折线宽度。
            color=domain_colors[domain],  # 当前 domain 的线条颜色。
            label=domain_labels[domain],  # 图例中显示的文字。
        )

        if annotate_values:
            for x_value, y_value in zip(budgets, values):
                # Target-Self 的四个点全是 1.0000，全部标会挤在一条线上。
                # 这里只标 K=1 和 K=50，表达“全程饱和”即可。
                if domain == "target_self" and x_value not in (1, 50):
                    continue

                offset = label_y_offsets[domain]
                # 给每个数据点旁边写出具体数值。
                # x_value 是数字标注的横向位置；y_value + offset 是纵向位置。
                # 如果某个数字压住线条，可以调 label_y_offsets 里的偏移量。
                ax.text(
                    x_value,  # 文字横坐标，与当前 K 对齐。
                    y_value + offset,  # 文字纵坐标，在真实数据点基础上上下错开。
                    f"{y_value:.3f}",  # 保留四位小数；想少一点可改成 .3f 或 .2f。
                    ha="center",  # 水平方向居中对齐。
                    va="bottom" if offset >= 0 else "top",  # 根据偏移方向决定文字贴在哪一侧。
                    fontsize=value_fontsize,  # 数值标注字号。
                    fontweight="bold",  # 数值加粗；如果想恢复普通粗细，改成 "normal"。
                    color=domain_colors[domain],  # 数值颜色与曲线颜色一致。
                    zorder=20,  # 让文字处在最上层，避免被线条、点或网格遮挡。
                    clip_on=False,  # 允许文字稍微超出坐标轴范围，避免顶部数值被裁剪。
                )

    # 设置子图标题。fontweight="bold" 表示加粗；如果想普通粗细，可以删掉。
    ax.set_title("Retrieval Success@K (raw-spectrum)", fontsize=title_fontsize)
    # 设置 x 轴标题。这里的 K (top-K) 对应检索候选数量。
    ax.set_xlabel("K (top-K)", fontsize=label_fontsize)
    # 设置 y 轴标题。Success@K 表示前 K 个候选中至少一个成功的比例。
    ax.set_ylabel("Success@K", fontsize=label_fontsize)
    # 只显示 budgets 中的四个刻度，避免横轴出现无意义的小数刻度。
    ax.set_xticks(budgets)
    # 设置 y 轴范围。上限 1.12 是为了给 1.0000 的文字留空间。
    ax.set_ylim(*y_lim)
    # 手动设置 y 轴刻度，让图更规整。
    ax.set_yticks(y_ticks)

    # 添加浅灰虚线网格，帮助读数。
    # axis="both" 表示横向和纵向网格都画；如果只想保留横向网格，改成 axis="y"。
    ax.grid(True, axis="both", linestyle="--", linewidth=grid_linewidth, alpha=grid_alpha)
    # 设置坐标轴刻度字号和刻度线宽度。
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, width=axis_linewidth)
    # 隐藏上边框和右边框，这是论文图常用的简洁风格。
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # 设置左边框和下边框的线宽。
    ax.spines["left"].set_linewidth(axis_linewidth)
    ax.spines["bottom"].set_linewidth(axis_linewidth)

    # 添加图例。
    # loc="lower right" 控制图例位置；如果挡住数据，可以改成 "lower center" 或 "best"。
    ax.legend(loc="lower right", fontsize=legend_fontsize, frameon=True, edgecolor="#999999")

    if show_letter:
        add_panel_letter(ax, "a", fontsize=panel_letter_fontsize)

    return fig, ax


# ============================================================
# 4. Panel b：raw-spectrum Success@K 分组柱状图
# ============================================================


def draw_panel_b(ax: plt.Axes | None, summary: pd.DataFrame, show_letter: bool = True) -> tuple[plt.Figure, plt.Axes]:
    """绘制 panel b。

    对应 caption：
    Success@K grouped comparison for the raw-spectrum baseline.

    数据来源：
    同样来自 baseline_summary.csv 中 method == raw_spectrum 的 Success@K。

    这张图想表达：
    - 每个 domain 内部比较 top1/top5/top10/top50；
    - Source 和 Target-H 随 K 增大提升明显；
    - Target-Self 在所有 K 下都是 1.0。
    """

    # ---------- 本 panel 的可调参数：单图尺寸 ----------
    # 这个尺寸只控制单独保存 panel b 时的大小。
    # 如果顶部竖排数字拥挤，可以把高度调大，例如 (6, 4.8)。
    panel_figsize = (5.5,4.5)

    # 如果外部没有传入 ax，说明现在是在单独画 panel a，就在这里创建画布。
    # 如果外部传入了 ax，说明现在是在组合图里画 panel a，就继续使用已有坐标轴。
    if ax is None:
        fig, ax = plt.subplots(figsize=panel_figsize)
    else:
        fig = ax.figure

    # ---------- 本 panel 的可调参数：字体 ----------
    font_family = "Arial"
    title_fontsize = 16
    label_fontsize = 16         #控制坐标轴标题大小。比如 x 轴的 K (top-K)，y 轴的 Success@K：
    tick_fontsize = 16         #控制坐标轴刻度数字大小。比如 x 轴上的 1, 5, 10, 50，y 轴上的 0.2, 0.4, 0.6：
    legend_fontsize = 16       #控制图例文字大小。比如 Source (in-domain)、Target-H 这些：
    value_fontsize = 14         #控制图中数值标注大小。比如曲线点旁边的 0.7167、柱子上方的 0.9377：
    panel_letter_fontsize = 12   #控制子图左上角的 panel 字母大小。比如 a、b、c、d、e：

    # ---------- 本 panel 的可调参数：柱状图设置 ----------
    budgets = [1, 5, 10, 50]
    domain_order = ["source", "target_h", "target_self"]
    domain_labels = {
        "source": "Source",
        "target_h": "Target-H",
        "target_self": "Target-Self",
    }
    budget_colors = {
        1: "#81AFD4",
        5: "#CAE0E8",
        10: "#DE7C7D",
        50: "#FBCFBB",
    }
    bar_width = 0.2
    bar_edge_width = 0.4
    bar_alpha = 0.8

    # ---------- 本 panel 的可调参数：坐标轴、网格、标注 ----------
    y_lim = (0.5, 1.08)
    y_ticks = [0.6, 0.7, 0.8, 0.9, 1.0]
    grid_linewidth = 0.7
    grid_alpha = 0.35
    axis_linewidth = 0.9
    annotate_values = True
    annotation_rotation = 90
    annotation_y_offset = 0.018

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_family, "DejaVu Sans"]

    x_positions = np.arange(len(domain_order))
    # offsets 用来把同一个 domain 下的 4 根柱子左右错开。
    # bar_width 越大，柱子越宽；offsets 的范围也随之变大。
    offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, len(budgets))

    for offset, budget in zip(offsets, budgets):
        # 取出当前 budget 下三个 domain 的 Success@K。
        # 例如 budget=5 时，这里读取 success_at_5。
        values = [
            float(get_summary_row(summary, domain, "raw_spectrum")[f"success_at_{budget}"])
            for domain in domain_order
        ]

        # 画一组柱子。
        # x_positions 是三个 domain 的中心位置，offset 用来把 Top-1/5/10/50 错开。
        # 如果你想让柱子之间更松，可以减小 bar_width；如果想更紧，可以增大。
        bars = ax.bar(
            x_positions + offset,  # 每根柱子的横坐标位置。
            values,  # 每根柱子的高度，也就是 Success@K。
            width=bar_width,  # 柱子的宽度。
            color=budget_colors[budget],  # 当前 K 对应的颜色。
            edgecolor="white",  # 柱子边缘颜色。白色边缘可以区分相邻柱子。
            linewidth=bar_edge_width,  # 柱子边缘线宽。
            alpha=bar_alpha,  # 透明度；1.0 表示不透明。
            label=f"Top-{budget}",  # 图例文字。
        )

        # if annotate_values:
        #     for bar, value in zip(bars, values):
        #         # 在柱子顶部写具体数值。
        #         # bar.get_x() + bar.get_width()/2 表示柱子中心位置。
        #         # value + annotation_y_offset 表示文字略高于柱顶。
        #         ax.text(
        #             bar.get_x() + bar.get_width() / 2,
        #             value + annotation_y_offset,
        #             f"{value:.4f}",
        #             ha="center",  # 水平居中。
        #             va="bottom",  # 文字底部贴近柱顶。
        #             rotation=annotation_rotation,  # 旋转 90 度，避免数字横向太拥挤。
        #             fontsize=value_fontsize,  # 数值字号。
        #             color=budget_colors[budget],  # 数值颜色与柱子一致。
        #         )

    # 设置标题。如果标题太长，可以改成两行，例如 "Success@K\n(raw-spectrum baseline)"。
    ax.set_title("Success@K (raw-spectrum)", fontsize=title_fontsize)
    # 设置 y 轴标签。
    ax.set_ylabel("Success@K", fontsize=label_fontsize)
    # 设置 x 轴刻度位置为三个 domain 的中心。
    ax.set_xticks(x_positions)
    # 设置 x 轴刻度文字。这里用 Source / Target-H / Target-Self。
    ax.set_xticklabels([domain_labels[d] for d in domain_order], fontsize=tick_fontsize)
    # 设置 y 轴范围。1.08 给顶部竖排数字留一点空间。
    ax.set_ylim(*y_lim)
    # 固定 y 轴刻度，方便和其它图保持一致。
    ax.set_yticks(y_ticks)

    # 只画 y 方向网格，因为柱状图通常不需要 x 方向网格。
    ax.grid(True, axis="y", linestyle="--", linewidth=grid_linewidth, alpha=grid_alpha)
    # 设置坐标轴刻度字号和刻度线宽。
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, width=axis_linewidth)
    # 去掉顶部和右侧边框，减少视觉负担。
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # 设置左侧和底部边框线宽。
    ax.spines["left"].set_linewidth(axis_linewidth)
    ax.spines["bottom"].set_linewidth(axis_linewidth)

    # 添加图例。
    # bbox_to_anchor 控制图例相对坐标轴的位置；这里放在图下方。
    # ncol=4 表示四个图例项横向排成一行。
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=legend_fontsize,
        frameon=False,
    )

    if show_letter:
        add_panel_letter(ax, "b", fontsize=panel_letter_fontsize)

    return fig, ax


# ============================================================
# 5. Panel c：raw-spectrum top-1 amplitude MSE
# ============================================================


def draw_panel_c(ax: plt.Axes | None, summary: pd.DataFrame, show_letter: bool = True) -> tuple[plt.Figure, plt.Axes]:
    """绘制 panel c。

    对应 caption：
    Top-1 amplitude MSE, showing the larger retrieval error of Target-H compared with Source
    and the saturated Target-Self case.

    数据来源：
    baseline_summary.csv 中 method == raw_spectrum 的 mean_top1_amplitude_mse。

    这张图想表达：
    - Target-H 的 top1 MSE 大于 Source，说明 topology shift 更难；
    - Target-Self 的 top1 MSE 极小，说明当前 retrieval-only 已经饱和。
    """

    # ---------- 本 panel 的可调参数：单图尺寸 ----------
    # panel c 元素比较少，可以稍微窄一点；如果想更宽，就改这里。
    # panel c 的 x 轴有 Target-H / Target-Self 两个较长标签。
    # 如果标签拥挤，可以在这里加宽，例如 (4.8, 4.5)、(5.2, 4.5)。
    panel_figsize = (3.5, 4.5)

    # 如果外部没有传入 ax，说明现在是在单独画 panel a，就在这里创建画布。
    # 如果外部传入了 ax，说明现在是在组合图里画 panel a，就继续使用已有坐标轴。
    if ax is None:
        fig, ax = plt.subplots(figsize=panel_figsize)
    else:
        fig = ax.figure

    # ---------- 本 panel 的可调参数：字体 ----------
    font_family = "Arial"
    title_fontsize = 16
    label_fontsize = 16         #控制坐标轴标题大小。比如 x 轴的 K (top-K)，y 轴的 Success@K：
    tick_fontsize = 16         #控制坐标轴刻度数字大小。比如 x 轴上的 1, 5, 10, 50，y 轴上的 0.2, 0.4, 0.6：
    legend_fontsize = 16       #控制图例文字大小。比如 Source (in-domain)、Target-H 这些：
    value_fontsize = 14         #控制图中数值标注大小。比如曲线点旁边的 0.7167、柱子上方的 0.9377：
    panel_letter_fontsize = 12   #控制子图左上角的 panel 字母大小。比如 a、b、c、d、e：

    # ---------- 本 panel 的可调参数：柱状图 ----------
    domain_order = ["source", "target_h", "target_self"]
    domain_labels = {
        "source": "Source",
        "target_h": "Target-H",
        "target_self": "Target-Self",
    }
    domain_colors = {
        "source": "#81AFD4",
        "target_h": "#DE7C7D",
        "target_self": "#CAE0E8",

        # "source": "#cce4c2",
        # "target_h": "#efbcb8",
        # "target_self": "#aecce4",
    }
    bar_width = 0.4       #控制柱子的宽度。
    bar_alpha = 0.7        #控制柱子的透明度。
    bar_edge_width = 0.4   #控制柱子边框线宽。

    # ---------- 本 panel 的可调参数：log 坐标 ----------
    y_lim = (1e-11, 1e0)
    grid_linewidth = 0.7
    grid_alpha = 0.35
    axis_linewidth = 0.9
    annotate_values = True
    annotation_multiplier = 1.8

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_family, "DejaVu Sans"]

    # x_positions 是三根柱子的横坐标位置：0, 1, 2。
    x_positions = np.arange(len(domain_order))* 0.6
    # 从 summary 表中读取三个 domain 的 raw-spectrum mean_top1_amplitude_mse。
    values = [
        float(get_summary_row(summary, domain, "raw_spectrum")["mean_top1_amplitude_mse"])
        for domain in domain_order
    ]
    # 按 domain 顺序取颜色，保证 Source / Target-H / Target-Self 和 panel a 保持一致。
    colors = [domain_colors[d] for d in domain_order]

    # 画 MSE 柱状图。
    # 注意这里后面会设置 y 轴为 log，因此柱高虽然是原始 MSE，但视觉上按数量级显示。
    bars = ax.bar(
        x_positions,  # 三根柱子的横坐标。
        values,  # 三根柱子的高度，即 mean top-1 amplitude MSE。
        width=bar_width,  # 柱子宽度。
        color=colors,  # 每个 domain 的颜色。
        edgecolor="white",  # 白色边缘让柱子边界更清楚。
        linewidth=bar_edge_width,  # 柱子边缘线宽。
        alpha=bar_alpha,  # 柱子透明度。
    )

    if annotate_values:
        for bar, value, domain in zip(bars, values, domain_order):
            # 在柱子上方写出具体 MSE 数值。
            # 因为 y 轴是 log，文字 y 坐标使用 value * annotation_multiplier，
            # 比 value + 常数更稳定。
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # 文字横坐标：柱子中心。
                value * annotation_multiplier,  # 文字纵坐标：柱高乘一个倍数。
                f"{value:.2e}",  # 科学计数法显示，例如 1.62e-03。
                ha="center",  # 水平居中。
                va="bottom",  # 文字底部贴近指定 y 坐标。
                fontsize=value_fontsize,  # 标注字号。
                fontweight="bold",  # 数值加粗；如果想恢复普通粗细，改成 "normal"。
                color=domain_colors[domain],  # 数值颜色与曲线颜色一致。
                zorder=20,  # 让文字处在最上层，避免被线条、点或网格遮挡。
            )

    # 设置标题。
    ax.set_title("Top-1 MSE (raw-spectrum)", fontsize=title_fontsize)
    # 设置 y 轴标签。
    ax.set_ylabel("Top-1 MSE", fontsize=label_fontsize)
    # 设置 x 轴刻度位置。
    ax.set_xticks(x_positions)
    # 设置 x 轴刻度文字。
    # x 轴标签稍微旋转 15°，避免 Target-H 和 Target-Self 挤在一起。
    # 如果你想水平显示，改 rotation=0；如果还拥挤，可以改成 rotation=30。
    ax.set_xticklabels(
        [domain_labels[d] for d in domain_order],
        fontsize=tick_fontsize,
        rotation=15,
        ha="right",
    )
    # 设置 y 轴为 log scale，这是因为 Target-Self 的 MSE 比另外两个小很多。
    # 如果改成线性坐标，Target-Self 会几乎看不见。
    ax.set_yscale("log")
    # 设置 log y 轴范围。下限越小，越能显示 Target-Self；上限通常设为 1。
    ax.set_ylim(*y_lim)

    # 只画 y 方向虚线网格，用于辅助读取数量级。
    ax.grid(True, axis="y", linestyle="--", linewidth=grid_linewidth, alpha=grid_alpha)
    # 设置刻度文字大小和刻度线宽。
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, width=axis_linewidth)
    # 隐藏上边框和右边框。
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # 设置左边框和底边框线宽。
    ax.spines["left"].set_linewidth(axis_linewidth)
    ax.spines["bottom"].set_linewidth(axis_linewidth)

    if show_letter:
        add_panel_letter(ax, "c", fontsize=panel_letter_fontsize)

    return fig, ax


# ============================================================
# 6. Panel d 数据计算：Target-H top-K best-MSE 分布
# ============================================================


def compute_target_h_best_mse_distributions() -> dict[int, np.ndarray]:
    """重新计算 Target-H 在 top-K 检索种子内的 best amplitude MSE 分布。

    为什么不直接读取 query_level_results.csv？
    - query_level_results.csv 里保存了 top50 里的最终 best MSE；
    - 但是 panel d 需要分别画 top1、top5、top10、top50 的 best-MSE 分布；
    - 因此这里重新做一次 Target-H raw-spectrum top50 检索，然后对每个 query
      分别计算前 K 个候选中的最低 MSE。

    返回值：
    - 字典 {1: array, 5: array, 10: array, 50: array}
    - 每个 array 的长度等于 Target-H test query 数量。
    """

    budgets = [1, 5, 10, 50]
    target_domain = "target_h"
    retrieval_batch_size = 128

    # load_config 当前会把 config 里的相对路径解析成绝对路径。
    # 这里虽然不直接使用 cfg，但 load_config 会验证配置文件可读。
    _ = load_config(CONFIG_PATH)

    bank = load_memory_bank(MEMORY_BANK_PATH)
    query_data = load_processed_split(PROCESSED_ROOT, target_domain, "test")

    top_indices = retrieve_raw_spectrum(
        bank_spectra=bank["spectra"],
        query_spectra=query_data["spectra"],
        top_k=max(budgets),
        batch_size=retrieval_batch_size,
    )

    distributions: dict[int, list[float]] = {budget: [] for budget in budgets}

    for query_index, candidate_indices in enumerate(top_indices):
        target_spectrum = query_data["spectra"][query_index]

        # 先算这个 query 的 top50 候选各自的 amplitude MSE。
        # 后面 top1/top5/top10/top50 都可以从同一组 mse_values 里取前 K 的最小值。
        mse_values = np.array(
            [
                amplitude_mse(bank["spectra"][candidate_index], target_spectrum)
                for candidate_index in candidate_indices
            ],
            dtype=np.float64,
        )

        for budget in budgets:
            best_mse = float(np.min(mse_values[:budget]))
            distributions[budget].append(best_mse)

    return {budget: np.array(values, dtype=np.float64) for budget, values in distributions.items()}


# ============================================================
# 7. Panel d：Target-H best-MSE distribution within top-K seeds
# ============================================================


def draw_panel_d(
    ax: plt.Axes | None,
    best_mse_by_budget: dict[int, np.ndarray],
    show_letter: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """绘制 panel d。

    对应 caption：
    Target-H best-MSE distribution within top-K retrieved seeds.
    Larger retrieved neighbourhoods progressively expose lower-error candidates;
    the dashed line marks the amplitude-MSE success threshold.

    这张图想表达：
    - 当 K 从 1 增加到 50 时，retrieval neighbourhood 变大；
    - 更大的 neighbourhood 理论上能暴露更多低误差候选；
    - 红色虚线是 amplitude MSE 成功阈值 0.03。
    """

    # ---------- 本 panel 的可调参数：单图尺寸 ----------
    # panel d 横向有 4 个分布和右侧阈值文字，所以建议比其它图更宽。
    # 例如 (6.4, 4.2)、(7.2, 4.5)、(8, 5) 都可以。
    panel_figsize = (5,4.5)

    # 如果外部没有传入 ax，说明现在是在单独画 panel a，就在这里创建画布。
    # 如果外部传入了 ax，说明现在是在组合图里画 panel a，就继续使用已有坐标轴。
    if ax is None:
        fig, ax = plt.subplots(figsize=panel_figsize)
    else:
        fig = ax.figure

    # ---------- 本 panel 的可调参数：字体 ----------
    font_family = "Arial"
    title_fontsize = 16
    label_fontsize = 16         #控制坐标轴标题大小。比如 x 轴的 K (top-K)，y 轴的 Success@K：
    tick_fontsize = 16         #控制坐标轴刻度数字大小。比如 x 轴上的 1, 5, 10, 50，y 轴上的 0.2, 0.4, 0.6：
    legend_fontsize = 16       #控制图例文字大小。比如 Source (in-domain)、Target-H 这些：
    value_fontsize = 14         #控制图中数值标注大小。比如曲线点旁边的 0.7167、柱子上方的 0.9377：
    panel_letter_fontsize = 12   #控制子图左上角的 panel 字母大小。比如 a、b、c、d、e：

    # ---------- 本 panel 的可调参数：字体 ----------
    threshold_fontsize = 14


    # ---------- 本 panel 的可调参数：小提琴图 ----------
    budgets = [1, 5, 10, 50]

    # 每个 top-K 小提琴图使用一种颜色。
    # 字典的 key 是 K，value 是颜色。
    # 例如 1: "#81AFD4" 表示 top-1 的小提琴图用蓝色。
    # 如果你想换颜色，只需要改这里，不需要改下面绘图逻辑。
    budget_colors = {
        1: "#81AFD4",
        5: "#CAE0E8",
        10: "#DE7C7D",
        50: "#FBCFBB",
    }

    # 小提琴图边框也使用同色调，但可以稍微不透明一点。
    # 当前直接复用 budget_colors；如果你想边框更深，可以单独写一个 budget_edge_colors。
    budget_edge_colors = {
        1: "#81AFD4",
        5: "#CAE0E8",
        10: "#DE7C7D",
        50: "#FBCFBB",
    }

    # 小提琴图主体透明度。
    # 1.0 是完全不透明；0.48 是半透明，能让内部四分位线更清楚。
    violin_alpha = 0.48

    # 小提琴图宽度。
    # 数值越大，小提琴越胖；数值越小，小提琴越窄。
    violin_width = 0.62

    # ---------- 本 panel 的可调参数：中位数和四分位线 ----------
    # 四分位线颜色是否跟随对应 top-K 的小提琴颜色。
    # True：top1/top5/top10/top50 的四分位线分别使用 budget_colors 中的颜色。
    # False：四个小提琴都使用 fixed_quartile_line_color。
    use_budget_color_for_quartile = True

    # 如果 use_budget_color_for_quartile=False，就使用这个固定颜色。
    fixed_quartile_line_color = "#222222"

    # 四分位线宽度。
    # 这里画的是 Q1 到 Q3 的竖线，表示中间 50% 数据范围。
    quartile_line_width = 4.0

    # 中位数点大小。
    median_marker_size = 42

    # 中位数点填充颜色。
    # "white" 是白心点；如果想让中位数点也填充同色，可以在下面 scatter 里改成 median_color。
    median_marker_face = "white"

    # 中位数点边框是否使用对应 top-K 的颜色。
    use_budget_color_for_median = True

    # 如果 use_budget_color_for_median=False，就使用这个固定边框颜色。
    fixed_median_marker_edge = "#222222"

    # ---------- 本 panel 的可调参数：中位数趋势线 ----------
    # 你提到“把提琴图中间那个点连成一条线”，这里就是控制这条线。
    # 它连接 top1/top5/top10/top50 四个分布的中位数点。
    # 这样即使小提琴主体差异不特别夸张，也能清楚看到 best MSE 随 K 增大整体下降。
    draw_median_trend_line = True
    median_trend_line_color = "#DEE3EF"   # 趋势线颜色；想更醒目可以改成 "#000000"。
    median_trend_line_width = 2.2         # 趋势线粗细。
    median_trend_line_style = "-"         # 趋势线样式；也可以改成 "--"。
    median_trend_marker = "o"             # 趋势线上的点型。
    median_trend_marker_size = 5.5        # 趋势线点大小。
    median_trend_zorder = 9               # 趋势线图层，数值越大越在上面。

    # ---------- 本 panel 的可调参数：MSE log 坐标 ----------
    mse_clip_min = 1e-10
    y_lim_log10 = (-5.3, 0)
    y_ticks_log10 = [-5, -4, -3, -2, -1, 0]

    # ---------- 本 panel 的可调参数：成功阈值线 ----------
    success_threshold_mse = 0.03
    threshold_line_color = "#4FB1B2"      #控制成功阈值线颜色。
    threshold_line_width = 2        #控制成功阈值线粗细。
    threshold_line_dash = (5, 3)      #控制虚线样式。意思是：画 5 个单位，空 3 个单位
    threshold_text_x_offset = 0.48    #控制红色说明文字距离最后一个小提琴图多远。

    # ---------- 本 panel 的可调参数：坐标轴和网格 ----------
    grid_linewidth = 0.7
    grid_alpha = 0.35
    axis_linewidth = 0.9

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_family, "DejaVu Sans"]

    # positions 是四个小提琴图在 x 轴上的位置：1, 2, 3, 4。
    # 这里的位置和 budgets 的实际数值 1/5/10/50 不同；
    # 后面会用 set_xticklabels 把它们显示成 1/5/10/50。
    positions = np.arange(1, len(budgets) + 1)

    # 因为 MSE 跨数量级变化，这里先取 log10 再画小提琴图。
    log_data = []
    for budget in budgets:
        # clip 是为了避免 MSE 极小或为 0 时 log10 出现 -inf。
        # mse_clip_min 越小，越能保留极低误差；但过小可能让 y 轴范围太大。
        clipped_mse = np.clip(best_mse_by_budget[budget], mse_clip_min, None)
        # 小提琴图实际画的是 log10(MSE)，这样纵轴可以显示 10^-5, 10^-4 等数量级。
        log_data.append(np.log10(clipped_mse))

    # 画小提琴图。
    # log_data 是四组分布数据，分别对应 top1/top5/top10/top50。
    # widths 控制小提琴图横向宽度；越大越宽。
    violin = ax.violinplot(
        log_data,  # 四组 log10(MSE) 分布。
        positions=positions,  # 四个小提琴图的 x 轴位置。
        widths=violin_width,  # 小提琴宽度。
        showmeans=False,  # 不显示默认均值线，因为我们手动画更清楚。
        showmedians=False,  # 不显示默认中位数线。
        showextrema=False,  # 不显示默认最大最小值竖线。
    )

    # 修改每个小提琴图主体的颜色、边框和透明度。
    # violin["bodies"] 的顺序与 log_data / budgets 的顺序一致：
    # 第 1 个 body 对应 top-1，第 2 个对应 top-5，依此类推。
    for body, budget in zip(violin["bodies"], budgets):
        # 设置小提琴主体填充颜色。
        body.set_facecolor(budget_colors[budget])
        # 设置小提琴边框颜色。
        body.set_edgecolor(budget_edge_colors[budget])
        # 设置透明度。
        body.set_alpha(violin_alpha)
        # 设置小提琴边框线宽。
        body.set_linewidth(1.1)

    # 手动画四分位范围和中位数点，样式更容易微调。
    # 同时把每个 K 的中位数保存下来，后面连成趋势线。
    median_positions = []
    median_values = []

    for pos, values, budget in zip(positions, log_data, budgets):
        # q1/q2/q3 分别是一四分位数、中位数、三四分位数。
        q1, q2, q3 = np.percentile(values, [25, 50, 75])
        median_positions.append(pos)
        median_values.append(q2)

        # 当前 top-K 对应的颜色。
        median_color = budget_colors[budget]
        quartile_color = budget_colors[budget] if use_budget_color_for_quartile else fixed_quartile_line_color
        median_edge_color = budget_colors[budget] if use_budget_color_for_median else fixed_median_marker_edge

        # 画 q1 到 q3 的粗黑竖线，表示中间 50% 数据范围。
        # 如果你觉得黑线太粗，改 quartile_line_width。
        ax.plot(
            [pos, pos],
            [q1, q3],
            color=quartile_color,
            linewidth=quartile_line_width,
            solid_capstyle="butt",
            zorder=5,
        )
        # 画中位数点。白心黑边比纯黑点更清楚。
        # 如果你想改成横线，可以用 ax.hlines(q2, pos-0.1, pos+0.1)。
        ax.scatter(
            [pos],
            [q2],
            color=median_marker_face,
            edgecolor=median_edge_color,
            linewidth=0.9,
            s=median_marker_size,
            zorder=6,
        )

    # 连接四个中位数点，形成一条趋势线。
    # 注意：这里连接的是 log10(MSE) 坐标下的中位数。
    # 如果线向下，表示 top-K 增大后，中位数 best-MSE 下降。
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

    # 把成功阈值 0.03 转换到 log10 坐标。
    threshold_log10 = np.log10(success_threshold_mse)
    # 画红色虚线，表示 amplitude-MSE success threshold。
    # linestyle="--" 和 dashes 控制虚线形态。
    ax.axhline(
        threshold_log10,
        color=threshold_line_color,
        linestyle="--",
        linewidth=threshold_line_width,
        dashes=threshold_line_dash,
    )
    # 在虚线右侧添加阈值说明文字。
    # threshold_text_x_offset 控制文字离最后一个小提琴图的距离。
    threshold_text_x = np.mean(positions)
    threshold_text_y = threshold_log10 + 0.25

    ax.text(
        threshold_text_x,
        threshold_text_y,
        f"Success threshold\n(MSE = {success_threshold_mse:g})",
        color=threshold_line_color,
        fontsize=threshold_fontsize,
        va="bottom",
        ha="center",
    )

    # 设置标题。
    ax.set_title("Target-H: best MSE within top-K seeds", fontsize=title_fontsize)
    # 设置 x 轴标题。
    ax.set_xlabel("Top-K seeds", fontsize=label_fontsize)
    # 设置 y 轴标题。虽然数据已经 log10，但标签写 log scale 更符合读者理解。
    ax.set_ylabel("Best amp-MSE (log scale)", fontsize=label_fontsize)
    # 设置 x 轴刻度位置。
    ax.set_xticks(positions)
    # 把 1/2/3/4 这四个位置显示成真实 top-K 数字。
    ax.set_xticklabels([str(k) for k in budgets], fontsize=tick_fontsize)
    # 设置 y 轴范围，对应 10^-5.3 到 10^0.05。
    ax.set_ylim(*y_lim_log10)
    # 设置 y 轴刻度位置，这里是 log10 值。
    ax.set_yticks(y_ticks_log10)
    # 把 log10 刻度显示成 10^n 的数学格式。
    ax.set_yticklabels([rf"$10^{{{tick}}}$" for tick in y_ticks_log10], fontsize=tick_fontsize)

    # 添加 y 方向网格线，便于看不同数量级。
    ax.grid(True, axis="y", linestyle="--", linewidth=grid_linewidth, alpha=grid_alpha)
    # 设置刻度样式。
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, width=axis_linewidth)
    # 隐藏顶部和右侧边框。
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # 设置左侧和底部边框线宽。
    ax.spines["left"].set_linewidth(axis_linewidth)
    ax.spines["bottom"].set_linewidth(axis_linewidth)

    if show_letter:
        add_panel_letter(ax, "d", fontsize=panel_letter_fontsize, x=-0.12, y=1.08)

    return fig, ax


# ============================================================
# 8. Panel e 辅助函数：生成雷达图角度
# ============================================================


def closed_radar_angles(num_axes: int) -> np.ndarray:
    """生成闭合雷达图角度。

    例如有 4 个轴时，返回 5 个角度：
    前 4 个是各轴位置，第 5 个重复第 1 个，用于闭合曲线。
    """

    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False)
    return np.concatenate([angles, angles[:1]])


# ============================================================
# 9. Panel e：retrieval representation ablation 雷达图
# ============================================================


def draw_panel_e(ax: plt.Axes | None, summary: pd.DataFrame, show_letter: bool = True) -> tuple[plt.Figure, plt.Axes]:
    """绘制 panel e。

    对应 caption：
    Retrieval representation ablation in terms of Success@50.
    Raw-spectrum retrieval provides the strongest candidate-generation baseline,
    whereas physics-summary and domain-aware retrieval are retained as diagnostic representations.

    数据来源：
    baseline_summary.csv 中三种 method 的 success_at_50。

    这张图想表达：
    - raw-spectrum 是最强的候选生成 baseline；
    - physics-summary 和 domain-aware 当前不是最强性能方法；
    - 但它们仍然有诊断意义，后续给 planner/critic 使用。
    """

    # ---------- 本 panel 的可调参数：单图尺寸 ----------
    # 雷达图通常适合接近正方形。
    # 如果图例放右侧后显得拥挤，可以改成 (6.2, 5.0) 或 (7, 5.2)。
    panel_figsize = (7.5, 4.5)

    if ax is None:
        fig, ax = plt.subplots(figsize=panel_figsize, subplot_kw={"projection": "polar"})
    else:
        fig = ax.figure

    # ---------- 本 panel 的可调参数：字体 ----------

    font_family = "Arial"
    title_fontsize = 16
    tick_fontsize = 16         #控制坐标轴刻度数字大小。比如 x 轴上的 1, 5, 10, 50，y 轴上的 0.2, 0.4, 0.6：
    legend_fontsize = 16       #控制图例文字大小。比如 Source (in-domain)、Target-H 这些：
    value_fontsize = 14         #控制图中数值标注大小。比如曲线点旁边的 0.7167、柱子上方的 0.9377：
    panel_letter_fontsize = 16   #控制子图左上角的 panel 字母大小。比如 a、b、c、d、e：


    # ---------- 本 panel 的可调参数：雷达图轴 ----------
    domain_order = ["source", "target_h", "target_self"]
    radar_labels = ["Source", "Target-H", "Target-Self", "Overall\n(avg.)"]

    # 雷达图半径范围。
    # 默认雷达图常从 0 开始，但本图所有 Success@50 都接近 1，
    # 如果从 0 开始，0.94 和 0.98 的差异会被压得很小。
    # 因此这里使用局部放大坐标，例如 0.93 到 1.00。
    # 如果你想更聚焦，可以改成：
    #   radar_rmin = 0.95
    #   radar_rmax = 1.00
    # 但注意 Target-H 的 physics_summary 大约是 0.9384，
    # 如果 rmin 高于它，该点会被裁到图外。
    radar_rmin = 0.93
    radar_rmax = 1.00

    # 雷达图半径刻度。
    # 这里不再使用 0, 0.25, 0.5, 0.75, 1.0，
    # 而是使用局部放大的 0.94, 0.96, 0.98, 1.00。
    # 如果你坚持从 0.95 起，可以改成 [0.95, 0.96, 0.97, 0.98, 0.99, 1.00]。
    radar_rticks = [0.94, 0.96, 0.98, 1.00]

    grid_linewidth = 0.6
    grid_alpha = 0.35
    axis_linewidth = 0.9

    # ---------- 本 panel 的可调参数：三种 retrieval 表征 ----------
    methods = ["raw_spectrum", "physics_summary", "domain_aware"]
    method_labels = {
        "raw_spectrum": "Raw spectrum",
        "physics_summary": "Physics summary",
        "domain_aware": "Domain-aware",
    }
    method_colors = {
        "raw_spectrum": "#CAE0E8",
        "physics_summary": "#DE7C7D",
        "domain_aware": "#FBCFBB",
    }
    method_markers = {
        "raw_spectrum": "o",
        "physics_summary": "s",
        "domain_aware": "D",
    }
    line_width = 2.5
    marker_size = 4.5
    fill_alpha = 0.2

    # ---------- 本 panel 的可调参数：数值标注 ----------
    annotate_values = True
    # 三种方法的曲线很接近，所以给不同方法的文字半径错开。
    # 由于现在雷达图只显示 0.93-1.00 的局部范围，
    # 偏移量不能太大，否则数值文字容易跑出图外。
    method_value_offsets = {
        "raw_spectrum": 0.006,
        "physics_summary": 0.000,
        "domain_aware": -0.006,
    }

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_family, "DejaVu Sans"]

    # 生成雷达图角度。
    # radar_labels 有 4 个轴，所以 angles 会有 5 个值，最后一个重复第一个，用于闭合曲线。
    angles = closed_radar_angles(len(radar_labels))

    for method in methods:
        # 读取 Source / Target-H / Target-Self 三个 domain 的 Success@50。
        domain_values = [
            float(get_summary_row(summary, domain, method)["success_at_50"])
            for domain in domain_order
        ]

        # 第四个轴是 Overall average，也就是三个 domain 的平均值。
        values = domain_values + [float(np.mean(domain_values))]

        # 雷达图曲线需要闭合，所以最后再重复第一个点。
        closed_values = np.array(values + values[:1], dtype=np.float64)

        # 画当前 retrieval method 的雷达图折线。
        # angles 是四个轴加一个闭合点的位置，closed_values 是对应的 Success@50。
        # 如果你想让某个方法更突出，可以单独给它更大的 linewidth 或 marker_size。
        ax.plot(
            angles,  # 雷达图角度位置。
            closed_values,  # 每个角度上的数值。
            color=method_colors[method],  # 当前方法的线条颜色。
            linewidth=line_width,  # 线宽。
            marker=method_markers[method],  # 点型。
            markersize=marker_size,  # 点大小。
            label=method_labels[method],  # 图例文字。
        )
        # 给折线内部填充淡色。
        # alpha 很小是为了避免三种方法互相遮挡。
        # 如果你不想要填充，可以注释掉这一行。
        ax.fill(angles, closed_values, color=method_colors[method], alpha=fill_alpha)

        # if annotate_values:
        #     value_offset = method_value_offsets[method]
        #     for angle, value in zip(angles[:-1], values):
        #         # 数值标注的半径位置。
        #         # np.clip 用来避免文字超出雷达图最外圈。
        #         # 数值文字的半径位置限制在当前局部坐标范围内。
        #         # radar_rmin + 0.002 防止文字压到最内圈边界。
        #         # radar_rmax - 0.002 防止文字超过外圈。
        #         text_radius = float(np.clip(value + value_offset, radar_rmin + 0.002, radar_rmax - 0.002))
        #         # 在每个雷达图顶点附近写具体 Success@50 数值。
        #         # 如果觉得拥挤，可以把 annotate_values 改为 False。
        #         ax.text(
        #             angle,  # 文字角度。
        #             text_radius,  # 文字距离圆心的半径。
        #             f"{value:.4f}",  # 数值文本。
        #             fontsize=value_fontsize,  # 数值字号。
        #             color=method_colors[method],  # 数值颜色与方法颜色一致。
        #             ha="center",  # 水平居中。
        #             va="center",  # 垂直居中。
        #         )

    # 设置雷达图标题。pad 控制标题和雷达图主体之间的距离。
    ax.set_title("Retrieval representation (Success@50)", fontsize=title_fontsize, pad=18)

    # 让第一个轴 Source 位于正上方，并按照顺时针方向排列。
    # 默认雷达图第一个轴在右侧，这两句把它旋转到顶部并改成顺时针。
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 不同 matplotlib 版本里 set_theticks 兼容性不同，因此使用 set_xticks。
    # angles[:-1] 去掉最后那个闭合重复角度。
    ax.set_xticks(angles[:-1])
    # 设置四个轴的文字标签。
    ax.set_xticklabels(radar_labels, fontsize=tick_fontsize)

    # 设置雷达图半径范围。Success@50 最大为 1，所以 1.05 给数值标注留空间。
    # 设置雷达图半径范围。
    # 这里从 radar_rmin 开始，而不是从 0 开始，相当于把 0.93-1.00 这一段局部放大。
    ax.set_ylim(radar_rmin, radar_rmax)
    # 设置半径刻度。
    ax.set_yticks(radar_rticks)
    # 设置半径刻度显示文本。
    ax.set_yticklabels([f"{tick:.2f}" for tick in radar_rticks], fontsize=value_fontsize)
    # 把 0.94 / 0.96 / 0.98 / 1.00 这些半径刻度文字放到最上层。
    # zorder 越大，越晚绘制，越不容易被填充色、曲线或网格盖住。
    # clip_on=False 表示即使刻度文字稍微超出坐标轴边界，也不要裁剪掉。
    for radial_tick_label in ax.get_yticklabels():
        radial_tick_label.set_zorder(30)
        radial_tick_label.set_clip_on(False)
    # 设置雷达图网格线。雷达图里 linestyle="-" 通常比虚线更清楚。
    ax.grid(True, linestyle="-", linewidth=grid_linewidth, alpha=grid_alpha)
    # 设置雷达图最外圈边框线宽。
    ax.spines["polar"].set_linewidth(axis_linewidth)

    # ---------- 自定义图例 ----------
    # matplotlib 默认 legend 是“线条 + 文字”左右并排。
    # 你现在想要的是竖向结构：
    #   蓝色线条
    #   Raw spectrum
    #
    #   红色线条
    #   Physics summary
    #
    #   绿色线条
    #   Domain-aware
    #
    # 所以这里不用 ax.legend，而是用 ax.plot + ax.text 在坐标轴外侧手动画图例。
    #
    # transform=ax.transAxes 表示下面的坐标使用“坐标轴相对坐标”：
    # x=0 是图左边，x=1 是图右边；y=0 是图底部，y=1 是图顶部。
    # x 大于 1 表示画到雷达图右侧外面。
    manual_legend_x = 1.42          # 自定义图例整体的横向位置；越大越靠右。
    manual_legend_y_start = 0.80    # 第一个图例项线条的 y 位置；越大越靠上。
    manual_legend_item_gap = 0.30   # 不同图例项之间的竖向间距。
    manual_legend_line_length = 0.15  # 每条图例线的长度。
    manual_legend_text_gap = 0.05    # 线条和下方文字之间的距离。
    manual_legend_line_width = line_width
    manual_legend_marker_size = marker_size

    for legend_index, method in enumerate(methods):
        # 当前图例项的线条 y 坐标。
        line_y = manual_legend_y_start - legend_index * manual_legend_item_gap

        # 画图例中的一小段彩色线条。
        # [manual_legend_x, manual_legend_x + manual_legend_line_length] 控制线条左右端点。
        # [line_y, line_y] 表示这条线是水平的。
        ax.plot(
            [manual_legend_x, manual_legend_x + manual_legend_line_length],
            [line_y, line_y],
            transform=ax.transAxes,
            color=method_colors[method],
            linewidth=manual_legend_line_width,
            marker=method_markers[method],
            markersize=manual_legend_marker_size,
            clip_on=False,  # 允许画在坐标轴外面，否则右侧图例可能被裁掉。
        )

        # 在彩色线条下方写方法名称。
        ax.text(
            manual_legend_x + manual_legend_line_length / 2,
            line_y - manual_legend_text_gap,
            method_labels[method],
            transform=ax.transAxes,
            fontsize=legend_fontsize,
            color="black",
            ha="center",
            va="top",
            clip_on=False,
        )

    if show_letter:
        add_panel_letter(ax, "e", fontsize=panel_letter_fontsize, x=-0.12, y=1.10)

    return fig, ax


# ============================================================
# 10. 组合图：把 a-e 放在一张总图中
# ============================================================


def draw_combined_figure(summary: pd.DataFrame, best_mse_by_budget: dict[int, np.ndarray]) -> plt.Figure:
    """绘制 a-e 五个 panel 的组合图。

    注意：
    - 组合图自己的尺寸、网格比例、间距都在这里单独设置；
    - 每个 panel 内部仍然使用自己的颜色、字号、坐标轴设置。
    """

    # ---------- 组合图可调参数：总图尺寸 ----------
    # 这里就是组合图的画布尺寸，格式是 (宽, 高)，单位是 inch。
    # 例如：
    # - (13.8, 8.2)：当前横向较宽，适合 a-e 五个 panel 放在一张图里；
    # - (12, 8)：整体更紧凑；
    # - (16, 9)：更适合宽屏展示或后期裁剪。
    combined_figsize = (13.8, 8.2)

    # ---------- 组合图可调参数：子图布局 ----------
    # height_ratios 控制上下两行的高度比例。
    # width_ratios 控制三列的宽度比例。
    # hspace 控制上下两行之间的距离。
    # wspace 控制左右子图之间的距离。
    height_ratios = [1.0, 1.05]
    width_ratios = [1.25, 1.25, 1.0]
    hspace = 0.48
    wspace = 0.34

    # 创建总图画布。
    # figsize=(宽, 高)，单位是英寸；如果整体图太挤，可以先增大这里。
    fig = plt.figure(figsize=combined_figsize)
    # add_gridspec 用来定义 2 行 3 列的布局。
    # height_ratios 控制上下两行高度比例；width_ratios 控制三列宽度比例。
    # hspace/wspace 控制子图之间的竖向/横向间距。
    grid = fig.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=hspace,
        wspace=wspace,
    )

    # 第一行三个子图：a、b、c。
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    # 第二行左侧两个格子合并给 panel d，因为小提琴图横向信息比较多。
    ax_d = fig.add_subplot(grid[1, 0:2])
    # 第二行右侧是 panel e 雷达图，所以需要 projection="polar"。
    ax_e = fig.add_subplot(grid[1, 2], projection="polar")

    # 分别调用每个 panel 的绘图函数。
    # show_letter=True 表示在组合图里显示 a/b/c/d/e 标号。
    draw_panel_a(ax_a, summary, show_letter=True)
    draw_panel_b(ax_b, summary, show_letter=True)
    draw_panel_c(ax_c, summary, show_letter=True)
    draw_panel_d(ax_d, best_mse_by_budget, show_letter=True)
    draw_panel_e(ax_e, summary, show_letter=True)

    return fig


# ============================================================
# 11. 单独 panel 输出：每张图单独保存，方便论文排版
# ============================================================


def draw_single_panels(summary: pd.DataFrame, best_mse_by_budget: dict[int, np.ndarray]) -> None:
    """把每个 panel 单独保存一份。

    单图尺寸也不走全局统一参数，而是在这里逐个设置。
    如果后面你想让 panel d 更宽，直接改 fig_d 的 figsize 即可。
    """

    # 现在每个 panel 的 figsize 都放在对应 draw_panel_* 函数内部。
    # 这里传 ax=None，表示让 draw_panel_* 自己创建画布和坐标轴。
    # 例如想改 panel a 的尺寸，就进入 draw_panel_a 里面改 panel_figsize。

    fig_a, ax_a = draw_panel_a(None, summary, show_letter=False)
    save_figure(fig_a, "panel_a_raw_success_at_k")

    fig_b, ax_b = draw_panel_b(None, summary, show_letter=False)
    save_figure(fig_b, "panel_b_raw_grouped_success_at_k")

    fig_c, ax_c = draw_panel_c(None, summary, show_letter=False)
    save_figure(fig_c, "panel_c_raw_top1_mse")

    fig_d, ax_d = draw_panel_d(None, best_mse_by_budget, show_letter=False)
    save_figure(fig_d, "panel_d_target_h_best_mse_distribution")

    fig_e, ax_e = draw_panel_e(None, summary, show_letter=False)
    save_figure(fig_e, "panel_e_representation_ablation_radar")

    if not SHOW_FIGURES:
        # 如果不弹窗，就主动关闭这些单独 panel，避免内存里积累太多 figure。
        # 如果 SHOW_FIGURES=True，这里不关闭，后面的 plt.show() 会一起展示。
        plt.close(fig_a)
        plt.close(fig_b)
        plt.close(fig_c)
        plt.close(fig_d)
        plt.close(fig_e)


# ============================================================
# 12. 主程序入口
# ============================================================


def main() -> None:
    """脚本主入口。

    运行顺序：
    1. 设置基础 matplotlib 配置；
    2. 创建输出目录；
    3. 读取 retrieval baseline 汇总表；
    4. 重新计算 Target-H top-K best-MSE 分布；
    5. 绘制组合图；
    6. 绘制每个单独 panel。
    """

    setup_basic_matplotlib()
    ensure_output_dir()

    print(f"[Phase01 Figure] 读取 retrieval 汇总表: {SUMMARY_CSV}")
    summary = load_summary_table()

    print("[Phase01 Figure] 计算 Target-H top-K best-MSE 分布，稍等...")
    best_mse_by_budget = compute_target_h_best_mse_distributions()

    print("[Phase01 Figure] 绘制 a-e 组合图...")
    combined_fig = draw_combined_figure(summary, best_mse_by_budget)
    save_figure(combined_fig, "phase01_retrieval_only_evidence_combined")

    print("[Phase01 Figure] 绘制每个单独 panel...")
    draw_single_panels(summary, best_mse_by_budget)

    print(f"[Phase01 Figure] 图片已保存到: {OUTPUT_DIR}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(combined_fig)


if __name__ == "__main__":
    main()
