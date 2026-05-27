"""
Phase 6 device-level metalens comparison figure.

这个文件是“手动运行版”绘图代码，位置和 plot_phase3.py、plot_phase4.py、
plot_phase5.py、plot_phase6_meta-filter.py 一样，都放在 MY-LLM-Framework 根目录。

运行方式：
1. 在 VS Code / PyCharm 中直接点击运行本文件；
2. 或者在终端中运行：
   python plot_phase6_device_field_comparison.py

这张图把旧 metalens 图里的 a/b/c/f 信息和新的三方法 fast-mode CST
field comparison 信息合并到同一张高信息密度总图：

  a. Metalens phase profile
  b. Phase error along aperture
  c. Selected unit amplitude
  d. Manual fast-mode CST field map
  e. DeepSeek fast-mode CST field map
  f. GLM-5.1 fast-mode CST field map
  g. Qualitative device-level field metrics
  h. Focus-location error
  i. On-axis field profile
  j. Focal-plane field profile

注意：
这些 CST 结果是 fast-mode 定性对比结果，适合支持 device-level qualitative
evidence，不要写成完全网格收敛的最终 CST benchmark。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt


# =============================================================================
# 1. 路径设置：这里是你最常改的地方
# =============================================================================

# PROJECT_ROOT 是当前 MY-LLM-Framework 文件夹。
PROJECT_ROOT = Path(__file__).resolve().parent

# 为了让根目录脚本可以直接 import src/my_llm_framework 里的工具函数。
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 图片输出目录。
# 运行后会保存 composite PNG/PDF、standalone panels、source-data CSV 和 README。
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "phase6_device_field_comparison_three_selectors_nonfast_substrate2_v1"

# 是否保存 PNG / PDF。
SAVE_PNG = True
SAVE_PDF = True

# 是否在运行结束后直接弹出 matplotlib 图窗。
# 如果你只是批量导图，可以改成 False。
SHOW_FIGURES = True
# 第一排 a/b/c 三张图的图例字号。
# 注意：后面不会再对 legend 做全局 +2，因此这里就是最终图例字号。
FIRST_ROW_LEGEND_FONTSIZE = 14

# 其他图例的最终字号。
LEGEND_FONTSIZE = 14

# 全图基础字体整体放大幅度。
# 坐标轴名称、坐标数字、colorbar label、panel label 等整体 +2。
FONT_SIZE_DELTA = 2

# 每个 panel 标题在全局 +2 的基础上再额外增大。
TITLE_EXTRA_DELTA = 2

TITLE_PAD = 12

def increase_figure_fontsizes(fig: plt.Figure, delta: int = FONT_SIZE_DELTA) -> None:
    """把当前 figure 中已经存在的文字统一放大 delta pt。

    这个安全版本只遍历 Text 对象一次，避免 tick label 和 legend 被重复放大。
    """

    for text in fig.findobj(match=plt.Text):
        current_size = text.get_fontsize()
        text.set_fontsize(current_size + delta)

def refine_title_and_legend_fonts(
    fig: plt.Figure,
    title_extra_delta: int = TITLE_EXTRA_DELTA,
    legend_fontsize: int = LEGEND_FONTSIZE,
    title_pad: int = TITLE_PAD,
) -> None:
    """单独微调标题和图例字号。"""

    for ax in fig.axes:
        # 增大每个子图标题，不加粗，并控制标题和图之间的距离。
        old_title = ax.get_title()
        if old_title:
            current_size = ax.title.get_fontsize()
            ax.set_title(
                old_title,
                fontsize=current_size + title_extra_delta,
                fontweight="normal",
                pad=title_pad,
            )

        # 减小图例字体。
        legend = ax.get_legend()
        if legend is not None:
            for legend_text in legend.get_texts():
                legend_text.set_fontsize(legend_fontsize)

            legend_title = legend.get_title()
            if legend_title is not None:
                legend_title.set_fontsize(legend_fontsize)
# =============================================================================
# 2. 三种方法的输入文件：这里对应 Manual / DeepSeek / GLM-5.1
# =============================================================================

# 每个 RUN 是一个字符串：
#   方法名=fast-mode CST 场导出文件=metalens design JSON
#
# 注意这里的 field export 是没有后缀的 CST ASCII export 文件：
#   phase5_hero_metalens_field_export_expected
#
# 如果你以后重新导出了 non-fast 或新版本，只需要改这里的路径。
RUNS = [
    (
        "Manual="
        "outputs/phase6_metalens_manual_source_v1/cst_results_nonfast_substrate2_execute/phase5_hero_metalens_field_export_expected="
        "outputs/phase6_metalens_manual_source_v1/phase6_metalens_design.json"
    ),
    (
        "DeepSeek="
        "outputs/phase6_metalens_deepseek_source_v1/cst_results_nonfast_substrate2_execute/phase5_hero_metalens_field_export_expected="
        "outputs/phase6_metalens_deepseek_source_v1/phase6_metalens_design.json"
    ),
    (
        "GLM-5.1="
        "outputs/phase6_metalens_glm51_source_v1/cst_results_nonfast_substrate2_execute/phase5_hero_metalens_field_export_expected="
        "outputs/phase6_metalens_glm51_source_v1/phase6_metalens_design.json"
    ),
]


# =============================================================================
# 3. 复用底层读取函数和 panel 绘制函数
# =============================================================================

# 这里导入的是已经写好的底层函数。
# 当前根目录文件负责“手动运行 + 统一总图布局”；
# src 里的文件继续保留，方便以后命令行批量绘图时复用同一套函数。
from my_llm_framework.scripts.plot_phase6_device_field_comparison import (
    _load_runs,
    _safe_filename,
    _write_readme,
    _write_source_data,
    draw_field_magnitude_metrics,
    draw_field_map,
    draw_focal_plane_profile,
    draw_location_metrics,
    draw_on_axis_profile,
    draw_phase_error_profile,
    draw_phase_profile,
    draw_selected_amplitude,
    draw_strength_metrics,
    save_figure,
    setup_basic_matplotlib,
)


def draw_composite() -> plt.Figure:
    """绘制一张高信息密度总图。

    这个 def 是你后面最常改的地方：
    - figsize 控制整张图大小；
    - add_gridspec 的 3, 12 表示 3 行 12 列；
    - wspace / hspace 控制子图横向、纵向间距；
    - height_ratios 控制三行高度比例；
    - 每个 fig.add_subplot(gs[行, 列]) 决定 panel 放在哪里。
    """

    loaded, metrics, common_max = _load_runs(RUNS)
    focal_length = float(metrics["design_focal_length_um"].iloc[0])

    # 保存这张图用到的 source data，方便论文 source-data 或后期复查。
    _write_source_data(OUTPUT_DIR, loaded, metrics)

    # -------------------- 总图尺寸和布局 --------------------
    # 字体整体放大后，高度不宜压得太狠。
    # 原版 figsize=(22.0, 14.2), hspace=0.52。
    # 这里把高度降到 13.2，同时把 hspace 降到 0.32，
    # 可以明显减小三排之间的空隙。
    fig = plt.figure(figsize=(22.0, 13.6))

    gs = fig.add_gridspec(
        3,
        12,
        left=0.050,
        right=0.940,
        bottom=0.065,
        top=0.955,
        wspace=0.72,
        hspace=0.40,
        height_ratios=[0.98, 1.08, 0.96],
    )

    # -------------------- 第一行：a/b/c 三个设计侧图 --------------------
    # a：相位剖面；b：相位误差；c：选中单元幅度。
    ax_phase = fig.add_subplot(gs[0, 0:4])
    ax_phase_error = fig.add_subplot(gs[0, 4:8])
    ax_amplitude = fig.add_subplot(gs[0, 8:12])

    draw_phase_profile(ax_phase, loaded, panel_label="a")
    draw_phase_error_profile(ax_phase_error, loaded, panel_label="b")
    draw_selected_amplitude(ax_amplitude, loaded, panel_label="c")

    for ax in [ax_phase, ax_phase_error, ax_amplitude]:
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(FIRST_ROW_LEGEND_FONTSIZE)
            if legend.get_title() is not None:
                legend.get_title().set_fontsize(FIRST_ROW_LEGEND_FONTSIZE)

    # -------------------- 第二行：d/e/f 三种方法的场图 --------------------
    # d/e/f：Manual、DeepSeek、GLM-5.1 的 fast-mode CST field map。
    ax_maps = [
        fig.add_subplot(gs[1, 0:4]),
        fig.add_subplot(gs[1, 4:8]),
        fig.add_subplot(gs[1, 8:12]),
    ]

    last_image = None
    for idx, ax_map in enumerate(ax_maps):
        if idx < len(loaded):
            last_image = draw_field_map(
                ax_map,
                loaded[idx],
                common_max=common_max,
                focal_length_um=focal_length,
                panel_label=chr(ord("d") + idx),
            )
        else:
            ax_map.axis("off")

    # 三张场图共用一个竖向 colorbar，放到第二排最右侧外面。
    # hspace 变小后，colorbar 略微上移并缩短一点，避免和第三排视觉上靠太近。
    if last_image is not None:
        cbar_ax = fig.add_axes([0.955, 0.405, 0.010, 0.245])
        cbar = fig.colorbar(last_image, cax=cbar_ax, orientation="vertical")
        cbar.set_label("Common-normalized |E|", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

    # -------------------- 第三行：g/h/i/j 四个诊断图 --------------------
    # g：旧图 f 的场强指标；
    # h：焦点位置误差；
    # i：on-axis field profile；
    # j：focal-plane field profile。
    ax_field_magnitude = fig.add_subplot(gs[2, 0:3])
    ax_location = fig.add_subplot(gs[2, 3:6])
    ax_on_axis = fig.add_subplot(gs[2, 6:9])
    ax_focal_plane = fig.add_subplot(gs[2, 9:12])

    draw_field_magnitude_metrics(ax_field_magnitude, metrics, panel_label="g")
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

    # 关键修改：所有 panel 绘制完成之后，统一把文字放大 +2。
    increase_figure_fontsizes(fig, delta=FONT_SIZE_DELTA)
    refine_title_and_legend_fonts(
        fig,
        title_extra_delta=TITLE_EXTRA_DELTA,
        legend_fontsize=LEGEND_FONTSIZE,
    )

    # 第一排 legend 单独略大一点，但仍比之前小。
    for ax in [ax_phase, ax_phase_error, ax_amplitude]:
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(FIRST_ROW_LEGEND_FONTSIZE)
            if legend.get_title() is not None:
                legend.get_title().set_fontsize(FIRST_ROW_LEGEND_FONTSIZE)

    _write_readme(OUTPUT_DIR, loaded)
    return fig


def save_standalone_panels() -> None:
    """额外保存每一个 panel 的单图版本。

    总图是主要结果；standalone panel 只是为了后期 PPT、AI、PowerPoint
    手动排版时方便取用。
    """

    loaded, metrics, common_max = _load_runs(RUNS)
    focal_length = float(metrics["design_focal_length_um"].iloc[0])
    formats = get_save_formats()

    standalone_specs: list[tuple[str, plt.Figure]] = []

    fig_phase, ax_phase = plt.subplots(figsize=(5.8, 4.5))
    draw_phase_profile(ax_phase, loaded, panel_label=None)
    standalone_specs.append(("panel_a_phase_profile", fig_phase))

    fig_error, ax_error = plt.subplots(figsize=(5.8, 4.5))
    draw_phase_error_profile(ax_error, loaded, panel_label=None)
    standalone_specs.append(("panel_b_phase_error", fig_error))

    fig_amp, ax_amp = plt.subplots(figsize=(5.8, 4.5))
    draw_selected_amplitude(ax_amp, loaded, panel_label=None)
    standalone_specs.append(("panel_c_selected_amplitude", fig_amp))

    for idx, item in enumerate(loaded):
        fig_field, ax_field = plt.subplots(figsize=(5.6, 4.6))
        image = draw_field_map(
            ax_field,
            item,
            common_max=common_max,
            focal_length_um=focal_length,
            panel_label=None,
        )
        cbar = fig_field.colorbar(image, ax=ax_field, fraction=0.045, pad=0.025)
        cbar.set_label("Common-normalized |E|", fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        standalone_specs.append(
            (
                f"panel_{chr(ord('d') + idx)}_{_safe_filename(item['label'])}_field_map",
                fig_field,
            )
        )

    fig_metric, ax_metric = plt.subplots(figsize=(5.8, 4.5))
    draw_field_magnitude_metrics(ax_metric, metrics, panel_label=None)
    standalone_specs.append(("panel_g_field_magnitude_metrics", fig_metric))

    fig_location, ax_location = plt.subplots(figsize=(6.2, 4.8))
    draw_location_metrics(ax_location, metrics, panel_label=None)
    standalone_specs.append(("panel_h_focus_location_error", fig_location))

    fig_on, ax_on = plt.subplots(figsize=(6.4, 4.8))
    draw_on_axis_profile(
        ax_on,
        loaded,
        common_max=common_max,
        focal_length_um=focal_length,
        panel_label=None,
    )
    standalone_specs.append(("panel_i_on_axis_profile", fig_on))

    fig_fp, ax_fp = plt.subplots(figsize=(6.4, 4.8))
    draw_focal_plane_profile(
        ax_fp,
        loaded,
        common_max=common_max,
        focal_length_um=focal_length,
        panel_label=None,
    )
    standalone_specs.append(("panel_j_focal_plane_profile", fig_fp))

    # 这个额外 panel 是 normalized focusing strength，不放进总图，但保留备用。
    fig_strength, ax_strength = plt.subplots(figsize=(6.2, 4.8))
    draw_strength_metrics(ax_strength, metrics, common_max=common_max, panel_label=None)
    standalone_specs.append(("panel_extra_normalized_focusing_strength", fig_strength))

    for stem, panel_fig in standalone_specs:
        # standalone panels 也同步字体 +2，并单独增大标题、减小图例。
        increase_figure_fontsizes(panel_fig, delta=FONT_SIZE_DELTA)
        refine_title_and_legend_fonts(
            panel_fig,
            title_extra_delta=TITLE_EXTRA_DELTA,
            legend_fontsize=LEGEND_FONTSIZE,
        )
        save_figure(panel_fig, OUTPUT_DIR, stem, formats=formats, tight=True)
        plt.close(panel_fig)


def get_save_formats() -> list[str]:
    """根据 SAVE_PNG / SAVE_PDF 返回要保存的格式。"""

    formats: list[str] = []
    if SAVE_PNG:
        formats.append("png")
    if SAVE_PDF:
        formats.append("pdf")
    if not formats:
        raise ValueError("SAVE_PNG 和 SAVE_PDF 不能同时为 False。")
    return formats


def main() -> None:
    """手动运行入口。"""

    # 应用论文绘图字体和 PDF 字体设置。
    setup_basic_matplotlib()

    # 切换到 PROJECT_ROOT，保证 RUNS 里的相对路径从 MY-LLM-Framework 开始解析。
    old_cwd = Path.cwd()
    try:
        import os

        os.chdir(PROJECT_ROOT)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        fig = draw_composite()
        save_figure(
            fig,
            OUTPUT_DIR,
            "phase6_device_field_comparison",
            formats=get_save_formats(),
            tight=False,
        )

        save_standalone_panels()
    finally:
        os.chdir(old_cwd)

    print(f"Saved Phase 6 device comparison figure: {OUTPUT_DIR.resolve()}")

    # 手动绘图脚本默认展示图像，和 plot_phase3/4/5/6 的习惯一致。
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
