from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

# 服务器通常没有图形界面。Agg 后端可以直接把图片保存成 png/pdf，
# 不需要弹出窗口，也不需要远程桌面。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from my_llm_framework.config import load_config
from my_llm_framework.data.processed import load_processed_split


# =============================================================================
# 论文出图风格参数
# =============================================================================
# 下面这些参数是专门留给你后期美化图片用的。
# 如果要改字体、字号、颜色、坐标轴标题、图例大小、点大小，优先改这里。

FIG_DPI = 300

# 用户希望使用 Arial。若当前机器没有 Arial，matplotlib 会自动回退到可用字体。
FONT_FAMILY = "Arial"
TITLE_FONT_SIZE = 14
AXIS_LABEL_FONT_SIZE = 12
TICK_LABEL_FONT_SIZE = 10
LEGEND_FONT_SIZE = 10
ANNOTATION_FONT_SIZE = 9

# 单张图的默认尺寸。单位是 inch，不是像素。
FIGSIZE_BAR = (6.2, 4.0)
FIGSIZE_HIST = (5.8, 4.0)
FIGSIZE_SPECTRUM = (6.2, 4.0)
FIGSIZE_PCA = (5.2, 4.8)

# 颜色统一放在这里，后续可以替换成论文统一配色。
DOMAIN_COLORS = {
    "source": "#4C78A8",
    "target_h": "#F58518",
    "target_self": "#54A24B",
}

# 图例中显示的名字也集中管理，避免每个函数里手写。
DOMAIN_LABELS = {
    "source": "Source",
    "target_h": "Target-H",
    "target_self": "Target-Self",
}

# histogram 参数。density=True 表示画概率密度，而不是绝对数量。
# 这样 Source 样本多、Target-H 样本少时，比较的是分布形状，而不是谁样本更多。
HIST_BINS = 45
HIST_ALPHA = 0.55
HIST_DENSITY = True
HIST_LINE_WIDTH = 1.0

# spectrum 样例图参数。
SPECTRUM_LINE_WIDTH = 1.0
SPECTRUM_ALPHA = 0.78
SPECTRUM_LABEL_PREFIX = "E"

# 谱线 example 默认少画几条，图会更干净。
# 每条线使用不同颜色，并在图例中显示 E01/E02/...，不再在线条末端写编号。
DEFAULT_EXAMPLES_PER_DOMAIN = 5
SPECTRUM_COLORMAP = "tab10"

# PCA 散点图参数。
PCA_POINT_SIZE = 12
PCA_ALPHA = 0.62
PCA_EDGE_WIDTH = 0.0

# Target-Self 在三域 PCA 中经常高度重叠。
# 为了避免 250 个点看起来像 2-3 个点，给它单独设置更大的点和边框。
PCA_DOMAIN_POINT_SIZE = {
    "source": 12,
    "target_h": 12,
    "target_self": 34,
}
PCA_DOMAIN_ALPHA = {
    "source": 0.58,
    "target_h": 0.58,
    "target_self": 0.86,
}
PCA_DOMAIN_EDGE_COLOR = {
    "source": "none",
    "target_h": "none",
    "target_self": "#222222",
}
PCA_DOMAIN_EDGE_WIDTH = {
    "source": 0.0,
    "target_h": 0.0,
    "target_self": 0.35,
}

# 只对指定 domain 的 PCA 散点做“显示用抖动”。
# 注意：这不改变 PCA 坐标表中的原始坐标，只是为了让重叠点在图上可见。
PCA_JITTER_DOMAINS = {"target_self"}
PCA_JITTER_STD = 0.025

# 坐标轴和网格参数。
GRID_ALPHA = 0.25
AXIS_LINE_WIDTH = 1.0

# 默认画哪些 summary 指标。
# 如果你想增加相位相关图，可以在命令行用 --summary-metrics 覆盖。
DEFAULT_SUMMARY_METRICS = [
    "amp_min",
    "amp_mean",
    "center_frequency_thz",
    "bandwidth_thz",
]

# summary 指标在坐标轴上的显示名称。
# 这里的文字不会影响数据，只影响图片标签。
SUMMARY_AXIS_LABELS = {
    "amp_min": "Minimum amplitude",
    "amp_mean": "Mean amplitude",
    "center_frequency_thz": "Center frequency (THz)",
    "bandwidth_thz": "Bandwidth (THz)",
    "amp_max": "Maximum amplitude",
    "amp_std": "Amplitude standard deviation",
    "suppression_depth_db": "Suppression depth (dB)",
    "q_factor": "Q factor",
    "phase_span": "Phase span",
    "phase_mean": "Mean phase",
}

# PCA 坐标轴标题。后期想改成中文或论文格式，改这里即可。
PCA_X_LABEL = "PC1"
PCA_Y_LABEL = "PC2"

# summary histogram 分组：
# - Source 和 Target-H 频率范围都是 30-60 THz，适合放在同一张图比较
# - Target-Self 频率范围是 0.5-1 THz，必须单独画，否则横坐标被 Source/H 淹没
SUMMARY_HIST_GROUPS = {
    "source_h": ["source", "target_h"],
    "target_self": ["target_self"],
}


def configure_plot_style() -> None:
    """设置全局绘图风格。

    这一步只影响当前脚本生成的图。
    字体、字号、坐标轴粗细都集中在这里，便于后期统一调论文风格。
    """
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,
            "xtick.labelsize": TICK_LABEL_FONT_SIZE,
            "ytick.labelsize": TICK_LABEL_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "axes.linewidth": AXIS_LINE_WIDTH,
            "savefig.dpi": FIG_DPI,
            "figure.dpi": FIG_DPI,
        }
    )


def load_memory_bank(path: str | Path) -> dict:
    """读取 memory bank，并把 real/imag 还原成复数频谱。"""
    data = np.load(path, allow_pickle=False)
    return {
        "domains": data["domains"].astype(str),
        "spectra": (data["spectra_real"] + 1j * data["spectra_imag"]).astype(np.complex64),
        "summary_features": data["summary_features"].astype(np.float32),
        "summary_names": data["summary_names"].astype(str).tolist(),
    }


def safe_name(text: str) -> str:
    """把指标名或分组名转换成适合文件名的字符串。"""
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_").lower()


def domain_label(domain: str) -> str:
    """返回画图时使用的 domain 名称。"""
    return DOMAIN_LABELS.get(domain, domain)


def domain_color(domain: str) -> str:
    """返回画图时使用的 domain 颜色。"""
    return DOMAIN_COLORS.get(domain, "#666666")


def parameter_columns(domain_cfg: dict) -> list[str]:
    """返回导出 example 参数表时使用的列名。

    `param_names` 中包含 fixed_1 / fixed_2 这类 padding 维度。
    为了让表格更诚实，只导出 active_param_dim 对应的真实物理参数。
    """
    active_dim = int(domain_cfg.get("active_param_dim", domain_cfg.get("param_dim", 0)))
    return list(domain_cfg.get("param_names", []))[:active_dim]


def apply_axis_style(ax: plt.Axes) -> None:
    """统一坐标轴样式。"""
    ax.grid(alpha=GRID_ALPHA)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINE_WIDTH)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """保存单张图，并关闭 figure 释放内存。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def ordered_domains(domains: np.ndarray) -> list[str]:
    """返回绘制顺序。

    Target-Self 放在最后画，避免被 Source / Target-H 盖住。
    """
    preferred = ["source", "target_h", "target_self"]
    present = set(domains.astype(str))
    ordered = [domain for domain in preferred if domain in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def plot_domain_counts(bank: dict, output_dir: Path) -> None:
    """画 memory bank 中每个 domain 的样本数。"""
    domains, counts = np.unique(bank["domains"], return_counts=True)
    table = pd.DataFrame({"domain": domains, "count": counts})
    table.to_csv(output_dir / "domain_counts.csv", index=False)

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    colors = [domain_color(domain) for domain in domains]
    labels = [domain_label(domain) for domain in domains]
    ax.bar(labels, counts, color=colors)
    ax.set_xlabel("Domain")
    ax.set_ylabel("Number of samples")
    ax.set_title("Memory Bank Domain Counts")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    save_figure(fig, output_dir / "domain_counts.png")


def plot_spectrum_examples(cfg: dict, processed_root: Path, output_dir: Path, examples_per_domain: int, seed: int) -> None:
    """每个 domain 随机画几条 train 频谱，并导出对应结构参数表。

    这些图单独保存，每个 domain 一张，方便检查：

    - 频率范围是否正确
    - 振幅是否在合理范围
    - 不同 domain 的谱线形态是否明显不同

    图中每条线会标成 E01、E02、E03 ...
    同时输出 `spectra_examples_<domain>_parameters.csv`。
    CSV 中记录每条线对应的 source_index、结构参数和 summary 指标。
    这样后续写论文或汇报时，可以说清楚“这条谱线对应哪个结构”。
    """
    rng = np.random.default_rng(seed)

    for domain_name in cfg["domains"]:
        domain_cfg = cfg["domains"][domain_name]
        split = load_processed_split(processed_root, domain_name, "train")
        spectra = split["spectra"]
        freq = split["frequency"]
        count = min(examples_per_domain, len(spectra))
        indices = rng.choice(len(spectra), size=count, replace=False)
        param_names = parameter_columns(domain_cfg)
        active_dim = len(param_names)

        fig, ax = plt.subplots(figsize=FIGSIZE_SPECTRUM)
        rows = []
        colors = plt.get_cmap(SPECTRUM_COLORMAP)(np.linspace(0.0, 1.0, count))

        for line_number, (idx, line_color) in enumerate(zip(indices, colors), start=1):
            label = f"{SPECTRUM_LABEL_PREFIX}{line_number:02d}"
            amplitude = np.abs(spectra[idx])

            ax.plot(
                freq,
                amplitude,
                linewidth=SPECTRUM_LINE_WIDTH,
                alpha=SPECTRUM_ALPHA,
                color=line_color,
                label=label,
            )

            row = {
                "example_id": label,
                "domain": domain_name,
                "split": "train",
                "row_in_split": int(idx),
                "source_index": int(split["source_indices"][idx]),
            }

            # normalized 参数用于和模型输入对应。
            for param_i, param_name in enumerate(param_names):
                row[f"{param_name}_norm"] = float(split["params"][idx, param_i])
                row[f"{param_name}_raw"] = float(split["params_raw"][idx, param_i])

            # summary 指标用于解释这条线的物理特征。
            for summary_i, summary_name in enumerate(split["summary_names"]):
                row[summary_name] = float(split["summary_features"][idx, summary_i])

            rows.append(row)

        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Amplitude Examples: {domain_label(domain_name)}")
        ax.legend(frameon=False, loc="best", ncol=2)
        apply_axis_style(ax)
        save_figure(fig, output_dir / f"spectra_examples_{domain_name}.png")

        pd.DataFrame(rows).to_csv(output_dir / f"spectra_examples_{domain_name}_parameters.csv", index=False)


def plot_summary_histograms(bank: dict, output_dir: Path, metrics: list[str]) -> None:
    """把 physics summary 直方图拆成多张单独图片。

    旧版本把所有 domain 和多个指标画在一张总图里，会有两个问题：

    1. Target-Self 的频率范围是 0.5-1 THz，而 Source/H 是 30-60 THz。
       三者放在同一横坐标下，Target-Self 会被压在最左侧，看不清。
    2. Source 样本数远多于 Target-Self，如果画绝对数量，少数 domain 会被淹没。

    所以这里改成：

    - Source + Target-H 一组
    - Target-Self 单独一组
    - 每个 summary metric 单独一张图片
    - 默认使用 density=True，比较分布形状而不是样本数量
    """
    names = bank["summary_names"]
    features = bank["summary_features"]
    domains = bank["domains"]

    valid_metrics = [metric for metric in metrics if metric in names]
    if not valid_metrics:
        raise ValueError(f"没有可绘制的 summary metric。可用字段: {names}")

    hist_dir = output_dir / "summary_histograms"

    for group_name, group_domains in SUMMARY_HIST_GROUPS.items():
        for metric in valid_metrics:
            col = names.index(metric)
            fig, ax = plt.subplots(figsize=FIGSIZE_HIST)

            plotted = False
            for domain in group_domains:
                mask = domains == domain
                values = features[mask, col]
                values = values[np.isfinite(values)]
                if len(values) == 0:
                    continue

                ax.hist(
                    values,
                    bins=HIST_BINS,
                    density=HIST_DENSITY,
                    alpha=HIST_ALPHA,
                    linewidth=HIST_LINE_WIDTH,
                    color=domain_color(domain),
                    label=domain_label(domain),
                )
                plotted = True

            if not plotted:
                plt.close(fig)
                continue

            ax.set_xlabel(SUMMARY_AXIS_LABELS.get(metric, metric))
            ax.set_ylabel("Density" if HIST_DENSITY else "Count")
            ax.set_title(f"{SUMMARY_AXIS_LABELS.get(metric, metric)}: {group_name}")
            ax.legend(frameon=False)
            apply_axis_style(ax)
            save_figure(fig, hist_dir / f"{group_name}_{safe_name(metric)}.png")


def balanced_domain_indices(domains: np.ndarray, samples_per_domain: int, seed: int) -> np.ndarray:
    """按 domain 均衡抽样。

    原始 memory bank 中 Source/H 样本很多，Target-Self 样本较少。
    如果直接随机抽样，PCA 图几乎全是 Source/H，Target-Self 视觉上会消失。

    这里对每个 domain 最多抽 `samples_per_domain` 个点。
    如果某个 domain 样本不足，就全部使用。
    """
    rng = np.random.default_rng(seed)
    selected = []

    for domain in np.unique(domains):
        domain_indices = np.flatnonzero(domains == domain)
        count = min(samples_per_domain, len(domain_indices))
        selected.append(rng.choice(domain_indices, size=count, replace=False))

    return np.concatenate(selected)


def pca_coordinates(features: np.ndarray) -> np.ndarray:
    """用 numpy SVD 计算二维 PCA 坐标。

    这里不用 sklearn，是为了让服务器环境依赖尽量少。
    """
    finite = np.all(np.isfinite(features), axis=1)
    clean = features[finite]
    normalized = (clean - clean.mean(axis=0, keepdims=True)) / (clean.std(axis=0, keepdims=True) + 1e-8)
    _, _, vt = np.linalg.svd(normalized, full_matrices=False)
    coords = normalized @ vt[:2].T
    return coords, finite


def plot_pca_scatter(
    features: np.ndarray,
    domains: np.ndarray,
    output_path: Path,
    title: str,
    seed: int,
) -> None:
    """保存一张 PCA 散点图。"""
    coords, finite = pca_coordinates(features)
    domains = domains[finite]
    rng = np.random.default_rng(seed)

    plot_coords = coords.copy()
    for domain in PCA_JITTER_DOMAINS:
        mask = domains == domain
        if np.any(mask):
            plot_coords[mask] += rng.normal(0.0, PCA_JITTER_STD, size=plot_coords[mask].shape)

    fig, ax = plt.subplots(figsize=FIGSIZE_PCA)
    for domain in ordered_domains(domains):
        mask = domains == domain
        ax.scatter(
            plot_coords[mask, 0],
            plot_coords[mask, 1],
            s=PCA_DOMAIN_POINT_SIZE.get(domain, PCA_POINT_SIZE),
            alpha=PCA_DOMAIN_ALPHA.get(domain, PCA_ALPHA),
            linewidths=PCA_DOMAIN_EDGE_WIDTH.get(domain, PCA_EDGE_WIDTH),
            edgecolors=PCA_DOMAIN_EDGE_COLOR.get(domain, "none"),
            color=domain_color(domain),
            label=domain_label(domain),
        )

    ax.set_xlabel(PCA_X_LABEL)
    ax.set_ylabel(PCA_Y_LABEL)
    ax.set_title(title)
    ax.legend(frameon=False, markerscale=1.6)
    apply_axis_style(ax)
    save_figure(fig, output_path)


def plot_summary_pca(bank: dict, output_dir: Path, samples_per_domain: int, seed: int) -> None:
    """画均衡抽样后的 PCA 图。

    输出两张单独图片：

    - `summary_pca_balanced_all.png`：Source、Target-H、Target-Self 三个域均衡抽样后一起画
    - `summary_pca_source_h.png`：只画 Source 和 Target-H，避免 Target-Self 频段差异过大影响观察

    这样既能看三域整体结构，也能看 30-60 THz 内 Source/H 的相对关系。
    """
    features = bank["summary_features"]
    domains = bank["domains"]
    pca_dir = output_dir / "summary_pca"

    balanced_idx = balanced_domain_indices(domains, samples_per_domain=samples_per_domain, seed=seed)
    plot_pca_scatter(
        features=features[balanced_idx],
        domains=domains[balanced_idx],
        output_path=pca_dir / "summary_pca_balanced_all.png",
        title=f"Physics-Summary PCA, Balanced {samples_per_domain} per Domain",
        seed=seed,
    )

    source_h_mask = np.isin(domains, ["source", "target_h"])
    source_h_domains = domains[source_h_mask]
    source_h_features = features[source_h_mask]
    source_h_idx = balanced_domain_indices(source_h_domains, samples_per_domain=samples_per_domain, seed=seed)
    plot_pca_scatter(
        features=source_h_features[source_h_idx],
        domains=source_h_domains[source_h_idx],
        output_path=pca_dir / "summary_pca_source_h.png",
        title=f"Physics-Summary PCA, Source vs Target-H",
        seed=seed,
    )

    target_self_mask = domains == "target_self"
    target_self_domains = domains[target_self_mask]
    target_self_features = features[target_self_mask]
    target_self_count = min(samples_per_domain, len(target_self_domains))
    rng = np.random.default_rng(seed)
    target_self_idx = rng.choice(len(target_self_domains), size=target_self_count, replace=False)
    plot_pca_scatter(
        features=target_self_features[target_self_idx],
        domains=target_self_domains[target_self_idx],
        output_path=pca_dir / "summary_pca_target_self_only.png",
        title=f"Physics-Summary PCA, Target-Self Only",
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="实验配置 JSON")
    parser.add_argument("--memory-bank", required=True, help="memory bank npz 文件")
    parser.add_argument("--processed-root", default=None, help="processed dataset 目录；默认使用配置中的 processed_root")
    parser.add_argument("--output-dir", required=True, help="图片输出目录")
    parser.add_argument("--examples-per-domain", type=int, default=DEFAULT_EXAMPLES_PER_DOMAIN, help="每个 domain 画多少条谱线样例")
    parser.add_argument("--pca-samples-per-domain", type=int, default=2000, help="PCA 图中每个 domain 最多抽多少个点")
    parser.add_argument("--summary-metrics", nargs="+", default=DEFAULT_SUMMARY_METRICS, help="要绘制直方图的 physics summary 字段")
    parser.add_argument("--seed", type=int, default=48, help="可视化抽样随机种子")
    args = parser.parse_args()

    configure_plot_style()

    cfg = load_config(args.config)
    processed_root = Path(args.processed_root) if args.processed_root else Path(cfg["_processed_root_abs"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bank = load_memory_bank(args.memory_bank)
    plot_domain_counts(bank, output_dir)
    plot_spectrum_examples(cfg, processed_root, output_dir, args.examples_per_domain, args.seed)
    plot_summary_histograms(bank, output_dir, args.summary_metrics)
    plot_summary_pca(bank, output_dir, args.pca_samples_per_domain, args.seed)

    print(f"Saved memory bank figures: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
