from __future__ import annotations

import numpy as np


# physics summary 是这篇新工作的关键中间层。
# 它不是为了替代原始谱线，而是把连续复数谱线压缩成
# LLM planner / numeric critic / retrieval 更容易使用的物理证据。
#
# 这些字段后续会变成 memory bank 中的 summary_features。
SUMMARY_NAMES = [
    "amp_min",
    "amp_max",
    "amp_mean",
    "amp_std",
    "center_frequency_thz",
    "suppression_depth_db",
    "bandwidth_thz",
    "q_factor",
    "phase_span",
    "phase_mean",
]


def amplitude(spectrum: np.ndarray) -> np.ndarray:
    """计算复数谱线的振幅。

    输入是 complex spectrum，例如 real + 1j * imag。
    输出是 |spectrum|。
    """
    return np.abs(spectrum)


def phase(spectrum: np.ndarray) -> np.ndarray:
    """计算复数谱线的相位，并做 unwrap。

    `np.angle` 得到的相位在 -pi 到 pi 之间，会有跳变。
    `np.unwrap` 可以把这些跳变展开，使 phase_span 更平滑。
    """
    return np.unwrap(np.angle(spectrum))


def engineering_specs(freq: np.ndarray, amp: np.ndarray) -> dict[str, float]:
    """从振幅谱线中提取工程指标。

    这里沿用旧框架中滤波器设计的思路：

    - center_frequency_thz：振幅最低点对应的频率
    - suppression_depth_db：最低振幅对应的 dB 深度
    - bandwidth_thz：按半高宽近似计算的带宽
    - q_factor：中心频率 / 带宽

    这些指标不仅用于评价，也用于 retrieval 的 physics summary。
    """
    min_idx = int(np.argmin(amp))
    f0 = float(freq[min_idx])

    # 防止 log10(0) 导致 -inf。
    min_amp = float(max(amp[min_idx], 1e-12))
    depth_db = float(20.0 * np.log10(min_amp))

    # half_level 是带阻滤波器的半高位置。
    # 这里假设背景振幅约为 1，最低点为 min_amp。
    half_level = 0.5 * (1.0 + min_amp)

    # 从中心最低点向左找半高交叉点。
    left = None
    for i in range(min_idx, 0, -1):
        if amp[i - 1] > half_level and amp[i] <= half_level:
            left = i
            break

    # 从中心最低点向右找半高交叉点。
    right = None
    for i in range(min_idx, len(amp) - 1):
        if amp[i] <= half_level and amp[i + 1] > half_level:
            right = i
            break

    bandwidth = 0.0
    if left is not None and right is not None:
        # 用线性插值获得比离散频点更细一点的左右交叉频率。
        f_left = float(np.interp(half_level, [amp[left], amp[left - 1]], [freq[left], freq[left - 1]]))
        f_right = float(np.interp(half_level, [amp[right], amp[right + 1]], [freq[right], freq[right + 1]]))
        bandwidth = max(0.0, f_right - f_left)

    # 带宽为 0 时 Q factor 不可用，用 inf 表示。
    q_factor = float(f0 / bandwidth) if bandwidth > 1e-9 else float("inf")
    return {
        "center_frequency_thz": f0,
        "suppression_depth_db": depth_db,
        "bandwidth_thz": float(bandwidth),
        "q_factor": q_factor,
    }


def summarize_spectrum(spectrum: np.ndarray, freq: np.ndarray) -> dict[str, float]:
    """把一条复数谱线转换成一个 physics summary 字典。"""
    amp = amplitude(spectrum)
    ph = phase(spectrum)
    specs = engineering_specs(freq, amp)

    # 先放通用统计量，再合并工程指标。
    summary = {
        "amp_min": float(np.min(amp)),
        "amp_max": float(np.max(amp)),
        "amp_mean": float(np.mean(amp)),
        "amp_std": float(np.std(amp)),
        "phase_span": float(np.max(ph) - np.min(ph)),
        "phase_mean": float(np.mean(ph)),
    }
    summary.update(specs)
    return summary


def summary_to_vector(summary: dict[str, float]) -> np.ndarray:
    """按固定字段顺序把 summary 字典变成数值向量。

    固定顺序很重要，因为 retrieval 计算距离时要求每一列含义一致。
    如果某个指标是 inf 或 nan，就替换成较大的数值，避免距离计算崩掉。
    """
    values = []
    for name in SUMMARY_NAMES:
        value = float(summary.get(name, 0.0))
        if not np.isfinite(value):
            value = 1e6
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def summarize_batch(spectra: np.ndarray, freq: np.ndarray) -> tuple[list[dict[str, float]], np.ndarray]:
    """批量构造 physics summary。

    返回两个对象：

    - summaries：字典列表，便于人阅读和写报告
    - matrix：二维数组，便于 retrieval 做距离计算
    """
    summaries = [summarize_spectrum(s, freq) for s in spectra]
    matrix = np.vstack([summary_to_vector(s) for s in summaries]).astype(np.float32)
    return summaries, matrix

