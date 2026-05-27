from __future__ import annotations

import numpy as np

from my_llm_framework.features.physics_summary import amplitude, engineering_specs


def amplitude_mse(candidate: np.ndarray, target: np.ndarray) -> float:
    """计算候选谱线和目标谱线的振幅 MSE。"""
    return float(np.mean((amplitude(candidate) - amplitude(target)) ** 2))


def amplitude_mae(candidate: np.ndarray, target: np.ndarray) -> float:
    """计算候选谱线和目标谱线的振幅 MAE。"""
    return float(np.mean(np.abs(amplitude(candidate) - amplitude(target))))


def spectrum_metric_bundle(candidate: np.ndarray, target: np.ndarray, freq: np.ndarray) -> dict[str, float]:
    """一次性计算论文中常用的谱线误差指标。

    输入：

    - candidate：候选设计得到的复数谱线
    - target：目标复数谱线
    - freq：频率网格

    输出：

    - amplitude_mse：振幅均方误差
    - amplitude_mae：振幅平均绝对误差
    - center_frequency_error_thz：中心频率误差
    - bandwidth_error_thz：带宽误差
    - suppression_depth_error_db：抑制深度误差
    """
    cand_amp = amplitude(candidate)
    target_amp = amplitude(target)

    cand_specs = engineering_specs(freq, cand_amp)
    target_specs = engineering_specs(freq, target_amp)

    return {
        "amplitude_mse": amplitude_mse(candidate, target),
        "amplitude_mae": amplitude_mae(candidate, target),
        "center_frequency_error_thz": abs(cand_specs["center_frequency_thz"] - target_specs["center_frequency_thz"]),
        "bandwidth_error_thz": abs(cand_specs["bandwidth_thz"] - target_specs["bandwidth_thz"]),
        "suppression_depth_error_db": abs(cand_specs["suppression_depth_db"] - target_specs["suppression_depth_db"]),
    }


def is_success(metrics: dict[str, float], thresholds: dict[str, float]) -> bool:
    """根据阈值判断一次候选设计是否成功。

    thresholds 来自 `configs/experiment_v1.json`。
    目前只判断 metrics 中实际存在的项，方便后续逐步扩展指标。
    """
    for key, limit in thresholds.items():
        # constraint_satisfaction 是更高层的约束满足率，不在这里直接判断。
        if key == "constraint_satisfaction":
            continue
        if key in metrics and metrics[key] > float(limit):
            return False
    return True


def success_at_budget(success_flags: list[bool], budget: int) -> float:
    """计算 Success@B。

    如果前 B 次尝试中任意一次成功，则 Success@B = 1，否则为 0。
    后续批量任务时，对多个任务求平均即可得到论文表格中的 Success@B。
    """
    if not success_flags:
        return 0.0
    return float(any(success_flags[:budget]))

