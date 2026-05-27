from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class PSOResult:
    """PSO 单次优化结果。"""

    best_position: np.ndarray
    best_value: float
    nfev: int
    success: bool
    message: str


def optimize_pso(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
    swarm_size: int = 24,
    maxiter: int = 40,
    seed_radius: float = 0.12,
    inertia: float = 0.72,
    cognitive: float = 1.45,
    social: float = 1.45,
    rng: np.random.Generator | None = None,
    objective_batch: Callable[[np.ndarray], np.ndarray] | None = None,
) -> PSOResult:
    """围绕 retrieval seed 做局部 PSO。

    这里的 PSO 不是全局大范围随机搜索，而是以 retrieval seed 为中心初始化一个小 swarm，
    在这个 seed 附近微调结构参数。这样它对应的是：

    `retrieval seed -> local optimizer -> refined candidate`

    `objective_batch` 是加速入口。如果 surrogate 支持批量预测，PSO 每一代可以一次性评估
    整个 swarm，而不是逐个粒子调用 PNN。这对 GPU 推理非常重要。
    """
    rng = rng or np.random.default_rng()
    x0 = np.asarray(x0, dtype=np.float32)
    dim = len(x0)

    lower = np.asarray([b[0] for b in bounds], dtype=np.float32)
    upper = np.asarray([b[1] for b in bounds], dtype=np.float32)
    span = upper - lower

    def evaluate_positions(positions_to_eval: np.ndarray) -> np.ndarray:
        """评估一批粒子的 objective 值。"""
        if objective_batch is not None:
            values = np.asarray(objective_batch(positions_to_eval), dtype=np.float64).reshape(-1)
        else:
            values = np.asarray([float(objective(p)) for p in positions_to_eval], dtype=np.float64)
        if len(values) != len(positions_to_eval):
            raise ValueError("objective_batch 返回长度与粒子数量不一致")
        return values

    # 初始化粒子：第一个粒子就是 retrieval seed，其余粒子在 seed 附近扰动。
    positions = x0[None, :] + rng.uniform(-seed_radius, seed_radius, size=(swarm_size, dim)).astype(np.float32) * span
    positions[0] = x0
    positions = np.clip(positions, lower, upper)
    velocities = rng.normal(0.0, 0.05, size=(swarm_size, dim)).astype(np.float32) * span

    values = evaluate_positions(positions)
    nfev = int(swarm_size)

    personal_best_positions = positions.copy()
    personal_best_values = values.copy()
    best_idx = int(np.argmin(values))
    global_best_position = positions[best_idx].copy()
    global_best_value = float(values[best_idx])

    for _ in range(int(maxiter)):
        r1 = rng.random(size=(swarm_size, dim), dtype=np.float32)
        r2 = rng.random(size=(swarm_size, dim), dtype=np.float32)
        velocities = (
            inertia * velocities
            + cognitive * r1 * (personal_best_positions - positions)
            + social * r2 * (global_best_position[None, :] - positions)
        )
        positions = np.clip(positions + velocities, lower, upper)
        values = evaluate_positions(positions)
        nfev += int(swarm_size)

        improved = values < personal_best_values
        personal_best_positions[improved] = positions[improved]
        personal_best_values[improved] = values[improved]

        best_idx = int(np.argmin(personal_best_values))
        if float(personal_best_values[best_idx]) < global_best_value:
            global_best_value = float(personal_best_values[best_idx])
            global_best_position = personal_best_positions[best_idx].copy()

    return PSOResult(
        best_position=global_best_position.astype(np.float32),
        best_value=global_best_value,
        nfev=nfev,
        success=True,
        message="PSO finished",
    )
