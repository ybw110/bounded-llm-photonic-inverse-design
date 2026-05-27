from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class CMAESResult:
    """CMA-ES 单次优化结果。"""

    best_position: np.ndarray
    best_value: float
    nfev: int
    success: bool
    message: str


def optimize_cma_es(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
    population_size: int = 24,
    maxiter: int = 30,
    sigma0: float = 0.08,
    rng: np.random.Generator | None = None,
    objective_batch: Callable[[np.ndarray], np.ndarray] | None = None,
) -> CMAESResult:
    """围绕一个初始 seed 做有界 CMA-ES 局部优化。

    CMA-ES 是 Covariance Matrix Adaptation Evolution Strategy
    （协方差矩阵自适应进化策略）。它维护一个多元高斯分布，
    每一代从该分布采样候选点，根据表现好的候选点更新均值和协方差矩阵。

    当前实现面向低维超表面参数空间，使用完整协方差矩阵；候选点超出
    `[0, 1]` 边界时直接 clip 回合法范围。为了和 PSO 公平对照，
    `population_size` 对应 PSO 的 `swarm_size`，`maxiter` 对应 PSO 的迭代轮数。
    """
    rng = rng or np.random.default_rng()
    x0 = np.asarray(x0, dtype=np.float32)
    n_dim = len(x0)

    lower = np.asarray([b[0] for b in bounds], dtype=np.float32)
    upper = np.asarray([b[1] for b in bounds], dtype=np.float32)

    lam = max(4, int(population_size))
    mu = max(1, lam // 2)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=np.float64))
    weights = weights / np.sum(weights)
    mueff = float(1.0 / np.sum(weights * weights))

    # 经典 CMA-ES 参数。当前维度很低，这组默认参数足够稳定。
    cc = float((4.0 + mueff / n_dim) / (n_dim + 4.0 + 2.0 * mueff / n_dim))
    cs = float((mueff + 2.0) / (n_dim + mueff + 5.0))
    c1 = float(2.0 / ((n_dim + 1.3) ** 2 + mueff))
    cmu = float(min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n_dim + 2.0) ** 2 + mueff)))
    damps = float(1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n_dim + 1.0)) - 1.0) + cs)
    chi_n = float(np.sqrt(n_dim) * (1.0 - 1.0 / (4.0 * n_dim) + 1.0 / (21.0 * n_dim * n_dim)))

    mean = np.clip(x0.astype(np.float64), lower, upper)
    sigma = float(max(sigma0, 1e-6))
    cov = np.eye(n_dim, dtype=np.float64)
    pc = np.zeros(n_dim, dtype=np.float64)
    ps = np.zeros(n_dim, dtype=np.float64)
    b_mat = np.eye(n_dim, dtype=np.float64)
    d_vec = np.ones(n_dim, dtype=np.float64)
    inv_sqrt_cov = np.eye(n_dim, dtype=np.float64)

    best_position = mean.astype(np.float32)
    best_value = float(objective(best_position))
    nfev = 1

    def evaluate_positions(positions: np.ndarray) -> np.ndarray:
        if objective_batch is not None:
            values = np.asarray(objective_batch(positions), dtype=np.float64).reshape(-1)
        else:
            values = np.asarray([float(objective(p)) for p in positions], dtype=np.float64)
        if len(values) != len(positions):
            raise ValueError("objective_batch 返回长度与候选点数量不一致")
        return values

    for generation in range(int(maxiter)):
        z = rng.normal(size=(lam, n_dim))
        y = z @ (b_mat * d_vec).T
        candidates = np.clip(mean[None, :] + sigma * y, lower, upper).astype(np.float32)

        # 确保第一代显式评估原始 seed，避免 CMA-ES 初始采样刚好远离好起点。
        if generation == 0:
            candidates[0] = best_position

        values = evaluate_positions(candidates)
        nfev += int(lam)

        order = np.argsort(values)
        if float(values[order[0]]) < best_value:
            best_value = float(values[order[0]])
            best_position = candidates[order[0]].astype(np.float32)

        old_mean = mean.copy()
        selected = candidates[order[:mu]].astype(np.float64)
        mean = np.sum(selected * weights[:, None], axis=0)
        y_w = (mean - old_mean) / sigma

        ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (inv_sqrt_cov @ y_w)
        norm_ps = float(np.linalg.norm(ps))
        hsig_threshold = (1.4 + 2.0 / (n_dim + 1.0)) * chi_n
        hsig = float(norm_ps / np.sqrt(1.0 - (1.0 - cs) ** (2.0 * (generation + 1))) < hsig_threshold)
        pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * y_w

        y_selected = (selected - old_mean[None, :]) / sigma
        rank_mu = np.zeros_like(cov)
        for weight, y_i in zip(weights, y_selected):
            rank_mu += weight * np.outer(y_i, y_i)

        cov = (
            (1.0 - c1 - cmu + (1.0 - hsig) * c1 * cc * (2.0 - cc)) * cov
            + c1 * np.outer(pc, pc)
            + cmu * rank_mu
        )
        cov = 0.5 * (cov + cov.T)

        sigma *= float(np.exp((cs / damps) * (norm_ps / chi_n - 1.0)))
        sigma = float(np.clip(sigma, 1e-5, 0.5))

        # 重新分解协方差矩阵，同时截断极小特征值，避免数值不稳定。
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-12)
        d_vec = np.sqrt(eigvals)
        b_mat = eigvecs
        inv_sqrt_cov = b_mat @ np.diag(1.0 / d_vec) @ b_mat.T

    return CMAESResult(
        best_position=best_position.astype(np.float32),
        best_value=best_value,
        nfev=int(nfev),
        success=True,
        message="CMA-ES finished",
    )
