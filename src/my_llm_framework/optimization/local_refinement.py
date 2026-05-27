from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from my_llm_framework.evaluation.metrics import amplitude_mse, spectrum_metric_bundle
from my_llm_framework.optimization.cma_es import optimize_cma_es
from my_llm_framework.optimization.pso import optimize_pso


@dataclass
class LocalOptimizationResult:
    """单个 seed 局部优化后的结果。"""

    seed_rank: int
    seed_params: np.ndarray
    optimized_params: np.ndarray
    predicted_spectrum: np.ndarray
    objective_value: float
    nfev: int
    success: bool
    message: str
    metrics: dict[str, float]


def optimize_from_seed(
    surrogate: Any,
    seed_params: np.ndarray,
    target_spectrum: np.ndarray,
    frequency: np.ndarray,
    seed_rank: int,
    maxiter: int = 35,
    method: str = "Powell",
    optimizer: str = "powell",
    swarm_size: int = 24,
    seed_radius: float = 0.12,
    inertia: float = 0.72,
    cognitive: float = 1.45,
    social: float = 1.45,
    rng: np.random.Generator | None = None,
) -> LocalOptimizationResult:
    """从一个 retrieval seed 出发做局部优化。

    当前优化目标是 amplitude MSE：

    `objective(x) = MSE(|surrogate(x)|, |target|)`

    也就是说，PSO 每次修改结构参数后，先由 PNN 预测频谱，再比较预测频谱
    和目标频谱的振幅差异。center frequency / bandwidth 仍然会在最终结果里报告，
    但不直接作为优化目标，因为这类指标对很小的谱线扰动更敏感。
    """
    active_dim = surrogate.active_dim
    x0 = np.asarray(seed_params[:active_dim], dtype=np.float32)
    bounds = [(0.0, 1.0)] * active_dim

    target_amplitude = np.abs(target_spectrum).astype(np.float32).reshape(1, -1)

    def objective_batch(x_batch: np.ndarray) -> np.ndarray:
        """批量 objective：一次评估多个候选结构参数。"""
        clipped = np.clip(np.asarray(x_batch, dtype=np.float32), 0.0, 1.0)
        if hasattr(surrogate, "predict_batch"):
            predicted = surrogate.predict_batch(clipped)
        else:
            predicted = np.stack([surrogate.predict_one(x) for x in clipped], axis=0)
        return np.mean((np.abs(predicted).astype(np.float32) - target_amplitude) ** 2, axis=1)

    def objective(x: np.ndarray) -> float:
        """单点 objective，供 scipy optimizer 或不支持批量的优化器使用。"""
        return float(objective_batch(np.asarray(x, dtype=np.float32).reshape(1, -1))[0])

    optimizer_name = optimizer.lower()
    if optimizer_name == "pso":
        result = optimize_pso(
            objective=objective,
            objective_batch=objective_batch,
            x0=x0,
            bounds=bounds,
            swarm_size=swarm_size,
            maxiter=maxiter,
            seed_radius=seed_radius,
            inertia=inertia,
            cognitive=cognitive,
            social=social,
            rng=rng,
        )
        optimized = result.best_position
        objective_value = result.best_value
        nfev = result.nfev
        success = result.success
        message = result.message
    elif optimizer_name in {"cmaes", "cma-es", "cma_es"}:
        result = optimize_cma_es(
            objective=objective,
            objective_batch=objective_batch,
            x0=x0,
            bounds=bounds,
            population_size=swarm_size,
            maxiter=maxiter,
            sigma0=seed_radius,
            rng=rng,
        )
        optimized = result.best_position
        objective_value = result.best_value
        nfev = result.nfev
        success = result.success
        message = result.message
    else:
        result = minimize(
            objective,
            x0=x0,
            method=method,
            bounds=bounds,
            options={"maxiter": int(maxiter), "disp": False},
        )
        optimized = np.clip(result.x.astype(np.float32), 0.0, 1.0)
        objective_value = float(result.fun)
        nfev = int(getattr(result, "nfev", 0))
        success = bool(result.success)
        message = str(result.message)

    predicted = surrogate.predict_one(optimized)
    metrics = spectrum_metric_bundle(predicted, target_spectrum, frequency)

    return LocalOptimizationResult(
        seed_rank=int(seed_rank),
        seed_params=x0,
        optimized_params=optimized,
        predicted_spectrum=predicted,
        objective_value=float(objective_value),
        nfev=int(nfev),
        success=bool(success),
        message=str(message),
        metrics=metrics,
    )
