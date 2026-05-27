from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from my_llm_framework.config import load_config, resolve_project_path
from my_llm_framework.data.processed import load_processed_split
from my_llm_framework.evaluation.metrics import is_success, spectrum_metric_bundle
from my_llm_framework.optimization.local_refinement import optimize_from_seed
from my_llm_framework.scripts.evaluate_retrieval_optimizer import (
    QUERY_ARRAY_KEYS,
    save_partial,
    should_accept_optimization,
    slice_query_data,
    thresholds_for_domain,
)
from my_llm_framework.surrogates.pnn_forward import PNNForwardSurrogate


def make_random_seeds(
    num_queries: int,
    num_seeds: int,
    active_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成完全随机的归一化参数种子。

    返回形状为 `(num_queries, num_seeds, active_dim)`。
    每个参数都在 `[0, 1]` 内均匀采样。
    """
    return rng.random((num_queries, num_seeds, active_dim), dtype=np.float32)


def make_lhs_seeds(
    num_queries: int,
    num_seeds: int,
    active_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成 LHS 种子。

    LHS 是 Latin Hypercube Sampling（拉丁超立方采样）。和完全随机相比，
    它会让每个参数维度上的采样点更均匀地覆盖 `[0, 1]` 区间。

    这里先生成 `num_queries * num_seeds` 个 LHS 点，再按 query 分组。
    """
    total = int(num_queries) * int(num_seeds)
    samples = np.empty((total, active_dim), dtype=np.float32)
    for dim in range(active_dim):
        values = (np.arange(total, dtype=np.float32) + rng.random(total, dtype=np.float32)) / float(total)
        samples[:, dim] = values[rng.permutation(total)]
    return samples.reshape(num_queries, num_seeds, active_dim)


def generate_seed_tensor(
    strategy: str,
    num_queries: int,
    num_seeds: int,
    active_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """根据策略生成 seed 参数张量。"""
    if strategy == "random":
        return make_random_seeds(num_queries, num_seeds, active_dim, rng)
    if strategy == "lhs":
        return make_lhs_seeds(num_queries, num_seeds, active_dim, rng)
    raise ValueError(f"未知 seed_strategy: {strategy}")


def evaluate_seed_strategy_for_domain(
    cfg: dict,
    processed_root: str | Path,
    domain: str,
    query_split: str,
    seed_strategy: str,
    pnn_model_root: str | Path,
    num_seeds: int,
    maxiter: int,
    optimizer: str,
    method: str,
    swarm_size: int,
    seed_radius: float,
    inertia: float,
    cognitive: float,
    social: float,
    accept_only_if_improved: bool,
    acceptance_policy: str,
    max_queries: int | None,
    start_index: int,
    end_index: int | None,
    save_every: int,
    partial_output_path: Path | None,
    device: str | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """评测一个 domain 上的 Random/LHS seed + PNN + PSO baseline。

    这条 baseline 不使用 memory bank 检索，因此可以直接回答：

    `retrieval seed 是否比盲目初始化更好？`
    """
    domain_cfg = cfg["domains"][domain]
    active_dim = int(domain_cfg["active_param_dim"])
    thresholds = thresholds_for_domain(cfg, domain)

    query_data_full = load_processed_split(processed_root, domain, query_split)
    query_data, query_offset, query_end = slice_query_data(
        query_data_full,
        start_index=start_index,
        end_index=end_index,
        max_queries=max_queries,
    )

    rng = np.random.default_rng(seed)
    surrogate = PNNForwardSurrogate(Path(pnn_model_root) / domain / "model_best.pt", device=device)
    seed_tensor = generate_seed_tensor(
        strategy=seed_strategy,
        num_queries=len(query_data["spectra"]),
        num_seeds=num_seeds,
        active_dim=active_dim,
        rng=rng,
    )

    rows: list[dict[str, Any]] = []
    initial_success = []
    optimized_success = []
    final_success = []
    initial_mse = []
    optimized_mse = []
    final_mse = []
    nfev_values = []
    improvement_values = []
    accepted_flags = []
    success_regression_flags = []
    success_gain_flags = []

    for row_in_slice in range(len(query_data["spectra"])):
        query_idx = query_offset + row_in_slice
        target_spectrum = query_data["spectra"][row_in_slice]
        frequency = query_data["frequency"]
        seed_params = seed_tensor[row_in_slice]

        # 先只用 PNN 评估种子本身，作为 Random/LHS + PNN evaluation baseline。
        seed_spectra = surrogate.predict_batch(seed_params)
        seed_metrics = [spectrum_metric_bundle(spec, target_spectrum, frequency) for spec in seed_spectra]
        best_seed_idx = int(np.argmin([m["amplitude_mse"] for m in seed_metrics]))
        initial_metrics = seed_metrics[best_seed_idx]
        initial_is_success = is_success(initial_metrics, thresholds)

        best_result = None
        for seed_rank, params in enumerate(seed_params, start=1):
            result = optimize_from_seed(
                surrogate=surrogate,
                seed_params=params,
                target_spectrum=target_spectrum,
                frequency=frequency,
                seed_rank=seed_rank,
                maxiter=maxiter,
                method=method,
                optimizer=optimizer,
                swarm_size=swarm_size,
                seed_radius=seed_radius,
                inertia=inertia,
                cognitive=cognitive,
                social=social,
                rng=rng,
            )
            if best_result is None or result.metrics["amplitude_mse"] < best_result.metrics["amplitude_mse"]:
                best_result = result

        if best_result is None:
            raise RuntimeError("没有得到任何局部优化结果")

        optimized_is_success = is_success(best_result.metrics, thresholds)
        optimized_improvement = initial_metrics["amplitude_mse"] - best_result.metrics["amplitude_mse"]
        accepted, acceptance_reason = should_accept_optimization(
            initial_metrics=initial_metrics,
            optimized_metrics=best_result.metrics,
            initial_is_success=initial_is_success,
            optimized_is_success=optimized_is_success,
            accept_only_if_improved=accept_only_if_improved,
            acceptance_policy=acceptance_policy,
        )
        final_metrics = best_result.metrics if accepted else initial_metrics
        final_is_success = is_success(final_metrics, thresholds)
        final_improvement = initial_metrics["amplitude_mse"] - final_metrics["amplitude_mse"]
        success_regression = bool(initial_is_success and not final_is_success)
        success_gain = bool((not initial_is_success) and final_is_success)

        initial_success.append(initial_is_success)
        optimized_success.append(optimized_is_success)
        final_success.append(final_is_success)
        initial_mse.append(initial_metrics["amplitude_mse"])
        optimized_mse.append(best_result.metrics["amplitude_mse"])
        final_mse.append(final_metrics["amplitude_mse"])
        nfev_values.append(best_result.nfev)
        improvement_values.append(final_improvement)
        accepted_flags.append(accepted)
        success_regression_flags.append(success_regression)
        success_gain_flags.append(success_gain)

        row = {
            "domain": domain,
            "query_index": int(query_idx),
            "row_in_slice": int(row_in_slice),
            "query_slice_start": int(query_offset),
            "query_slice_end": int(query_end),
            "source_index": int(query_data["source_indices"][row_in_slice]),
            "seed_strategy": seed_strategy,
            "num_seeds": int(num_seeds),
            "best_initial_seed_rank": int(best_seed_idx + 1),
            "best_optimized_seed_rank": int(best_result.seed_rank),
            "initial_best_seed_amplitude_mse": initial_metrics["amplitude_mse"],
            "optimized_amplitude_mse": best_result.metrics["amplitude_mse"],
            "final_amplitude_mse": final_metrics["amplitude_mse"],
            "optimized_amplitude_mse_improvement": optimized_improvement,
            "final_amplitude_mse_improvement": final_improvement,
            "initial_best_seed_success": bool(initial_is_success),
            "optimized_success": bool(optimized_is_success),
            "final_success": bool(final_is_success),
            "accepted_optimization": bool(accepted),
            "acceptance_policy": acceptance_policy,
            "acceptance_reason": acceptance_reason,
            "success_regression": success_regression,
            "success_gain": success_gain,
            "optimizer_nfev": int(best_result.nfev),
            "optimizer_success_flag": bool(best_result.success),
            "optimizer_message": best_result.message,
        }
        for key, value in best_result.metrics.items():
            row[f"optimized_{key}"] = value
        for i, value in enumerate(seed_params[best_seed_idx], start=1):
            row[f"initial_param_{i}_norm"] = float(value)
        for i, value in enumerate(best_result.optimized_params, start=1):
            row[f"optimized_param_{i}_norm"] = float(value)
        rows.append(row)

        if save_every > 0 and partial_output_path is not None and len(rows) % save_every == 0:
            save_partial(rows, partial_output_path)

    if partial_output_path is not None:
        save_partial(rows, partial_output_path)

    summary = {
        "domain": domain,
        "num_queries": len(rows),
        "query_slice_start": int(query_offset),
        "query_slice_end": int(query_end),
        "baseline_family": "seed_plus_optimizer",
        "seed_strategy": seed_strategy,
        "num_seeds": int(num_seeds),
        "top_seeds": int(num_seeds),
        "maxiter": int(maxiter),
        "surrogate_type": "pnn_forward",
        "optimizer": optimizer,
        "optimizer_method": method,
        "swarm_size": int(swarm_size),
        "seed_radius": float(seed_radius),
        "evaluation_source": "pnn_forward",
        "acceptance_policy": acceptance_policy,
        "mean_initial_top1_amplitude_mse": float(np.mean(initial_mse)),
        "mean_optimized_amplitude_mse": float(np.mean(optimized_mse)),
        "mean_final_amplitude_mse": float(np.mean(final_mse)),
        "median_optimized_amplitude_mse": float(np.median(optimized_mse)),
        "median_final_amplitude_mse": float(np.median(final_mse)),
        "mean_amplitude_mse_improvement": float(np.mean(improvement_values)),
        "success_rate_initial_top1": float(np.mean(initial_success)),
        "success_rate_optimized": float(np.mean(optimized_success)),
        "success_rate_final": float(np.mean(final_success)),
        "accepted_optimization_rate": float(np.mean(accepted_flags)),
        "success_gain_count": int(np.sum(success_gain_flags)),
        "success_regression_count": int(np.sum(success_regression_flags)),
        "net_success_gain_count": int(np.sum(success_gain_flags) - np.sum(success_regression_flags)),
        "mean_optimizer_nfev": float(np.mean(nfev_values)),
    }
    return rows, summary


def evaluate_seed_optimizer_baselines(
    cfg: dict,
    processed_root: str | Path,
    domains: list[str],
    query_split: str,
    output_dir: str | Path,
    seed_strategies: list[str],
    pnn_model_root: str | Path,
    num_seeds: int,
    maxiter: int,
    optimizer: str,
    method: str,
    swarm_size: int,
    seed_radius: float,
    inertia: float,
    cognitive: float,
    social: float,
    accept_only_if_improved: bool,
    acceptance_policy: str,
    max_queries_per_domain: int | None,
    start_index: int,
    end_index: int | None,
    save_every: int,
    device: str | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """批量评测 Random/LHS seed baseline。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summaries = []
    for domain in domains:
        for strategy_idx, seed_strategy in enumerate(seed_strategies):
            partial_path = output_dir / f"partial_{domain}_{seed_strategy}_query_results.csv"
            rows, summary = evaluate_seed_strategy_for_domain(
                cfg=cfg,
                processed_root=processed_root,
                domain=domain,
                query_split=query_split,
                seed_strategy=seed_strategy,
                pnn_model_root=pnn_model_root,
                num_seeds=num_seeds,
                maxiter=maxiter,
                optimizer=optimizer,
                method=method,
                swarm_size=swarm_size,
                seed_radius=seed_radius,
                inertia=inertia,
                cognitive=cognitive,
                social=social,
                accept_only_if_improved=accept_only_if_improved,
                acceptance_policy=acceptance_policy,
                max_queries=max_queries_per_domain,
                start_index=start_index,
                end_index=end_index,
                save_every=save_every,
                partial_output_path=partial_path,
                device=device,
                seed=seed + strategy_idx * 1009,
            )
            all_rows.extend(rows)
            summaries.append(summary)

    query_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summaries)
    query_df.to_csv(output_dir / "seed_optimizer_query_results.csv", index=False)
    summary_df.to_csv(output_dir / "seed_optimizer_summary.csv", index=False)
    return query_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="实验配置 JSON")
    parser.add_argument("--processed-root", default=None, help="processed dataset 目录；默认使用配置中的 processed_root")
    parser.add_argument("--domains", nargs="+", default=None, help="要评测的 domain；默认评测全部 domain")
    parser.add_argument("--query-split", default="test", choices=["train", "val", "test"], help="query split")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--seed-strategies", nargs="+", default=["random", "lhs"], choices=["random", "lhs"])
    parser.add_argument("--pnn-model-root", default=None, help="PNN 模型根目录，下面应有 {domain}/model_best.pt")
    parser.add_argument("--num-seeds", type=int, default=3, help="每个 query 使用多少个 Random/LHS seed")
    parser.add_argument("--maxiter", type=int, default=None, help="局部优化最大迭代数")
    parser.add_argument("--optimizer", default=None, choices=["pso", "cmaes", "powell"], help="优化器类型")
    parser.add_argument("--method", default=None, help="scipy.optimize.minimize 方法，例如 Powell 或 L-BFGS-B")
    parser.add_argument("--swarm-size", type=int, default=None, help="PSO 粒子数量")
    parser.add_argument("--seed-radius", type=float, default=None, help="PSO 初始粒子围绕 seed 的扰动半径")
    parser.add_argument("--inertia", type=float, default=None, help="PSO 惯性权重")
    parser.add_argument("--cognitive", type=float, default=None, help="PSO 个体学习系数")
    parser.add_argument("--social", type=float, default=None, help="PSO 群体学习系数")
    parser.add_argument("--accept-worse", action="store_true", help="允许接受比初始 seed 更差的优化结果；默认不接受")
    parser.add_argument("--acceptance-policy", default=None, choices=["no_success_regression", "mse_only"])
    parser.add_argument("--device", default=None, help="PNN 推理设备，例如 cuda、cuda:0 或 cpu")
    parser.add_argument("--seed", type=int, default=48, help="随机种子")
    parser.add_argument("--max-queries-per-domain", type=int, default=None, help="调试时限制每个 domain 的 query 数")
    parser.add_argument("--start-index", type=int, default=0, help="从 query split 的第几个样本开始跑")
    parser.add_argument("--end-index", type=int, default=None, help="跑到 query split 的第几个样本之前")
    parser.add_argument("--save-every", type=int, default=100, help="每完成多少条 query 保存一次 partial CSV；设为 0 可关闭")
    args = parser.parse_args()

    cfg = load_config(args.config)
    opt_cfg = cfg.get("local_optimizer", {})
    processed_root = Path(args.processed_root) if args.processed_root else Path(cfg["_processed_root_abs"])
    pnn_model_root = (
        Path(args.pnn_model_root)
        if args.pnn_model_root
        else resolve_project_path(cfg, opt_cfg.get("pnn_model_root", "outputs/phase2_models/pnn_forward"))
    )

    _, summary_df = evaluate_seed_optimizer_baselines(
        cfg=cfg,
        processed_root=processed_root,
        domains=args.domains or list(cfg["domains"].keys()),
        query_split=args.query_split,
        output_dir=args.output_dir,
        seed_strategies=args.seed_strategies,
        pnn_model_root=pnn_model_root,
        num_seeds=int(args.num_seeds),
        maxiter=int(args.maxiter if args.maxiter is not None else opt_cfg.get("maxiter", 30)),
        optimizer=str(args.optimizer if args.optimizer is not None else opt_cfg.get("optimizer", "pso")),
        method=str(args.method if args.method is not None else opt_cfg.get("method", "Powell")),
        swarm_size=int(args.swarm_size if args.swarm_size is not None else opt_cfg.get("swarm_size", 24)),
        seed_radius=float(args.seed_radius if args.seed_radius is not None else opt_cfg.get("seed_radius", 0.08)),
        inertia=float(args.inertia if args.inertia is not None else opt_cfg.get("inertia", 0.72)),
        cognitive=float(args.cognitive if args.cognitive is not None else opt_cfg.get("cognitive", 1.45)),
        social=float(args.social if args.social is not None else opt_cfg.get("social", 1.45)),
        accept_only_if_improved=not args.accept_worse and bool(opt_cfg.get("accept_only_if_improved", True)),
        acceptance_policy=str(args.acceptance_policy if args.acceptance_policy is not None else opt_cfg.get("acceptance_policy", "no_success_regression")),
        max_queries_per_domain=args.max_queries_per_domain,
        start_index=args.start_index,
        end_index=args.end_index,
        save_every=int(args.save_every),
        device=args.device,
        seed=int(args.seed),
    )

    print(f"Saved Random/LHS seed baseline results: {Path(args.output_dir).resolve()}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
