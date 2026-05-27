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
from my_llm_framework.retrieval.methods import retrieve_raw_spectrum
from my_llm_framework.surrogates.knn_forward import KNNForwardSurrogate
from my_llm_framework.surrogates.pnn_forward import PNNForwardSurrogate


QUERY_ARRAY_KEYS = ["params", "params_raw", "spectra", "summary_features", "source_indices"]


def load_memory_bank(path: str | Path) -> dict[str, Any]:
    """读取 memory bank，并把 real/imag 还原成复数频谱。"""
    data = np.load(path, allow_pickle=False)
    return {
        "domains": data["domains"].astype(str),
        "params": data["params"].astype(np.float32),
        "params_raw": data["params_raw"].astype(np.float32),
        "spectra": (data["spectra_real"] + 1j * data["spectra_imag"]).astype(np.complex64),
        "summary_features": data["summary_features"].astype(np.float32),
    }


def thresholds_for_domain(cfg: dict, domain: str) -> dict[str, float]:
    """读取某个 domain 的成功阈值。"""
    return cfg.get("success_thresholds_by_domain", {}).get(domain, cfg.get("success_thresholds", {}))


def domain_bank_view(bank: dict[str, Any], domain: str) -> dict[str, Any]:
    """从完整 memory bank 里取出某个 domain 的子库。"""
    mask = bank["domains"] == domain
    if not np.any(mask):
        raise ValueError(f"memory bank 中没有 domain={domain} 的样本")
    return {
        "domains": bank["domains"][mask],
        "params": bank["params"][mask],
        "params_raw": bank["params_raw"][mask],
        "spectra": bank["spectra"][mask],
        "summary_features": bank["summary_features"][mask],
    }


def slice_query_data(
    query_data: dict[str, Any],
    start_index: int,
    end_index: int | None,
    max_queries: int | None,
) -> tuple[dict[str, Any], int, int]:
    """按 query index 切片，方便把完整 test split 拆成多个小任务运行。

    返回值里的第二、第三项分别是真实的起始和结束位置。结果表中的 `query_index`
    会继续使用全局 index，便于后续合并不同切片。
    """
    total = len(query_data["spectra"])
    start = max(0, int(start_index))
    end = total if end_index is None else min(total, int(end_index))
    if max_queries is not None:
        end = min(end, start + int(max_queries))
    if start >= end:
        raise ValueError(f"query 切片为空：start={start}, end={end}, total={total}")

    sliced = dict(query_data)
    for key in QUERY_ARRAY_KEYS:
        sliced[key] = query_data[key][start:end]
    return sliced, start, end


def save_partial(rows: list[dict[str, Any]], output_path: Path) -> None:
    """中途保存已经完成的 query 结果。

    PSO 实验可能跑很久，中途落盘可以避免 Ctrl+C 或服务器中断后完全丢失进度。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def should_accept_optimization(
    initial_metrics: dict[str, float],
    optimized_metrics: dict[str, float],
    initial_is_success: bool,
    optimized_is_success: bool,
    accept_only_if_improved: bool,
    acceptance_policy: str,
) -> tuple[bool, str]:
    """判断是否接受局部优化结果。"""
    optimized_improvement = initial_metrics["amplitude_mse"] - optimized_metrics["amplitude_mse"]
    mse_gate = True if not accept_only_if_improved else optimized_improvement > 0.0
    if not mse_gate:
        return False, "rejected_not_improved"

    if acceptance_policy == "mse_only":
        return True, "accepted_mse_improved"

    if acceptance_policy == "no_success_regression":
        if initial_is_success and not optimized_is_success:
            return False, "rejected_success_regression"
        return True, "accepted_no_success_regression"

    raise ValueError(f"未知 acceptance_policy: {acceptance_policy}")


def evaluate_domain_retrieval_optimizer(
    cfg: dict,
    bank: dict[str, Any],
    processed_root: str | Path,
    domain: str,
    query_split: str,
    surrogate_type: str,
    pnn_model_root: str | Path,
    top_seeds: int,
    knn_k: int,
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
    batch_size: int,
    max_queries: int | None,
    start_index: int,
    end_index: int | None,
    save_every: int,
    partial_output_path: Path | None,
    device: str | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """评测单个 domain 上的 retrieval + local optimizer。

    当前流程：
    1. 同 domain train memory bank 中做 raw-spectrum retrieval，得到 top seeds。
    2. 从每个 seed 参数出发，用 PSO 或 Powell 在归一化参数空间里局部搜索。
    3. 用 PNN forward surrogate 预测候选参数的频谱。
    4. 只在优化结果的 amplitude MSE 更小时接受，否则保留 retrieval top1。

    这一步是强非 LLM baseline，用来证明后续 LLM planner 的增益不是来自简单优化器。
    """
    domain_cfg = cfg["domains"][domain]
    active_dim = int(domain_cfg["active_param_dim"])
    thresholds = thresholds_for_domain(cfg, domain)

    domain_bank = domain_bank_view(bank, domain)
    query_data_full = load_processed_split(processed_root, domain, query_split)
    query_data, query_offset, query_end = slice_query_data(
        query_data_full,
        start_index=start_index,
        end_index=end_index,
        max_queries=max_queries,
    )

    if surrogate_type == "pnn_forward":
        checkpoint = Path(pnn_model_root) / domain / "model_best.pt"
        surrogate = PNNForwardSurrogate(checkpoint, device=device)
    elif surrogate_type == "knn_forward":
        surrogate = KNNForwardSurrogate(
            params=domain_bank["params"],
            spectra=domain_bank["spectra"],
            active_dim=active_dim,
            k=knn_k,
        )
    else:
        raise ValueError(f"未知 surrogate_type: {surrogate_type}")

    seed_indices = retrieve_raw_spectrum(
        bank_spectra=domain_bank["spectra"],
        query_spectra=query_data["spectra"],
        top_k=top_seeds,
        batch_size=batch_size,
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
    rng = np.random.default_rng(seed)

    for row_in_slice, local_seed_indices in enumerate(seed_indices):
        query_idx = query_offset + row_in_slice
        target_spectrum = query_data["spectra"][row_in_slice]
        frequency = query_data["frequency"]

        first_seed_idx = int(local_seed_indices[0])
        initial_metrics = spectrum_metric_bundle(domain_bank["spectra"][first_seed_idx], target_spectrum, frequency)
        initial_is_success = is_success(initial_metrics, thresholds)

        best_result = None
        for seed_rank, local_seed_idx in enumerate(local_seed_indices, start=1):
            result = optimize_from_seed(
                surrogate=surrogate,
                seed_params=domain_bank["params"][local_seed_idx],
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

        # 保守接受策略：PNN surrogate 不是 CST，局部优化也可能走到代理模型不可靠的位置。
        # 因此默认只有优化后的 amplitude MSE 确实更小，才接受优化结果。
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
            "top_seeds": int(top_seeds),
            "best_seed_rank": int(best_result.seed_rank),
            "initial_top1_amplitude_mse": initial_metrics["amplitude_mse"],
            "optimized_amplitude_mse": best_result.metrics["amplitude_mse"],
            "final_amplitude_mse": final_metrics["amplitude_mse"],
            "optimized_amplitude_mse_improvement": optimized_improvement,
            "final_amplitude_mse_improvement": final_improvement,
            "initial_top1_success": bool(initial_is_success),
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
        "top_seeds": int(top_seeds),
        "knn_k": int(knn_k),
        "maxiter": int(maxiter),
        "surrogate_type": surrogate_type,
        "optimizer": optimizer,
        "optimizer_method": method,
        "swarm_size": int(swarm_size),
        "seed_radius": float(seed_radius),
        "evaluation_source": surrogate_type,
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


def evaluate_retrieval_optimizer(
    cfg: dict,
    memory_bank_path: str | Path,
    processed_root: str | Path,
    domains: list[str],
    query_split: str,
    output_dir: str | Path,
    top_seeds: int,
    knn_k: int,
    maxiter: int,
    surrogate_type: str,
    pnn_model_root: str | Path,
    optimizer: str,
    method: str,
    swarm_size: int,
    seed_radius: float,
    inertia: float,
    cognitive: float,
    social: float,
    accept_only_if_improved: bool,
    acceptance_policy: str,
    batch_size: int,
    max_queries_per_domain: int | None,
    start_index: int,
    end_index: int | None,
    save_every: int,
    device: str | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """批量评测 retrieval + local optimizer。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bank = load_memory_bank(memory_bank_path)

    all_rows = []
    summaries = []
    for domain in domains:
        partial_output_path = output_dir / f"partial_{domain}_query_results.csv"
        rows, summary = evaluate_domain_retrieval_optimizer(
            cfg=cfg,
            bank=bank,
            processed_root=processed_root,
            domain=domain,
            query_split=query_split,
            surrogate_type=surrogate_type,
            pnn_model_root=pnn_model_root,
            top_seeds=top_seeds,
            knn_k=knn_k,
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
            batch_size=batch_size,
            max_queries=max_queries_per_domain,
            start_index=start_index,
            end_index=end_index,
            save_every=save_every,
            partial_output_path=partial_output_path,
            device=device,
            seed=seed,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    query_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summaries)
    query_df.to_csv(output_dir / "retrieval_optimizer_query_results.csv", index=False)
    summary_df.to_csv(output_dir / "retrieval_optimizer_summary.csv", index=False)
    return query_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="实验配置 JSON")
    parser.add_argument("--memory-bank", required=True, help="train memory bank npz")
    parser.add_argument("--processed-root", default=None, help="processed dataset 目录；默认使用配置中的 processed_root")
    parser.add_argument("--domains", nargs="+", default=None, help="要评测的 domain；默认评测全部 domain")
    parser.add_argument("--query-split", default="test", choices=["train", "val", "test"], help="query split")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--top-seeds", type=int, default=None, help="每个 query 使用多少个 retrieval seed")
    parser.add_argument("--surrogate-type", default=None, choices=["pnn_forward", "knn_forward"], help="频谱评估器类型")
    parser.add_argument("--pnn-model-root", default=None, help="PNN 模型根目录，下面应有 {domain}/model_best.pt")
    parser.add_argument("--knn-k", type=int, default=None, help="KNN surrogate 的邻居数量")
    parser.add_argument("--maxiter", type=int, default=None, help="局部优化最大迭代数")
    parser.add_argument("--optimizer", default=None, choices=["pso", "cmaes", "powell"], help="优化器类型")
    parser.add_argument("--method", default=None, help="scipy.optimize.minimize 方法，例如 Powell 或 L-BFGS-B")
    parser.add_argument("--swarm-size", type=int, default=None, help="PSO 粒子数量")
    parser.add_argument("--seed-radius", type=float, default=None, help="PSO 初始粒子围绕 retrieval seed 的扰动半径")
    parser.add_argument("--inertia", type=float, default=None, help="PSO 惯性权重")
    parser.add_argument("--cognitive", type=float, default=None, help="PSO 个体学习系数")
    parser.add_argument("--social", type=float, default=None, help="PSO 群体学习系数")
    parser.add_argument("--accept-worse", action="store_true", help="允许接受比 retrieval seed 更差的优化结果；默认不接受")
    parser.add_argument(
        "--acceptance-policy",
        default=None,
        choices=["no_success_regression", "mse_only"],
        help="优化结果接受策略；no_success_regression 会避免把已成功样本优化成失败",
    )
    parser.add_argument("--device", default=None, help="PNN 推理设备，例如 cuda、cuda:0 或 cpu")
    parser.add_argument("--seed", type=int, default=48, help="PSO 随机种子")
    parser.add_argument("--batch-size", type=int, default=128, help="retrieval 批大小")
    parser.add_argument("--max-queries-per-domain", type=int, default=None, help="调试时限制每个 domain 的 query 数")
    parser.add_argument("--start-index", type=int, default=0, help="从 query split 的第几个样本开始跑，便于分片")
    parser.add_argument("--end-index", type=int, default=None, help="跑到 query split 的第几个样本之前，便于分片")
    parser.add_argument("--save-every", type=int, default=25, help="每完成多少条 query 保存一次 partial CSV；设为 0 可关闭")
    args = parser.parse_args()

    cfg = load_config(args.config)
    opt_cfg = cfg.get("local_optimizer", {})
    processed_root = Path(args.processed_root) if args.processed_root else Path(cfg["_processed_root_abs"])
    pnn_model_root = (
        Path(args.pnn_model_root)
        if args.pnn_model_root
        else resolve_project_path(cfg, opt_cfg.get("pnn_model_root", "outputs/phase2_models/pnn_forward"))
    )

    _, summary_df = evaluate_retrieval_optimizer(
        cfg=cfg,
        memory_bank_path=args.memory_bank,
        processed_root=processed_root,
        domains=args.domains or list(cfg["domains"].keys()),
        query_split=args.query_split,
        output_dir=args.output_dir,
        top_seeds=int(args.top_seeds if args.top_seeds is not None else opt_cfg.get("top_seeds", 1)),
        surrogate_type=str(args.surrogate_type if args.surrogate_type is not None else opt_cfg.get("surrogate_type", "pnn_forward")),
        pnn_model_root=pnn_model_root,
        knn_k=int(args.knn_k if args.knn_k is not None else opt_cfg.get("knn_k", 24)),
        maxiter=int(args.maxiter if args.maxiter is not None else opt_cfg.get("maxiter", 15)),
        optimizer=str(args.optimizer if args.optimizer is not None else opt_cfg.get("optimizer", "pso")),
        method=str(args.method if args.method is not None else opt_cfg.get("method", "Powell")),
        swarm_size=int(args.swarm_size if args.swarm_size is not None else opt_cfg.get("swarm_size", 12)),
        seed_radius=float(args.seed_radius if args.seed_radius is not None else opt_cfg.get("seed_radius", 0.08)),
        inertia=float(args.inertia if args.inertia is not None else opt_cfg.get("inertia", 0.72)),
        cognitive=float(args.cognitive if args.cognitive is not None else opt_cfg.get("cognitive", 1.45)),
        social=float(args.social if args.social is not None else opt_cfg.get("social", 1.45)),
        accept_only_if_improved=not args.accept_worse and bool(opt_cfg.get("accept_only_if_improved", True)),
        acceptance_policy=str(args.acceptance_policy if args.acceptance_policy is not None else opt_cfg.get("acceptance_policy", "no_success_regression")),
        batch_size=args.batch_size,
        max_queries_per_domain=args.max_queries_per_domain,
        start_index=args.start_index,
        end_index=args.end_index,
        save_every=int(args.save_every),
        device=args.device,
        seed=int(args.seed),
    )

    print(f"Saved retrieval + optimizer results: {Path(args.output_dir).resolve()}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
