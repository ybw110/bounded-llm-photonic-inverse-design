from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from my_llm_framework.config import load_config
from my_llm_framework.data.processed import load_processed_split
from my_llm_framework.evaluation.metrics import is_success, spectrum_metric_bundle
from my_llm_framework.retrieval.domain_aware import RetrievalWeights
from my_llm_framework.retrieval.methods import (
    retrieve_domain_aware,
    retrieve_physics_summary,
    retrieve_raw_spectrum,
)


METHODS = ("raw_spectrum", "physics_summary", "domain_aware")


def load_memory_bank(path: str | Path) -> dict[str, Any]:
    """读取 memory bank，并还原复数频谱。"""
    data = np.load(path, allow_pickle=False)
    return {
        "domains": data["domains"].astype(str),
        "params": data["params"].astype(np.float32),
        "params_raw": data["params_raw"].astype(np.float32),
        "spectra": (data["spectra_real"] + 1j * data["spectra_imag"]).astype(np.complex64),
        "summary_features": data["summary_features"].astype(np.float32),
        "summary_names": data["summary_names"].astype(str).tolist(),
    }


def thresholds_for_domain(cfg: dict, domain: str) -> dict[str, float]:
    """读取某个 domain 的成功判据。"""
    by_domain = cfg.get("success_thresholds_by_domain", {})
    return by_domain.get(domain, cfg.get("success_thresholds", {}))


def retrieve_for_method(
    method: str,
    bank: dict[str, Any],
    query_data: dict[str, Any],
    query_domain: str,
    cfg: dict,
    top_k: int,
    batch_size: int,
) -> np.ndarray:
    """执行一种 retrieval baseline，返回每个 query 的 top-k memory index。"""
    if method == "raw_spectrum":
        return retrieve_raw_spectrum(
            bank_spectra=bank["spectra"],
            query_spectra=query_data["spectra"],
            top_k=top_k,
            batch_size=batch_size,
        )

    if method == "physics_summary":
        return retrieve_physics_summary(
            bank_summary=bank["summary_features"],
            query_summary=query_data["summary_features"],
            top_k=top_k,
            batch_size=batch_size,
        )

    if method == "domain_aware":
        weights_cfg = cfg.get("retrieval", {})
        return retrieve_domain_aware(
            bank_summary=bank["summary_features"],
            bank_domains=bank["domains"],
            query_summary=query_data["summary_features"],
            query_domain=query_domain,
            domain_specs=cfg["domains"],
            top_k=top_k,
            batch_size=batch_size,
            weights=RetrievalWeights(
                summary_weight=float(weights_cfg.get("summary_weight", 0.70)),
                domain_weight=float(weights_cfg.get("domain_weight", 0.20)),
                constraint_weight=float(weights_cfg.get("constraint_weight", 0.10)),
            ),
        )

    raise ValueError(f"未知 retrieval method: {method}")


def evaluate_topk(
    bank: dict[str, Any],
    query_data: dict[str, Any],
    top_indices: np.ndarray,
    budgets: list[int],
    thresholds: dict[str, float],
    target_domain: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """根据 top-k 候选计算 query-level 和 summary-level 指标。"""
    query_rows: list[dict[str, Any]] = []
    max_budget = max(budgets)

    first_success_ranks = []
    first_target_domain_success_ranks = []
    best_mse_values = []
    top1_mse_values = []
    top1_target_domain_values = []

    for query_idx, candidate_indices in enumerate(top_indices):
        target_spectrum = query_data["spectra"][query_idx]
        freq = query_data["frequency"]
        success_flags = []
        target_domain_success_flags = []
        metrics_at_rank = []

        for rank, candidate_idx in enumerate(candidate_indices[:max_budget], start=1):
            candidate_spectrum = bank["spectra"][candidate_idx]
            metrics = spectrum_metric_bundle(candidate_spectrum, target_spectrum, freq)
            success = is_success(metrics, thresholds)
            success_flags.append(success)
            if target_domain is None:
                target_domain_success_flags.append(False)
            else:
                target_domain_success_flags.append(success and str(bank["domains"][candidate_idx]) == target_domain)
            metrics_at_rank.append(metrics)

        first_success = next((rank for rank, flag in enumerate(success_flags, start=1) if flag), None)
        first_target_domain_success = next(
            (rank for rank, flag in enumerate(target_domain_success_flags, start=1) if flag),
            None,
        )
        first_success_ranks.append(first_success)
        first_target_domain_success_ranks.append(first_target_domain_success)

        mse_values = [m["amplitude_mse"] for m in metrics_at_rank]
        best_mse = float(np.min(mse_values)) if mse_values else float("nan")
        top1_mse = float(mse_values[0]) if mse_values else float("nan")
        best_mse_values.append(best_mse)
        top1_mse_values.append(top1_mse)
        top1_is_target_domain = bool(
            target_domain is not None
            and len(candidate_indices) > 0
            and str(bank["domains"][candidate_indices[0]]) == target_domain
        )
        top1_target_domain_values.append(top1_is_target_domain)

        row: dict[str, Any] = {
            "query_index": query_idx,
            "source_index": int(query_data["source_indices"][query_idx]),
            "top1_bank_index": int(candidate_indices[0]) if len(candidate_indices) else -1,
            "top1_bank_domain": str(bank["domains"][candidate_indices[0]]) if len(candidate_indices) else "",
            "top1_is_target_domain": top1_is_target_domain,
            "top1_amplitude_mse": top1_mse,
            "best_amplitude_mse_at_max_budget": best_mse,
            "first_success_rank": first_success if first_success is not None else "",
            "first_success_bank_domain": (
                str(bank["domains"][candidate_indices[first_success - 1]])
                if first_success is not None
                else ""
            ),
            "first_target_domain_success_rank": first_target_domain_success if first_target_domain_success is not None else "",
        }
        for budget in budgets:
            row[f"success_at_{budget}"] = bool(any(success_flags[:budget]))
            if target_domain is not None:
                row[f"target_domain_success_at_{budget}"] = bool(any(target_domain_success_flags[:budget]))
        query_rows.append(row)

    summary: dict[str, Any] = {
        "num_queries": len(query_rows),
        "mean_top1_amplitude_mse": float(np.nanmean(top1_mse_values)) if top1_mse_values else float("nan"),
        "mean_best_amplitude_mse": float(np.nanmean(best_mse_values)) if best_mse_values else float("nan"),
        "median_best_amplitude_mse": float(np.nanmedian(best_mse_values)) if best_mse_values else float("nan"),
        "top1_target_domain_rate": float(np.mean(top1_target_domain_values)) if target_domain is not None else float("nan"),
    }

    success_ranks_numeric = [rank for rank in first_success_ranks if rank is not None]
    summary["mean_budget_to_success"] = float(np.mean(success_ranks_numeric)) if success_ranks_numeric else float("nan")
    target_domain_success_ranks_numeric = [rank for rank in first_target_domain_success_ranks if rank is not None]
    summary["mean_budget_to_target_domain_success"] = (
        float(np.mean(target_domain_success_ranks_numeric)) if target_domain_success_ranks_numeric else float("nan")
    )

    for budget in budgets:
        summary[f"success_at_{budget}"] = float(np.mean([row[f"success_at_{budget}"] for row in query_rows]))
        if target_domain is not None:
            summary[f"target_domain_success_at_{budget}"] = float(
                np.mean([row[f"target_domain_success_at_{budget}"] for row in query_rows])
            )

    return query_rows, summary


def evaluate_retrieval_baselines(
    cfg: dict,
    memory_bank_path: str | Path,
    processed_root: str | Path,
    query_split: str,
    methods: list[str],
    budgets: list[int],
    output_dir: str | Path,
    batch_size: int,
    max_queries_per_domain: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """批量评测 retrieval baselines，并输出表格。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bank = load_memory_bank(memory_bank_path)
    top_k = max(budgets)

    all_query_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for domain_name in cfg["domains"]:
        query_data = load_processed_split(processed_root, domain_name, query_split)
        if max_queries_per_domain is not None:
            for key in ["params", "params_raw", "spectra", "summary_features", "source_indices"]:
                query_data[key] = query_data[key][:max_queries_per_domain]

        thresholds = thresholds_for_domain(cfg, domain_name)

        for method in methods:
            top_indices = retrieve_for_method(
                method=method,
                bank=bank,
                query_data=query_data,
                query_domain=domain_name,
                cfg=cfg,
                top_k=top_k,
                batch_size=batch_size,
            )

            query_rows, summary = evaluate_topk(
                bank=bank,
                query_data=query_data,
                top_indices=top_indices,
                budgets=budgets,
                thresholds=thresholds,
            )

            for row in query_rows:
                row["domain"] = domain_name
                row["method"] = method
                all_query_rows.append(row)

            summary["domain"] = domain_name
            summary["method"] = method
            summary_rows.append(summary)

    query_df = pd.DataFrame(all_query_rows)
    summary_df = pd.DataFrame(summary_rows)

    query_df.to_csv(output_dir / "query_level_results.csv", index=False)
    summary_df.to_csv(output_dir / "baseline_summary.csv", index=False)
    return query_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="实验配置 JSON")
    parser.add_argument("--memory-bank", required=True, help="由 build_memory_bank 生成的 train memory bank")
    parser.add_argument("--processed-root", default=None, help="processed dataset 目录；默认使用配置中的 processed_root")
    parser.add_argument("--query-split", default="test", choices=["train", "val", "test"], help="用哪个 split 作为 query")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS), help="要评测的 retrieval 方法")
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 5, 10, 50], help="Success@B 的预算列表")
    parser.add_argument("--output-dir", required=True, help="评测表格输出目录")
    parser.add_argument("--batch-size", type=int, default=128, help="批量检索时每批 query 数")
    parser.add_argument("--max-queries-per-domain", type=int, default=None, help="开发排错时限制每个 domain 的 query 数；正式实验不要设置")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(args.processed_root) if args.processed_root else Path(cfg["_processed_root_abs"])

    _, summary_df = evaluate_retrieval_baselines(
        cfg=cfg,
        memory_bank_path=args.memory_bank,
        processed_root=processed_root,
        query_split=args.query_split,
        methods=args.methods,
        budgets=sorted(args.budgets),
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_queries_per_domain=args.max_queries_per_domain,
    )

    print(f"Saved retrieval baseline tables: {Path(args.output_dir).resolve()}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
