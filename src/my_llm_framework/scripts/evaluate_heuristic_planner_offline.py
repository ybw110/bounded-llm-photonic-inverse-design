from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_raw_retrieval_rows(retrieval_query_path: str | Path, domain: str) -> pd.DataFrame:
    """读取 raw-spectrum retrieval-only 的 query-level 结果。"""
    df = pd.read_csv(retrieval_query_path)
    df = df[(df["domain"] == domain) & (df["method"] == "raw_spectrum")].copy()
    if df.empty:
        raise ValueError(f"没有找到 domain={domain}, method=raw_spectrum 的 retrieval query rows")
    return df.sort_values("query_index").reset_index(drop=True)


def load_optimizer_rows(path: str | Path, method_name: str) -> pd.DataFrame:
    """读取某个 optimizer 的 query-level 结果。"""
    df = pd.read_csv(path).copy()
    df["planner_candidate_method"] = method_name
    return df.sort_values("query_index").reset_index(drop=True)


def optimizer_view(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """只保留 planner 需要的 optimizer 列，并显式添加前缀。"""
    required = ["domain", "query_index", "final_success", "final_amplitude_mse"]
    missing = [key for key in required if key not in df.columns]
    if missing:
        raise ValueError(f"optimizer query results 缺少列: {missing}")
    view = df[required].copy()
    return view.rename(
        columns={
            "final_success": f"{prefix}_final_success",
            "final_amplitude_mse": f"{prefix}_final_amplitude_mse",
        }
    )


def choose_final_candidate(
    retrieval_row: pd.Series,
    pso_row: pd.Series,
    cmaes_row: pd.Series,
) -> dict[str, Any]:
    """按照固定人工规则选择最终候选。

    规则：

    1. 如果 raw-spectrum retrieval top1 成功，直接停止。
    2. 如果 retrieval top1 失败，尝试 PSO final。
    3. 如果 PSO final 仍失败，尝试 CMA-ES final。
    4. 如果仍失败，返回三者中 final amplitude MSE 最低的候选。
    """
    retrieval_success = bool(retrieval_row["success_at_1"])
    retrieval_mse = float(retrieval_row["top1_amplitude_mse"])
    pso_success = bool(pso_row["final_success"])
    pso_mse = float(pso_row["final_amplitude_mse"])
    cmaes_success = bool(cmaes_row["final_success"])
    cmaes_mse = float(cmaes_row["final_amplitude_mse"])

    if retrieval_success:
        return {
            "planner_final_method": "raw_spectrum_top1",
            "planner_stop_reason": "retrieval_top1_success",
            "planner_steps_used": 1,
            "planner_success": True,
            "planner_final_amplitude_mse": retrieval_mse,
        }

    if pso_success:
        return {
            "planner_final_method": "retrieval_pnn_pso",
            "planner_stop_reason": "pso_success",
            "planner_steps_used": 2,
            "planner_success": True,
            "planner_final_amplitude_mse": pso_mse,
        }

    if cmaes_success:
        return {
            "planner_final_method": "retrieval_pnn_cmaes",
            "planner_stop_reason": "cmaes_success",
            "planner_steps_used": 3,
            "planner_success": True,
            "planner_final_amplitude_mse": cmaes_mse,
        }

    candidates = [
        ("raw_spectrum_top1", retrieval_mse),
        ("retrieval_pnn_pso", pso_mse),
        ("retrieval_pnn_cmaes", cmaes_mse),
    ]
    best_method, best_mse = min(candidates, key=lambda item: item[1])
    return {
        "planner_final_method": best_method,
        "planner_stop_reason": "all_failed_choose_lowest_mse",
        "planner_steps_used": 3,
        "planner_success": False,
        "planner_final_amplitude_mse": float(best_mse),
    }


def evaluate_domain(
    domain: str,
    retrieval_query_path: str | Path,
    pso_query_path: str | Path,
    cmaes_query_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """离线评估一个 domain 的 heuristic planner。"""
    retrieval_df = load_raw_retrieval_rows(retrieval_query_path, domain)
    pso_df = optimizer_view(load_optimizer_rows(pso_query_path, "retrieval_pnn_pso"), "pso")
    cmaes_df = optimizer_view(load_optimizer_rows(cmaes_query_path, "retrieval_pnn_cmaes"), "cmaes")

    merged = retrieval_df.merge(
        pso_df,
        on=["domain", "query_index"],
    ).merge(
        cmaes_df,
        on=["domain", "query_index"],
    )
    if len(merged) != len(retrieval_df):
        raise ValueError(f"{domain} 合并后 query 数量不一致: retrieval={len(retrieval_df)}, merged={len(merged)}")

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        # merge 后，CMA-ES 的列名多数带 _cmaes；PSO 的列名多数带 _pso。
        retrieval_row = pd.Series(
            {
                "success_at_1": row["success_at_1"],
                "top1_amplitude_mse": row["top1_amplitude_mse"],
            }
        )
        pso_row = pd.Series(
            {
                "final_success": row["pso_final_success"],
                "final_amplitude_mse": row["pso_final_amplitude_mse"],
            }
        )
        cmaes_row = pd.Series(
            {
                "final_success": row["cmaes_final_success"],
                "final_amplitude_mse": row["cmaes_final_amplitude_mse"],
            }
        )
        decision = choose_final_candidate(retrieval_row, pso_row, cmaes_row)

        rows.append(
            {
                "domain": domain,
                "query_index": int(row["query_index"]),
                "source_index": int(row["source_index"]),
                "retrieval_top1_success": bool(row["success_at_1"]),
                "retrieval_top1_amplitude_mse": float(row["top1_amplitude_mse"]),
                "pso_final_success": bool(row["pso_final_success"]),
                "pso_final_amplitude_mse": float(row["pso_final_amplitude_mse"]),
                "cmaes_final_success": bool(row["cmaes_final_success"]),
                "cmaes_final_amplitude_mse": float(row["cmaes_final_amplitude_mse"]),
                **decision,
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "domain": domain,
        "num_queries": len(df),
        "baseline_family": "heuristic_planner",
        "method": "retrieval_then_pso_then_cmaes",
        "setting": "offline_rule_v1",
        "evaluation_source": "offline_existing_results",
        "success_rate_final": float(df["planner_success"].mean()),
        "mean_final_amplitude_mse": float(df["planner_final_amplitude_mse"].mean()),
        "median_final_amplitude_mse": float(df["planner_final_amplitude_mse"].median()),
        "mean_steps_used": float(df["planner_steps_used"].mean()),
        "retrieval_top1_success_rate": float(df["retrieval_top1_success"].mean()),
        "pso_final_success_rate": float(df["pso_final_success"].mean()),
        "cmaes_final_success_rate": float(df["cmaes_final_success"].mean()),
        "retrieval_stop_count": int((df["planner_stop_reason"] == "retrieval_top1_success").sum()),
        "pso_stop_count": int((df["planner_stop_reason"] == "pso_success").sum()),
        "cmaes_stop_count": int((df["planner_stop_reason"] == "cmaes_success").sum()),
        "failed_count": int((~df["planner_success"]).sum()),
    }
    return rows, summary


def evaluate_heuristic_planner(
    domain_specs: list[tuple[str, Path, Path]],
    retrieval_query_path: str | Path,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """评估多个 domain 的离线 heuristic planner。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summaries = []
    for domain, pso_query_path, cmaes_query_path in domain_specs:
        rows, summary = evaluate_domain(
            domain=domain,
            retrieval_query_path=retrieval_query_path,
            pso_query_path=pso_query_path,
            cmaes_query_path=cmaes_query_path,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    query_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summaries)
    query_df.to_csv(output_dir / "heuristic_planner_query_results.csv", index=False)
    summary_df.to_csv(output_dir / "heuristic_planner_summary.csv", index=False)
    return query_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-query-results", required=True, help="phase01_retrieval_baselines_v1/query_level_results.csv")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--source-pso", required=True, help="Source PSO query results CSV")
    parser.add_argument("--source-cmaes", required=True, help="Source CMA-ES query results CSV")
    parser.add_argument("--target-h-pso", required=True, help="Target-H PSO query results CSV")
    parser.add_argument("--target-h-cmaes", required=True, help="Target-H CMA-ES query results CSV")
    parser.add_argument("--target-self-pso", required=True, help="Target-Self PSO query results CSV")
    parser.add_argument("--target-self-cmaes", required=True, help="Target-Self CMA-ES query results CSV")
    args = parser.parse_args()

    _, summary_df = evaluate_heuristic_planner(
        domain_specs=[
            ("source", Path(args.source_pso), Path(args.source_cmaes)),
            ("target_h", Path(args.target_h_pso), Path(args.target_h_cmaes)),
            ("target_self", Path(args.target_self_pso), Path(args.target_self_cmaes)),
        ],
        retrieval_query_path=args.retrieval_query_results,
        output_dir=args.output_dir,
    )

    print(f"Saved offline heuristic planner results: {Path(args.output_dir).resolve()}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
