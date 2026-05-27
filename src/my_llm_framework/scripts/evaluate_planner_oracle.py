from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from my_llm_framework.planning.offline_evaluator import (
    load_optimizer_view,
    load_raw_retrieval_rows,
)


CANDIDATE_SPECS = [
    ("raw_spectrum_top1", "retrieval_top1_success", "retrieval_top1_amplitude_mse", 1),
    ("retrieval_pnn_pso", "pso_final_success", "pso_final_amplitude_mse", 2),
    ("retrieval_pnn_cmaes", "cmaes_final_success", "cmaes_final_amplitude_mse", 3),
    ("retrieval_pnn_pso_expand_top_seeds", "expand_pso_final_success", "expand_pso_final_amplitude_mse", 3),
]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _candidate_rows(row: pd.Series) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for method, success_key, mse_key, cost in CANDIDATE_SPECS:
        if success_key not in row or mse_key not in row or pd.isna(row[mse_key]):
            continue
        candidates.append(
            {
                "method": method,
                "success": _as_bool(row[success_key]),
                "amplitude_mse": float(row[mse_key]),
                "steps_used": cost,
            }
        )
    return candidates


def choose_oracle_candidate(candidates: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No oracle candidates were provided")

    if policy == "success_then_mse":
        successful = [candidate for candidate in candidates if candidate["success"]]
        pool = successful or candidates
        return min(pool, key=lambda item: (item["amplitude_mse"], item["steps_used"]))

    if policy == "success_then_budget":
        successful = [candidate for candidate in candidates if candidate["success"]]
        pool = successful or candidates
        return min(pool, key=lambda item: (item["steps_used"], item["amplitude_mse"]))

    if policy == "min_mse":
        return min(candidates, key=lambda item: (item["amplitude_mse"], item["steps_used"]))

    raise ValueError(f"Unknown oracle policy: {policy}")


def load_domain_pool(
    *,
    domain: str,
    retrieval_query_path: str | Path,
    pso_query_path: str | Path,
    cmaes_query_path: str | Path,
    expand_pso_query_path: str | Path | None = None,
    max_queries: int | None = None,
) -> pd.DataFrame:
    retrieval_df = load_raw_retrieval_rows(retrieval_query_path, domain)
    if max_queries is not None:
        retrieval_df = retrieval_df.head(max_queries).copy()
    retrieval_df = retrieval_df.rename(
        columns={
            "success_at_1": "retrieval_top1_success",
            "top1_amplitude_mse": "retrieval_top1_amplitude_mse",
        }
    )

    pso_df = load_optimizer_view(pso_query_path, "pso")
    cmaes_df = load_optimizer_view(cmaes_query_path, "cmaes")
    merged = retrieval_df.merge(pso_df, on=["domain", "query_index"]).merge(
        cmaes_df,
        on=["domain", "query_index"],
    )
    if expand_pso_query_path is not None:
        expand_df = load_optimizer_view(expand_pso_query_path, "expand_pso")
        merged = merged.merge(expand_df, on=["domain", "query_index"], how="left")
    if len(merged) != len(retrieval_df):
        raise ValueError(f"Merged row count mismatch for {domain}: retrieval={len(retrieval_df)}, merged={len(merged)}")
    return merged


def evaluate_oracle_domain(
    *,
    domain: str,
    retrieval_query_path: str | Path,
    pso_query_path: str | Path,
    cmaes_query_path: str | Path,
    expand_pso_query_path: str | Path | None,
    max_queries: int | None,
    oracle_policy: str,
) -> list[dict[str, Any]]:
    pool = load_domain_pool(
        domain=domain,
        retrieval_query_path=retrieval_query_path,
        pso_query_path=pso_query_path,
        cmaes_query_path=cmaes_query_path,
        expand_pso_query_path=expand_pso_query_path,
        max_queries=max_queries,
    )

    rows: list[dict[str, Any]] = []
    for _, row in pool.iterrows():
        candidates = _candidate_rows(row)
        oracle = choose_oracle_candidate(candidates, oracle_policy)
        methods_by_success = [candidate["method"] for candidate in candidates if candidate["success"]]
        rows.append(
            {
                "domain": domain,
                "query_index": int(row["query_index"]),
                "source_index": int(row["source_index"]) if "source_index" in row and pd.notna(row["source_index"]) else None,
                "oracle_policy": oracle_policy,
                "oracle_success": bool(oracle["success"]),
                "oracle_final_method": oracle["method"],
                "oracle_final_amplitude_mse": float(oracle["amplitude_mse"]),
                "oracle_steps_used": int(oracle["steps_used"]),
                "oracle_successful_methods": "|".join(methods_by_success),
                "retrieval_top1_success": _as_bool(row["retrieval_top1_success"]),
                "retrieval_top1_amplitude_mse": float(row["retrieval_top1_amplitude_mse"]),
                "pso_final_success": _as_bool(row["pso_final_success"]),
                "pso_final_amplitude_mse": float(row["pso_final_amplitude_mse"]),
                "cmaes_final_success": _as_bool(row["cmaes_final_success"]),
                "cmaes_final_amplitude_mse": float(row["cmaes_final_amplitude_mse"]),
                "expand_pso_available": "expand_pso_final_success" in row and pd.notna(row.get("expand_pso_final_success")),
                "expand_pso_final_success": _as_bool(row["expand_pso_final_success"])
                if "expand_pso_final_success" in row and pd.notna(row["expand_pso_final_success"])
                else False,
                "expand_pso_final_amplitude_mse": float(row["expand_pso_final_amplitude_mse"])
                if "expand_pso_final_amplitude_mse" in row and pd.notna(row["expand_pso_final_amplitude_mse"])
                else None,
            }
        )
    return rows


def summarize_oracle(query_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, group in query_df.groupby("domain", sort=False):
        rows.append(
            {
                "domain": domain,
                "num_queries": len(group),
                "oracle_policy": str(group["oracle_policy"].iloc[0]),
                "success_rate_final": float(group["oracle_success"].mean()),
                "mean_final_amplitude_mse": float(group["oracle_final_amplitude_mse"].mean()),
                "median_final_amplitude_mse": float(group["oracle_final_amplitude_mse"].median()),
                "mean_steps_used": float(group["oracle_steps_used"].mean()),
                "retrieval_top1_success_rate": float(group["retrieval_top1_success"].mean()),
                "pso_final_success_rate": float(group["pso_final_success"].mean()),
                "cmaes_final_success_rate": float(group["cmaes_final_success"].mean()),
                "expand_pso_available_rate": float(group["expand_pso_available"].mean()),
                "expand_pso_final_success_rate": float(group["expand_pso_final_success"].mean()),
            }
        )
    return pd.DataFrame(rows)


def parse_baseline_specs(values: list[str] | None) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Baseline must be formatted as name=path, got: {value}")
        name, path = value.split("=", 1)
        specs.append((name, Path(path)))
    return specs


def compare_baselines(oracle_df: pd.DataFrame, baseline_specs: list[tuple[str, Path]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for baseline_name, path in baseline_specs:
        baseline = pd.read_csv(path)
        merged = oracle_df.merge(
            baseline,
            on=["domain", "query_index"],
            how="inner",
            suffixes=("_oracle", "_baseline"),
        )
        if merged.empty:
            raise ValueError(f"No overlapping rows between oracle and baseline {baseline_name}: {path}")
        for domain, group in merged.groupby("domain", sort=False):
            baseline_success = group["planner_success"].astype(bool)
            oracle_success = group["oracle_success"].astype(bool)
            baseline_mse = group["planner_final_amplitude_mse"].astype(float)
            oracle_mse = group["oracle_final_amplitude_mse"].astype(float)
            baseline_steps = group["planner_steps_used"].astype(float)
            oracle_steps = group["oracle_steps_used"].astype(float)
            rows.append(
                {
                    "baseline": baseline_name,
                    "domain": domain,
                    "num_queries": len(group),
                    "baseline_success_rate": float(baseline_success.mean()),
                    "oracle_success_rate": float(oracle_success.mean()),
                    "success_rate_gap": float(oracle_success.mean() - baseline_success.mean()),
                    "recoverable_failed_count": int((oracle_success & ~baseline_success).sum()),
                    "baseline_mean_mse": float(baseline_mse.mean()),
                    "oracle_mean_mse": float(oracle_mse.mean()),
                    "mean_mse_gap": float(baseline_mse.mean() - oracle_mse.mean()),
                    "oracle_lower_mse_fraction": float((oracle_mse < baseline_mse).mean()),
                    "baseline_mean_steps": float(baseline_steps.mean()),
                    "oracle_mean_steps": float(oracle_steps.mean()),
                    "mean_steps_gap": float(baseline_steps.mean() - oracle_steps.mean()),
                }
            )
    return pd.DataFrame(rows)


def evaluate_planner_oracle(
    *,
    domain_specs: list[tuple[str, Path, Path, Path | None]],
    retrieval_query_path: str | Path,
    output_dir: str | Path,
    max_queries: int | None,
    oracle_policy: str,
    baseline_specs: list[tuple[str, Path]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for domain, pso_path, cmaes_path, expand_path in domain_specs:
        rows.extend(
            evaluate_oracle_domain(
                domain=domain,
                retrieval_query_path=retrieval_query_path,
                pso_query_path=pso_path,
                cmaes_query_path=cmaes_path,
                expand_pso_query_path=expand_path,
                max_queries=max_queries,
                oracle_policy=oracle_policy,
            )
        )

    query_df = pd.DataFrame(rows)
    summary_df = summarize_oracle(query_df)
    gap_df = compare_baselines(query_df, baseline_specs) if baseline_specs else pd.DataFrame()
    query_df.to_csv(output_dir / "planner_oracle_query_results.csv", index=False)
    summary_df.to_csv(output_dir / "planner_oracle_summary.csv", index=False)
    if not gap_df.empty:
        gap_df.to_csv(output_dir / "planner_oracle_gap_summary.csv", index=False)
    return query_df, summary_df, gap_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-query-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--oracle-policy",
        choices=["success_then_mse", "success_then_budget", "min_mse"],
        default="success_then_mse",
    )
    parser.add_argument("--baseline", action="append", default=None, help="name=planner_offline_query_results.csv")
    parser.add_argument("--source-pso", required=True)
    parser.add_argument("--source-cmaes", required=True)
    parser.add_argument("--source-expand-pso", default=None)
    parser.add_argument("--target-h-pso", required=True)
    parser.add_argument("--target-h-cmaes", required=True)
    parser.add_argument("--target-h-expand-pso", default=None)
    parser.add_argument("--target-self-pso", required=True)
    parser.add_argument("--target-self-cmaes", required=True)
    parser.add_argument("--target-self-expand-pso", default=None)
    args = parser.parse_args()

    _, summary_df, gap_df = evaluate_planner_oracle(
        domain_specs=[
            ("source", Path(args.source_pso), Path(args.source_cmaes), Path(args.source_expand_pso) if args.source_expand_pso else None),
            ("target_h", Path(args.target_h_pso), Path(args.target_h_cmaes), Path(args.target_h_expand_pso) if args.target_h_expand_pso else None),
            (
                "target_self",
                Path(args.target_self_pso),
                Path(args.target_self_cmaes),
                Path(args.target_self_expand_pso) if args.target_self_expand_pso else None,
            ),
        ],
        retrieval_query_path=args.retrieval_query_results,
        output_dir=args.output_dir,
        max_queries=args.max_queries,
        oracle_policy=args.oracle_policy,
        baseline_specs=parse_baseline_specs(args.baseline),
    )

    print(f"Saved planner oracle analysis: {Path(args.output_dir).resolve()}")
    print(summary_df.to_string(index=False))
    if not gap_df.empty:
        print("\nOracle gap summary:")
        print(gap_df.to_string(index=False))


if __name__ == "__main__":
    main()
