from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def safe_get(row: pd.Series, key: str, default: Any = None) -> Any:
    """从一行 DataFrame 里安全取值，避免不同 baseline 列名不完全一致时报错。"""
    return row[key] if key in row.index else default


def retrieval_rows(retrieval_summary_path: str | Path) -> list[dict[str, Any]]:
    """把 retrieval-only baseline summary 转成 Phase 2 统一表格行。"""
    path = Path(retrieval_summary_path)
    if not path.exists():
        raise FileNotFoundError(f"retrieval summary 不存在: {path}")

    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "domain": row["domain"],
                "baseline_family": "retrieval_only",
                "method": row["method"],
                "setting": "top-k retrieval",
                "num_queries": int(row["num_queries"]),
                "success_rate_primary": float(row["success_at_1"]),
                "success_rate_initial": float(row["success_at_1"]),
                "success_rate_optimized": None,
                "success_rate_final": float(row["success_at_1"]),
                "success_at_1": float(row["success_at_1"]),
                "success_at_5": float(row["success_at_5"]),
                "success_at_10": float(row["success_at_10"]),
                "success_at_50": float(row["success_at_50"]),
                "mean_initial_mse": float(row["mean_top1_amplitude_mse"]),
                "mean_optimized_mse": None,
                "mean_final_mse": float(row["mean_top1_amplitude_mse"]),
                "median_final_mse": None,
                "mean_mse_improvement": 0.0,
                "accepted_optimization_rate": None,
                "success_gain_count": 0,
                "success_regression_count": 0,
                "net_success_gain_count": 0,
                "top_seeds": None,
                "swarm_size": None,
                "maxiter": None,
                "seed_radius": None,
                "acceptance_policy": None,
                "evaluation_source": "memory_bank_true_spectra",
                "result_path": str(path),
            }
        )
    return rows


def optimizer_rows(optimizer_dirs: list[str | Path]) -> list[dict[str, Any]]:
    """把一个或多个 optimizer 输出目录转成 Phase 2 统一表格行。

    支持两类目录：

    - `retrieval_optimizer_summary.csv`：retrieval seed + optimizer。
    - `seed_optimizer_summary.csv`：Random / LHS seed + optimizer。
    """
    rows: list[dict[str, Any]] = []
    for directory in optimizer_dirs:
        directory = Path(directory)
        retrieval_summary_path = directory / "retrieval_optimizer_summary.csv"
        seed_summary_path = directory / "seed_optimizer_summary.csv"
        inn_summary_path = directory / "inn_inverse_summary.csv"
        heuristic_summary_path = directory / "heuristic_planner_summary.csv"
        if retrieval_summary_path.exists():
            summary_path = retrieval_summary_path
            summary_kind = "optimizer"
        elif seed_summary_path.exists():
            summary_path = seed_summary_path
            summary_kind = "optimizer"
        elif inn_summary_path.exists():
            summary_path = inn_summary_path
            summary_kind = "direct_inverse"
        elif heuristic_summary_path.exists():
            summary_path = heuristic_summary_path
            summary_kind = "heuristic_planner"
        else:
            raise FileNotFoundError(
                f"summary 不存在: {retrieval_summary_path}、{seed_summary_path}、{inn_summary_path} 或 {heuristic_summary_path}"
            )

        df = pd.read_csv(summary_path)
        for _, row in df.iterrows():
            if summary_kind == "direct_inverse":
                rows.append(
                    {
                        "domain": row["domain"],
                        "baseline_family": "direct_inverse",
                        "method": row["method"],
                        "setting": row["setting"],
                        "num_queries": int(row["num_queries"]),
                        "success_rate_primary": float(row["success_rate_final"]),
                        "success_rate_initial": None,
                        "success_rate_optimized": None,
                        "success_rate_final": float(row["success_rate_final"]),
                        "success_at_1": None,
                        "success_at_5": None,
                        "success_at_10": None,
                        "success_at_50": None,
                        "mean_initial_mse": None,
                        "mean_optimized_mse": None,
                        "mean_final_mse": float(row["mean_final_amplitude_mse"]),
                        "median_final_mse": float(row["median_final_amplitude_mse"]),
                        "mean_mse_improvement": None,
                        "accepted_optimization_rate": None,
                        "success_gain_count": None,
                        "success_regression_count": None,
                        "net_success_gain_count": None,
                        "top_seeds": None,
                        "swarm_size": None,
                        "maxiter": None,
                        "seed_radius": None,
                        "acceptance_policy": None,
                        "evaluation_source": row["evaluation_source"],
                        "result_path": str(summary_path),
                    }
                )
                continue

            if summary_kind == "heuristic_planner":
                rows.append(
                    {
                        "domain": row["domain"],
                        "baseline_family": "heuristic_planner",
                        "method": row["method"],
                        "setting": row["setting"],
                        "num_queries": int(row["num_queries"]),
                        "success_rate_primary": float(row["success_rate_final"]),
                        "success_rate_initial": float(row["retrieval_top1_success_rate"]),
                        "success_rate_optimized": None,
                        "success_rate_final": float(row["success_rate_final"]),
                        "success_at_1": None,
                        "success_at_5": None,
                        "success_at_10": None,
                        "success_at_50": None,
                        "mean_initial_mse": None,
                        "mean_optimized_mse": None,
                        "mean_final_mse": float(row["mean_final_amplitude_mse"]),
                        "median_final_mse": float(row["median_final_amplitude_mse"]),
                        "mean_mse_improvement": None,
                        "accepted_optimization_rate": None,
                        "success_gain_count": None,
                        "success_regression_count": None,
                        "net_success_gain_count": None,
                        "top_seeds": None,
                        "swarm_size": None,
                        "maxiter": None,
                        "seed_radius": None,
                        "acceptance_policy": None,
                        "evaluation_source": row["evaluation_source"],
                        "result_path": str(summary_path),
                    }
                )
                continue

            seed_strategy = safe_get(row, "seed_strategy", None)
            if seed_strategy is None or pd.isna(seed_strategy):
                baseline_family = "retrieval_plus_optimizer"
                method = f"{row['surrogate_type']}+{row['optimizer']}"
            else:
                baseline_family = "seed_plus_optimizer"
                method = f"{seed_strategy}_seed+{row['surrogate_type']}+{row['optimizer']}"
            setting = (
                f"top{int(row['top_seeds'])}_swarm{int(row['swarm_size'])}"
                f"_iter{int(row['maxiter'])}_r{float(row['seed_radius']):g}"
            )
            rows.append(
                {
                    "domain": row["domain"],
                    "baseline_family": baseline_family,
                    "method": method,
                    "setting": setting,
                    "num_queries": int(row["num_queries"]),
                    "success_rate_primary": float(row["success_rate_final"]),
                    "success_rate_initial": float(row["success_rate_initial_top1"]),
                    "success_rate_optimized": float(row["success_rate_optimized"]),
                    "success_rate_final": float(row["success_rate_final"]),
                    "success_at_1": None,
                    "success_at_5": None,
                    "success_at_10": None,
                    "success_at_50": None,
                    "mean_initial_mse": float(row["mean_initial_top1_amplitude_mse"]),
                    "mean_optimized_mse": float(row["mean_optimized_amplitude_mse"]),
                    "mean_final_mse": float(row["mean_final_amplitude_mse"]),
                    "median_final_mse": float(row["median_final_amplitude_mse"]),
                    "mean_mse_improvement": float(row["mean_amplitude_mse_improvement"]),
                    "accepted_optimization_rate": float(row["accepted_optimization_rate"]),
                    "success_gain_count": int(safe_get(row, "success_gain_count", 0)),
                    "success_regression_count": int(safe_get(row, "success_regression_count", 0)),
                    "net_success_gain_count": int(safe_get(row, "net_success_gain_count", 0)),
                    "top_seeds": int(row["top_seeds"]),
                    "swarm_size": int(row["swarm_size"]),
                    "maxiter": int(row["maxiter"]),
                    "seed_radius": float(row["seed_radius"]),
                    "acceptance_policy": safe_get(row, "acceptance_policy", None),
                    "evaluation_source": safe_get(row, "evaluation_source", row["surrogate_type"]),
                    "result_path": str(summary_path),
                }
            )
    return rows


def summarize_phase2(
    retrieval_summary_path: str | Path,
    optimizer_dirs: list[str | Path],
    output_path: str | Path,
) -> pd.DataFrame:
    """生成 Phase 2 baseline 汇总表。"""
    rows = retrieval_rows(retrieval_summary_path)
    rows.extend(optimizer_rows(optimizer_dirs))

    df = pd.DataFrame(rows)
    df = df.sort_values(["domain", "baseline_family", "method", "setting"], kind="stable")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-summary", required=True, help="retrieval-only baseline_summary.csv")
    parser.add_argument(
        "--optimizer-dirs",
        nargs="+",
        required=True,
        help="一个或多个 retrieval + optimizer 输出目录",
    )
    parser.add_argument("--output", required=True, help="输出 phase2_baseline_summary.csv")
    args = parser.parse_args()

    df = summarize_phase2(
        retrieval_summary_path=args.retrieval_summary,
        optimizer_dirs=args.optimizer_dirs,
        output_path=args.output,
    )

    print(f"Saved Phase 2 baseline summary: {Path(args.output).resolve()}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
