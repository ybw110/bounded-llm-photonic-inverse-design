from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from my_llm_framework.scripts.evaluate_planner_oracle import (
    _as_bool,
    load_domain_pool,
    parse_baseline_specs,
)


METHOD_SPECS = {
    "raw_spectrum_top1": ("retrieval_top1_success", "retrieval_top1_amplitude_mse", 1),
    "retrieval_pnn_pso": ("pso_final_success", "pso_final_amplitude_mse", 2),
    "retrieval_pnn_cmaes": ("cmaes_final_success", "cmaes_final_amplitude_mse", 3),
    "retrieval_pnn_pso_expand_top_seeds": (
        "expand_pso_final_success",
        "expand_pso_final_amplitude_mse",
        3,
    ),
}

REFINE_ORDERS = [
    "",
    "cmaes",
    "expand",
    "cmaes,expand",
    "expand,cmaes",
]


def _candidate(row: pd.Series, method: str) -> dict[str, Any] | None:
    success_col, mse_col, steps = METHOD_SPECS[method]
    if success_col not in row or mse_col not in row or pd.isna(row[mse_col]):
        return None
    return {
        "method": method,
        "success": _as_bool(row[success_col]),
        "amplitude_mse": float(row[mse_col]),
        "steps_used": int(steps),
    }


def _best_revealed(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [candidate for candidate in candidates if candidate["success"]]
    pool = successful or candidates
    return min(pool, key=lambda item: (item["amplitude_mse"], item["steps_used"]))


def simulate_rule_baseline(row: pd.Series, *, budget_limit: int) -> dict[str, Any]:
    retrieval = _candidate(row, "raw_spectrum_top1")
    pso = _candidate(row, "retrieval_pnn_pso")
    cmaes = _candidate(row, "retrieval_pnn_cmaes")
    if retrieval is None or pso is None or cmaes is None:
        raise ValueError("Rule baseline requires retrieval, PSO, and CMA-ES candidates")

    revealed = [retrieval]
    actions: list[str] = []
    steps_used = 1

    if retrieval["success"]:
        final = retrieval
        actions.append("stop_with_retrieval")
    else:
        revealed.append(pso)
        steps_used = min(2, budget_limit)
        actions.append("run_pso")
        if pso["success"] or steps_used >= budget_limit:
            final = _best_revealed(revealed)
            actions.append("stop_with_best_mse")
        else:
            revealed.append(cmaes)
            steps_used = min(3, budget_limit)
            actions.append("run_cmaes")
            final = _best_revealed(revealed)
            actions.append("stop_with_best_mse")

    return {
        "success": bool(final["success"]),
        "final_method": final["method"],
        "final_amplitude_mse": float(final["amplitude_mse"]),
        "steps_used": int(steps_used),
        "action_sequence": "|".join(actions),
    }


def simulate_mse_selector(row: pd.Series, policy: dict[str, Any], *, budget_limit: int) -> dict[str, Any]:
    retrieval = _candidate(row, "raw_spectrum_top1")
    pso = _candidate(row, "retrieval_pnn_pso")
    if retrieval is None or pso is None:
        raise ValueError("MSE-aware selector requires retrieval and PSO candidates")

    revealed = [retrieval]
    actions: list[str] = []
    steps_used = 1

    retrieval_stop_mse = float(policy["retrieval_stop_mse"])
    post_optimizer_stop_mse = float(policy["post_optimizer_stop_mse"])
    refine_order = str(policy["refine_order"])

    if (
        policy.get("force_stop_target_self", True)
        and str(row["domain"]) == "target_self"
        and retrieval["success"]
    ):
        final = retrieval
        actions.append("stop_with_retrieval")
        return _selector_result(final, steps_used, actions)

    if retrieval["success"] and retrieval["amplitude_mse"] <= retrieval_stop_mse:
        final = retrieval
        actions.append("stop_with_retrieval")
        return _selector_result(final, steps_used, actions)

    revealed.append(pso)
    steps_used = min(2, budget_limit)
    actions.append("run_pso")
    current_best = _best_revealed(revealed)
    if steps_used >= budget_limit or (
        current_best["success"] and current_best["amplitude_mse"] <= post_optimizer_stop_mse
    ):
        actions.append("stop_with_best_mse")
        return _selector_result(current_best, steps_used, actions)

    for short_name in [item for item in refine_order.split(",") if item]:
        if steps_used >= budget_limit:
            break
        method = {
            "cmaes": "retrieval_pnn_cmaes",
            "expand": "retrieval_pnn_pso_expand_top_seeds",
        }[short_name]
        candidate = _candidate(row, method)
        if candidate is None:
            continue
        revealed.append(candidate)
        steps_used = min(steps_used + 1, budget_limit)
        actions.append("expand_top_seeds" if short_name == "expand" else "run_cmaes")
        current_best = _best_revealed(revealed)
        if current_best["success"] and current_best["amplitude_mse"] <= post_optimizer_stop_mse:
            break

    actions.append("stop_with_best_mse")
    return _selector_result(_best_revealed(revealed), steps_used, actions)


def _selector_result(final: dict[str, Any], steps_used: int, actions: list[str]) -> dict[str, Any]:
    return {
        "success": bool(final["success"]),
        "final_method": final["method"],
        "final_amplitude_mse": float(final["amplitude_mse"]),
        "steps_used": int(steps_used),
        "action_sequence": "|".join(actions),
    }


def threshold_grid(values: pd.Series) -> list[float]:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    finite = finite[finite >= 0.0]
    if finite.empty:
        return [0.0, float("inf")]
    quantiles = finite.quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
    rounded = {round(float(value), 12) for value in quantiles}
    return sorted(rounded) + [float("inf")]


def split_label(query_index: int, split_mode: str) -> str:
    if split_mode == "even_odd":
        return "calibration" if int(query_index) % 2 == 0 else "evaluation"
    raise ValueError(f"Unknown split mode: {split_mode}")


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "success_rate": 0.0,
            "mean_final_amplitude_mse": float("nan"),
            "median_final_amplitude_mse": float("nan"),
            "mean_steps_used": float("nan"),
        }
    df = pd.DataFrame(rows)
    return {
        "success_rate": float(df["success"].mean()),
        "mean_final_amplitude_mse": float(df["final_amplitude_mse"].mean()),
        "median_final_amplitude_mse": float(df["final_amplitude_mse"].median()),
        "mean_steps_used": float(df["steps_used"].mean()),
    }


def _evaluate_policy(
    pool: pd.DataFrame,
    policy: dict[str, Any],
    *,
    budget_limit: int,
) -> list[dict[str, Any]]:
    return [
        simulate_mse_selector(row, policy, budget_limit=budget_limit)
        for _, row in pool.iterrows()
    ]


def tune_domain_policy(
    pool: pd.DataFrame,
    *,
    domain: str,
    budget_limit: int,
    success_tolerance: float,
    max_mean_steps: float | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    calibration = pool[pool["split"] == "calibration"].copy()
    if calibration.empty:
        raise ValueError(f"No calibration rows for domain={domain}")

    baseline_rows = [
        simulate_rule_baseline(row, budget_limit=budget_limit)
        for _, row in calibration.iterrows()
    ]
    baseline_metrics = _metrics(baseline_rows)
    retrieval_thresholds = threshold_grid(calibration["retrieval_top1_amplitude_mse"])
    post_thresholds = threshold_grid(
        pd.concat(
            [
                calibration["pso_final_amplitude_mse"],
                calibration["cmaes_final_amplitude_mse"],
                calibration["expand_pso_final_amplitude_mse"]
                if "expand_pso_final_amplitude_mse" in calibration
                else pd.Series(dtype=float),
            ],
            ignore_index=True,
        )
    )
    has_expand = (
        "expand_pso_final_amplitude_mse" in calibration
        and calibration["expand_pso_final_amplitude_mse"].notna().any()
    )
    orders = REFINE_ORDERS if has_expand else ["", "cmaes"]

    trial_rows: list[dict[str, Any]] = []
    for retrieval_stop_mse in retrieval_thresholds:
        for post_optimizer_stop_mse in post_thresholds:
            for refine_order in orders:
                policy = {
                    "domain": domain,
                    "retrieval_stop_mse": retrieval_stop_mse,
                    "post_optimizer_stop_mse": post_optimizer_stop_mse,
                    "refine_order": refine_order,
                    "force_stop_target_self": True,
                }
                rows = _evaluate_policy(calibration, policy, budget_limit=budget_limit)
                metrics = _metrics(rows)
                feasible = metrics["success_rate"] + 1e-12 >= (
                    baseline_metrics["success_rate"] - success_tolerance
                )
                if max_mean_steps is not None:
                    feasible = feasible and metrics["mean_steps_used"] <= max_mean_steps + 1e-12
                trial_rows.append(
                    {
                        **policy,
                        "is_feasible": bool(feasible),
                        "calibration_success_rate": metrics["success_rate"],
                        "calibration_mean_mse": metrics["mean_final_amplitude_mse"],
                        "calibration_median_mse": metrics["median_final_amplitude_mse"],
                        "calibration_mean_steps": metrics["mean_steps_used"],
                        "baseline_calibration_success_rate": baseline_metrics["success_rate"],
                        "baseline_calibration_mean_mse": baseline_metrics["mean_final_amplitude_mse"],
                        "baseline_calibration_mean_steps": baseline_metrics["mean_steps_used"],
                    }
                )

    trials = pd.DataFrame(trial_rows)
    feasible_trials = trials[trials["is_feasible"]].copy()
    if feasible_trials.empty:
        feasible_trials = trials.copy()
    chosen = feasible_trials.sort_values(
        [
            "calibration_mean_mse",
            "calibration_success_rate",
            "calibration_mean_steps",
        ],
        ascending=[True, False, True],
    ).iloc[0].to_dict()
    return chosen, trials


def summarize_selector(query_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (domain, split), group in query_df.groupby(["domain", "split"], sort=False):
        selector_success = group["selector_success"].astype(bool)
        baseline_success = group["baseline_success"].astype(bool)
        selector_mse = group["selector_final_amplitude_mse"].astype(float)
        baseline_mse = group["baseline_final_amplitude_mse"].astype(float)
        selector_steps = group["selector_steps_used"].astype(float)
        baseline_steps = group["baseline_steps_used"].astype(float)
        rows.append(
            {
                "domain": domain,
                "split": split,
                "num_queries": len(group),
                "selector_success_rate": float(selector_success.mean()),
                "baseline_success_rate": float(baseline_success.mean()),
                "success_rate_delta": float(selector_success.mean() - baseline_success.mean()),
                "selector_mean_final_amplitude_mse": float(selector_mse.mean()),
                "baseline_mean_final_amplitude_mse": float(baseline_mse.mean()),
                "mean_mse_improvement": float(baseline_mse.mean() - selector_mse.mean()),
                "selector_lower_mse_fraction": float((selector_mse < baseline_mse).mean()),
                "selector_mean_steps_used": float(selector_steps.mean()),
                "baseline_mean_steps_used": float(baseline_steps.mean()),
                "mean_steps_delta": float(selector_steps.mean() - baseline_steps.mean()),
            }
        )
    return pd.DataFrame(rows)


def compare_external_baselines(query_df: pd.DataFrame, baseline_specs: list[tuple[str, Path]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for baseline_name, path in baseline_specs:
        baseline = pd.read_csv(path)
        merged = query_df.merge(
            baseline,
            on=["domain", "query_index"],
            how="inner",
            suffixes=("_selector", "_external"),
        )
        if merged.empty:
            raise ValueError(f"No overlapping rows with external baseline {baseline_name}: {path}")
        for (domain, split), group in merged.groupby(["domain", "split"], sort=False):
            selector_success = group["selector_success"].astype(bool)
            external_success = group["planner_success"].astype(bool)
            selector_mse = group["selector_final_amplitude_mse"].astype(float)
            external_mse = group["planner_final_amplitude_mse"].astype(float)
            selector_steps = group["selector_steps_used"].astype(float)
            external_steps = group["planner_steps_used"].astype(float)
            rows.append(
                {
                    "baseline": baseline_name,
                    "domain": domain,
                    "split": split,
                    "num_queries": len(group),
                    "selector_success_rate": float(selector_success.mean()),
                    "baseline_success_rate": float(external_success.mean()),
                    "success_rate_delta": float(selector_success.mean() - external_success.mean()),
                    "selector_mean_mse": float(selector_mse.mean()),
                    "baseline_mean_mse": float(external_mse.mean()),
                    "mean_mse_improvement": float(external_mse.mean() - selector_mse.mean()),
                    "selector_lower_mse_fraction": float((selector_mse < external_mse).mean()),
                    "selector_mean_steps": float(selector_steps.mean()),
                    "baseline_mean_steps": float(external_steps.mean()),
                    "mean_steps_delta": float(selector_steps.mean() - external_steps.mean()),
                }
            )
    return pd.DataFrame(rows)


def evaluate_mse_aware_selector(
    *,
    domain_specs: list[tuple[str, Path, Path, Path | None]],
    retrieval_query_path: str | Path,
    output_dir: str | Path,
    max_queries: int | None,
    split_mode: str,
    budget_limit: int,
    success_tolerance: float,
    max_mean_steps: float | None,
    baseline_specs: list[tuple[str, Path]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_query_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    trial_frames: list[pd.DataFrame] = []

    for domain, pso_path, cmaes_path, expand_path in domain_specs:
        pool = load_domain_pool(
            domain=domain,
            retrieval_query_path=retrieval_query_path,
            pso_query_path=pso_path,
            cmaes_query_path=cmaes_path,
            expand_pso_query_path=expand_path,
            max_queries=max_queries,
        )
        pool["split"] = pool["query_index"].apply(lambda value: split_label(int(value), split_mode))
        chosen_policy, trials = tune_domain_policy(
            pool,
            domain=domain,
            budget_limit=budget_limit,
            success_tolerance=success_tolerance,
            max_mean_steps=max_mean_steps,
        )
        trials["domain"] = domain
        trial_frames.append(trials)
        policy_rows.append(chosen_policy)

        for _, row in pool.iterrows():
            selector = simulate_mse_selector(row, chosen_policy, budget_limit=budget_limit)
            baseline = simulate_rule_baseline(row, budget_limit=budget_limit)
            all_query_rows.append(
                {
                    "domain": domain,
                    "query_index": int(row["query_index"]),
                    "source_index": int(row["source_index"]) if "source_index" in row and pd.notna(row["source_index"]) else None,
                    "split": row["split"],
                    "selector_policy_id": (
                        f"{domain}:retrieval<={chosen_policy['retrieval_stop_mse']}:"
                        f"post<={chosen_policy['post_optimizer_stop_mse']}:"
                        f"order={chosen_policy['refine_order']}"
                    ),
                    "selector_success": selector["success"],
                    "selector_final_method": selector["final_method"],
                    "selector_final_amplitude_mse": selector["final_amplitude_mse"],
                    "selector_steps_used": selector["steps_used"],
                    "selector_action_sequence": selector["action_sequence"],
                    "baseline_success": baseline["success"],
                    "baseline_final_method": baseline["final_method"],
                    "baseline_final_amplitude_mse": baseline["final_amplitude_mse"],
                    "baseline_steps_used": baseline["steps_used"],
                    "baseline_action_sequence": baseline["action_sequence"],
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

    query_df = pd.DataFrame(all_query_rows)
    summary_df = summarize_selector(query_df)
    policy_df = pd.DataFrame(policy_rows)
    trial_df = pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()
    gap_df = compare_external_baselines(query_df, baseline_specs) if baseline_specs else pd.DataFrame()

    query_df.to_csv(output_dir / "mse_aware_selector_query_results.csv", index=False)
    summary_df.to_csv(output_dir / "mse_aware_selector_summary.csv", index=False)
    policy_df.to_csv(output_dir / "mse_aware_selector_policy_summary.csv", index=False)
    trial_df.to_csv(output_dir / "mse_aware_selector_trials.csv", index=False)
    if not gap_df.empty:
        gap_df.to_csv(output_dir / "mse_aware_selector_gap_summary.csv", index=False)
    return query_df, summary_df, policy_df, gap_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-query-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--split-mode", choices=["even_odd"], default="even_odd")
    parser.add_argument("--budget-limit", type=int, default=4)
    parser.add_argument("--success-tolerance", type=float, default=0.0)
    parser.add_argument("--max-mean-steps", type=float, default=2.5)
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

    _, summary_df, policy_df, gap_df = evaluate_mse_aware_selector(
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
        split_mode=args.split_mode,
        budget_limit=args.budget_limit,
        success_tolerance=args.success_tolerance,
        max_mean_steps=args.max_mean_steps,
        baseline_specs=parse_baseline_specs(args.baseline),
    )

    print(f"Saved MSE-aware selector analysis: {Path(args.output_dir).resolve()}")
    print(summary_df.to_string(index=False))
    print("\nChosen selector policies:")
    print(
        policy_df[
            [
                "domain",
                "retrieval_stop_mse",
                "post_optimizer_stop_mse",
                "refine_order",
                "calibration_success_rate",
                "calibration_mean_mse",
                "calibration_mean_steps",
                "baseline_calibration_success_rate",
                "baseline_calibration_mean_mse",
                "baseline_calibration_mean_steps",
            ]
        ].to_string(index=False)
    )
    if not gap_df.empty:
        print("\nExternal baseline comparison:")
        print(gap_df.to_string(index=False))


if __name__ == "__main__":
    main()
