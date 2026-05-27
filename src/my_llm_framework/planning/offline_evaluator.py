from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from my_llm_framework.critics.numeric_critic import critique_offline_candidate
from my_llm_framework.schemas import (
    CandidateMethod,
    PlannerAction,
    PlannerActionType,
    PlannerCandidateObservation,
    PlannerObservation,
)


class PlannerPolicy(Protocol):
    name: str

    def decide(self, observation: PlannerObservation) -> PlannerAction:
        ...


class OfflineRulePlannerPolicy:
    """Deterministic policy used until an API-backed LLM planner is connected."""

    name = "offline_rule_planner_v1"

    def decide(self, observation: PlannerObservation) -> PlannerAction:
        if observation.retrieval_top1.success:
            return PlannerAction(
                action_type=PlannerActionType.STOP_WITH_RETRIEVAL.value,
                reason="Retrieval top1 already satisfies the numeric critic.",
                candidate_source=CandidateMethod.RAW_SPECTRUM_TOP1.value,
            )

        if observation.remaining_budget <= 0:
            return PlannerAction(
                action_type=PlannerActionType.STOP_WITH_BEST_MSE.value,
                reason="Budget exhausted; choose the best revealed candidate by amplitude MSE.",
            )

        if observation.pso_final is None:
            return PlannerAction(
                action_type=PlannerActionType.RUN_PSO.value,
                reason="Retrieval failed; evaluate the PSO result from the offline pool.",
                budget_to_spend=1,
                candidate_source=CandidateMethod.RAW_SPECTRUM_TOP1.value,
            )

        if observation.pso_final.success:
            return PlannerAction(
                action_type=PlannerActionType.STOP_WITH_BEST_MSE.value,
                reason="PSO candidate satisfies the numeric critic.",
                candidate_source=CandidateMethod.RETRIEVAL_PNN_PSO.value,
            )

        if observation.remaining_budget <= 0:
            return PlannerAction(
                action_type=PlannerActionType.STOP_WITH_BEST_MSE.value,
                reason="Budget exhausted after PSO; choose the best revealed candidate.",
            )

        if observation.cmaes_final is None:
            return PlannerAction(
                action_type=PlannerActionType.RUN_CMAES.value,
                reason="PSO failed or did not improve enough; evaluate CMA-ES from the offline pool.",
                budget_to_spend=1,
                candidate_source=CandidateMethod.RETRIEVAL_PNN_PSO.value,
            )

        return PlannerAction(
            action_type=PlannerActionType.STOP_WITH_BEST_MSE.value,
            reason="All offline candidates have been revealed; choose the best amplitude MSE.",
        )


class ExpandAwareRulePlannerPolicy:
    """Rule planner that uses expand_top_seeds after a no-gain PSO result."""

    name = "expand_aware_rule_planner_v1"

    def decide(self, observation: PlannerObservation) -> PlannerAction:
        if observation.retrieval_top1.success:
            return PlannerAction(
                action_type=PlannerActionType.STOP_WITH_RETRIEVAL.value,
                reason="Retrieval top1 already satisfies the numeric critic.",
                candidate_source=CandidateMethod.RAW_SPECTRUM_TOP1.value,
            )

        if observation.remaining_budget <= 0:
            return PlannerAction(
                action_type=PlannerActionType.STOP_WITH_BEST_MSE.value,
                reason="Budget exhausted; choose the best revealed candidate by amplitude MSE.",
            )

        if observation.pso_final is None:
            return PlannerAction(
                action_type=PlannerActionType.RUN_PSO.value,
                reason="Retrieval failed; evaluate the regular top3 PSO result first.",
                budget_to_spend=1,
                candidate_source=CandidateMethod.RAW_SPECTRUM_TOP1.value,
            )

        if observation.pso_final.success:
            return PlannerAction(
                action_type=PlannerActionType.STOP_WITH_BEST_MSE.value,
                reason="Regular PSO candidate satisfies the numeric critic.",
                candidate_source=CandidateMethod.RETRIEVAL_PNN_PSO.value,
            )

        if observation.expand_top_seeds_final is None and observation.remaining_budget > 0:
            return PlannerAction(
                action_type=PlannerActionType.EXPAND_TOP_SEEDS.value,
                reason="Regular PSO did not recover; reveal the top10 seed expansion result.",
                budget_to_spend=1,
                candidate_source=CandidateMethod.RETRIEVAL_PNN_PSO.value,
                top_k=10,
            )

        if observation.expand_top_seeds_final and observation.expand_top_seeds_final.success:
            return PlannerAction(
                action_type=PlannerActionType.STOP_WITH_BEST_MSE.value,
                reason="Expanded-seed PSO candidate satisfies the numeric critic.",
                candidate_source=CandidateMethod.RETRIEVAL_PNN_PSO_EXPAND_TOP_SEEDS.value,
            )

        if observation.cmaes_final is None and observation.remaining_budget > 0:
            return PlannerAction(
                action_type=PlannerActionType.RUN_CMAES.value,
                reason="Expanded-seed PSO still failed; reveal CMA-ES if budget remains.",
                budget_to_spend=1,
                candidate_source=CandidateMethod.RETRIEVAL_PNN_PSO_EXPAND_TOP_SEEDS.value,
            )

        return PlannerAction(
            action_type=PlannerActionType.STOP_WITH_BEST_MSE.value,
            reason="All useful offline candidates have been revealed; choose the best amplitude MSE.",
        )


def load_thresholds_by_domain(config_path: str | Path) -> dict[str, dict[str, float]]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if "success_thresholds_by_domain" in config:
        return {
            domain: {key: float(value) for key, value in thresholds.items()}
            for domain, thresholds in config["success_thresholds_by_domain"].items()
        }
    shared = {key: float(value) for key, value in config["success_thresholds"].items()}
    return {domain: shared for domain in config["domains"]}


def load_raw_retrieval_rows(retrieval_query_path: str | Path, domain: str) -> pd.DataFrame:
    df = pd.read_csv(retrieval_query_path)
    df = df[(df["domain"] == domain) & (df["method"] == "raw_spectrum")].copy()
    if df.empty:
        raise ValueError(f"No raw_spectrum retrieval rows found for domain={domain}")
    return df.sort_values("query_index").reset_index(drop=True)


def load_optimizer_view(path: str | Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    keys = ["domain", "query_index"]
    optional = [
        "source_index",
        "initial_top1_amplitude_mse",
        "final_success",
        "final_amplitude_mse",
        "accepted_optimization",
        "acceptance_reason",
        "optimizer_nfev",
        "optimized_amplitude_mae",
        "optimized_center_frequency_error_thz",
        "optimized_bandwidth_error_thz",
        "optimized_suppression_depth_error_db",
    ]
    cols = keys + [col for col in optional if col in df.columns]
    view = df[cols].copy()
    rename = {col: f"{prefix}_{col}" for col in cols if col not in keys}
    return view.rename(columns=rename)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _extra_optimizer_metrics(row: pd.Series, prefix: str) -> dict[str, float]:
    accepted_key = f"{prefix}_accepted_optimization"
    if accepted_key in row and not _as_bool(row[accepted_key]):
        return {}

    mapping = {
        f"{prefix}_optimized_amplitude_mae": "amplitude_mae",
        f"{prefix}_optimized_center_frequency_error_thz": "center_frequency_error_thz",
        f"{prefix}_optimized_bandwidth_error_thz": "bandwidth_error_thz",
        f"{prefix}_optimized_suppression_depth_error_db": "suppression_depth_error_db",
    }
    metrics: dict[str, float] = {}
    for source_key, target_key in mapping.items():
        if source_key in row and pd.notna(row[source_key]):
            metrics[target_key] = float(row[source_key])
    return metrics


def make_candidate_observation(
    *,
    method: CandidateMethod,
    success: bool,
    amplitude_mse: float,
    thresholds: dict[str, float],
    extra_metrics: dict[str, Any] | None = None,
    initial_amplitude_mse: float | None = None,
    accepted_optimization: bool | None = None,
) -> PlannerCandidateObservation:
    critic = critique_offline_candidate(
        success=success,
        amplitude_mse=amplitude_mse,
        thresholds=thresholds,
        extra_metrics=extra_metrics,
        initial_amplitude_mse=initial_amplitude_mse,
        accepted_optimization=accepted_optimization,
    )
    return PlannerCandidateObservation(
        method=method.value,
        available=True,
        success=success,
        amplitude_mse=float(amplitude_mse),
        metrics=critic.metrics,
        critic=critic,
        budget_cost=1,
    )


def build_candidates(row: pd.Series, thresholds: dict[str, float]) -> dict[str, PlannerCandidateObservation]:
    retrieval = make_candidate_observation(
        method=CandidateMethod.RAW_SPECTRUM_TOP1,
        success=_as_bool(row["success_at_1"]),
        amplitude_mse=float(row["top1_amplitude_mse"]),
        thresholds=thresholds,
    )
    pso = make_candidate_observation(
        method=CandidateMethod.RETRIEVAL_PNN_PSO,
        success=_as_bool(row["pso_final_success"]),
        amplitude_mse=float(row["pso_final_amplitude_mse"]),
        thresholds=thresholds,
        extra_metrics=_extra_optimizer_metrics(row, "pso"),
        initial_amplitude_mse=float(row["pso_initial_top1_amplitude_mse"])
        if "pso_initial_top1_amplitude_mse" in row
        else float(row["top1_amplitude_mse"]),
        accepted_optimization=_as_bool(row["pso_accepted_optimization"])
        if "pso_accepted_optimization" in row
        else None,
    )
    cmaes = make_candidate_observation(
        method=CandidateMethod.RETRIEVAL_PNN_CMAES,
        success=_as_bool(row["cmaes_final_success"]),
        amplitude_mse=float(row["cmaes_final_amplitude_mse"]),
        thresholds=thresholds,
        extra_metrics=_extra_optimizer_metrics(row, "cmaes"),
        initial_amplitude_mse=float(row["cmaes_initial_top1_amplitude_mse"])
        if "cmaes_initial_top1_amplitude_mse" in row
        else float(row["top1_amplitude_mse"]),
        accepted_optimization=_as_bool(row["cmaes_accepted_optimization"])
        if "cmaes_accepted_optimization" in row
        else None,
    )
    candidates = {
        CandidateMethod.RAW_SPECTRUM_TOP1.value: retrieval,
        CandidateMethod.RETRIEVAL_PNN_PSO.value: pso,
        CandidateMethod.RETRIEVAL_PNN_CMAES.value: cmaes,
    }
    if "expand_pso_final_success" in row and pd.notna(row["expand_pso_final_success"]):
        expand_pso = make_candidate_observation(
            method=CandidateMethod.RETRIEVAL_PNN_PSO_EXPAND_TOP_SEEDS,
            success=_as_bool(row["expand_pso_final_success"]),
            amplitude_mse=float(row["expand_pso_final_amplitude_mse"]),
            thresholds=thresholds,
            extra_metrics=_extra_optimizer_metrics(row, "expand_pso"),
            initial_amplitude_mse=float(row["expand_pso_initial_top1_amplitude_mse"])
            if "expand_pso_initial_top1_amplitude_mse" in row
            else float(row["top1_amplitude_mse"]),
            accepted_optimization=_as_bool(row["expand_pso_accepted_optimization"])
            if "expand_pso_accepted_optimization" in row
            else None,
        )
        candidates[CandidateMethod.RETRIEVAL_PNN_PSO_EXPAND_TOP_SEEDS.value] = expand_pso
    return candidates


def _best_candidate(candidates: list[PlannerCandidateObservation]) -> PlannerCandidateObservation:
    successful = [candidate for candidate in candidates if candidate.success]
    if successful:
        return min(successful, key=lambda candidate: candidate.amplitude_mse)
    return min(candidates, key=lambda candidate: candidate.amplitude_mse)


def _make_observation(
    *,
    row: pd.Series,
    budget_used: int,
    budget_limit: int,
    revealed: set[str],
    candidates: dict[str, PlannerCandidateObservation],
    history: list[dict[str, Any]],
) -> PlannerObservation:
    available = [candidates[name] for name in revealed]
    current_best = _best_candidate(available)
    return PlannerObservation(
        domain=str(row["domain"]),
        query_index=int(row["query_index"]),
        source_index=int(row["source_index"]) if "source_index" in row and pd.notna(row["source_index"]) else None,
        budget_used=budget_used,
        budget_limit=budget_limit,
        retrieval_top1=candidates[CandidateMethod.RAW_SPECTRUM_TOP1.value],
        pso_final=candidates[CandidateMethod.RETRIEVAL_PNN_PSO.value]
        if CandidateMethod.RETRIEVAL_PNN_PSO.value in revealed
        else None,
        cmaes_final=candidates[CandidateMethod.RETRIEVAL_PNN_CMAES.value]
        if CandidateMethod.RETRIEVAL_PNN_CMAES.value in revealed
        else None,
        expand_top_seeds_final=candidates.get(CandidateMethod.RETRIEVAL_PNN_PSO_EXPAND_TOP_SEEDS.value)
        if CandidateMethod.RETRIEVAL_PNN_PSO_EXPAND_TOP_SEEDS.value in revealed
        else None,
        current_best_method=current_best.method,
        current_best_amplitude_mse=current_best.amplitude_mse,
        current_best_success=current_best.success,
        history=[step["action"] for step in history],
    )


def replay_query(
    row: pd.Series,
    thresholds: dict[str, float],
    policy: PlannerPolicy,
    *,
    budget_limit: int = 3,
) -> dict[str, Any]:
    candidates = build_candidates(row, thresholds)
    revealed = {CandidateMethod.RAW_SPECTRUM_TOP1.value}
    budget_used = 1
    history: list[dict[str, Any]] = []
    final_candidate: PlannerCandidateObservation | None = None
    stop_action: PlannerAction | None = None

    for _ in range(max(1, budget_limit + 2)):
        observation = _make_observation(
            row=row,
            budget_used=budget_used,
            budget_limit=budget_limit,
            revealed=revealed,
            candidates=candidates,
            history=history,
        )
        action = policy.decide(observation)
        step = {
            "observation": observation.to_dict(),
            "action": action.to_dict(),
        }
        history.append(step)

        if action.action_type == PlannerActionType.STOP_WITH_RETRIEVAL.value:
            final_candidate = candidates[CandidateMethod.RAW_SPECTRUM_TOP1.value]
            stop_action = action
            break

        if action.action_type in {PlannerActionType.STOP_WITH_BEST_MSE.value, PlannerActionType.STOP_FAIL.value}:
            final_candidate = _best_candidate([candidates[name] for name in revealed])
            stop_action = action
            break

        if action.action_type == PlannerActionType.RUN_PSO.value:
            if CandidateMethod.RETRIEVAL_PNN_PSO.value not in revealed:
                revealed.add(CandidateMethod.RETRIEVAL_PNN_PSO.value)
                budget_used += 1
            continue

        if action.action_type == PlannerActionType.RUN_CMAES.value:
            if CandidateMethod.RETRIEVAL_PNN_CMAES.value not in revealed:
                revealed.add(CandidateMethod.RETRIEVAL_PNN_CMAES.value)
                budget_used += 1
            continue

        if action.action_type in {
            PlannerActionType.EXPAND_TOP_SEEDS.value,
            PlannerActionType.SWITCH_DOMAIN_PRIOR.value,
        }:
            if (
                action.action_type == PlannerActionType.EXPAND_TOP_SEEDS.value
                and CandidateMethod.RETRIEVAL_PNN_PSO_EXPAND_TOP_SEEDS.value in candidates
                and CandidateMethod.RETRIEVAL_PNN_PSO_EXPAND_TOP_SEEDS.value not in revealed
            ):
                revealed.add(CandidateMethod.RETRIEVAL_PNN_PSO_EXPAND_TOP_SEEDS.value)
                budget_used += 1
            else:
                budget_used += min(action.budget_to_spend, max(0, budget_limit - budget_used))
            continue

    if final_candidate is None:
        final_candidate = _best_candidate([candidates[name] for name in revealed])
        stop_action = PlannerAction(
            action_type=PlannerActionType.STOP_WITH_BEST_MSE.value,
            reason="Replay guard reached; choose the best revealed candidate.",
        )

    return {
        "domain": str(row["domain"]),
        "query_index": int(row["query_index"]),
        "source_index": int(row["source_index"]) if "source_index" in row and pd.notna(row["source_index"]) else None,
        "planner_policy": policy.name,
        "planner_success": bool(final_candidate.success),
        "planner_final_method": final_candidate.method,
        "planner_final_amplitude_mse": float(final_candidate.amplitude_mse),
        "planner_steps_used": int(budget_used),
        "planner_stop_action": stop_action.action_type if stop_action else "",
        "planner_stop_reason": stop_action.reason if stop_action else "",
        "final_failure_modes": "|".join(final_candidate.critic.failure_modes),
        "action_sequence": "|".join(step["action"]["action_type"] for step in history),
        "trace_json": json.dumps(history, ensure_ascii=True),
        "retrieval_top1_success": candidates[CandidateMethod.RAW_SPECTRUM_TOP1.value].success,
        "retrieval_top1_amplitude_mse": candidates[CandidateMethod.RAW_SPECTRUM_TOP1.value].amplitude_mse,
        "pso_final_success": candidates[CandidateMethod.RETRIEVAL_PNN_PSO.value].success,
        "pso_final_amplitude_mse": candidates[CandidateMethod.RETRIEVAL_PNN_PSO.value].amplitude_mse,
        "cmaes_final_success": candidates[CandidateMethod.RETRIEVAL_PNN_CMAES.value].success,
        "cmaes_final_amplitude_mse": candidates[CandidateMethod.RETRIEVAL_PNN_CMAES.value].amplitude_mse,
        "expand_top_seeds_final_success": candidates.get(
            CandidateMethod.RETRIEVAL_PNN_PSO_EXPAND_TOP_SEEDS.value,
            candidates[CandidateMethod.RETRIEVAL_PNN_PSO.value],
        ).success,
        "expand_top_seeds_final_amplitude_mse": candidates.get(
            CandidateMethod.RETRIEVAL_PNN_PSO_EXPAND_TOP_SEEDS.value,
            candidates[CandidateMethod.RETRIEVAL_PNN_PSO.value],
        ).amplitude_mse,
    }


def evaluate_domain(
    *,
    domain: str,
    retrieval_query_path: str | Path,
    pso_query_path: str | Path,
    cmaes_query_path: str | Path,
    expand_pso_query_path: str | Path | None,
    thresholds: dict[str, float],
    policy: PlannerPolicy,
    budget_limit: int = 3,
    max_queries: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    retrieval_df = load_raw_retrieval_rows(retrieval_query_path, domain)
    if max_queries is not None:
        retrieval_df = retrieval_df.head(max_queries).copy()
    pso_df = load_optimizer_view(pso_query_path, "pso")
    cmaes_df = load_optimizer_view(cmaes_query_path, "cmaes")
    merged = retrieval_df.merge(pso_df, on=["domain", "query_index"]).merge(
        cmaes_df,
        on=["domain", "query_index"],
    )
    if expand_pso_query_path is not None:
        expand_pso_df = load_optimizer_view(expand_pso_query_path, "expand_pso")
        merged = merged.merge(expand_pso_df, on=["domain", "query_index"], how="left")
    if len(merged) != len(retrieval_df):
        raise ValueError(f"Merged row count mismatch for {domain}: retrieval={len(retrieval_df)}, merged={len(merged)}")

    rows = [
        replay_query(row, thresholds, policy, budget_limit=budget_limit)
        for _, row in merged.iterrows()
    ]
    df = pd.DataFrame(rows)
    action_counts = Counter()
    for sequence in df["action_sequence"]:
        action_counts.update(str(sequence).split("|"))

    summary: dict[str, Any] = {
        "domain": domain,
        "num_queries": len(df),
        "planner_policy": policy.name,
        "budget_limit": budget_limit,
        "success_rate_final": float(df["planner_success"].mean()),
        "mean_final_amplitude_mse": float(df["planner_final_amplitude_mse"].mean()),
        "median_final_amplitude_mse": float(df["planner_final_amplitude_mse"].median()),
        "mean_steps_used": float(df["planner_steps_used"].mean()),
        "failure_recovery_rate": _failure_recovery_rate(df),
    }
    for action_type in PlannerActionType:
        summary[f"action_count_{action_type.value}"] = int(action_counts[action_type.value])
    return rows, summary


def _failure_recovery_rate(df: pd.DataFrame) -> float:
    failed_retrieval = ~df["retrieval_top1_success"].astype(bool)
    if int(failed_retrieval.sum()) == 0:
        return 0.0
    recovered = failed_retrieval & df["planner_success"].astype(bool)
    return float(recovered.sum() / failed_retrieval.sum())


def evaluate_offline_planner(
    *,
    domain_specs: list[tuple[str, Path, Path] | tuple[str, Path, Path, Path | None]],
    retrieval_query_path: str | Path,
    thresholds_by_domain: dict[str, dict[str, float]],
    output_dir: str | Path,
    policy: PlannerPolicy | None = None,
    budget_limit: int = 3,
    max_queries: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = policy or OfflineRulePlannerPolicy()

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for spec in domain_specs:
        domain, pso_query_path, cmaes_query_path = spec[:3]
        expand_pso_query_path = spec[3] if len(spec) > 3 else None
        rows, summary = evaluate_domain(
            domain=domain,
            retrieval_query_path=retrieval_query_path,
            pso_query_path=pso_query_path,
            cmaes_query_path=cmaes_query_path,
            expand_pso_query_path=expand_pso_query_path,
            thresholds=thresholds_by_domain[domain],
            policy=policy,
            budget_limit=budget_limit,
            max_queries=max_queries,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    query_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summaries)
    query_df.to_csv(output_dir / "planner_offline_query_results.csv", index=False)
    summary_df.to_csv(output_dir / "planner_offline_summary.csv", index=False)
    return query_df, summary_df
