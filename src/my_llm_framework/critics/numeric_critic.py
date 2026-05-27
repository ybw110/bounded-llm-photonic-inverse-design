from __future__ import annotations

from math import isfinite
from typing import Any

from my_llm_framework.evaluation.metrics import is_success, spectrum_metric_bundle
from my_llm_framework.schemas import Critique


ALREADY_SUCCESSFUL = "already_successful"
AMPLITUDE_MSE_TOO_HIGH = "amplitude_mse_too_high"
CENTER_FREQUENCY_SHIFTED = "center_frequency_shifted"
BANDWIDTH_MISMATCH = "bandwidth_mismatch"
SUPPRESSION_DEPTH_MISMATCH = "suppression_depth_mismatch"
OPTIMIZER_NO_GAIN = "optimizer_no_gain"
CONSTRAINTS_NOT_MET = "constraints_not_met"
METRICS_UNAVAILABLE = "metrics_unavailable"


METRIC_FAILURE_CODES = {
    "amplitude_mse": AMPLITUDE_MSE_TOO_HIGH,
    "center_frequency_error_thz": CENTER_FREQUENCY_SHIFTED,
    "bandwidth_error_thz": BANDWIDTH_MISMATCH,
    "suppression_depth_error_db": SUPPRESSION_DEPTH_MISMATCH,
}


def _clean_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    clean: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(number):
            clean[key] = number
    return clean


def critique_metrics(
    metrics: dict[str, Any],
    thresholds: dict[str, float],
    *,
    candidate_success: bool | None = None,
    initial_amplitude_mse: float | None = None,
    accepted_optimization: bool | None = None,
    min_optimizer_gain: float = 1e-12,
) -> Critique:
    """Convert numeric metrics into stable critic codes.

    The critic deliberately stays numeric and deterministic. It does not choose
    a full search policy; it summarizes why the current candidate is acceptable
    or why the planner should consider another action.
    """

    clean_metrics = _clean_metrics(metrics)
    if not clean_metrics:
        return Critique(
            passed=False,
            metrics={},
            failure_modes=[METRICS_UNAVAILABLE],
            recommended_action="run_pso",
            primary_failure_mode=METRICS_UNAVAILABLE,
            notes=["No finite metrics were provided to the critic."],
        )

    passed = bool(candidate_success) if candidate_success is not None else is_success(clean_metrics, thresholds)
    failure_modes: list[str] = []
    notes: list[str] = []

    if passed:
        failure_modes.append(ALREADY_SUCCESSFUL)
    else:
        for metric_name, limit in thresholds.items():
            if metric_name == "constraint_satisfaction":
                continue
            if metric_name not in clean_metrics:
                continue
            if clean_metrics[metric_name] > float(limit):
                failure_modes.append(METRIC_FAILURE_CODES.get(metric_name, CONSTRAINTS_NOT_MET))
                notes.append(f"{metric_name}={clean_metrics[metric_name]:.6g} exceeds {float(limit):.6g}")

        current_mse = clean_metrics.get("amplitude_mse")
        if initial_amplitude_mse is not None and current_mse is not None:
            gain = float(initial_amplitude_mse) - current_mse
            clean_metrics["amplitude_mse_gain"] = gain
            if gain <= min_optimizer_gain:
                failure_modes.append(OPTIMIZER_NO_GAIN)
                notes.append(f"amplitude_mse_gain={gain:.6g} is not positive enough")

        if accepted_optimization is False:
            failure_modes.append(OPTIMIZER_NO_GAIN)
            notes.append("optimizer result was not accepted by its acceptance policy")

        if not failure_modes:
            failure_modes.append(CONSTRAINTS_NOT_MET)

    failure_modes = list(dict.fromkeys(failure_modes))
    primary = failure_modes[0] if failure_modes else None
    return Critique(
        passed=passed,
        metrics=clean_metrics,
        failure_modes=failure_modes,
        recommended_action=recommend_action_from_failure_modes(failure_modes, passed=passed),
        primary_failure_mode=primary,
        notes=notes,
    )


def critique_offline_candidate(
    *,
    success: bool,
    amplitude_mse: float,
    thresholds: dict[str, float],
    extra_metrics: dict[str, Any] | None = None,
    initial_amplitude_mse: float | None = None,
    accepted_optimization: bool | None = None,
) -> Critique:
    """Critique a candidate represented by Phase 2 query-level CSV columns."""

    metrics = {"amplitude_mse": amplitude_mse}
    if extra_metrics:
        metrics.update(extra_metrics)
    return critique_metrics(
        metrics,
        thresholds,
        candidate_success=success,
        initial_amplitude_mse=initial_amplitude_mse,
        accepted_optimization=accepted_optimization,
    )


def recommend_action_from_failure_modes(failure_modes: list[str], *, passed: bool) -> str:
    if passed or ALREADY_SUCCESSFUL in failure_modes:
        return "stop_success"
    if CENTER_FREQUENCY_SHIFTED in failure_modes or BANDWIDTH_MISMATCH in failure_modes:
        return "run_cmaes"
    if OPTIMIZER_NO_GAIN in failure_modes:
        return "expand_top_seeds"
    if AMPLITUDE_MSE_TOO_HIGH in failure_modes:
        return "run_pso"
    return "run_pso"


def critique_candidate(candidate_spectrum, target_spectrum, freq, thresholds: dict[str, float]) -> dict[str, Any]:
    """Critique a full candidate spectrum and return a JSON-ready dict.

    This keeps backward compatibility with the earlier Phase 1/2 helper while
    using the richer Phase 3 Critique schema internally.
    """

    metrics = spectrum_metric_bundle(candidate_spectrum, target_spectrum, freq)
    return critique_metrics(metrics, thresholds).to_dict()
