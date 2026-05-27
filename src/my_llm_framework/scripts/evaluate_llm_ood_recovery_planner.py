from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHOD_ORDER = ("raw_spectrum", "domain_aware", "physics_summary")
SHIFT_TYPES = {
    "target_h": "topology_shift",
    "target_self": "material_frequency_shift",
}
ACTION_TYPES = (
    "stop_with_candidate",
    "select_recovery_candidate",
)


@dataclass
class OODAction:
    action_type: str
    reason: str
    method: str | None = None
    fewshot_size: int | str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "reason": self.reason,
            "method": self.method,
            "fewshot_size": self.fewshot_size,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OODAction":
        action_type = str(payload.get("action_type", ""))
        if action_type not in ACTION_TYPES:
            raise ValueError(f"Invalid action_type={action_type}; allowed={ACTION_TYPES}")
        return cls(
            action_type=action_type,
            reason=str(payload.get("reason", "")),
            method=payload.get("method"),
            fewshot_size=payload.get("fewshot_size"),
            metadata=dict(payload.get("metadata", {})),
        )


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _to_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value) or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fewshot_sort_value(label: Any) -> float:
    label_str = str(label)
    if label_str == "full":
        return float("inf")
    return float(label_str)


def _fewshot_cost(row: pd.Series) -> int:
    if str(row["fewshot_label"]) == "full":
        return int(_to_float(row.get("effective_fewshot_size"), 0.0))
    return int(_to_float(row.get("requested_fewshot_size"), 0.0))


def _normalize_label(value: int | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "full":
        return "full"
    try:
        return str(int(float(str(value))))
    except ValueError:
        return str(value)


def load_query_pool(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        df["pool_path"] = str(path)
        frames.append(df)
    pool = pd.concat(frames, ignore_index=True)
    pool["fewshot_label"] = pool["fewshot_label"].astype(str)
    pool["method"] = pool["method"].astype(str)
    pool["target_domain"] = pool["target_domain"].astype(str)
    pool["seed"] = pool["seed"].astype(int)
    pool["query_index"] = pool["query_index"].astype(int)
    pool["fewshot_sort_value"] = pool["fewshot_label"].map(_fewshot_sort_value)
    pool = pool.drop_duplicates(
        subset=["target_domain", "seed", "query_index", "method", "fewshot_label"],
        keep="last",
    )
    return pool


def candidate_payload(row: pd.Series, success_budget: int) -> dict[str, Any]:
    return {
        "method": str(row["method"]),
        "fewshot_label": str(row["fewshot_label"]),
        "fewshot_cost": _fewshot_cost(row),
        "top1_bank_domain": str(row.get("top1_bank_domain", "")),
        "top1_is_target_domain": _to_bool(row.get("top1_is_target_domain")),
        "top1_amplitude_mse": _to_float(row.get("top1_amplitude_mse")),
        "best_amplitude_mse_at_budget": _to_float(row.get("best_amplitude_mse_at_max_budget")),
        f"success_at_{success_budget}": _to_bool(row.get(f"success_at_{success_budget}")),
        f"target_domain_success_at_{success_budget}": _to_bool(row.get(f"target_domain_success_at_{success_budget}")),
        "first_success_rank": None if pd.isna(row.get("first_success_rank")) or row.get("first_success_rank") == "" else int(float(row.get("first_success_rank"))),
        "first_target_domain_success_rank": None
        if pd.isna(row.get("first_target_domain_success_rank")) or row.get("first_target_domain_success_rank") == ""
        else int(float(row.get("first_target_domain_success_rank"))),
    }


def critique_candidate(row: pd.Series, success_budget: int) -> dict[str, Any]:
    domain = str(row["target_domain"])
    shift_type = SHIFT_TYPES.get(domain, "ood_shift")
    success = _to_bool(row.get(f"success_at_{success_budget}"))
    target_success = _to_bool(row.get(f"target_domain_success_at_{success_budget}"))
    top1_target = _to_bool(row.get("top1_is_target_domain"))
    top1_mse = _to_float(row.get("top1_amplitude_mse"))
    fewshot_cost = _fewshot_cost(row)

    reasons: list[str] = []
    if target_success:
        reasons.append("already_successful")
    if success and not target_success:
        reasons.append("spectral_match_ok_but_topology_or_domain_invalid")
    if not top1_target:
        reasons.append("top1_source_domain_candidate")
    if not target_success and fewshot_cost == 0:
        reasons.append("zero_shot_target_memory_missing")
    if not target_success and domain == "target_h":
        reasons.append("target_topology_coverage_insufficient")
    if not target_success and domain == "target_self":
        reasons.append("material_frequency_calibration_missing")
    if np.isfinite(top1_mse) and top1_mse > 0.02:
        reasons.append("amplitude_mse_too_high")
    if not reasons:
        reasons.append("candidate_not_yet_diagnostic")

    recommended = "stop_with_candidate" if target_success else "select_recovery_candidate"
    return {
        "shift_type": shift_type,
        "passed": bool(target_success),
        "primary_reason": reasons[0],
        "failure_reasons": reasons,
        "recommended_action": recommended,
    }


def available_candidate_rows(
    group: pd.DataFrame,
    *,
    allowed_methods: set[str],
    allowed_fewshot_labels: set[str],
    allow_full: bool,
) -> pd.DataFrame:
    sub = group[group["method"].isin(allowed_methods) & group["fewshot_label"].isin(allowed_fewshot_labels)].copy()
    if not allow_full:
        sub = sub[sub["fewshot_label"] != "full"]
    return sub.sort_values(["fewshot_sort_value", "method"], kind="stable")


def build_available_actions(candidates: pd.DataFrame, current: pd.Series, success_budget: int) -> list[dict[str, Any]]:
    actions = [
        {
            "action_type": "stop_with_candidate",
            "method": str(current["method"]),
            "fewshot_size": str(current["fewshot_label"]),
            "description": "Stop and return the currently selected candidate.",
        }
    ]
    current_key = (str(current["method"]), str(current["fewshot_label"]))
    for _, row in candidates.iterrows():
        key = (str(row["method"]), str(row["fewshot_label"]))
        if key == current_key:
            continue
        payload = candidate_payload(row, success_budget)
        actions.append(
            {
                "action_type": "select_recovery_candidate",
                "method": key[0],
                "fewshot_size": key[1],
                "description": "Select this offline recovery candidate from the frozen Phase 4 pool.",
                "candidate_preview": {
                    "success_at_budget": payload[f"success_at_{success_budget}"],
                    "target_domain_success_at_budget": payload[f"target_domain_success_at_{success_budget}"],
                    "top1_is_target_domain": payload["top1_is_target_domain"],
                    "top1_amplitude_mse": payload["top1_amplitude_mse"],
                },
            }
        )
    return actions


class BaseOODPolicy:
    name = "base"

    def decide(self, observation: dict[str, Any]) -> OODAction:
        raise NotImplementedError


class SourceOnlyPolicy(BaseOODPolicy):
    name = "source_only_rule"

    def decide(self, observation: dict[str, Any]) -> OODAction:
        return OODAction("stop_with_candidate", reason="Use zero-shot source-memory retrieval.")


class FixedRecoveryPolicy(BaseOODPolicy):
    def __init__(self, *, method: str, fewshot_size: int | str) -> None:
        self.method = method
        self.fewshot_size = _normalize_label(fewshot_size)
        self.name = f"fixed_{method}_{self.fewshot_size}_rule"

    def decide(self, observation: dict[str, Any]) -> OODAction:
        current = observation["current_candidate"]
        if current["method"] == self.method and str(current["fewshot_label"]) == str(self.fewshot_size):
            return OODAction("stop_with_candidate", reason="Fixed recovery candidate has been selected.")
        return OODAction(
            "select_recovery_candidate",
            reason=f"Select fixed recovery candidate {self.method} with {self.fewshot_size} target samples.",
            method=self.method,
            fewshot_size=self.fewshot_size,
        )


class ShiftAwareRecoveryPolicy(BaseOODPolicy):
    def __init__(self, *, target_h_size: int, target_self_size: int, method: str = "domain_aware") -> None:
        self.target_h_size = int(target_h_size)
        self.target_self_size = int(target_self_size)
        self.method = method
        self.name = f"shift_aware_{method}_h{target_h_size}_self{target_self_size}_rule"

    def decide(self, observation: dict[str, Any]) -> OODAction:
        domain = observation["target_domain"]
        target_size = self.target_h_size if domain == "target_h" else self.target_self_size
        current = observation["current_candidate"]
        if current["method"] == self.method and str(current["fewshot_label"]) == str(target_size):
            return OODAction("stop_with_candidate", reason="Shift-aware recovery candidate has been selected.")
        return OODAction(
            "select_recovery_candidate",
            reason=f"Use shift-aware few-shot recovery for {domain}.",
            method=self.method,
            fewshot_size=target_size,
        )


class OracleRecoveryPolicy(BaseOODPolicy):
    name = "oracle_recovery_policy"

    def decide(self, observation: dict[str, Any]) -> OODAction:
        best = observation["oracle_best_candidate"]
        current = observation["current_candidate"]
        if current["method"] == best["method"] and str(current["fewshot_label"]) == str(best["fewshot_label"]):
            return OODAction("stop_with_candidate", reason="Oracle best candidate has been selected.")
        return OODAction(
            "select_recovery_candidate",
            reason="Oracle selects the best available offline candidate under the action/sample constraints.",
            method=best["method"],
            fewshot_size=best["fewshot_label"],
            metadata={"oracle": True},
        )


class OpenAICompatibleOODPolicy(BaseOODPolicy):
    """OOD planner policy using an OpenAI-compatible chat-completions API."""

    def __init__(
        self,
        *,
        provider_label: str,
        api_key_env: str,
        model: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
        use_response_format: bool,
        fallback_policy: BaseOODPolicy,
    ) -> None:
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(f"Missing {provider_label} API key. Set {api_key_env}.")
        self.provider_label = provider_label
        self.api_key_env = api_key_env
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = float(timeout_seconds)
        self.use_response_format = bool(use_response_format)
        self.fallback_policy = fallback_policy
        safe_model = str(model).replace("/", "_").replace(":", "_")
        self.name = f"{provider_label}_{safe_model}_ood_recovery_planner"

    def decide(self, observation: dict[str, Any]) -> OODAction:
        content = ""
        try:
            payload = self._chat_payload(observation)
            content = self._post_chat_completion(payload)
            action = OODAction.from_dict(_loads_json_object(content))
            _validate_action_against_observation(action, observation)
            action.metadata = dict(action.metadata or {})
            action.metadata.setdefault("llm_provider", self.provider_label)
            action.metadata.setdefault("llm_model", self.model)
            return action
        except Exception as exc:
            action = self.fallback_policy.decide(observation)
            action.metadata = dict(action.metadata or {})
            action.metadata.update(
                {
                    "llm_provider": self.provider_label,
                    "fallback_policy": self.fallback_policy.name,
                    "fallback_error": f"{type(exc).__name__}: {exc}",
                    "raw_llm_response_snippet": str(content)[:300],
                    "llm_model": self.model,
                }
            )
            return action

    def _chat_payload(self, observation: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _ood_planner_system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": "Choose one OOD recovery action from available_actions.",
                            "observation": observation,
                        },
                        ensure_ascii=True,
                        allow_nan=False,
                    ),
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.use_response_format:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _post_chat_completion(self, payload: dict[str, Any]) -> str:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.provider_label} API HTTP {exc.code}: {body[:500]}") from exc
        response_json = json.loads(body)
        return str(response_json["choices"][0]["message"]["content"])


class DeepSeekOODPolicy(OpenAICompatibleOODPolicy):
    def __init__(
        self,
        *,
        api_key_env: str,
        model: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
        use_response_format: bool,
        fallback_policy: BaseOODPolicy,
    ) -> None:
        super().__init__(
            provider_label="deepseek",
            api_key_env=api_key_env,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            use_response_format=use_response_format,
            fallback_policy=fallback_policy,
        )


class GLMOODPolicy(OpenAICompatibleOODPolicy):
    def __init__(
        self,
        *,
        api_key_env: str,
        model: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
        use_response_format: bool,
        fallback_policy: BaseOODPolicy,
    ) -> None:
        super().__init__(
            provider_label="glm",
            api_key_env=api_key_env,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            use_response_format=use_response_format,
            fallback_policy=fallback_policy,
        )


def _ood_planner_system_prompt() -> str:
    return (
        "You are an evidence-conditioned OOD recovery planner for metasurface inverse design. "
        "You do not generate spectra or parameters. You only select one candidate from a frozen offline Phase 4 pool. "
        "Return exactly one JSON object and no prose. "
        "Allowed action_type values are: stop_with_candidate, select_recovery_candidate. "
        "Prefer target-domain-valid recovery over source-domain spectral matches when the target domain has topology shift. "
        "For material/frequency shift, small target-memory calibration can be enough. "
        "Use no more target samples than needed. "
        "The JSON schema is: "
        '{"action_type":"select_recovery_candidate","method":"domain_aware","fewshot_size":200,'
        '"reason":"short reason","metadata":{}}'
    )


def _loads_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


def _validate_action_against_observation(action: OODAction, observation: dict[str, Any]) -> None:
    if action.action_type == "stop_with_candidate":
        return
    method = str(action.method)
    fewshot_label = _normalize_label(action.fewshot_size)
    current = observation.get("current_candidate", {})
    if method == str(current.get("method")) and str(fewshot_label) == str(current.get("fewshot_label")):
        action.action_type = "stop_with_candidate"
        action.reason = action.reason or "LLM re-selected the current candidate; treating this as stop."
        return
    available = {
        (str(item.get("method")), str(item.get("fewshot_size")))
        for item in observation.get("available_actions", [])
        if item.get("action_type") == "select_recovery_candidate"
    }
    if (method, str(fewshot_label)) not in available:
        raise ValueError(f"Action selects unavailable candidate method={method}, fewshot_size={fewshot_label}")


def select_oracle_best(candidates: pd.DataFrame, success_budget: int) -> pd.Series:
    scored = candidates.copy()
    scored["_target_success"] = scored[f"target_domain_success_at_{success_budget}"].map(_to_bool).astype(int)
    scored["_success"] = scored[f"success_at_{success_budget}"].map(_to_bool).astype(int)
    scored["_top1_target"] = scored["top1_is_target_domain"].map(_to_bool).astype(int)
    scored["_best_mse"] = pd.to_numeric(scored["best_amplitude_mse_at_max_budget"], errors="coerce").fillna(np.inf)
    scored["_top1_mse"] = pd.to_numeric(scored["top1_amplitude_mse"], errors="coerce").fillna(np.inf)
    scored["_fewshot_cost"] = scored.apply(_fewshot_cost, axis=1)
    scored["_method_rank"] = scored["method"].map({method: i for i, method in enumerate(METHOD_ORDER)}).fillna(99)
    scored = scored.sort_values(
        [
            "_target_success",
            "_success",
            "_top1_target",
            "_fewshot_cost",
            "_best_mse",
            "_top1_mse",
            "_method_rank",
        ],
        ascending=[False, False, False, True, True, True, True],
        kind="stable",
    )
    return scored.iloc[0]


def build_observation(
    *,
    target_domain: str,
    seed: int,
    query_index: int,
    current: pd.Series,
    candidates: pd.DataFrame,
    history: list[dict[str, Any]],
    budget_used: int,
    budget_limit: int,
    success_budget: int,
    oracle_best: pd.Series,
) -> dict[str, Any]:
    current_payload = candidate_payload(current, success_budget)
    oracle_payload = candidate_payload(oracle_best, success_budget)
    observation = {
        "target_domain": target_domain,
        "shift_type": SHIFT_TYPES.get(target_domain, "ood_shift"),
        "seed": int(seed),
        "query_index": int(query_index),
        "budget_used": int(budget_used),
        "budget_limit": int(budget_limit),
        "success_budget": int(success_budget),
        "current_candidate": current_payload,
        "numeric_critic": critique_candidate(current, success_budget),
        "available_actions": build_available_actions(candidates, current, success_budget),
        "history": history[-4:],
        "planner_objective": {
            "primary": f"maximize target_domain_success_at_{success_budget}",
            "secondary": f"maximize success_at_{success_budget}",
            "tertiary": "minimize top1 and best amplitude MSE",
            "cost": "use as few target-domain samples and planner steps as possible",
        },
        "oracle_best_candidate": {
            "method": oracle_payload["method"],
            "fewshot_label": oracle_payload["fewshot_label"],
            f"success_at_{success_budget}": oracle_payload[f"success_at_{success_budget}"],
            f"target_domain_success_at_{success_budget}": oracle_payload[f"target_domain_success_at_{success_budget}"],
            "top1_is_target_domain": oracle_payload["top1_is_target_domain"],
            "top1_amplitude_mse": oracle_payload["top1_amplitude_mse"],
        },
    }
    return _json_clean(observation)


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def apply_action(action: OODAction, current: pd.Series, candidates: pd.DataFrame) -> pd.Series:
    if action.action_type == "stop_with_candidate":
        return current
    method = str(action.method)
    fewshot_label = str(_normalize_label(action.fewshot_size))
    match = candidates[(candidates["method"] == method) & (candidates["fewshot_label"] == fewshot_label)]
    if match.empty:
        raise ValueError(f"No candidate for method={method}, fewshot_label={fewshot_label}")
    return match.iloc[0]


def replay_one_query(
    *,
    group: pd.DataFrame,
    policy: BaseOODPolicy,
    budget_limit: int,
    success_budget: int,
    allowed_methods: set[str],
    allowed_fewshot_labels: set[str],
    allow_full: bool,
    initial_method: str,
    initial_fewshot_label: str,
) -> dict[str, Any]:
    target_domain = str(group["target_domain"].iloc[0])
    seed = int(group["seed"].iloc[0])
    query_index = int(group["query_index"].iloc[0])
    candidates = available_candidate_rows(
        group,
        allowed_methods=allowed_methods,
        allowed_fewshot_labels=allowed_fewshot_labels,
        allow_full=allow_full,
    )
    if candidates.empty:
        raise ValueError(f"No candidates available for target={target_domain}, seed={seed}, query={query_index}")

    initial = candidates[(candidates["method"] == initial_method) & (candidates["fewshot_label"] == initial_fewshot_label)]
    if initial.empty:
        initial = candidates.iloc[[0]]
    current = initial.iloc[0]
    oracle_best = select_oracle_best(candidates, success_budget)
    history: list[dict[str, Any]] = []
    budget_used = 0
    stop_reason = "budget_exhausted"

    while budget_used < budget_limit:
        observation = build_observation(
            target_domain=target_domain,
            seed=seed,
            query_index=query_index,
            current=current,
            candidates=candidates,
            history=history,
            budget_used=budget_used,
            budget_limit=budget_limit,
            success_budget=success_budget,
            oracle_best=oracle_best,
        )
        action = policy.decide(observation)
        action_dict = action.to_dict()
        history.append(action_dict)
        if action.action_type == "stop_with_candidate":
            stop_reason = action.reason
            break
        next_candidate = apply_action(action, current, candidates)
        budget_used += 1
        if str(next_candidate["method"]) == str(current["method"]) and str(next_candidate["fewshot_label"]) == str(current["fewshot_label"]):
            stop_reason = "selected_current_candidate"
            break
        current = next_candidate
    else:
        stop_reason = "budget_exhausted"

    final_payload = candidate_payload(current, success_budget)
    oracle_payload = candidate_payload(oracle_best, success_budget)
    final_critic = critique_candidate(current, success_budget)
    used_fallback = any("fallback_error" in dict(item.get("metadata") or {}) for item in history)
    fallback_errors = [
        str(dict(item.get("metadata") or {}).get("fallback_error"))
        for item in history
        if "fallback_error" in dict(item.get("metadata") or {})
    ]
    return {
        "planner_policy": policy.name,
        "planner_provider": str(getattr(policy, "provider_label", "")),
        "planner_model": str(getattr(policy, "model", "")),
        "target_domain": target_domain,
        "shift_type": SHIFT_TYPES.get(target_domain, "ood_shift"),
        "seed": seed,
        "query_index": query_index,
        "planner_steps_used": int(budget_used),
        "planner_stop_reason": stop_reason,
        "planner_used_fallback": bool(used_fallback),
        "planner_fallback_errors": "|".join(fallback_errors),
        "planner_actions": json.dumps(history, ensure_ascii=True),
        "final_method": final_payload["method"],
        "final_fewshot_label": final_payload["fewshot_label"],
        "final_fewshot_cost": final_payload["fewshot_cost"],
        f"final_success_at_{success_budget}": final_payload[f"success_at_{success_budget}"],
        f"final_target_domain_success_at_{success_budget}": final_payload[f"target_domain_success_at_{success_budget}"],
        "final_top1_is_target_domain": final_payload["top1_is_target_domain"],
        "final_top1_amplitude_mse": final_payload["top1_amplitude_mse"],
        "final_best_amplitude_mse_at_budget": final_payload["best_amplitude_mse_at_budget"],
        "final_first_success_rank": final_payload["first_success_rank"],
        "final_first_target_domain_success_rank": final_payload["first_target_domain_success_rank"],
        "final_critic_primary_reason": final_critic["primary_reason"],
        "final_critic_reasons": "|".join(final_critic["failure_reasons"]),
        "oracle_method": oracle_payload["method"],
        "oracle_fewshot_label": oracle_payload["fewshot_label"],
        "oracle_fewshot_cost": oracle_payload["fewshot_cost"],
        f"oracle_success_at_{success_budget}": oracle_payload[f"success_at_{success_budget}"],
        f"oracle_target_domain_success_at_{success_budget}": oracle_payload[f"target_domain_success_at_{success_budget}"],
        "oracle_top1_amplitude_mse": oracle_payload["top1_amplitude_mse"],
        "oracle_best_amplitude_mse_at_budget": oracle_payload["best_amplitude_mse_at_budget"],
        "matched_oracle_choice": final_payload["method"] == oracle_payload["method"]
        and final_payload["fewshot_label"] == oracle_payload["fewshot_label"],
    }


def summarize_replay(query_df: pd.DataFrame, success_budget: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    success_col = f"final_success_at_{success_budget}"
    target_success_col = f"final_target_domain_success_at_{success_budget}"
    oracle_target_col = f"oracle_target_domain_success_at_{success_budget}"

    for target_domain, group in query_df.groupby("target_domain", sort=False):
        row = {
            "planner_policy": str(group["planner_policy"].iloc[0]),
            "planner_provider": str(group["planner_provider"].iloc[0]) if "planner_provider" in group.columns else "",
            "planner_model": str(group["planner_model"].iloc[0]) if "planner_model" in group.columns else "",
            "target_domain": target_domain,
            "shift_type": str(group["shift_type"].iloc[0]),
            "num_replays": int(len(group)),
            f"success_at_{success_budget}": float(group[success_col].astype(bool).mean()),
            f"target_domain_success_at_{success_budget}": float(group[target_success_col].astype(bool).mean()),
            "top1_target_domain_rate": float(group["final_top1_is_target_domain"].astype(bool).mean()),
            "mean_final_top1_amplitude_mse": float(pd.to_numeric(group["final_top1_amplitude_mse"], errors="coerce").mean()),
            "mean_final_best_amplitude_mse_at_budget": float(
                pd.to_numeric(group["final_best_amplitude_mse_at_budget"], errors="coerce").mean()
            ),
            "mean_steps_used": float(pd.to_numeric(group["planner_steps_used"], errors="coerce").mean()),
            "mean_fewshot_cost": float(pd.to_numeric(group["final_fewshot_cost"], errors="coerce").mean()),
            "oracle_match_rate": float(group["matched_oracle_choice"].astype(bool).mean()),
            "fallback_or_invalid_action_rate": float(group["planner_used_fallback"].astype(bool).mean())
            if "planner_used_fallback" in group.columns
            else 0.0,
            f"oracle_target_domain_success_at_{success_budget}": float(group[oracle_target_col].astype(bool).mean()),
            "oracle_mean_fewshot_cost": float(pd.to_numeric(group["oracle_fewshot_cost"], errors="coerce").mean()),
        }
        row[f"oracle_gap_target_domain_success_at_{success_budget}"] = (
            row[f"oracle_target_domain_success_at_{success_budget}"] - row[f"target_domain_success_at_{success_budget}"]
        )
        rows.append(row)

        total = len(group)
        for col, prefix in [("final_method", "method"), ("final_fewshot_label", "fewshot"), ("final_critic_primary_reason", "critic")]:
            counts = group[col].value_counts(dropna=False)
            for value, count in counts.items():
                action_rows.append(
                    {
                        "planner_policy": str(group["planner_policy"].iloc[0]),
                        "target_domain": target_domain,
                        "category": prefix,
                        "value": str(value),
                        "count": int(count),
                        "proportion": float(count / total) if total else 0.0,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(action_rows)


def evaluate_ood_recovery_planner(
    *,
    pool_paths: list[Path],
    output_dir: Path,
    policy: BaseOODPolicy,
    success_budget: int,
    budget_limit: int,
    allowed_methods: list[str],
    allowed_fewshot_sizes: list[str],
    allow_full: bool,
    target_domains: list[str] | None,
    seeds: list[int] | None,
    max_queries_per_domain: int | None,
    initial_method: str,
    initial_fewshot_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pool = load_query_pool(pool_paths)
    if target_domains:
        pool = pool[pool["target_domain"].isin(target_domains)]
    if seeds:
        pool = pool[pool["seed"].isin(seeds)]

    allowed_method_set = set(allowed_methods)
    allowed_fewshot_labels = {_normalize_label(size) for size in allowed_fewshot_sizes}
    if allow_full:
        allowed_fewshot_labels.add("full")
    allowed_fewshot_labels = {str(label) for label in allowed_fewshot_labels if label is not None}

    groups: list[pd.DataFrame] = []
    for (_, _, _), group in pool.groupby(["target_domain", "seed", "query_index"], sort=False):
        if not set(group["method"]).intersection(allowed_method_set):
            continue
        if not set(group["fewshot_label"]).intersection(allowed_fewshot_labels):
            continue
        groups.append(group)

    if max_queries_per_domain is not None:
        limited: list[pd.DataFrame] = []
        counts: dict[tuple[str, int], int] = {}
        for group in groups:
            key = (str(group["target_domain"].iloc[0]), int(group["seed"].iloc[0]))
            count = counts.get(key, 0)
            if count >= max_queries_per_domain:
                continue
            limited.append(group)
            counts[key] = count + 1
        groups = limited

    rows = [
        replay_one_query(
            group=group,
            policy=policy,
            budget_limit=budget_limit,
            success_budget=success_budget,
            allowed_methods=allowed_method_set,
            allowed_fewshot_labels=allowed_fewshot_labels,
            allow_full=allow_full,
            initial_method=initial_method,
            initial_fewshot_label=initial_fewshot_label,
        )
        for group in groups
    ]
    query_df = pd.DataFrame(rows)
    summary_df, action_df = summarize_replay(query_df, success_budget)
    query_df.to_csv(output_dir / "ood_recovery_planner_query_results.csv", index=False)
    summary_df.to_csv(output_dir / "ood_recovery_planner_summary.csv", index=False)
    action_df.to_csv(output_dir / "ood_recovery_planner_action_summary.csv", index=False)
    return query_df, summary_df, action_df


def build_policy(args: argparse.Namespace) -> BaseOODPolicy:
    if args.planner_policy == "source_only":
        return SourceOnlyPolicy()
    if args.planner_policy == "fixed":
        return FixedRecoveryPolicy(method=args.fixed_method, fewshot_size=args.fixed_fewshot_size)
    if args.planner_policy == "shift_aware":
        return ShiftAwareRecoveryPolicy(
            target_h_size=args.shift_aware_target_h_size,
            target_self_size=args.shift_aware_target_self_size,
            method=args.fixed_method,
        )
    if args.planner_policy == "oracle":
        return OracleRecoveryPolicy()
    if args.planner_policy == "deepseek":
        fallback = ShiftAwareRecoveryPolicy(
            target_h_size=args.shift_aware_target_h_size,
            target_self_size=args.shift_aware_target_self_size,
            method=args.fixed_method,
        )
        return DeepSeekOODPolicy(
            api_key_env=args.deepseek_api_key_env,
            model=args.deepseek_model,
            base_url=args.deepseek_base_url,
            temperature=args.deepseek_temperature,
            max_tokens=args.deepseek_max_tokens,
            timeout_seconds=args.deepseek_timeout_seconds,
            use_response_format=not args.disable_response_format,
            fallback_policy=fallback,
        )
    if args.planner_policy == "glm":
        fallback = ShiftAwareRecoveryPolicy(
            target_h_size=args.shift_aware_target_h_size,
            target_self_size=args.shift_aware_target_self_size,
            method=args.fixed_method,
        )
        return GLMOODPolicy(
            api_key_env=args.glm_api_key_env,
            model=args.glm_model,
            base_url=args.glm_base_url,
            temperature=args.glm_temperature,
            max_tokens=args.glm_max_tokens,
            timeout_seconds=args.glm_timeout_seconds,
            use_response_format=not args.disable_response_format,
            fallback_policy=fallback,
        )
    raise ValueError(f"Unknown planner policy: {args.planner_policy}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-results", nargs="+", required=True, help="One or more Phase 4 fewshot_retrieval_query_results.csv files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--planner-policy",
        choices=["source_only", "fixed", "shift_aware", "oracle", "deepseek", "glm"],
        default="shift_aware",
    )
    parser.add_argument("--success-budget", type=int, default=50)
    parser.add_argument("--budget-limit", type=int, default=2)
    parser.add_argument("--allowed-methods", nargs="+", default=["raw_spectrum", "domain_aware"], choices=list(METHOD_ORDER))
    parser.add_argument("--allowed-fewshot-sizes", nargs="+", default=["0", "20", "50", "100", "200"])
    parser.add_argument("--allow-full-target-action", action="store_true")
    parser.add_argument("--target-domains", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--max-queries-per-domain", type=int, default=None)
    parser.add_argument("--initial-method", default="raw_spectrum", choices=list(METHOD_ORDER))
    parser.add_argument("--initial-fewshot-size", default="0")
    parser.add_argument("--fixed-method", default="domain_aware", choices=list(METHOD_ORDER))
    parser.add_argument("--fixed-fewshot-size", default="200")
    parser.add_argument("--shift-aware-target-h-size", type=int, default=200)
    parser.add_argument("--shift-aware-target-self-size", type=int, default=20)
    parser.add_argument("--deepseek-model", default="deepseek-chat")
    parser.add_argument("--deepseek-base-url", default="https://api.deepseek.com")
    parser.add_argument("--deepseek-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--deepseek-temperature", type=float, default=0.0)
    parser.add_argument("--deepseek-max-tokens", type=int, default=512)
    parser.add_argument("--deepseek-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--glm-model", default="glm-4-flash")
    parser.add_argument("--glm-base-url", default="https://open.bigmodel.cn/api/paas/v4")
    parser.add_argument("--glm-api-key-env", default="GLM_API_KEY")
    parser.add_argument("--glm-temperature", type=float, default=0.0)
    parser.add_argument("--glm-max-tokens", type=int, default=512)
    parser.add_argument("--glm-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--disable-response-format",
        action="store_true",
        help="Do not send OpenAI JSON response_format; useful for providers/models that reject this parameter.",
    )
    args = parser.parse_args()

    policy = build_policy(args)
    _, summary_df, action_df = evaluate_ood_recovery_planner(
        pool_paths=[Path(path) for path in args.query_results],
        output_dir=Path(args.output_dir),
        policy=policy,
        success_budget=args.success_budget,
        budget_limit=args.budget_limit,
        allowed_methods=args.allowed_methods,
        allowed_fewshot_sizes=args.allowed_fewshot_sizes,
        allow_full=args.allow_full_target_action,
        target_domains=args.target_domains,
        seeds=args.seeds,
        max_queries_per_domain=args.max_queries_per_domain,
        initial_method=args.initial_method,
        initial_fewshot_label=str(_normalize_label(args.initial_fewshot_size)),
    )

    print(f"Saved LLM-OOD recovery planner replay: {Path(args.output_dir).resolve()}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print("\nAction summary:")
    print(action_df.to_string(index=False))


if __name__ == "__main__":
    main()
