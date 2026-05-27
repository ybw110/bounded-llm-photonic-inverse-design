from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from my_llm_framework.planning.offline_evaluator import OfflineRulePlannerPolicy
from my_llm_framework.schemas import PlannerAction, PlannerActionType, PlannerObservation


OFFLINE_EXECUTABLE_ACTIONS = (
    PlannerActionType.STOP_WITH_RETRIEVAL.value,
    PlannerActionType.RUN_PSO.value,
    PlannerActionType.RUN_CMAES.value,
    PlannerActionType.STOP_WITH_BEST_MSE.value,
)

OFFLINE_EXPAND_EXECUTABLE_ACTIONS = (
    PlannerActionType.STOP_WITH_RETRIEVAL.value,
    PlannerActionType.RUN_PSO.value,
    PlannerActionType.RUN_CMAES.value,
    PlannerActionType.EXPAND_TOP_SEEDS.value,
    PlannerActionType.STOP_WITH_BEST_MSE.value,
)

ALL_PLANNER_ACTIONS = tuple(action.value for action in PlannerActionType)


class DeepSeekPlannerPolicy:
    """LLM planner policy using DeepSeek's OpenAI-compatible chat API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "DEEPSEEK_API_KEY",
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_seconds: float = 60.0,
        fallback_policy: OfflineRulePlannerPolicy | None = None,
        allowed_action_types: tuple[str, ...] = OFFLINE_EXECUTABLE_ACTIONS,
    ) -> None:
        self.api_key = api_key or os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(f"Missing DeepSeek API key. Set {api_key_env}.")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.fallback_policy = fallback_policy or OfflineRulePlannerPolicy()
        self.allowed_action_types = tuple(allowed_action_types)
        self.name = f"deepseek_{model}_planner"

    def decide(self, observation: PlannerObservation) -> PlannerAction:
        try:
            payload = self._chat_payload(observation)
            content = self._post_chat_completion(payload)
            action = PlannerAction.from_dict(_loads_json_object(content))
            if action.action_type not in self.allowed_action_types:
                raise ValueError(f"LLM returned non-executable action_type={action.action_type}")
            if action.action_type in {
                PlannerActionType.STOP_WITH_RETRIEVAL.value,
                PlannerActionType.STOP_WITH_BEST_MSE.value,
                PlannerActionType.STOP_FAIL.value,
            }:
                action.budget_to_spend = 0
            action.metadata.setdefault("llm_model", self.model)
            action.metadata.setdefault("allowed_action_types", list(self.allowed_action_types))
            return action
        except Exception as exc:
            action = self.fallback_policy.decide(observation)
            action.metadata.update(
                {
                    "llm_model": self.model,
                    "fallback_policy": self.fallback_policy.name,
                    "fallback_error": f"{type(exc).__name__}: {exc}",
                }
            )
            return action

    def _chat_payload(self, observation: PlannerObservation) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _system_prompt(self.allowed_action_types)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": "Choose the next planner action as JSON.",
                            "observation": _compact_observation(observation),
                        },
                        ensure_ascii=True,
                    ),
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
        }

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
            raise RuntimeError(f"DeepSeek API HTTP {exc.code}: {body[:500]}") from exc
        response_json = json.loads(body)
        return str(response_json["choices"][0]["message"]["content"])


def _system_prompt(allowed_action_types: tuple[str, ...]) -> str:
    return (
        "You are a numeric evidence planner for offline metasurface inverse-design replay. "
        "Return exactly one valid json object and no extra prose. "
        "The response must be json. "
        "Allowed action_type values are exactly: "
        f"{list(allowed_action_types)}. "
        "Do not use any action_type outside this list. "
        "The offline pool can reveal retrieval_top1, pso_final, cmaes_final, and optionally expand_top_seeds_final. "
        "Use stop_with_retrieval only when retrieval_top1 is successful. "
        "Use run_pso only when pso_final is not yet revealed and retrieval_top1 failed. "
        "Use expand_top_seeds only when it is allowed, regular pso_final is revealed and failed, "
        "and expand_top_seeds_final is not yet revealed. "
        "Use run_cmaes only when cmaes_final is not yet revealed and previous revealed candidates failed. "
        "Use stop_with_best_mse when all useful revealed candidates are exhausted or budget is gone. "
        "The json object must match this schema: "
        '{"action_type":"run_pso","reason":"short reason","budget_to_spend":1,'
        '"candidate_source":"raw_spectrum_top1","top_k":null,"domain_prior":null,"metadata":{}}'
    )


def _compact_observation(observation: PlannerObservation) -> dict[str, Any]:
    payload = observation.to_dict()
    payload["history"] = [
        {
            "action_type": item.get("action_type"),
            "reason": item.get("reason"),
            "budget_to_spend": item.get("budget_to_spend"),
            "candidate_source": item.get("candidate_source"),
        }
        for item in payload.get("history", [])
    ]
    return payload


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
