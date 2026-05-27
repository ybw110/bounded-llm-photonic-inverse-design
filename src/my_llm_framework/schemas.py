from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


# 这个文件只放“数据结构定义”，不放具体算法。
# 这样做的好处是：以后你看代码时，可以先看这里，知道整个框架里
# 常见对象长什么样，再去看 data / retrieval / planning 等模块。


@dataclass(frozen=True)
class DomainSpec:
    """描述一个数据域或设计域。

    例如：

    - source：旧 LPR 工作中的源域 cylinder，30-60 THz
    - target_h：H-shaped，代表 topology shift
    - target_self：自构 cylinder/self 域，代表 material-frequency shift

    这些信息既用于数据读取，也用于 domain-aware retrieval。
    """

    name: str
    display_name: str
    role: str
    topology: str
    shift_type: str
    frequency_min_thz: float
    frequency_max_thz: float
    raw_points: int
    param_dim: int
    active_param_dim: int
    param_names: list[str]


@dataclass
class DesignSpec:
    """一个被结构化后的设计需求。

    未来 LLM spec parser 的任务，就是把自然语言需求转换成这个对象。
    例如用户说“设计一个 H 型 45 THz 附近的带阻滤波器”，最终应该被解析为：

    - target_domain = "target_h"
    - objective = "bandstop_filter"
    - constraints = {"center_frequency_thz": 45.0, ...}
    - budget = 50
    """

    target_domain: str
    objective: str
    budget: int
    constraints: dict[str, Any] = field(default_factory=dict)
    success_thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class SpectrumRecord:
    """memory bank 中的一条历史样本。

    每条记录包含：

    - 属于哪个 domain
    - 结构参数
    - 复数谱线
    - 对应频率网格
    - 由谱线压缩得到的 physics summary
    """

    domain: str
    params: np.ndarray
    spectrum: np.ndarray
    frequency: np.ndarray
    summary: dict[str, float]
    source_index: int | None = None


@dataclass
class RetrievalResult:
    """一次检索返回的候选结果。

    total score 被拆成三部分，是为了论文里能解释清楚：

    - summary_score：物理摘要相似度
    - domain_score：域相似度
    - constraint_score：约束兼容性，当前只是预留接口
    """

    index: int
    domain: str
    score: float
    summary_score: float
    domain_score: float
    constraint_score: float
    params: np.ndarray
    summary: dict[str, float]


@dataclass
class Candidate:
    """设计流程中的候选解。

    检索、逆向模型、优化器、LLM planner 都可能产生 Candidate。
    这里统一格式，方便后续评测和 critic 使用。
    """

    params: np.ndarray
    domain: str
    source: str
    spectrum: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Critique:
    """numeric critic 对候选解的评价结果。"""

    passed: bool
    metrics: dict[str, float]
    failure_modes: list[str]
    recommended_action: str
    primary_failure_mode: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "metrics": self.metrics,
            "failure_modes": self.failure_modes,
            "recommended_action": self.recommended_action,
            "primary_failure_mode": self.primary_failure_mode,
            "notes": self.notes,
        }


class PlannerActionType(str, Enum):
    """Actions that an offline or LLM planner is allowed to emit."""

    STOP_WITH_RETRIEVAL = "stop_with_retrieval"
    RUN_PSO = "run_pso"
    RUN_CMAES = "run_cmaes"
    EXPAND_TOP_SEEDS = "expand_top_seeds"
    SWITCH_DOMAIN_PRIOR = "switch_domain_prior"
    STOP_WITH_BEST_MSE = "stop_with_best_mse"
    STOP_FAIL = "stop_fail"


class CandidateMethod(str, Enum):
    """Candidate sources available in the Phase 3 offline result pool."""

    RAW_SPECTRUM_TOP1 = "raw_spectrum_top1"
    RETRIEVAL_PNN_PSO = "retrieval_pnn_pso"
    RETRIEVAL_PNN_CMAES = "retrieval_pnn_cmaes"
    RETRIEVAL_PNN_PSO_EXPAND_TOP_SEEDS = "retrieval_pnn_pso_expand_top_seeds"


@dataclass
class PlannerAction:
    """Structured planner decision.

    This is API-neutral: a rule planner can construct it directly, while a
    future LLM planner can be asked to emit the same shape as JSON.
    """

    action_type: str
    reason: str
    budget_to_spend: int = 0
    candidate_source: str | None = None
    top_k: int | None = None
    domain_prior: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        PlannerActionType(self.action_type)
        if self.budget_to_spend < 0:
            raise ValueError("budget_to_spend must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "reason": self.reason,
            "budget_to_spend": self.budget_to_spend,
            "candidate_source": self.candidate_source,
            "top_k": self.top_k,
            "domain_prior": self.domain_prior,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlannerAction":
        return cls(
            action_type=str(payload["action_type"]),
            reason=str(payload.get("reason", "")),
            budget_to_spend=int(payload.get("budget_to_spend", 0)),
            candidate_source=payload.get("candidate_source"),
            top_k=payload.get("top_k"),
            domain_prior=payload.get("domain_prior"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class PlannerCandidateObservation:
    """One candidate result exposed to the planner."""

    method: str
    available: bool
    success: bool
    amplitude_mse: float
    critic: Critique
    metrics: dict[str, float] = field(default_factory=dict)
    budget_cost: int = 1

    def __post_init__(self) -> None:
        CandidateMethod(self.method)
        if self.budget_cost < 0:
            raise ValueError("budget_cost must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "available": self.available,
            "success": self.success,
            "amplitude_mse": self.amplitude_mse,
            "metrics": self.metrics,
            "critic": self.critic.to_dict(),
            "budget_cost": self.budget_cost,
        }


@dataclass
class PlannerObservation:
    """Compact state shown to a planner at one query replay step."""

    domain: str
    query_index: int
    source_index: int | None
    budget_used: int
    budget_limit: int
    retrieval_top1: PlannerCandidateObservation
    pso_final: PlannerCandidateObservation | None = None
    cmaes_final: PlannerCandidateObservation | None = None
    expand_top_seeds_final: PlannerCandidateObservation | None = None
    current_best_method: str | None = None
    current_best_amplitude_mse: float | None = None
    current_best_success: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def remaining_budget(self) -> int:
        return max(0, self.budget_limit - self.budget_used)

    def available_candidates(self) -> list[PlannerCandidateObservation]:
        candidates = [self.retrieval_top1, self.pso_final, self.cmaes_final, self.expand_top_seeds_final]
        return [candidate for candidate in candidates if candidate is not None and candidate.available]

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "query_index": self.query_index,
            "source_index": self.source_index,
            "budget_used": self.budget_used,
            "budget_limit": self.budget_limit,
            "remaining_budget": self.remaining_budget,
            "retrieval_top1": self.retrieval_top1.to_dict(),
            "pso_final": self.pso_final.to_dict() if self.pso_final else None,
            "cmaes_final": self.cmaes_final.to_dict() if self.cmaes_final else None,
            "expand_top_seeds_final": self.expand_top_seeds_final.to_dict() if self.expand_top_seeds_final else None,
            "current_best_method": self.current_best_method,
            "current_best_amplitude_mse": self.current_best_amplitude_mse,
            "current_best_success": self.current_best_success,
            "history": list(self.history),
        }
