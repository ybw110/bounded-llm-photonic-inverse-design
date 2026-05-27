from __future__ import annotations


def plan_next_step(retrieval_results: list[dict], critic_feedback: dict | None, remaining_budget: int) -> dict:
    """一个不使用 LLM 的规则式 planner。

    这个模块很重要，因为它会成为 LLM planner 的强基线。
    如果 full LLM planner 不能稳定超过 heuristic planner，就说明 LLM 的独立贡献不足。

    输入：

    - retrieval_results：检索模块返回的候选列表
    - critic_feedback：numeric critic 对上一轮候选的反馈
    - remaining_budget：剩余评估预算

    输出是一个结构化 plan，字段尽量和 prompts/planner.md 中的 LLM 输出一致。
    这样后续替换成 LLM planner 时，外部流程不需要大改。
    """
    # 预算耗尽时直接失败停止。
    if remaining_budget <= 0:
        return {
            "next_action": "stop_fail",
            "reason": "No budget remains.",
            "budget_to_spend": 0,
        }

    # 如果 critic 已经接受候选，就成功停止。
    if critic_feedback and critic_feedback.get("passed"):
        return {
            "next_action": "stop_success",
            "reason": "Numeric critic accepted the candidate.",
            "budget_to_spend": 0,
        }

    # 如果 critic 指出的是可局部修正的谱线指标，比如中心频率或带宽，
    # 就围绕当前最佳候选做局部 refinement。
    if critic_feedback and critic_feedback.get("recommended_action") in {"shift_center_frequency", "adjust_bandwidth"}:
        return {
            "next_action": "local_refine",
            "reason": "The critic found a correctable spectral-spec mismatch.",
            "candidate_source": "best_retrieval",
            "search_radius": "medium",
            "budget_to_spend": min(10, remaining_budget),
        }

    # 如果 top retrieval score 较高，说明 memory bank 中已有接近目标的样本。
    # 此时直接围绕 top retrieval 做小范围局部搜索。
    if retrieval_results and retrieval_results[0]["score"] > 0.65:
        return {
            "next_action": "local_refine",
            "reason": "Top retrieval score is high enough to refine locally.",
            "candidate_source": "top_retrieval",
            "search_radius": "small",
            "budget_to_spend": min(8, remaining_budget),
        }

    # 如果检索证据较弱，就先扩大检索或切换记忆来源，而不是盲目优化。
    return {
        "next_action": "retrieve_more",
        "reason": "Current evidence is weak; broaden memory retrieval before optimization.",
        "candidate_source": "memory_bank",
        "search_radius": "large",
        "budget_to_spend": min(5, remaining_budget),
    }

