from __future__ import annotations

import argparse
from pathlib import Path

from my_llm_framework.planning.offline_evaluator import (
    ExpandAwareRulePlannerPolicy,
    OfflineRulePlannerPolicy,
    evaluate_offline_planner,
    load_thresholds_by_domain,
)
from my_llm_framework.planning.llm_policy import DeepSeekPlannerPolicy
from my_llm_framework.planning.llm_policy import ALL_PLANNER_ACTIONS
from my_llm_framework.planning.llm_policy import OFFLINE_EXPAND_EXECUTABLE_ACTIONS
from my_llm_framework.planning.llm_policy import OFFLINE_EXECUTABLE_ACTIONS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_v1.json")
    parser.add_argument("--retrieval-query-results", required=True, help="phase01_retrieval_baselines_v1/query_level_results.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget-limit", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=None, help="Optional per-domain smoke-test limit")
    parser.add_argument("--planner-policy", choices=["rule", "rule_expand", "deepseek"], default="rule")
    parser.add_argument("--deepseek-model", default="deepseek-chat")
    parser.add_argument("--deepseek-base-url", default="https://api.deepseek.com")
    parser.add_argument("--deepseek-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument(
        "--allow-exploratory-actions",
        action="store_true",
        help="Allow expand_top_seeds/switch_domain_prior even though the current offline pool cannot reveal them.",
    )
    parser.add_argument(
        "--enable-expand-top-seeds",
        action="store_true",
        help="Allow expand_top_seeds as a real action backed by expand PSO result pools.",
    )
    parser.add_argument("--source-pso", required=True, help="Source PSO query results CSV")
    parser.add_argument("--source-cmaes", required=True, help="Source CMA-ES query results CSV")
    parser.add_argument("--source-expand-pso", default=None, help="Source expanded top-seed PSO query results CSV")
    parser.add_argument("--target-h-pso", required=True, help="Target-H PSO query results CSV")
    parser.add_argument("--target-h-cmaes", required=True, help="Target-H CMA-ES query results CSV")
    parser.add_argument("--target-h-expand-pso", default=None, help="Target-H expanded top-seed PSO query results CSV")
    parser.add_argument("--target-self-pso", required=True, help="Target-Self PSO query results CSV")
    parser.add_argument("--target-self-cmaes", required=True, help="Target-Self CMA-ES query results CSV")
    parser.add_argument("--target-self-expand-pso", default=None, help="Target-Self expanded top-seed PSO query results CSV")
    args = parser.parse_args()

    thresholds_by_domain = load_thresholds_by_domain(args.config)
    if args.planner_policy == "deepseek":
        if args.allow_exploratory_actions:
            allowed_action_types = ALL_PLANNER_ACTIONS
        elif args.enable_expand_top_seeds:
            allowed_action_types = OFFLINE_EXPAND_EXECUTABLE_ACTIONS
        else:
            allowed_action_types = OFFLINE_EXECUTABLE_ACTIONS
        policy = DeepSeekPlannerPolicy(
            api_key_env=args.deepseek_api_key_env,
            model=args.deepseek_model,
            base_url=args.deepseek_base_url,
            allowed_action_types=allowed_action_types,
        )
    elif args.planner_policy == "rule_expand":
        policy = ExpandAwareRulePlannerPolicy()
    else:
        policy = OfflineRulePlannerPolicy()

    _, summary_df = evaluate_offline_planner(
        domain_specs=[
            (
                "source",
                Path(args.source_pso),
                Path(args.source_cmaes),
                Path(args.source_expand_pso) if args.source_expand_pso else None,
            ),
            (
                "target_h",
                Path(args.target_h_pso),
                Path(args.target_h_cmaes),
                Path(args.target_h_expand_pso) if args.target_h_expand_pso else None,
            ),
            (
                "target_self",
                Path(args.target_self_pso),
                Path(args.target_self_cmaes),
                Path(args.target_self_expand_pso) if args.target_self_expand_pso else None,
            ),
        ],
        retrieval_query_path=args.retrieval_query_results,
        thresholds_by_domain=thresholds_by_domain,
        output_dir=args.output_dir,
        policy=policy,
        budget_limit=args.budget_limit,
        max_queries=args.max_queries,
    )

    print(f"Saved Phase 3 offline planner replay: {Path(args.output_dir).resolve()}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
