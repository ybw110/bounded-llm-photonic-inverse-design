from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from my_llm_framework.config import load_config
from my_llm_framework.scripts.evaluate_fewshot_retrieval_recovery import (
    evaluate_fewshot_retrieval_recovery,
)


SUCCESS_BUDGET = 50
METHOD_RANK = {"domain_aware": 0, "raw_spectrum": 1, "physics_summary": 2}
SHIFT_TYPES = {
    "target_h": "topology_shift",
    "target_self": "material_frequency_shift",
}


@dataclass(frozen=True)
class CandidateKey:
    target_domain: str
    seed: int
    query_index: int
    method: str
    fewshot_label: str


def normalize_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def fewshot_cost(row: pd.Series | dict[str, Any]) -> int:
    label = normalize_label(row["fewshot_label"] if "fewshot_label" in row else row["final_fewshot_label"])
    if label == "full":
        return 10**9
    if label == "":
        return 0
    return int(float(label))


def format_tau(tau: float) -> str:
    return f"{tau:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def display_tau(tau: float) -> str:
    if abs(tau) < 0.01:
        return f"{tau:.0e}"
    return f"{tau:.2f}"


def set_amplitude_threshold(cfg: dict[str, Any], tau: float) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.setdefault("success_thresholds", {})["amplitude_mse"] = float(tau)
    by_domain = out.setdefault("success_thresholds_by_domain", {})
    for domain_cfg in by_domain.values():
        domain_cfg["amplitude_mse"] = float(tau)
    return out


def run_or_load_pool(
    *,
    base_cfg: dict[str, Any],
    processed_root: Path,
    output_root: Path,
    tau: float,
    reuse_existing: bool,
    max_queries_per_domain: int,
    target_domains: list[str],
) -> pd.DataFrame:
    tau_tag = format_tau(tau)
    domain_tag = "_".join(target_domains)
    pool_dir = output_root / f"pool_{domain_tag}_tau_{tau_tag}"
    legacy_pool_dir = output_root / f"pool_tau_{tau_tag}"
    query_path = pool_dir / "fewshot_retrieval_query_results.csv"
    legacy_query_path = legacy_pool_dir / "fewshot_retrieval_query_results.csv"
    config_dir = output_root / "threshold_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg_tau = set_amplitude_threshold(base_cfg, tau)
    config_path = config_dir / f"experiment_tau_{tau_tag}.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in cfg_tau.items() if not str(k).startswith("_")},
            f,
            indent=2,
            ensure_ascii=False,
        )

    if reuse_existing and query_path.exists():
        return pd.read_csv(query_path)
    if reuse_existing and set(target_domains) == {"target_h", "target_self"} and legacy_query_path.exists():
        return pd.read_csv(legacy_query_path)

    query_df, _, _ = evaluate_fewshot_retrieval_recovery(
        cfg_tau,
        processed_root,
        target_domains=target_domains,
        base_domains=["source"],
        query_split="test",
        methods=["raw_spectrum", "domain_aware"],
        fewshot_sizes=[0, 20, 50, 100, 200],
        seeds=[48],
        budgets=[SUCCESS_BUDGET],
        output_dir=pool_dir,
        batch_size=512,
        max_queries_per_domain=max_queries_per_domain,
        include_full_target=False,
    )
    return query_df


def candidate_sort_key(row: pd.Series) -> tuple[int, int, int, int, float, float, int]:
    return (
        int(bool_value(row[f"target_domain_success_at_{SUCCESS_BUDGET}"])),
        int(bool_value(row[f"success_at_{SUCCESS_BUDGET}"])),
        int(bool_value(row["top1_is_target_domain"])),
        -fewshot_cost(row),
        -float(pd.to_numeric(row["best_amplitude_mse_at_max_budget"], errors="coerce")),
        -float(pd.to_numeric(row["top1_amplitude_mse"], errors="coerce")),
        -METHOD_RANK.get(str(row["method"]), 99),
    )


def select_oracle(group: pd.DataFrame) -> pd.Series:
    scored = group.copy()
    scored["_target_success"] = scored[f"target_domain_success_at_{SUCCESS_BUDGET}"].map(bool_value).astype(int)
    scored["_success"] = scored[f"success_at_{SUCCESS_BUDGET}"].map(bool_value).astype(int)
    scored["_top1_target"] = scored["top1_is_target_domain"].map(bool_value).astype(int)
    scored["_cost"] = scored.apply(fewshot_cost, axis=1)
    scored["_best_mse"] = pd.to_numeric(scored["best_amplitude_mse_at_max_budget"], errors="coerce").fillna(np.inf)
    scored["_top1_mse"] = pd.to_numeric(scored["top1_amplitude_mse"], errors="coerce").fillna(np.inf)
    scored["_method_rank"] = scored["method"].map(METHOD_RANK).fillna(99)
    scored = scored.sort_values(
        ["_target_success", "_success", "_top1_target", "_cost", "_best_mse", "_top1_mse", "_method_rank"],
        ascending=[False, False, False, True, True, True, True],
        kind="stable",
    )
    return scored.iloc[0]


def make_lookup(pool: pd.DataFrame) -> dict[CandidateKey, pd.Series]:
    lookup: dict[CandidateKey, pd.Series] = {}
    for _, row in pool.iterrows():
        key = CandidateKey(
            target_domain=str(row["target_domain"]),
            seed=int(row["seed"]),
            query_index=int(row["query_index"]),
            method=str(row["method"]),
            fewshot_label=normalize_label(row["fewshot_label"]),
        )
        lookup[key] = row
    return lookup


def selected_candidate(
    *,
    group: pd.DataFrame,
    method: str,
    fewshot_label: str,
) -> pd.Series:
    fewshot_label = normalize_label(fewshot_label)
    match = group[(group["method"] == method) & (group["fewshot_label_norm"] == fewshot_label)]
    if match.empty:
        raise ValueError(
            f"Missing candidate method={method}, fewshot={fewshot_label} for "
            f"{group['target_domain'].iloc[0]} seed={group['seed'].iloc[0]} query={group['query_index'].iloc[0]}"
        )
    return match.iloc[0]


def row_from_candidate(policy: str, run_label: str, tau: float, candidate: pd.Series, oracle: pd.Series) -> dict[str, Any]:
    final_method = str(candidate["method"])
    final_label = normalize_label(candidate["fewshot_label"])
    oracle_method = str(oracle["method"])
    oracle_label = normalize_label(oracle["fewshot_label"])
    return {
        "tau": float(tau),
        "policy": policy,
        "run_label": run_label,
        "target_domain": str(candidate["target_domain"]),
        "shift_type": SHIFT_TYPES.get(str(candidate["target_domain"]), "ood_shift"),
        "seed": int(candidate["seed"]),
        "query_index": int(candidate["query_index"]),
        "final_method": final_method,
        "final_fewshot_label": final_label,
        "final_fewshot_cost": fewshot_cost(candidate),
        f"final_success_at_{SUCCESS_BUDGET}": bool_value(candidate[f"success_at_{SUCCESS_BUDGET}"]),
        f"final_target_domain_success_at_{SUCCESS_BUDGET}": bool_value(
            candidate[f"target_domain_success_at_{SUCCESS_BUDGET}"]
        ),
        "final_top1_is_target_domain": bool_value(candidate["top1_is_target_domain"]),
        "final_top1_amplitude_mse": float(pd.to_numeric(candidate["top1_amplitude_mse"], errors="coerce")),
        "final_best_amplitude_mse_at_budget": float(
            pd.to_numeric(candidate["best_amplitude_mse_at_max_budget"], errors="coerce")
        ),
        "oracle_method": oracle_method,
        "oracle_fewshot_label": oracle_label,
        "oracle_fewshot_cost": fewshot_cost(oracle),
        f"oracle_success_at_{SUCCESS_BUDGET}": bool_value(oracle[f"success_at_{SUCCESS_BUDGET}"]),
        f"oracle_target_domain_success_at_{SUCCESS_BUDGET}": bool_value(
            oracle[f"target_domain_success_at_{SUCCESS_BUDGET}"]
        ),
        "matched_oracle_choice": final_method == oracle_method and final_label == oracle_label,
    }


def evaluate_fixed_policies(pool: pd.DataFrame, tau: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    policy_specs = {
        "source_only_raw0": {"target_h": ("raw_spectrum", "0"), "target_self": ("raw_spectrum", "0")},
        "fixed_H50_self20": {"target_h": ("domain_aware", "50"), "target_self": ("domain_aware", "20")},
        "fixed_H100_self20": {"target_h": ("domain_aware", "100"), "target_self": ("domain_aware", "20")},
        "fixed_H200_self20": {"target_h": ("domain_aware", "200"), "target_self": ("domain_aware", "20")},
    }
    for _, group in pool.groupby(["target_domain", "seed", "query_index"], sort=False):
        oracle = select_oracle(group)
        target_domain = str(group["target_domain"].iloc[0])
        for policy, by_domain in policy_specs.items():
            method, label = by_domain[target_domain]
            candidate = selected_candidate(group=group, method=method, fewshot_label=label)
            rows.append(row_from_candidate(policy, "deterministic", tau, candidate, oracle))

        oracle_candidate = oracle
        rows.append(row_from_candidate("action_space_oracle", "deterministic", tau, oracle_candidate, oracle))
        rows.append(row_from_candidate("preview_greedy_cost_heuristic", "deterministic", tau, oracle_candidate, oracle))
    return rows


def evaluate_existing_llm_runs(
    *,
    pool: pd.DataFrame,
    tau: float,
    project_root: Path,
) -> list[dict[str, Any]]:
    lookup = make_lookup(pool)
    oracle_by_query: dict[tuple[str, int, int], pd.Series] = {}
    for key, group in pool.groupby(["target_domain", "seed", "query_index"], sort=False):
        oracle_by_query[(str(key[0]), int(key[1]), int(key[2]))] = select_oracle(group)

    run_dirs = [
        ("deepseek_chat", "repeat1", "phase4b_ood_planner_deepseek_500_repeat1_v1"),
        ("deepseek_chat", "repeat2", "phase4b_ood_planner_deepseek_500_repeat2_v1"),
        ("deepseek_chat", "repeat3", "phase4b_ood_planner_deepseek_500_repeat3_v1"),
        ("glm_5_1", "repeat1", "phase4b_ood_planner_glm51_500_repeat1_v1"),
        ("glm_5_1", "repeat2", "phase4b_ood_planner_glm51_500_repeat2_v1"),
        ("glm_5_1", "repeat3", "phase4b_ood_planner_glm51_500_repeat3_v1"),
    ]
    rows: list[dict[str, Any]] = []
    available_domains = set(str(domain) for domain in pool["target_domain"].unique())
    for policy, run_label, dirname in run_dirs:
        path = project_root / "outputs" / dirname / "ood_recovery_planner_query_results.csv"
        if not path.exists():
            print(f"[warn] missing LLM replay file: {path}")
            continue
        decisions = pd.read_csv(path)
        decisions = decisions[decisions["target_domain"].isin(available_domains)]
        for _, decision in decisions.iterrows():
            target_domain = str(decision["target_domain"])
            seed = int(decision["seed"])
            query_index = int(decision["query_index"])
            method = str(decision["final_method"])
            label = normalize_label(decision["final_fewshot_label"])
            key = CandidateKey(target_domain, seed, query_index, method, label)
            candidate = lookup.get(key)
            if candidate is None:
                raise ValueError(f"Missing candidate for existing LLM decision: {key}")
            oracle = oracle_by_query[(target_domain, seed, query_index)]
            rows.append(row_from_candidate(policy, run_label, tau, candidate, oracle))
            rows[-1]["planner_used_fallback"] = bool_value(decision.get("planner_used_fallback", False))
            rows[-1]["planner_steps_used"] = float(pd.to_numeric(decision.get("planner_steps_used", 0), errors="coerce"))
    return rows


def summarize_by_run(query_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (tau, policy, run_label, target_domain), group in query_rows.groupby(
        ["tau", "policy", "run_label", "target_domain"], sort=False
    ):
        row = {
            "tau": float(tau),
            "policy": policy,
            "run_label": run_label,
            "target_domain": target_domain,
            "shift_type": str(group["shift_type"].iloc[0]),
            "num_replays": int(len(group)),
            f"success_at_{SUCCESS_BUDGET}": float(group[f"final_success_at_{SUCCESS_BUDGET}"].astype(bool).mean()),
            f"target_domain_success_at_{SUCCESS_BUDGET}": float(
                group[f"final_target_domain_success_at_{SUCCESS_BUDGET}"].astype(bool).mean()
            ),
            "top1_target_domain_rate": float(group["final_top1_is_target_domain"].astype(bool).mean()),
            "mean_final_top1_amplitude_mse": float(
                pd.to_numeric(group["final_top1_amplitude_mse"], errors="coerce").mean()
            ),
            "mean_final_best_amplitude_mse_at_budget": float(
                pd.to_numeric(group["final_best_amplitude_mse_at_budget"], errors="coerce").mean()
            ),
            "mean_fewshot_cost": float(pd.to_numeric(group["final_fewshot_cost"], errors="coerce").mean()),
            "oracle_match_rate": float(group["matched_oracle_choice"].astype(bool).mean()),
            f"oracle_target_domain_success_at_{SUCCESS_BUDGET}": float(
                group[f"oracle_target_domain_success_at_{SUCCESS_BUDGET}"].astype(bool).mean()
            ),
            "oracle_mean_fewshot_cost": float(pd.to_numeric(group["oracle_fewshot_cost"], errors="coerce").mean()),
        }
        if "planner_used_fallback" in group:
            fallback_values = group["planner_used_fallback"].map(bool_value)
            row["fallback_or_invalid_action_rate"] = float(fallback_values.mean())
        else:
            row["fallback_or_invalid_action_rate"] = 0.0
        if "planner_steps_used" in group:
            steps = pd.to_numeric(group["planner_steps_used"], errors="coerce").fillna(0)
            row["mean_steps_used"] = float(steps.mean())
        else:
            row["mean_steps_used"] = 0.0
        row[f"oracle_gap_target_domain_success_at_{SUCCESS_BUDGET}"] = (
            row[f"oracle_target_domain_success_at_{SUCCESS_BUDGET}"]
            - row[f"target_domain_success_at_{SUCCESS_BUDGET}"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_across_runs(run_summary: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        f"success_at_{SUCCESS_BUDGET}",
        f"target_domain_success_at_{SUCCESS_BUDGET}",
        "top1_target_domain_rate",
        "mean_final_top1_amplitude_mse",
        "mean_final_best_amplitude_mse_at_budget",
        "mean_fewshot_cost",
        "oracle_match_rate",
        f"oracle_target_domain_success_at_{SUCCESS_BUDGET}",
        "oracle_mean_fewshot_cost",
        f"oracle_gap_target_domain_success_at_{SUCCESS_BUDGET}",
        "fallback_or_invalid_action_rate",
        "mean_steps_used",
    ]
    rows: list[dict[str, Any]] = []
    for (tau, policy, target_domain), group in run_summary.groupby(["tau", "policy", "target_domain"], sort=False):
        row: dict[str, Any] = {
            "tau": float(tau),
            "policy": policy,
            "target_domain": target_domain,
            "shift_type": str(group["shift_type"].iloc[0]),
            "num_runs": int(group["run_label"].nunique()),
            "num_replays_per_run": int(group["num_replays"].iloc[0]),
        }
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown_summary(summary: pd.DataFrame, output_path: Path) -> None:
    order = [
        "source_only_raw0",
        "fixed_H50_self20",
        "fixed_H100_self20",
        "fixed_H200_self20",
        "deepseek_chat",
        "glm_5_1",
        "preview_greedy_cost_heuristic",
        "action_space_oracle",
    ]
    lines = [
        "# NC threshold and policy sensitivity",
        "",
        "Success is re-scored at amplitude-MSE thresholds tau in {"
        + ", ".join(display_tau(float(tau)) for tau in sorted(summary["tau"].unique()))
        + "}; all other domain-specific success constraints are unchanged.",
        "The existing DeepSeek and GLM planner decisions are not regenerated. Their selected actions are replayed against each threshold-specific candidate pool.",
        "",
    ]
    for domain in ["target_h", "target_self"]:
        display = summary[summary["target_domain"] == domain].copy()
        if display.empty:
            continue
        lines.append(f"## {domain}")
        display["policy_order"] = display["policy"].map({p: i for i, p in enumerate(order)}).fillna(99)
        display = display.sort_values(["policy_order", "tau"])
        pivot = display.pivot_table(
            index="policy",
            columns="tau",
            values=f"target_domain_success_at_{SUCCESS_BUDGET}_mean",
            aggfunc="first",
        )
        pivot = pivot.reindex([p for p in order if p in pivot.index])
        header = ["Policy"] + [display_tau(float(tau)) for tau in pivot.columns]
        rows = []
        for policy, row in pivot.iterrows():
            rows.append([policy] + [f"{float(value):.3f}" if pd.notna(value) else "--" for value in row])
        widths = [max(len(str(item)) for item in col) for col in zip(header, *rows)]
        lines.append("| " + " | ".join(str(item).ljust(width) for item, width in zip(header, widths)) + " |")
        lines.append("| " + " | ".join("-" * width for width in widths) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(item).ljust(width) for item, width in zip(row, widths)) + " |")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_table(summary: pd.DataFrame, output_path: Path) -> None:
    order = [
        ("source_only_raw0", "Source only"),
        ("fixed_H50_self20", "Fixed H-50/Self-20"),
        ("fixed_H100_self20", "Fixed H-100/Self-20"),
        ("fixed_H200_self20", "Fixed H-200/Self-20"),
        ("deepseek_chat", "DeepSeek"),
        ("glm_5_1", "GLM-5.1"),
        ("preview_greedy_cost_heuristic", "Preview-greedy control"),
        ("action_space_oracle", "Action-space oracle"),
    ]
    tau_values = sorted(summary["tau"].unique())
    lines = [
        "\\begin{table}[t]",
        "\\caption{Threshold sensitivity of NC policy re-scoring. Entries are target-domain Success@50; the amplitude-MSE threshold $\\tau$ is varied while other domain-specific success constraints are kept fixed. Existing LLM decisions are re-scored rather than regenerated.}",
        "\\label{tab:si_nc_threshold_sensitivity}",
        "\\centering",
        "\\begin{tabular}{ll" + "r" * len(tau_values) + "r}",
        "\\toprule",
        "Domain & Policy & "
        + " & ".join([f"$\\tau={tau:.2f}$" for tau in tau_values])
        + " & Mean target samples \\\\",
        "\\midrule",
    ]
    for domain in ["target_h", "target_self"]:
        domain_df = summary[summary["target_domain"] == domain]
        for policy, label in order:
            policy_df = domain_df[domain_df["policy"] == policy]
            if policy_df.empty:
                continue
            values = []
            for tau in tau_values:
                row = policy_df[np.isclose(policy_df["tau"], tau)]
                if row.empty:
                    values.append("--")
                else:
                    values.append(f"{float(row[f'target_domain_success_at_{SUCCESS_BUDGET}_mean'].iloc[0]):.3f}")
            base = policy_df[np.isclose(policy_df["tau"], 0.03)]
            if base.empty:
                cost = "--"
            else:
                cost = f"{float(base['mean_fewshot_cost_mean'].iloc[0]):.1f}"
            domain_label = "Target-H" if domain == "target_h" else "Target-Self"
            lines.append(f"{domain_label} & {label} & " + " & ".join(values) + f" & {cost} \\\\")
        lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_v1.json")
    parser.add_argument("--output-dir", default="outputs/phase4_threshold_sensitivity_nc_v1")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.01, 0.02, 0.03, 0.05])
    parser.add_argument("--target-domains", nargs="+", default=["target_h", "target_self"])
    parser.add_argument("--max-queries-per-domain", type=int, default=500)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(cfg["_project_root"])
    processed_root = Path(cfg["_processed_root_abs"])
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for tau in args.thresholds:
        pool = run_or_load_pool(
            base_cfg=cfg,
            processed_root=processed_root,
            output_root=output_root,
            tau=tau,
            reuse_existing=args.reuse_existing,
            max_queries_per_domain=args.max_queries_per_domain,
            target_domains=args.target_domains,
        )
        pool = pool.copy()
        pool["fewshot_label_norm"] = pool["fewshot_label"].map(normalize_label)
        all_rows.extend(evaluate_fixed_policies(pool, tau))
        all_rows.extend(evaluate_existing_llm_runs(pool=pool, tau=tau, project_root=project_root))

    query_rows = pd.DataFrame(all_rows)
    run_summary = summarize_by_run(query_rows)
    summary = summarize_across_runs(run_summary)

    query_rows.to_csv(output_root / "threshold_policy_query_results.csv", index=False)
    run_summary.to_csv(output_root / "threshold_policy_summary_by_run.csv", index=False)
    summary.to_csv(output_root / "threshold_policy_summary.csv", index=False)
    write_markdown_summary(summary, output_root / "threshold_policy_summary.md")
    write_latex_table(summary, output_root / "threshold_policy_sensitivity_table.tex")

    print(f"Saved NC threshold and policy sensitivity outputs to {output_root.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
