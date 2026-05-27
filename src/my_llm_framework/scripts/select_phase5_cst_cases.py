from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from my_llm_framework.config import load_config
from my_llm_framework.data.processed import load_processed_split
from my_llm_framework.evaluation.metrics import is_success, spectrum_metric_bundle
from my_llm_framework.features.physics_summary import amplitude
from my_llm_framework.scripts.evaluate_fewshot_retrieval_recovery import build_fewshot_memory_bank
from my_llm_framework.scripts.evaluate_retrieval_baselines import retrieve_for_method, thresholds_for_domain


DEFAULT_PLANNER_RESULTS = Path("outputs/phase4b_ood_planner_deepseek_500_v1/ood_recovery_planner_query_results.csv")
DEFAULT_QUERY_POOLS = [
    Path("outputs/phase4_fewshot_retrieval_methods_full_with_upper_v1/fewshot_retrieval_query_results.csv"),
    Path("outputs/phase4_fewshot_retrieval_target_h_extended_v1/fewshot_retrieval_query_results.csv"),
]


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value) or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _label_to_size(label: str) -> int:
    if str(label) == "full":
        return -1
    return int(float(str(label)))


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
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


def _parse_actions(value: Any) -> list[dict[str, Any]]:
    if pd.isna(value) or value == "":
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return [{"raw": str(value)}]
    return parsed if isinstance(parsed, list) else [parsed]


def _load_query_pool(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        df["pool_path"] = str(path)
        frames.append(df)
    pool = pd.concat(frames, ignore_index=True)
    pool["target_domain"] = pool["target_domain"].astype(str)
    pool["domain"] = pool["domain"].astype(str)
    pool["method"] = pool["method"].astype(str)
    pool["fewshot_label"] = pool["fewshot_label"].astype(str)
    pool["seed"] = pool["seed"].astype(int)
    pool["query_index"] = pool["query_index"].astype(int)
    return pool.drop_duplicates(
        subset=["target_domain", "seed", "query_index", "method", "fewshot_label"],
        keep="last",
    )


def _rank_or_none(value: Any) -> int | None:
    if pd.isna(value) or value == "":
        return None
    try:
        rank = int(float(value))
    except (TypeError, ValueError):
        return None
    return rank if rank > 0 else None


def _candidate_score(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    scored = df.copy()
    scored["_target_success"] = scored["final_target_domain_success_at_50"].map(_to_bool).astype(int)
    scored["_success"] = scored["final_success_at_50"].map(_to_bool).astype(int)
    scored["_oracle_match"] = scored["matched_oracle_choice"].map(_to_bool).astype(int)
    scored["_top1_target"] = scored["final_top1_is_target_domain"].map(_to_bool).astype(int)
    scored["_best_mse"] = pd.to_numeric(scored["final_best_amplitude_mse_at_budget"], errors="coerce").fillna(np.inf)
    scored["_top1_mse"] = pd.to_numeric(scored["final_top1_amplitude_mse"], errors="coerce").fillna(np.inf)
    scored["_fewshot_cost"] = pd.to_numeric(scored["final_fewshot_cost"], errors="coerce").fillna(np.inf)
    scored["_steps"] = pd.to_numeric(scored["planner_steps_used"], errors="coerce").fillna(np.inf)

    # For Target-H, prefer cases where the planner genuinely uses target-domain memory.
    if domain == "target_h":
        scored["_domain_priority"] = scored["final_critic_primary_reason"].astype(str).eq("already_successful").astype(int)
    else:
        scored["_domain_priority"] = 1

    return scored.sort_values(
        [
            "_target_success",
            "_success",
            "_oracle_match",
            "_top1_target",
            "_domain_priority",
            "_best_mse",
            "_fewshot_cost",
            "_steps",
            "query_index",
        ],
        ascending=[False, False, False, False, False, True, True, True, True],
        kind="stable",
    )


def select_planner_cases(
    planner_df: pd.DataFrame,
    *,
    target_h_cases: int,
    target_self_cases: int,
    target_h_invalid_source_cases: int,
    include_hard_failure: bool,
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for domain, count in [("target_h", target_h_cases), ("target_self", target_self_cases)]:
        domain_df = planner_df[planner_df["target_domain"].astype(str) == domain].copy()
        if domain_df.empty:
            continue
        domain_cases = _candidate_score(domain_df, domain).head(count).copy()
        domain_cases["_phase5_selection_group"] = f"{domain}_domain_valid"
        selected.append(domain_cases)

    if target_h_invalid_source_cases > 0:
        invalid = planner_df[
            (planner_df["target_domain"].astype(str) == "target_h")
            & (~planner_df["final_target_domain_success_at_50"].map(_to_bool))
            & (~planner_df["final_top1_is_target_domain"].map(_to_bool))
        ].copy()
        if not invalid.empty:
            invalid["_success"] = invalid["final_success_at_50"].map(_to_bool).astype(int)
            invalid["_best_mse"] = pd.to_numeric(invalid["final_best_amplitude_mse_at_budget"], errors="coerce").fillna(
                np.inf
            )
            invalid["_top1_mse"] = pd.to_numeric(invalid["final_top1_amplitude_mse"], errors="coerce").fillna(np.inf)
            invalid["_fewshot_cost"] = pd.to_numeric(invalid["final_fewshot_cost"], errors="coerce").fillna(np.inf)
            invalid["_steps"] = pd.to_numeric(invalid["planner_steps_used"], errors="coerce").fillna(np.inf)
            invalid_cases = invalid.sort_values(
                ["_success", "_best_mse", "_top1_mse", "_fewshot_cost", "_steps", "query_index"],
                ascending=[False, True, True, True, True, True],
                kind="stable",
            ).head(target_h_invalid_source_cases)
            invalid_cases = invalid_cases.copy()
            invalid_cases["_phase5_selection_group"] = "target_h_topology_invalid_source"
            selected.append(invalid_cases)

    if include_hard_failure:
        hard = planner_df[
            (planner_df["target_domain"].astype(str) == "target_h")
            & (~planner_df["final_target_domain_success_at_50"].map(_to_bool))
        ].copy()
        if not hard.empty:
            hard["_best_mse"] = pd.to_numeric(hard["final_best_amplitude_mse_at_budget"], errors="coerce").fillna(np.inf)
            hard_case = hard.sort_values(["_best_mse", "query_index"], kind="stable").head(1).copy()
            hard_case["_phase5_selection_group"] = "target_h_hard_failure_legacy"
            selected.append(hard_case)

    if not selected:
        raise ValueError("No Phase 5 cases selected. Check planner results and selection counts.")
    cases = pd.concat(selected, ignore_index=True)
    return cases.drop_duplicates(subset=["target_domain", "seed", "query_index"], keep="first").reset_index(drop=True)


def _slice_single_query(query_data: dict[str, Any], query_index: int) -> dict[str, Any]:
    sliced = dict(query_data)
    for key in ["params", "params_raw", "spectra", "summary_features", "source_indices"]:
        sliced[key] = sliced[key][query_index : query_index + 1]
    return sliced


def _param_columns(domain_cfg: dict[str, Any]) -> list[str]:
    active_dim = int(domain_cfg["active_param_dim"])
    return list(domain_cfg["param_names"])[:active_dim]


def _find_pool_row(pool: pd.DataFrame, planner_row: pd.Series) -> pd.Series:
    mask = (
        (pool["target_domain"].astype(str) == str(planner_row["target_domain"]))
        & (pool["seed"].astype(int) == int(planner_row["seed"]))
        & (pool["query_index"].astype(int) == int(planner_row["query_index"]))
        & (pool["method"].astype(str) == str(planner_row["final_method"]))
        & (pool["fewshot_label"].astype(str) == str(planner_row["final_fewshot_label"]))
    )
    match = pool[mask]
    if match.empty:
        raise ValueError(
            "Could not match planner final candidate in Phase 4 pool: "
            f"target={planner_row['target_domain']} seed={planner_row['seed']} "
            f"query={planner_row['query_index']} method={planner_row['final_method']} "
            f"fewshot={planner_row['final_fewshot_label']}"
        )
    return match.iloc[0]


def export_phase5_cases(
    cfg: dict[str, Any],
    planner_cases: pd.DataFrame,
    query_pool: pd.DataFrame,
    *,
    output_dir: Path,
    query_split: str,
    top_k: int,
    batch_size: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = output_dir / "cases"
    spectra_dir = output_dir / "spectra"
    cases_dir.mkdir(parents=True, exist_ok=True)
    spectra_dir.mkdir(parents=True, exist_ok=True)

    processed_root = Path(cfg["_processed_root_abs"])

    @lru_cache(maxsize=None)
    def load_query(domain: str) -> dict[str, Any]:
        return load_processed_split(processed_root, domain, query_split)

    @lru_cache(maxsize=None)
    def load_bank(target_domain: str, fewshot_label: str, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        return build_fewshot_memory_bank(
            cfg,
            processed_root,
            base_domains=["source"],
            target_domain=target_domain,
            fewshot_size=_label_to_size(fewshot_label),
            seed=seed,
        )

    candidate_rows: list[dict[str, Any]] = []
    for case_number, (_, planner_row) in enumerate(planner_cases.iterrows(), start=1):
        target_domain = str(planner_row["target_domain"])
        seed = int(planner_row["seed"])
        query_index = int(planner_row["query_index"])
        method = str(planner_row["final_method"])
        fewshot_label = str(planner_row["final_fewshot_label"])
        domain_cfg = cfg["domains"][target_domain]
        target_param_names = _param_columns(domain_cfg)
        thresholds = thresholds_for_domain(cfg, target_domain)

        pool_row = _find_pool_row(query_pool, planner_row)
        query_data = load_query(target_domain)
        single_query = _slice_single_query(query_data, query_index)
        bank, bank_meta = load_bank(target_domain, fewshot_label, seed)
        top_indices = retrieve_for_method(
            method=method,
            bank=bank,
            query_data=single_query,
            query_domain=target_domain,
            cfg=cfg,
            top_k=top_k,
            batch_size=batch_size,
        )[0]

        first_target_rank = _rank_or_none(pool_row.get("first_target_domain_success_rank"))
        first_success_rank = _rank_or_none(pool_row.get("first_success_rank"))
        if _to_bool(planner_row.get("final_target_domain_success_at_50")) and first_target_rank is not None:
            selected_rank = first_target_rank
            selected_rank_reason = "first_target_domain_success_rank"
        elif _to_bool(planner_row.get("final_success_at_50")) and first_success_rank is not None:
            selected_rank = first_success_rank
            selected_rank_reason = "first_success_rank"
        else:
            selected_rank = 1
            selected_rank_reason = "top1_or_near_miss"

        selected_rank = min(selected_rank, len(top_indices))
        bank_index = int(top_indices[selected_rank - 1])
        selected_bank_domain = str(bank["domains"][bank_index])
        candidate_domain_cfg = cfg["domains"][selected_bank_domain]
        param_names = _param_columns(candidate_domain_cfg)
        target_spectrum = query_data["spectra"][query_index]
        candidate_spectrum = bank["spectra"][bank_index]
        freq = query_data["frequency"]
        metrics = spectrum_metric_bundle(candidate_spectrum, target_spectrum, freq)
        success = is_success(metrics, thresholds)

        case_id = f"case_{case_number:03d}_{target_domain}_q{query_index:04d}_{method}_fs{fewshot_label}"
        spectra_path = spectra_dir / f"{case_id}_spectra.csv"
        spectra_df = pd.DataFrame(
            {
                "frequency_thz": freq,
                "target_real": np.real(target_spectrum),
                "target_imag": np.imag(target_spectrum),
                "target_amplitude": amplitude(target_spectrum),
                "candidate_real": np.real(candidate_spectrum),
                "candidate_imag": np.imag(candidate_spectrum),
                "candidate_amplitude": amplitude(candidate_spectrum),
            }
        )
        spectra_df.to_csv(spectra_path, index=False)

        params_norm = bank["params"][bank_index]
        params_raw = bank["params_raw"][bank_index]
        query_params_raw = query_data["params_raw"][query_index]
        row: dict[str, Any] = {
            "case_id": case_id,
            "phase5_selection_group": str(planner_row.get("_phase5_selection_group", "")),
            "target_domain": target_domain,
            "shift_type": str(planner_row.get("shift_type", domain_cfg.get("shift_type", ""))),
            "seed": seed,
            "query_index": query_index,
            "query_source_index": int(query_data["source_indices"][query_index]),
            "planner_policy": str(planner_row.get("planner_policy", "")),
            "planner_steps_used": int(_safe_float(planner_row.get("planner_steps_used"), 0)),
            "planner_stop_reason": str(planner_row.get("planner_stop_reason", "")),
            "planner_actions": str(planner_row.get("planner_actions", "")),
            "final_method": method,
            "final_fewshot_label": fewshot_label,
            "final_fewshot_cost": int(_safe_float(planner_row.get("final_fewshot_cost"), 0)),
            "selected_rank": selected_rank,
            "selected_rank_reason": selected_rank_reason,
            "selected_bank_index": bank_index,
            "selected_bank_domain": selected_bank_domain,
            "domain_valid": bool(selected_bank_domain == target_domain),
            "selected_bank_source_index": int(bank["source_indices"][bank_index]),
            "selected_success_recomputed": bool(success),
            "selected_amplitude_mse": float(metrics["amplitude_mse"]),
            "selected_amplitude_mae": float(metrics["amplitude_mae"]),
            "selected_center_frequency_error_thz": float(metrics["center_frequency_error_thz"]),
            "selected_bandwidth_error_thz": float(metrics["bandwidth_error_thz"]),
            "selected_suppression_depth_error_db": float(metrics["suppression_depth_error_db"]),
            "pool_top1_bank_index": int(pool_row["top1_bank_index"]),
            "pool_top1_bank_domain": str(pool_row["top1_bank_domain"]),
            "pool_top1_amplitude_mse": float(pool_row["top1_amplitude_mse"]),
            "pool_best_amplitude_mse_at_50": float(pool_row["best_amplitude_mse_at_max_budget"]),
            "spectra_csv": str(spectra_path),
            "case_json": str(cases_dir / f"{case_id}.json"),
            "pool_path": str(pool_row.get("pool_path", "")),
        }
        for idx, name in enumerate(param_names):
            row[f"{name}_norm"] = float(params_norm[idx])
            row[f"{name}_raw"] = float(params_raw[idx])
        for idx, name in enumerate(target_param_names):
            row[f"query_{target_domain}_{name}_raw"] = float(query_params_raw[idx])

        case_payload = {
            "case_id": case_id,
            "purpose": "Phase 5 CST case study candidate exported from frozen Phase 4 LLM-OOD planner evidence.",
            "target": {
                "domain": target_domain,
                "display_name": domain_cfg.get("display_name"),
                "shift_type": row["shift_type"],
                "query_index": query_index,
                "query_source_index": row["query_source_index"],
                "frequency_min_thz": domain_cfg.get("frequency_min_thz"),
                "frequency_max_thz": domain_cfg.get("frequency_max_thz"),
            },
            "planner_evidence": {
                "policy": row["planner_policy"],
                "steps_used": row["planner_steps_used"],
                "stop_reason": row["planner_stop_reason"],
                "actions": _parse_actions(planner_row.get("planner_actions")),
                "critic_primary_reason": str(planner_row.get("final_critic_primary_reason", "")),
                "critic_reasons": str(planner_row.get("final_critic_reasons", "")).split("|"),
                "matched_oracle_choice": _to_bool(planner_row.get("matched_oracle_choice")),
            },
            "retrieval_candidate": {
                "method": method,
                "fewshot_label": fewshot_label,
                "fewshot_cost": row["final_fewshot_cost"],
                "bank_meta": bank_meta,
                "selected_rank": selected_rank,
                "selected_rank_reason": selected_rank_reason,
                "selected_bank_index": bank_index,
                "selected_bank_domain": row["selected_bank_domain"],
                "domain_valid": row["domain_valid"],
                "selected_bank_source_index": row["selected_bank_source_index"],
            },
            "parameters": {
                "param_names": param_names,
                "candidate_norm": [float(params_norm[i]) for i in range(len(param_names))],
                "candidate_raw": [float(params_raw[i]) for i in range(len(param_names))],
                "candidate_domain": selected_bank_domain,
                "query_param_names": target_param_names,
                "query_raw": [float(query_params_raw[i]) for i in range(len(target_param_names))],
            },
            "surrogate_or_retrieval_metrics": {
                key: float(value) for key, value in metrics.items()
            },
            "files": {
                "spectra_csv": str(spectra_path),
                "pool_csv": str(pool_row.get("pool_path", "")),
            },
            "cst_status": "pending",
        }
        (cases_dir / f"{case_id}.json").write_text(
            json.dumps(_json_clean(case_payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        candidate_rows.append(row)

    candidates_df = pd.DataFrame(candidate_rows)
    candidates_df.to_csv(output_dir / "phase5_cst_candidates.csv", index=False)
    return candidates_df


def write_readme(output_dir: Path, candidates_df: pd.DataFrame, planner_results: Path, query_pools: list[Path]) -> None:
    selected_table = candidates_df[
        [
            "case_id",
            "phase5_selection_group",
            "target_domain",
            "final_method",
            "final_fewshot_label",
            "selected_rank",
            "selected_bank_domain",
            "domain_valid",
            "selected_amplitude_mse",
            "selected_center_frequency_error_thz",
            "selected_bandwidth_error_thz",
        ]
    ]
    lines = [
        "# Phase 5 CST Case Candidates",
        "",
        "本目录保存 Phase 5 CST case study 的候选设计。当前步骤只完成 A/B：从 Phase 4 LLM-OOD planner 结果中挑选代表性 query，并导出可交给 CST 的结构参数、目标谱线、候选谱线和 planner 轨迹。",
        "",
        "## Source files",
        "",
        f"- Planner results: `{planner_results}`",
    ]
    for path in query_pools:
        lines.append(f"- Phase 4 query pool: `{path}`")
    lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `phase5_cst_candidates.csv`: 所有候选 case 的汇总表，包含 domain、query、planner action、few-shot 设置、结构参数和重算的谱线误差。",
            "- `cases/*.json`: 每个 case 的完整证据包，供 CST 自动仿真脚本读取。",
            "- `spectra/*_spectra.csv`: 每个 case 的 target/candidate complex spectrum 和 amplitude curve。",
            "",
            "## Selected cases",
            "",
            "```text",
            selected_table.to_string(index=False),
            "```",
            "",
            "## Interpretation",
            "",
            "这些 case 用于论文中的 closed-loop CST validation。它们不是新的大规模统计结果，而是从已经冻结的 Phase 4 离线证据池中抽取的代表性物理验证样本。CST 完成后，应将真实仿真谱线回填到同一目录下的 `cst_results/`，再进行 surrogate/retrieval-vs-CST 的曲线和指标对比。",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_v1.json")
    parser.add_argument("--planner-results", type=Path, default=DEFAULT_PLANNER_RESULTS)
    parser.add_argument("--query-pools", nargs="+", type=Path, default=DEFAULT_QUERY_POOLS)
    parser.add_argument("--query-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase5_cst_case_study_v1"))
    parser.add_argument("--target-h-cases", type=int, default=2)
    parser.add_argument("--target-self-cases", type=int, default=2)
    parser.add_argument(
        "--target-h-invalid-source-cases",
        type=int,
        default=0,
        help=(
            "Select Target-H queries whose final planner candidate is spectrally useful but not target-domain valid. "
            "These source/cylinder near-misses support topology-invalid CST audit controls."
        ),
    )
    parser.add_argument("--include-hard-failure", action="store_true")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    cfg = load_config(args.config)
    planner_df = pd.read_csv(args.planner_results, low_memory=False)
    query_pool = _load_query_pool(args.query_pools)

    selected_cases = select_planner_cases(
        planner_df,
        target_h_cases=args.target_h_cases,
        target_self_cases=args.target_self_cases,
        target_h_invalid_source_cases=args.target_h_invalid_source_cases,
        include_hard_failure=args.include_hard_failure,
    )
    candidates_df = export_phase5_cases(
        cfg,
        selected_cases,
        query_pool,
        output_dir=args.output_dir,
        query_split=args.query_split,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )
    write_readme(args.output_dir, candidates_df, args.planner_results, args.query_pools)

    print(f"Saved Phase 5 CST candidates: {args.output_dir.resolve()}")
    print(candidates_df[["case_id", "target_domain", "final_fewshot_label", "selected_amplitude_mse"]].to_string(index=False))


if __name__ == "__main__":
    main()
