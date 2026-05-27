from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from my_llm_framework.config import load_config
from my_llm_framework.data.processed import load_processed_split
from my_llm_framework.scripts.evaluate_retrieval_baselines import (
    evaluate_topk,
    retrieve_for_method,
    thresholds_for_domain,
)


METHODS = ("raw_spectrum", "physics_summary", "domain_aware")


def _empty_bank_like(summary_names: list[str]) -> dict[str, Any]:
    return {
        "domains": [],
        "params": [],
        "params_raw": [],
        "spectra": [],
        "summary_features": [],
        "summary_names": summary_names,
        "source_indices": [],
    }


def _append_split_to_bank(bank: dict[str, Any], domain: str, split_data: dict[str, Any], indices: np.ndarray | None = None) -> None:
    if indices is None:
        indices = np.arange(len(split_data["params"]))
    if len(indices) == 0:
        return

    bank["domains"].extend([domain] * len(indices))
    bank["params"].append(split_data["params"][indices])
    bank["params_raw"].append(split_data["params_raw"][indices])
    bank["spectra"].append(split_data["spectra"][indices])
    bank["summary_features"].append(split_data["summary_features"][indices])
    bank["source_indices"].append(split_data["source_indices"][indices])


def _finalize_bank(bank: dict[str, Any]) -> dict[str, Any]:
    if not bank["domains"]:
        raise ValueError("memory bank is empty")
    return {
        "domains": np.asarray(bank["domains"]).astype(str),
        "params": np.vstack(bank["params"]).astype(np.float32),
        "params_raw": np.vstack(bank["params_raw"]).astype(np.float32),
        "spectra": np.vstack(bank["spectra"]).astype(np.complex64),
        "summary_features": np.vstack(bank["summary_features"]).astype(np.float32),
        "summary_names": bank["summary_names"],
        "source_indices": np.concatenate(bank["source_indices"]).astype(np.int64),
    }


def build_fewshot_memory_bank(
    cfg: dict,
    processed_root: Path,
    *,
    base_domains: list[str],
    target_domain: str,
    fewshot_size: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build source/base memory plus k target-domain train samples.

    `fewshot_size=0` is the zero-shot setting. `fewshot_size=-1` means all
    target-domain train samples and is useful as an upper reference.
    """
    if target_domain in base_domains:
        raise ValueError(
            f"target_domain={target_domain} is already included in base_domains={base_domains}. "
            "For OOD few-shot recovery, keep the target domain out of the base memory."
        )

    first_split = load_processed_split(processed_root, base_domains[0], "train")
    bank = _empty_bank_like(summary_names=first_split["summary_names"])

    for base_domain in base_domains:
        split_data = first_split if base_domain == base_domains[0] else load_processed_split(processed_root, base_domain, "train")
        _append_split_to_bank(bank, base_domain, split_data)

    target_train = load_processed_split(processed_root, target_domain, "train")
    target_train_count = len(target_train["params"])
    rng = np.random.default_rng(seed)

    if fewshot_size < 0:
        sampled_indices = np.arange(target_train_count)
        effective_fewshot_size = target_train_count
        fewshot_label = "full"
    else:
        effective_fewshot_size = min(int(fewshot_size), target_train_count)
        sampled_indices = (
            rng.choice(target_train_count, size=effective_fewshot_size, replace=False)
            if effective_fewshot_size > 0
            else np.asarray([], dtype=np.int64)
        )
        sampled_indices = np.sort(sampled_indices.astype(np.int64))
        fewshot_label = str(int(fewshot_size))

    _append_split_to_bank(bank, target_domain, target_train, sampled_indices)
    final_bank = _finalize_bank(bank)
    meta = {
        "base_domains": ",".join(base_domains),
        "target_domain": target_domain,
        "fewshot_label": fewshot_label,
        "requested_fewshot_size": int(fewshot_size),
        "effective_fewshot_size": int(effective_fewshot_size),
        "target_train_count": int(target_train_count),
        "memory_size": int(len(final_bank["domains"])),
        "seed": int(seed),
    }
    return final_bank, meta


def _slice_query_data(query_data: dict[str, Any], max_queries: int | None) -> dict[str, Any]:
    if max_queries is None:
        return query_data
    sliced = dict(query_data)
    for key in ["params", "params_raw", "spectra", "summary_features", "source_indices"]:
        sliced[key] = sliced[key][:max_queries]
    return sliced


def _add_zero_shot_gains(
    query_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    budgets: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_budget = max(budgets)
    query_df = query_df.copy()
    summary_df = summary_df.copy()

    query_df[f"zero_shot_success_at_{max_budget}"] = False
    query_df[f"zero_shot_failed_recovered_at_{max_budget}"] = False

    summary_extra: list[dict[str, Any]] = []

    group_cols = ["target_domain", "method", "seed"]
    for group_key, group in query_df.groupby(group_cols, sort=False):
        target_domain, method, seed = group_key
        group = group.reset_index(names="_orig_index")
        zero = group[group["fewshot_label"] == "0"][["query_index", f"success_at_{max_budget}"]].rename(
            columns={f"success_at_{max_budget}": f"zero_shot_success_at_{max_budget}"}
        )
        if zero.empty:
            continue

        merged = group.merge(zero, on="query_index", how="left", suffixes=("", "_zero"))
        zero_success = merged[f"zero_shot_success_at_{max_budget}_zero"].fillna(False).astype(bool)
        recovered = (~zero_success) & merged[f"success_at_{max_budget}"].astype(bool)

        query_df.loc[merged["_orig_index"], f"zero_shot_success_at_{max_budget}"] = zero_success.to_numpy()
        query_df.loc[merged["_orig_index"], f"zero_shot_failed_recovered_at_{max_budget}"] = recovered.to_numpy()

        zero_fail_count = int((~zero[f"zero_shot_success_at_{max_budget}"].astype(bool)).sum())
        for fewshot_label, fewshot_group in merged.groupby("fewshot_label", sort=False):
            idx = fewshot_group.index
            fewshot_zero_success = zero_success.loc[idx]
            fewshot_recovered = (~fewshot_zero_success) & fewshot_group[f"success_at_{max_budget}"].astype(bool)
            recovery_rate = float(fewshot_recovered.sum() / (~fewshot_zero_success).sum()) if int((~fewshot_zero_success).sum()) else 0.0
            summary_extra.append(
                {
                    "target_domain": target_domain,
                    "method": method,
                    "seed": int(seed),
                    "fewshot_label": fewshot_label,
                    "zero_shot_failed_count_at_max_budget": zero_fail_count,
                    f"failure_recovery_rate_at_{max_budget}": recovery_rate,
                }
            )

    if summary_extra:
        extra_df = pd.DataFrame(summary_extra)
        summary_df = summary_df.merge(extra_df, on=["target_domain", "method", "seed", "fewshot_label"], how="left")
    else:
        summary_df["zero_shot_failed_count_at_max_budget"] = 0
        summary_df[f"failure_recovery_rate_at_{max_budget}"] = 0.0

    for (target_domain, method, seed), group in summary_df.groupby(group_cols, sort=False):
        zero_rows = group[group["fewshot_label"] == "0"]
        if zero_rows.empty:
            continue
        zero_row = zero_rows.iloc[0]
        mask = (
            (summary_df["target_domain"] == target_domain)
            & (summary_df["method"] == method)
            & (summary_df["seed"] == seed)
        )
        for budget in budgets:
            col = f"success_at_{budget}"
            summary_df.loc[mask, f"zero_shot_{col}"] = float(zero_row[col])
            summary_df.loc[mask, f"ood_recovery_gain_at_{budget}"] = summary_df.loc[mask, col].astype(float) - float(zero_row[col])
        summary_df.loc[mask, "zero_shot_mean_budget_to_success"] = float(zero_row["mean_budget_to_success"])
        summary_df.loc[mask, "budget_to_success_delta_vs_zero_shot"] = (
            summary_df.loc[mask, "mean_budget_to_success"].astype(float) - float(zero_row["mean_budget_to_success"])
        )
        summary_df.loc[mask, "zero_shot_mean_top1_amplitude_mse"] = float(zero_row["mean_top1_amplitude_mse"])
        summary_df.loc[mask, "top1_mse_delta_vs_zero_shot"] = (
            summary_df.loc[mask, "mean_top1_amplitude_mse"].astype(float) - float(zero_row["mean_top1_amplitude_mse"])
        )

    return query_df, summary_df


def aggregate_summary(summary_df: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    metric_cols = [
        "num_queries",
        "effective_fewshot_size",
        "memory_size",
        "mean_top1_amplitude_mse",
        "mean_best_amplitude_mse",
        "median_best_amplitude_mse",
        "top1_target_domain_rate",
        "mean_budget_to_success",
        "mean_budget_to_target_domain_success",
        "budget_to_success_delta_vs_zero_shot",
        "top1_mse_delta_vs_zero_shot",
        f"failure_recovery_rate_at_{max(budgets)}",
    ]
    for budget in budgets:
        metric_cols.extend([f"success_at_{budget}", f"ood_recovery_gain_at_{budget}"])
        metric_cols.append(f"target_domain_success_at_{budget}")

    existing_metric_cols = [col for col in metric_cols if col in summary_df.columns]
    group_cols = ["target_domain", "method", "fewshot_label", "requested_fewshot_size"]

    rows: list[dict[str, Any]] = []
    for key, group in summary_df.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, key))
        row["num_seeds"] = int(group["seed"].nunique())
        for col in existing_metric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_fewshot_retrieval_recovery(
    cfg: dict,
    processed_root: Path,
    *,
    target_domains: list[str],
    base_domains: list[str],
    query_split: str,
    methods: list[str],
    fewshot_sizes: list[int],
    seeds: list[int],
    budgets: list[int],
    output_dir: Path,
    batch_size: int,
    max_queries_per_domain: int | None,
    include_full_target: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    sizes = list(dict.fromkeys(int(size) for size in fewshot_sizes))
    if 0 not in sizes:
        sizes.insert(0, 0)
    if include_full_target and -1 not in sizes:
        sizes.append(-1)

    all_query_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for target_domain in target_domains:
        query_data = load_processed_split(processed_root, target_domain, query_split)
        query_data = _slice_query_data(query_data, max_queries_per_domain)
        thresholds = thresholds_for_domain(cfg, target_domain)

        for seed in seeds:
            for fewshot_size in sizes:
                bank, meta = build_fewshot_memory_bank(
                    cfg,
                    processed_root,
                    base_domains=base_domains,
                    target_domain=target_domain,
                    fewshot_size=fewshot_size,
                    seed=seed,
                )

                for method in methods:
                    top_indices = retrieve_for_method(
                        method=method,
                        bank=bank,
                        query_data=query_data,
                        query_domain=target_domain,
                        cfg=cfg,
                        top_k=max(budgets),
                        batch_size=batch_size,
                    )
                    query_rows, summary = evaluate_topk(
                        bank=bank,
                        query_data=query_data,
                        top_indices=top_indices,
                        budgets=budgets,
                        thresholds=thresholds,
                        target_domain=target_domain,
                    )

                    for row in query_rows:
                        row.update(meta)
                        row["target_domain"] = target_domain
                        row["domain"] = target_domain
                        row["method"] = method
                        all_query_rows.append(row)

                    summary.update(meta)
                    summary["target_domain"] = target_domain
                    summary["domain"] = target_domain
                    summary["method"] = method
                    summary["query_split"] = query_split
                    summary["max_budget"] = max(budgets)
                    summary_rows.append(summary)

    query_df = pd.DataFrame(all_query_rows)
    summary_df = pd.DataFrame(summary_rows)
    query_df, summary_df = _add_zero_shot_gains(query_df, summary_df, budgets)
    aggregate_df = aggregate_summary(summary_df, budgets)

    query_df.to_csv(output_dir / "fewshot_retrieval_query_results.csv", index=False)
    summary_df.to_csv(output_dir / "fewshot_retrieval_summary_by_seed.csv", index=False)
    aggregate_df.to_csv(output_dir / "fewshot_retrieval_summary_aggregate.csv", index=False)
    return query_df, summary_df, aggregate_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="实验配置 JSON，例如 configs/experiment_v1.json")
    parser.add_argument("--processed-root", default=None, help="processed dataset 目录；默认使用配置中的 processed_root")
    parser.add_argument("--target-domains", nargs="+", default=["target_h", "target_self"], help="要评估 few-shot recovery 的 OOD target domains")
    parser.add_argument("--base-domains", nargs="+", default=["source"], help="zero-shot memory 使用哪些 base domains；默认只用 source")
    parser.add_argument("--query-split", default="test", choices=["train", "val", "test"], help="评估 query split")
    parser.add_argument("--methods", nargs="+", default=["raw_spectrum"], choices=list(METHODS), help="retrieval 方法")
    parser.add_argument("--fewshot-sizes", nargs="+", type=int, default=[0, 20, 50, 100, 200], help="加入 target train memory 的样本数")
    parser.add_argument("--seeds", nargs="+", type=int, default=[48, 49, 50], help="few-shot sample 抽样随机种子")
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 5, 10, 50], help="Success@B budgets")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--batch-size", type=int, default=512, help="retrieval 批大小")
    parser.add_argument("--max-queries-per-domain", type=int, default=None, help="smoke test 时限制每个 target domain 的 query 数")
    parser.add_argument("--include-full-target", action="store_true", help="额外加入 target train full-memory upper reference")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(args.processed_root) if args.processed_root else Path(cfg["_processed_root_abs"])

    _, summary_df, aggregate_df = evaluate_fewshot_retrieval_recovery(
        cfg,
        processed_root,
        target_domains=args.target_domains,
        base_domains=args.base_domains,
        query_split=args.query_split,
        methods=args.methods,
        fewshot_sizes=sorted(args.fewshot_sizes),
        seeds=args.seeds,
        budgets=sorted(args.budgets),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        max_queries_per_domain=args.max_queries_per_domain,
        include_full_target=args.include_full_target,
    )

    print(f"Saved Phase 4 few-shot retrieval recovery: {Path(args.output_dir).resolve()}")
    print("Per-seed summary:")
    print(summary_df.to_string(index=False))
    print("\nAggregate summary:")
    print(aggregate_df.to_string(index=False))


if __name__ == "__main__":
    main()
