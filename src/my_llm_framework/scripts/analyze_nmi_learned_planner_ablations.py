"""Learned planner controls for the NMI manuscript.

The script evaluates two lightweight non-language planner controls on the
few-shot OOD replay pool at the nominal success threshold:

1. exposed-preview decision-list ranker: candidate-level previews are visible;
2. preview-withheld decision-list ranker: only source-only diagnostics and action
   metadata are visible.

Predictions are out-of-fold at the query level. The outputs are intended for
the NMI version of the manuscript, where the goal is attribution rather than a
claim of unique language-model reasoning.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


SUCCESS_BUDGET = 50
TAU = 0.03
SHIFT_TYPES = {
    "target_h": "topology_shift",
    "target_self": "material_frequency_shift",
}
METHOD_RANK = {"domain_aware": 0, "raw_spectrum": 1}
EXPOSED_RANK_FEATURES = [
    "candidate_target_success",
    "candidate_success",
    "candidate_top1_target",
    "target_success_low_cost",
    "success_low_cost",
    "top1_target_low_cost",
    "candidate_best_quality",
    "candidate_top1_quality",
    "domain_aware",
    "low_cost",
]
WITHHELD_RANK_FEATURES = [
    "target_h_action_domain_aware_200",
    "target_h_action_domain_aware_100",
    "target_h_action_domain_aware_50",
    "target_h_action_domain_aware_20",
    "target_h_action_raw_spectrum_200",
    "target_h_action_raw_spectrum_100",
    "target_h_action_raw_spectrum_50",
    "target_h_action_raw_spectrum_20",
    "target_self_action_domain_aware_20",
    "target_self_action_raw_spectrum_20",
    "domain_aware",
    "low_cost",
    "high_cost",
    "source_raw_failed",
    "source_raw_not_target_valid",
    "source_domain_failed",
    "source_domain_not_target_valid",
]


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return "0"
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text or "0"


def fewshot_cost(row: pd.Series) -> int:
    label = normalize_label(row["fewshot_label"] if "fewshot_label" in row else row["final_fewshot_label"])
    if label.lower() in {"self20", "self-20"}:
        return 20
    try:
        return int(float(label))
    except ValueError:
        return int(float(row.get("effective_fewshot_size", 0)))


def as_float(row: pd.Series, column: str, default: float = math.nan) -> float:
    value = pd.to_numeric(row.get(column, default), errors="coerce")
    if pd.isna(value):
        return default
    return float(value)


def quality_from_mse(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, -math.log10(max(value, 1e-12)) / 12.0))


def select_oracle(group: pd.DataFrame) -> pd.Series:
    scored = group.copy()
    scored["_target_success"] = scored[f"target_domain_success_at_{SUCCESS_BUDGET}"].map(bool_value).astype(int)
    scored["_success"] = scored[f"success_at_{SUCCESS_BUDGET}"].map(bool_value).astype(int)
    scored["_top1_target"] = scored["top1_is_target_domain"].map(bool_value).astype(int)
    scored["_cost"] = scored.apply(fewshot_cost, axis=1)
    scored["_best_mse"] = pd.to_numeric(scored["best_amplitude_mse_at_max_budget"], errors="coerce").fillna(np.inf)
    scored["_top1_mse"] = pd.to_numeric(scored["top1_amplitude_mse"], errors="coerce").fillna(np.inf)
    scored["_method_rank"] = scored["method"].map(METHOD_RANK).fillna(99)
    return scored.sort_values(
        ["_target_success", "_success", "_top1_target", "_cost", "_best_mse", "_top1_mse", "_method_rank"],
        ascending=[False, False, False, True, True, True, True],
        kind="stable",
    ).iloc[0]


def query_key(row: pd.Series) -> tuple[str, int, int]:
    return str(row["target_domain"]), int(row["seed"]), int(row["query_index"])


def source_context(group: pd.DataFrame) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for method in ["raw_spectrum", "domain_aware"]:
        match = group[(group["method"] == method) & (group["fewshot_label_norm"] == "0")]
        if match.empty:
            out[method] = group.iloc[0]
        else:
            out[method] = match.iloc[0]
    return out


def source_features(row: pd.Series, prefix: str) -> list[float]:
    top1 = as_float(row, "top1_amplitude_mse", math.inf)
    best = as_float(row, "best_amplitude_mse_at_max_budget", math.inf)
    return [
        float(bool_value(row[f"success_at_{SUCCESS_BUDGET}"])),
        float(bool_value(row[f"target_domain_success_at_{SUCCESS_BUDGET}"])),
        float(bool_value(row["top1_is_target_domain"])),
        quality_from_mse(top1),
        quality_from_mse(best),
        float(str(row.get("top1_bank_domain", "")) == "target_h"),
        float(str(row.get("top1_bank_domain", "")) == "target_self"),
        float(str(row.get("top1_bank_domain", "")) == "source"),
    ]


def base_action_features(row: pd.Series, ctx: dict[str, pd.Series]) -> list[float]:
    target_h = float(row["target_domain"] == "target_h")
    target_self = float(row["target_domain"] == "target_self")
    domain_aware = float(row["method"] == "domain_aware")
    raw = float(row["method"] == "raw_spectrum")
    cost = fewshot_cost(row) / 200.0
    zero_cost = float(fewshot_cost(row) == 0)
    feats = [
        target_h,
        target_self,
        domain_aware,
        raw,
        cost,
        cost * cost,
        zero_cost,
        target_h * domain_aware,
        target_self * domain_aware,
        target_h * cost,
        target_self * cost,
        domain_aware * cost,
    ]
    feats.extend(source_features(ctx["raw_spectrum"], "raw0"))
    feats.extend(source_features(ctx["domain_aware"], "domain0"))
    return feats


def exposed_features(row: pd.Series, ctx: dict[str, pd.Series]) -> list[float]:
    feats = base_action_features(row, ctx)
    target_success = float(bool_value(row[f"target_domain_success_at_{SUCCESS_BUDGET}"]))
    success = float(bool_value(row[f"success_at_{SUCCESS_BUDGET}"]))
    top1_target = float(bool_value(row["top1_is_target_domain"]))
    best_q = quality_from_mse(as_float(row, "best_amplitude_mse_at_max_budget", math.inf))
    top1_q = quality_from_mse(as_float(row, "top1_amplitude_mse", math.inf))
    cost = fewshot_cost(row) / 200.0
    domain_aware = float(row["method"] == "domain_aware")
    feats.extend(
        [
            target_success,
            success,
            top1_target,
            best_q,
            top1_q,
            target_success * (1.0 - cost),
            success * (1.0 - cost),
            top1_target * (1.0 - cost),
            target_success * domain_aware,
            success * domain_aware,
        ]
    )
    return feats


def withheld_features(row: pd.Series, ctx: dict[str, pd.Series]) -> list[float]:
    return base_action_features(row, ctx)


def action_flags(row: pd.Series) -> dict[str, float]:
    method = str(row["method"])
    label = normalize_label(row["fewshot_label"])
    flags: dict[str, float] = {}
    for m in ["raw_spectrum", "domain_aware"]:
        for lbl in ["0", "20", "50", "100", "200"]:
            flags[f"action_{m}_{lbl}"] = float(method == m and label == lbl)
            flags[f"target_h_action_{m}_{lbl}"] = float(row["target_domain"] == "target_h" and method == m and label == lbl)
            flags[f"target_self_action_{m}_{lbl}"] = float(
                row["target_domain"] == "target_self" and method == m and label == lbl
            )
    return flags


def rank_feature_values(row: pd.Series, ctx: dict[str, pd.Series], *, exposed: bool) -> dict[str, float]:
    cost = fewshot_cost(row)
    source_raw = ctx["raw_spectrum"]
    source_domain = ctx["domain_aware"]
    source_raw_mse = quality_from_mse(as_float(source_raw, "top1_amplitude_mse", math.inf))
    source_domain_mse = quality_from_mse(as_float(source_domain, "top1_amplitude_mse", math.inf))
    values: dict[str, float] = {
        "target_h": float(row["target_domain"] == "target_h"),
        "target_self": float(row["target_domain"] == "target_self"),
        "domain_aware": float(row["method"] == "domain_aware"),
        "raw_spectrum": float(row["method"] == "raw_spectrum"),
        "low_cost": -float(cost),
        "high_cost": float(cost),
        "nonzero_cost": float(cost > 0),
        "self20_cost": float(cost == 20),
        "source_raw_quality": source_raw_mse,
        "source_domain_quality": source_domain_mse,
        "source_raw_failed": float(not bool_value(source_raw[f"success_at_{SUCCESS_BUDGET}"])),
        "source_raw_not_target_valid": float(not bool_value(source_raw["top1_is_target_domain"])),
        "source_domain_failed": float(not bool_value(source_domain[f"success_at_{SUCCESS_BUDGET}"])),
        "source_domain_not_target_valid": float(not bool_value(source_domain["top1_is_target_domain"])),
        "target_h_domain_aware": float(row["target_domain"] == "target_h" and row["method"] == "domain_aware"),
        "target_self_domain_aware": float(row["target_domain"] == "target_self" and row["method"] == "domain_aware"),
        "target_h_low_cost": float(row["target_domain"] == "target_h") * -float(cost),
        "target_self_low_cost": float(row["target_domain"] == "target_self") * -float(cost),
    }
    values.update(action_flags(row))
    if exposed:
        best_mse = as_float(row, "best_amplitude_mse_at_max_budget", math.inf)
        top1_mse = as_float(row, "top1_amplitude_mse", math.inf)
        values.update(
            {
                "candidate_target_success": float(bool_value(row[f"target_domain_success_at_{SUCCESS_BUDGET}"])),
                "candidate_success": float(bool_value(row[f"success_at_{SUCCESS_BUDGET}"])),
                "candidate_top1_target": float(bool_value(row["top1_is_target_domain"])),
                "candidate_best_quality": quality_from_mse(best_mse),
                "candidate_top1_quality": quality_from_mse(top1_mse),
                "target_success_low_cost": float(
                    bool_value(row[f"target_domain_success_at_{SUCCESS_BUDGET}"])
                )
                * -float(cost),
                "success_low_cost": float(bool_value(row[f"success_at_{SUCCESS_BUDGET}"])) * -float(cost),
                "top1_target_low_cost": float(bool_value(row["top1_is_target_domain"])) * -float(cost),
            }
        )
    return values


def add_rank_features(pool: pd.DataFrame, *, exposed: bool) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, float]] = []
    for _, group in pool.groupby(["target_domain", "seed", "query_index"], sort=False):
        ctx = source_context(group)
        for row_idx, row in group.iterrows():
            values = rank_feature_values(row, ctx, exposed=exposed)
            values["_row_index"] = float(row_idx)
            rows.append(values)
    feature_frame = pd.DataFrame(rows).fillna(0.0).set_index("_row_index")
    out = pool.reset_index(drop=True).copy()
    preferred = EXPOSED_RANK_FEATURES if exposed else WITHHELD_RANK_FEATURES
    feature_names = [name for name in preferred if name in feature_frame.columns]
    for name in feature_names:
        out[f"rank__{name}"] = feature_frame[name].reindex(out.index).astype(float).values
    return out, feature_names


def select_by_feature_order(group: pd.DataFrame, feature_order: list[str]) -> pd.Series:
    scored = group.copy()
    scored["_cost_for_tie"] = scored.apply(fewshot_cost, axis=1)
    scored["_method_rank_for_tie"] = scored["method"].map(METHOD_RANK).fillna(99)
    sort_cols = [f"rank__{name}" for name in feature_order] + ["_cost_for_tie", "_method_rank_for_tie"]
    ascending = [False for _ in feature_order] + [True, True]
    return scored.sort_values(sort_cols, ascending=ascending, kind="stable").iloc[0]


def oracle_match_for_order(pool: pd.DataFrame, feature_order: list[str], oracle_by_query: dict[tuple[str, int, int], pd.Series]) -> float:
    matches = 0
    total = 0
    for _, group in pool.groupby(["target_domain", "seed", "query_index"], sort=False):
        selected = select_by_feature_order(group, feature_order)
        oracle = oracle_by_query[query_key(group.iloc[0])]
        matches += int(str(selected["method"]) == str(oracle["method"]) and normalize_label(selected["fewshot_label"]) == normalize_label(oracle["fewshot_label"]))
        total += 1
    return matches / max(total, 1)


def learn_feature_order(
    train_pool: pd.DataFrame,
    feature_names: list[str],
    oracle_by_query: dict[tuple[str, int, int], pd.Series],
    *,
    max_steps: int = 8,
) -> list[str]:
    selected: list[str] = []
    remaining = list(feature_names)
    best_score = oracle_match_for_order(train_pool, selected, oracle_by_query)
    for _ in range(max_steps):
        best_name: str | None = None
        best_candidate_score = best_score
        for name in remaining:
            score = oracle_match_for_order(train_pool, selected + [name], oracle_by_query)
            if score > best_candidate_score + 1e-12:
                best_candidate_score = score
                best_name = name
        if best_name is None:
            break
        selected.append(best_name)
        remaining.remove(best_name)
        best_score = best_candidate_score
    return selected


def select_index_fast(
    indices: np.ndarray,
    feature_array: np.ndarray,
    order_indices: list[int],
    costs: np.ndarray,
    method_ranks: np.ndarray,
) -> int:
    best = int(indices[0])
    for raw_idx in indices[1:]:
        idx = int(raw_idx)
        better = False
        worse = False
        for feature_idx in order_indices:
            a = feature_array[idx, feature_idx]
            b = feature_array[best, feature_idx]
            if a > b + 1e-12:
                better = True
                break
            if a < b - 1e-12:
                worse = True
                break
        if not better and not worse:
            if costs[idx] < costs[best]:
                better = True
            elif costs[idx] == costs[best] and method_ranks[idx] < method_ranks[best]:
                better = True
        if better:
            best = idx
    return best


def match_rate_fast(
    groups: list[np.ndarray],
    oracle_indices: dict[int, int],
    feature_array: np.ndarray,
    order_indices: list[int],
    costs: np.ndarray,
    method_ranks: np.ndarray,
) -> float:
    matches = 0
    for group_id, indices in enumerate(groups):
        selected = select_index_fast(indices, feature_array, order_indices, costs, method_ranks)
        matches += int(selected == oracle_indices[group_id])
    return matches / max(len(groups), 1)


def reward_fast(
    groups: list[np.ndarray],
    feature_array: np.ndarray,
    order_indices: list[int],
    costs: np.ndarray,
    method_ranks: np.ndarray,
    target_successes: np.ndarray,
    successes: np.ndarray,
    top1_targets: np.ndarray,
    best_mses: np.ndarray,
) -> float:
    rewards: list[float] = []
    for indices in groups:
        selected = select_index_fast(indices, feature_array, order_indices, costs, method_ranks)
        mse_penalty = min(max(best_mses[selected], 0.0), 0.1) / 0.1
        rewards.append(
            float(target_successes[selected])
            + 0.10 * float(successes[selected])
            + 0.03 * float(top1_targets[selected])
            - 0.02 * float(costs[selected] / 200.0)
            - 0.005 * mse_penalty
        )
    return float(np.mean(rewards)) if rewards else 0.0


def learn_feature_order_fast(
    train_group_ids: list[int],
    groups: list[np.ndarray],
    feature_array: np.ndarray,
    feature_names: list[str],
    costs: np.ndarray,
    method_ranks: np.ndarray,
    target_successes: np.ndarray,
    successes: np.ndarray,
    top1_targets: np.ndarray,
    best_mses: np.ndarray,
    *,
    max_steps: int = 8,
) -> list[int]:
    selected: list[int] = []
    remaining = list(range(len(feature_names)))
    train_groups = [groups[i] for i in train_group_ids]
    best_score = reward_fast(
        train_groups,
        feature_array,
        selected,
        costs,
        method_ranks,
        target_successes,
        successes,
        top1_targets,
        best_mses,
    )
    for _ in range(max_steps):
        best_feature: int | None = None
        best_candidate_score = best_score
        for feature_idx in remaining:
            score = reward_fast(
                train_groups,
                feature_array,
                selected + [feature_idx],
                costs,
                method_ranks,
                target_successes,
                successes,
                top1_targets,
                best_mses,
            )
            if score > best_candidate_score + 1e-12:
                best_candidate_score = score
                best_feature = feature_idx
        if best_feature is None:
            break
        selected.append(best_feature)
        remaining.remove(best_feature)
        best_score = best_candidate_score
    return selected


def evaluate_decision_list_policy(pool: pd.DataFrame, *, policy: str, exposed: bool) -> pd.DataFrame:
    ranked_pool, feature_names = add_rank_features(pool, exposed=exposed)
    oracle_by_query: dict[tuple[str, int, int], pd.Series] = {}
    groups: list[np.ndarray] = []
    group_keys: list[tuple[str, int, int]] = []
    group_folds: list[int] = []
    oracle_indices: dict[int, int] = {}
    feature_array = ranked_pool[[f"rank__{name}" for name in feature_names]].to_numpy(dtype=float)
    costs = ranked_pool.apply(fewshot_cost, axis=1).to_numpy(dtype=float)
    method_ranks = ranked_pool["method"].map(METHOD_RANK).fillna(99).to_numpy(dtype=float)
    target_successes = ranked_pool[f"target_domain_success_at_{SUCCESS_BUDGET}"].map(bool_value).astype(float).to_numpy()
    successes = ranked_pool[f"success_at_{SUCCESS_BUDGET}"].map(bool_value).astype(float).to_numpy()
    top1_targets = ranked_pool["top1_is_target_domain"].map(bool_value).astype(float).to_numpy()
    best_mses = pd.to_numeric(ranked_pool["best_amplitude_mse_at_max_budget"], errors="coerce").fillna(0.1).to_numpy(dtype=float)
    for group_id, (_, group) in enumerate(ranked_pool.groupby(["target_domain", "seed", "query_index"], sort=False)):
        indices = group.index.to_numpy(dtype=int)
        groups.append(indices)
        key = query_key(group.iloc[0])
        group_keys.append(key)
        group_folds.append(int(group["query_index"].iloc[0]) % 5)
        oracle = select_oracle(group)
        oracle_by_query[key] = oracle
        oracle_match = group[
            (group["method"] == str(oracle["method"]))
            & (group["fewshot_label_norm"] == normalize_label(oracle["fewshot_label"]))
        ]
        oracle_indices[group_id] = int(oracle_match.index[0])
    rows: list[dict[str, object]] = []
    for fold in range(5):
        train_group_ids = [i for i, group_fold in enumerate(group_folds) if group_fold != fold]
        test_group_ids = [i for i, group_fold in enumerate(group_folds) if group_fold == fold]
        order = learn_feature_order_fast(
            train_group_ids,
            groups,
            feature_array,
            feature_names,
            costs,
            method_ranks,
            target_successes,
            successes,
            top1_targets,
            best_mses,
        )
        if not order:
            order = [feature_names.index("low_cost")] if "low_cost" in feature_names else []
        for group_id in test_group_ids:
            selected_idx = select_index_fast(groups[group_id], feature_array, order, costs, method_ranks)
            selected = ranked_pool.iloc[selected_idx]
            key = group_keys[group_id]
            rows.append(row_from_candidate(policy, selected, oracle_by_query[key]))
    return pd.DataFrame(rows)


def train_binary_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    iterations: int = 1600,
    learning_rate: float = 0.08,
    l2: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    x = (x_train - mean) / std
    x = np.c_[np.ones(len(x)), x]
    w = np.zeros(x.shape[1], dtype=float)
    positive = max(float(y_train.sum()), 1.0)
    negative = max(float(len(y_train) - y_train.sum()), 1.0)
    sample_weight = np.where(y_train > 0.5, negative / positive, 1.0)
    sample_weight = sample_weight / sample_weight.mean()
    for _ in range(iterations):
        logits = np.clip(x @ w, -40.0, 40.0)
        pred = 1.0 / (1.0 + np.exp(-logits))
        grad = (x.T @ ((pred - y_train) * sample_weight)) / len(y_train)
        grad[1:] += l2 * w[1:]
        w -= learning_rate * grad
    return w, mean, std


def train_pairwise_ranker(
    x_train: np.ndarray,
    y_train: np.ndarray,
    keys_train: list[tuple[str, int, int]],
    *,
    iterations: int = 2200,
    learning_rate: float = 0.06,
    l2: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train a small logistic ranker from oracle-vs-alternative pairs."""
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    x = (x_train - mean) / std
    by_query: dict[tuple[str, int, int], list[int]] = {}
    for i, key in enumerate(keys_train):
        by_query.setdefault(key, []).append(i)
    diffs: list[np.ndarray] = []
    for indices in by_query.values():
        positives = [i for i in indices if y_train[i] > 0.5]
        if not positives:
            continue
        pos = positives[0]
        for neg in indices:
            if neg != pos:
                diffs.append(x[pos] - x[neg])
    if not diffs:
        return np.zeros(x_train.shape[1], dtype=float), mean, std
    d = np.vstack(diffs)
    w = np.zeros(d.shape[1], dtype=float)
    for _ in range(iterations):
        margin = np.clip(d @ w, -40.0, 40.0)
        grad = -(d.T @ (1.0 / (1.0 + np.exp(margin)))) / len(d)
        grad += l2 * w
        w -= learning_rate * grad
    return w, mean, std


def predict_scores(x_test: np.ndarray, w: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = (x_test - mean) / std
    return x @ w


@dataclass
class TreeNode:
    score: float
    feature: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


def weighted_positive_rate(y: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    return float((weights * y).sum() / total)


def weighted_gini(y: np.ndarray, weights: np.ndarray) -> float:
    p = weighted_positive_rate(y, weights)
    return 2.0 * p * (1.0 - p)


def threshold_candidates(values: np.ndarray, max_thresholds: int = 24) -> np.ndarray:
    unique = np.unique(values[np.isfinite(values)])
    if len(unique) <= 1:
        return np.asarray([], dtype=float)
    mids = (unique[:-1] + unique[1:]) / 2.0
    if len(mids) <= max_thresholds:
        return mids
    qs = np.linspace(0.02, 0.98, max_thresholds)
    return np.unique(np.quantile(mids, qs))


def build_tree(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    depth: int = 0,
    max_depth: int = 5,
    min_leaf: int = 40,
    min_gain: float = 1e-5,
) -> TreeNode:
    score = weighted_positive_rate(y, weights)
    node = TreeNode(score=score)
    if depth >= max_depth or len(y) < 2 * min_leaf or score < 1e-9 or score > 1.0 - 1e-9:
        return node
    parent = weighted_gini(y, weights)
    best_gain = 0.0
    best_feature: int | None = None
    best_threshold: float | None = None
    best_left: np.ndarray | None = None
    total_weight = float(weights.sum())
    for feature in range(x.shape[1]):
        values = x[:, feature]
        for threshold in threshold_candidates(values):
            left = values <= threshold
            left_n = int(left.sum())
            right_n = len(y) - left_n
            if left_n < min_leaf or right_n < min_leaf:
                continue
            wl = weights[left].sum()
            wr = total_weight - wl
            if wl <= 0 or wr <= 0:
                continue
            impurity = (wl / total_weight) * weighted_gini(y[left], weights[left])
            impurity += (wr / total_weight) * weighted_gini(y[~left], weights[~left])
            gain = parent - impurity
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = float(threshold)
                best_left = left
    if best_feature is None or best_threshold is None or best_left is None or best_gain < min_gain:
        return node
    node.feature = best_feature
    node.threshold = best_threshold
    node.left = build_tree(
        x[best_left],
        y[best_left],
        weights[best_left],
        depth=depth + 1,
        max_depth=max_depth,
        min_leaf=min_leaf,
        min_gain=min_gain,
    )
    node.right = build_tree(
        x[~best_left],
        y[~best_left],
        weights[~best_left],
        depth=depth + 1,
        max_depth=max_depth,
        min_leaf=min_leaf,
        min_gain=min_gain,
    )
    return node


def train_tree_ranker(x_train: np.ndarray, y_train: np.ndarray) -> TreeNode:
    positive = max(float(y_train.sum()), 1.0)
    negative = max(float(len(y_train) - y_train.sum()), 1.0)
    weights = np.where(y_train > 0.5, negative / positive, 1.0)
    return build_tree(x_train, y_train, weights)


def predict_tree_scores(node: TreeNode, x_test: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(x_test), dtype=float)
    for i, row in enumerate(x_test):
        current = node
        while current.feature is not None and current.threshold is not None:
            if row[current.feature] <= current.threshold:
                if current.left is None:
                    break
                current = current.left
            else:
                if current.right is None:
                    break
                current = current.right
        scores[i] = current.score
    return scores


def row_from_candidate(policy: str, candidate: pd.Series, oracle: pd.Series) -> dict[str, object]:
    final_method = str(candidate["method"])
    final_label = normalize_label(candidate["fewshot_label"])
    oracle_method = str(oracle["method"])
    oracle_label = normalize_label(oracle["fewshot_label"])
    return {
        "tau": TAU,
        "policy": policy,
        "run_label": "5fold_oof",
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
        "final_top1_amplitude_mse": as_float(candidate, "top1_amplitude_mse"),
        "final_best_amplitude_mse_at_budget": as_float(candidate, "best_amplitude_mse_at_max_budget"),
        "oracle_method": oracle_method,
        "oracle_fewshot_label": oracle_label,
        "oracle_fewshot_cost": fewshot_cost(oracle),
        f"oracle_success_at_{SUCCESS_BUDGET}": bool_value(oracle[f"success_at_{SUCCESS_BUDGET}"]),
        f"oracle_target_domain_success_at_{SUCCESS_BUDGET}": bool_value(
            oracle[f"target_domain_success_at_{SUCCESS_BUDGET}"]
        ),
        "matched_oracle_choice": final_method == oracle_method and final_label == oracle_label,
        "planner_used_fallback": False,
        "planner_steps_used": 1,
    }


def build_design_matrix(
    pool: pd.DataFrame,
    feature_fn: Callable[[pd.Series, dict[str, pd.Series]], list[float]],
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int, int]], list[int], dict[tuple[str, int, int], pd.Series]]:
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    keys: list[tuple[str, int, int]] = []
    folds: list[int] = []
    oracle_by_query: dict[tuple[str, int, int], pd.Series] = {}
    for _, group in pool.groupby(["target_domain", "seed", "query_index"], sort=False):
        oracle = select_oracle(group)
        key = query_key(group.iloc[0])
        oracle_by_query[key] = oracle
        ctx = source_context(group)
        fold = int(group["query_index"].iloc[0]) % 5
        for _, row in group.iterrows():
            x_rows.append(feature_fn(row, ctx))
            y_rows.append(
                float(
                    str(row["method"]) == str(oracle["method"])
                    and normalize_label(row["fewshot_label"]) == normalize_label(oracle["fewshot_label"])
                )
            )
            keys.append(key)
            folds.append(fold)
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float), keys, folds, oracle_by_query


def evaluate_policy(
    pool: pd.DataFrame,
    *,
    policy: str,
    feature_fn: Callable[[pd.Series, dict[str, pd.Series]], list[float]],
) -> pd.DataFrame:
    x, y, keys, folds, oracle_by_query = build_design_matrix(pool, feature_fn)
    pool_indexed = pool.reset_index(drop=True).copy()
    all_rows: list[dict[str, object]] = []
    folds_arr = np.asarray(folds)
    keys_arr = np.asarray(keys, dtype=object)
    for fold in sorted(set(folds)):
        train_mask = folds_arr != fold
        test_mask = folds_arr == fold
        tree = train_tree_ranker(x[train_mask], y[train_mask])
        scores = predict_tree_scores(tree, x[test_mask])
        test_indices = np.flatnonzero(test_mask)
        scored = pool_indexed.iloc[test_indices].copy()
        scored["_score"] = scores
        scored["_cost_for_tie"] = scored.apply(fewshot_cost, axis=1)
        scored["_method_rank_for_tie"] = scored["method"].map(METHOD_RANK).fillna(99)
        for key_tuple, group in scored.groupby(["target_domain", "seed", "query_index"], sort=False):
            selected = group.sort_values(
                ["_score", "_cost_for_tie", "_method_rank_for_tie"],
                ascending=[False, True, True],
                kind="stable",
            ).iloc[0]
            key = str(key_tuple[0]), int(key_tuple[1]), int(key_tuple[2])
            all_rows.append(row_from_candidate(policy, selected, oracle_by_query[key]))
    return pd.DataFrame(all_rows)


def summarize(query_results: pd.DataFrame) -> pd.DataFrame:
    def bool_mean(series: pd.Series) -> float:
        return float(series.map(bool_value).mean())

    per_run_rows: list[dict[str, object]] = []
    for (policy, target_domain, run_label), group in query_results.groupby(
        ["policy", "target_domain", "run_label"], sort=False
    ):
        per_run_rows.append(
            {
                "policy": policy,
                "target_domain": target_domain,
                "shift_type": SHIFT_TYPES.get(str(target_domain), "ood_shift"),
                "run_label": run_label,
                "num_replays": len(group),
                f"success_at_{SUCCESS_BUDGET}": bool_mean(group[f"final_success_at_{SUCCESS_BUDGET}"]),
                f"target_domain_success_at_{SUCCESS_BUDGET}": bool_mean(
                    group[f"final_target_domain_success_at_{SUCCESS_BUDGET}"]
                ),
                "top1_target_domain_rate": bool_mean(group["final_top1_is_target_domain"]),
                "mean_final_top1_amplitude_mse": pd.to_numeric(
                    group["final_top1_amplitude_mse"], errors="coerce"
                ).mean(),
                "mean_final_best_amplitude_mse_at_budget": pd.to_numeric(
                    group["final_best_amplitude_mse_at_budget"], errors="coerce"
                ).mean(),
                "mean_fewshot_cost": pd.to_numeric(group["final_fewshot_cost"], errors="coerce").mean(),
                "oracle_match_rate": bool_mean(group["matched_oracle_choice"]),
                f"oracle_target_domain_success_at_{SUCCESS_BUDGET}": bool_mean(
                    group[f"oracle_target_domain_success_at_{SUCCESS_BUDGET}"]
                ),
                "oracle_mean_fewshot_cost": pd.to_numeric(group["oracle_fewshot_cost"], errors="coerce").mean(),
                "fallback_or_invalid_action_rate": bool_mean(group["planner_used_fallback"].where(group["planner_used_fallback"].notna(), False)),
            }
        )
    per_run = pd.DataFrame(per_run_rows)
    rows: list[dict[str, object]] = []
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
        "fallback_or_invalid_action_rate",
    ]
    for (policy, target_domain), group in per_run.groupby(["policy", "target_domain"], sort=False):
        row: dict[str, object] = {
            "policy": policy,
            "target_domain": target_domain,
            "shift_type": SHIFT_TYPES.get(str(target_domain), "ood_shift"),
            "num_runs": len(group),
            "num_replays_per_run": int(group["num_replays"].iloc[0]),
        }
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
        row[f"oracle_gap_target_domain_success_at_{SUCCESS_BUDGET}_mean"] = (
            row[f"oracle_target_domain_success_at_{SUCCESS_BUDGET}_mean"]
            - row[f"target_domain_success_at_{SUCCESS_BUDGET}_mean"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def format_metric(value: float) -> str:
    if abs(value) >= 0.01:
        return f"{value:.3f}"
    return f"{value:.2e}"


def write_latex_table(summary: pd.DataFrame, output_path: Path) -> None:
    label_map = {
        "learned_exposed_preview_decision_list": "Learned decision-list ranker, exposed previews",
        "learned_preview_withheld_decision_list": "Learned decision-list ranker, preview-withheld",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Out-of-fold lightweight learned planner controls.}",
        r"Both learned controls are transparent greedy decision-list rankers evaluated by five-fold out-of-fold prediction at the query level. The exposed-preview control learns a ranking over the same candidate-level previews as the replay interface. The preview-withheld control removes action-level candidate success, target-validity, and MSE previews, retaining only source-only diagnostics and action metadata.}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Policy & Domain & S@50 & Target S@50 & Best MSE@50 & Target samples & Oracle match \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        policy = label_map.get(str(row["policy"]), str(row["policy"]))
        domain = str(row["target_domain"]).replace("_", "-")
        lines.append(
            f"{policy} & {domain} & "
            f"{float(row[f'success_at_{SUCCESS_BUDGET}_mean']):.3f} & "
            f"{float(row[f'target_domain_success_at_{SUCCESS_BUDGET}_mean']):.3f} & "
            f"{format_metric(float(row['mean_final_best_amplitude_mse_at_budget_mean']))} & "
            f"{float(row['mean_fewshot_cost_mean']):.2f} & "
            f"{float(row['oracle_match_rate_mean']):.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool",
        type=Path,
        default=Path("MY-LLM-Framework/outputs/phase4_threshold_sensitivity_nc_v1/pool_tau_0p03/fewshot_retrieval_query_results.csv"),
    )
    parser.add_argument(
        "--existing-query-results",
        type=Path,
        default=Path("MY-LLM-Framework/outputs/phase4_threshold_sensitivity_nc_v1/threshold_policy_query_results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("MY-LLM-Framework/outputs/phase4c_nmi_learned_planner_ablations_v1"),
    )
    args = parser.parse_args()

    pool = pd.read_csv(args.pool)
    pool["fewshot_label_norm"] = pool["fewshot_label"].map(normalize_label)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exposed = evaluate_decision_list_policy(
        pool,
        policy="learned_exposed_preview_decision_list",
        exposed=True,
    )
    withheld = evaluate_decision_list_policy(
        pool,
        policy="learned_preview_withheld_decision_list",
        exposed=False,
    )
    learned = pd.concat([exposed, withheld], ignore_index=True)
    learned.to_csv(args.output_dir / "learned_planner_query_results.csv", index=False)
    learned_summary = summarize(learned)
    learned_summary.to_csv(args.output_dir / "learned_planner_summary.csv", index=False)
    write_latex_table(learned_summary, args.output_dir / "learned_planner_latex_table.tex")

    if args.existing_query_results.exists():
        existing = pd.read_csv(args.existing_query_results)
        existing = existing[pd.to_numeric(existing["tau"], errors="coerce").round(6) == TAU]
        comparison = pd.concat([existing, learned], ignore_index=True, sort=False)
        comparison.to_csv(args.output_dir / "nmi_policy_comparison_query_results.csv", index=False)
        comparison_summary = summarize(comparison)
        comparison_summary.to_csv(args.output_dir / "nmi_policy_comparison_summary.csv", index=False)

    print(f"Wrote learned planner ablations to {args.output_dir}")
    print(learned_summary.to_string(index=False))


if __name__ == "__main__":
    main()
