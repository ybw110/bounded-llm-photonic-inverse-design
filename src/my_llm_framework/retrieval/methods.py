from __future__ import annotations

from typing import Any

import numpy as np

from my_llm_framework.retrieval.batch import (
    apply_normalization,
    normalized_feature_matrix,
    topk_largest_scores,
    topk_smallest_l2,
)
from my_llm_framework.retrieval.domain_aware import RetrievalWeights, domain_score


def raw_spectrum_features(spectra: np.ndarray) -> np.ndarray:
    """把复数频谱转换成 raw-spectrum retrieval 使用的特征。

    这里直接拼接 real 和 imag：

    [real_1, ..., real_31, imag_1, ..., imag_31]

    它是最朴素的谱线最近邻 baseline。
    """
    return np.hstack([np.real(spectra), np.imag(spectra)]).astype(np.float32)


def retrieve_raw_spectrum(
    bank_spectra: np.ndarray,
    query_spectra: np.ndarray,
    top_k: int,
    batch_size: int,
) -> np.ndarray:
    """raw-spectrum nearest neighbor。"""
    bank_features, mean, std = normalized_feature_matrix(raw_spectrum_features(bank_spectra))
    query_features = apply_normalization(raw_spectrum_features(query_spectra), mean, std)
    return topk_smallest_l2(bank_features, query_features, top_k=top_k, batch_size=batch_size)


def retrieve_physics_summary(
    bank_summary: np.ndarray,
    query_summary: np.ndarray,
    top_k: int,
    batch_size: int,
) -> np.ndarray:
    """physics-summary nearest neighbor。"""
    bank_features, mean, std = normalized_feature_matrix(bank_summary)
    query_features = apply_normalization(query_summary, mean, std)
    return topk_smallest_l2(bank_features, query_features, top_k=top_k, batch_size=batch_size)


def retrieve_domain_aware(
    bank_summary: np.ndarray,
    bank_domains: np.ndarray,
    query_summary: np.ndarray,
    query_domain: str,
    domain_specs: dict[str, Any],
    top_k: int,
    batch_size: int,
    weights: RetrievalWeights | None = None,
) -> np.ndarray:
    """domain-aware retrieval。

    这一步使用 physics summary 作为物理相似度，再加上 domain similarity。
    它对应论文中要重点验证的 H1：domain 信息是否能改善 OOD 检索。
    """
    weights = weights or RetrievalWeights()
    bank_features, mean, std = normalized_feature_matrix(bank_summary)
    query_features = apply_normalization(query_summary, mean, std)

    domain_scores = np.asarray(
        [domain_score(query_domain, record_domain, domain_specs) for record_domain in bank_domains.astype(str)],
        dtype=np.float32,
    )
    constraint_scores = np.ones_like(domain_scores, dtype=np.float32)

    top_indices = np.empty((len(query_features), min(top_k, len(bank_features))), dtype=np.int64)
    bank_norm = np.sum(bank_features * bank_features, axis=1, keepdims=True).T

    for start in range(0, len(query_features), batch_size):
        end = min(start + batch_size, len(query_features))
        q = query_features[start:end]

        q_norm = np.sum(q * q, axis=1, keepdims=True)
        squared = np.maximum(q_norm + bank_norm - 2.0 * (q @ bank_features.T), 0.0)
        distances = np.sqrt(squared)
        summary_scores = np.exp(-distances).astype(np.float32)
        scores = (
            weights.summary_weight * summary_scores
            + weights.domain_weight * domain_scores[None, :]
            + weights.constraint_weight * constraint_scores[None, :]
        )
        top_indices[start:end] = topk_largest_scores(scores, top_k=top_indices.shape[1])

    return top_indices
