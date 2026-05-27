from __future__ import annotations

import numpy as np


def normalized_feature_matrix(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用 memory bank 自身统计量标准化特征矩阵。

    nearest neighbor 对不同量纲很敏感。raw-spectrum 的每个维度还比较一致，
    但 physics summary 中同时有振幅、频率、带宽、Q 值，必须标准化后再计算距离。
    """
    features = features.astype(np.float32)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    return (features - mean) / std, mean, std


def apply_normalization(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """把 query 特征按 memory bank 的均值和方差标准化。"""
    return (features.astype(np.float32) - mean) / std


def topk_smallest_l2(
    bank_features: np.ndarray,
    query_features: np.ndarray,
    top_k: int,
    batch_size: int = 128,
) -> np.ndarray:
    """批量计算 L2 最近邻。

    返回形状为 [num_queries, top_k] 的索引矩阵。
    这里使用矩阵乘法公式：

    ||q - b||^2 = ||q||^2 + ||b||^2 - 2 q b

    比逐个 query 写 Python 循环快很多，也避免一次性构造过大的三维差值数组。
    """
    bank = bank_features.astype(np.float32)
    queries = query_features.astype(np.float32)
    top_k = min(int(top_k), len(bank))

    bank_norm = np.sum(bank * bank, axis=1, keepdims=True).T
    top_indices = np.empty((len(queries), top_k), dtype=np.int64)

    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        q = queries[start:end]
        q_norm = np.sum(q * q, axis=1, keepdims=True)
        distances = np.maximum(q_norm + bank_norm - 2.0 * (q @ bank.T), 0.0)

        unsorted = np.argpartition(distances, kth=top_k - 1, axis=1)[:, :top_k]
        row = np.arange(end - start)[:, None]
        ordered = unsorted[row, np.argsort(distances[row, unsorted], axis=1)]
        top_indices[start:end] = ordered

    return top_indices


def topk_largest_scores(
    scores: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """从分数矩阵中取每行分数最高的 top-k 索引。"""
    top_k = min(int(top_k), scores.shape[1])
    unsorted = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
    row = np.arange(scores.shape[0])[:, None]
    ordered = unsorted[row, np.argsort(-scores[row, unsorted], axis=1)]
    return ordered.astype(np.int64)

