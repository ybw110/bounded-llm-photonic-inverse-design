from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from my_llm_framework.features.physics_summary import summarize_spectrum, summary_to_vector


def domain_score(query_domain: str, record_domain: str, domain_specs: dict) -> float:
    """计算两个 domain 的粗略相似度。

    这个函数被单条 demo 和批量评测共同使用。

    - 完全同域：1.0
    - topology 相同：加 0.45
    - 频段完全相同：加 0.35
    - role 相同：加 0.20
    """
    if query_domain == record_domain:
        return 1.0

    q = domain_specs.get(query_domain, {})
    r = domain_specs.get(record_domain, {})
    score = 0.0

    if q.get("topology") == r.get("topology"):
        score += 0.45

    if q.get("frequency_min_thz") == r.get("frequency_min_thz") and q.get("frequency_max_thz") == r.get("frequency_max_thz"):
        score += 0.35

    if q.get("role") == r.get("role"):
        score += 0.20

    return float(score)


@dataclass
class RetrievalWeights:
    """domain-aware retrieval 的三项权重。

    当前总分由三部分组成：

    - summary_weight：physics summary 相似度
    - domain_weight：域相似度
    - constraint_weight：约束兼容性，目前先设为 1，后续接 design spec
    """

    summary_weight: float = 0.70
    domain_weight: float = 0.20
    constraint_weight: float = 0.10


class DomainAwareRetriever:
    """面向 OOD 逆向设计的检索器。

    普通 nearest-neighbor 只看谱线或特征距离。
    这里额外加入 domain 信息，让检索结果既考虑“物理响应像不像”，
    也考虑“这个历史样本来自哪个结构族、哪个频段、哪个数据域”。
    """

    def __init__(
        self,
        summary_features: np.ndarray,
        domains: np.ndarray,
        params: np.ndarray,
        spectra: np.ndarray,
        domain_specs: dict,
        weights: RetrievalWeights | None = None,
    ) -> None:
        # memory bank 中每条样本的 physics summary 矩阵，形状 [N, summary_dim]。
        self.summary_features = summary_features.astype(np.float32)

        # 每条样本所属 domain，例如 source / target_h / target_self。
        self.domains = domains.astype(str)

        # 每条样本对应的结构参数和复数谱线。
        self.params = params.astype(np.float32)
        self.spectra = spectra.astype(np.complex64)

        # domain_specs 来自 experiment_v1.json，用于计算 domain similarity。
        self.domain_specs = domain_specs
        self.weights = weights or RetrievalWeights()

        # 对 summary features 做标准化。
        # 如果不标准化，数值范围大的指标会主导距离，例如 q_factor。
        self.mean = self.summary_features.mean(axis=0, keepdims=True)
        self.std = self.summary_features.std(axis=0, keepdims=True) + 1e-8
        self.normalized_features = (self.summary_features - self.mean) / self.std

    def query(
        self,
        target_spectrum: np.ndarray,
        target_frequency: np.ndarray,
        target_domain: str,
        top_k: int = 10,
        exclude_index: int | None = None,
    ) -> list[dict]:
        """检索和目标谱线最相关的历史样本。

        参数：

        - `target_spectrum`：目标复数谱线
        - `target_frequency`：目标谱线对应频率点
        - `target_domain`：目标任务希望设计到哪个域
        - `top_k`：返回多少个候选
        - `exclude_index`：如果 query 来自 memory bank 本身，可以排除它自己
        """
        # 先把 query 谱线转成同样的 physics summary。
        target_summary = summarize_spectrum(target_spectrum, target_frequency)
        target_vec = summary_to_vector(target_summary).reshape(1, -1)
        target_vec = (target_vec - self.mean) / self.std

        # summary distance 越小越相似。这里用 exp(-distance) 转成 0-1 附近的相似度。
        distances = np.linalg.norm(self.normalized_features - target_vec, axis=1)
        summary_scores = np.exp(-distances)

        # domain score 由 topology、频段、role 的相似性粗略构成。
        domain_scores = np.asarray(
            [self._domain_score(target_domain, d) for d in self.domains],
            dtype=np.float32,
        )

        # 约束兼容性先预留。后续接 design spec 后，可以检查制造约束、参数范围等。
        constraint_scores = np.ones_like(summary_scores, dtype=np.float32)

        total = (
            self.weights.summary_weight * summary_scores
            + self.weights.domain_weight * domain_scores
            + self.weights.constraint_weight * constraint_scores
        )

        # 防止检索 demo 中“自己检索到自己”，否则 top1 没有意义。
        if exclude_index is not None and 0 <= exclude_index < len(total):
            total[exclude_index] = -np.inf

        order = np.argsort(total)[::-1][:top_k]
        return [
            {
                "index": int(i),
                "domain": str(self.domains[i]),
                "score": float(total[i]),
                "summary_score": float(summary_scores[i]),
                "domain_score": float(domain_scores[i]),
                "constraint_score": float(constraint_scores[i]),
                "params": self.params[i].tolist(),
            }
            for i in order
        ]

    def _domain_score(self, query_domain: str, record_domain: str) -> float:
        """计算两个 domain 的粗略相似度。"""
        return domain_score(query_domain, record_domain, self.domain_specs)
