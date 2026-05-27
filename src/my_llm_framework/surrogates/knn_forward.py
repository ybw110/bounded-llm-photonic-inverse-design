from __future__ import annotations

import numpy as np


class KNNForwardSurrogate:
    """非训练版 forward surrogate。

    它用 memory bank 中同一 domain 的历史样本做局部插值：

    输入：归一化结构参数
    输出：预测复数频谱

    这个模型不需要 GPU，也不需要训练。它的作用是先把
    `retrieval -> local optimization -> predicted spectrum` 这条闭环跑通。
    后续可以把它替换成旧项目里的 PNN / DANN / PC-DANN forward model。
    """

    def __init__(
        self,
        params: np.ndarray,
        spectra: np.ndarray,
        active_dim: int,
        k: int = 24,
        eps: float = 1e-8,
    ) -> None:
        self.active_dim = int(active_dim)
        self.params = params[:, : self.active_dim].astype(np.float32)
        self.spectra = spectra.astype(np.complex64)
        self.k = min(int(k), len(self.params))
        self.eps = float(eps)

        if len(self.params) == 0:
            raise ValueError("KNNForwardSurrogate 收到空训练集")

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        """预测单个参数点的复数频谱。"""
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != self.active_dim:
            raise ValueError(f"输入参数维度 {x.shape[1]} 与 active_dim={self.active_dim} 不一致")

        distances = np.linalg.norm(self.params - x, axis=1)
        order = np.argpartition(distances, kth=self.k - 1)[: self.k]
        local_distances = distances[order]

        # 如果正好落在已有样本上，直接返回该样本频谱，避免除零。
        nearest = int(np.argmin(local_distances))
        if local_distances[nearest] < self.eps:
            return self.spectra[order[nearest]]

        weights = 1.0 / (local_distances + self.eps)
        weights = weights / np.sum(weights)
        return np.sum(self.spectra[order] * weights[:, None], axis=0).astype(np.complex64)

