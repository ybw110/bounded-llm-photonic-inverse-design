from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from my_llm_framework.models.pnn import PNNForwardModel


class PNNForwardSurrogate:
    """训练好的 PNN forward model 包装器。

    这个类把 `model_best.pt` 变成和 KNN surrogate 类似的接口：

    - `predict_one(x_norm)`：预测单个归一化参数点的复数频谱。
    - `predict_batch(x_norm_batch)`：一次预测一批归一化参数点的复数频谱。

    PSO 会频繁评估很多候选参数。如果一次只把一个点送进 PNN，
    GPU 会被大量小调用拖慢；批量预测可以显著减少 Python 调用开销，
    也更容易把 GPU 算力吃起来。
    """

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            # 兼容旧版 PyTorch：老版本没有 weights_only 参数。
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        self.active_dim = int(checkpoint["active_dim"])
        self.output_dim = int(checkpoint["output_dim"])

        self.model = PNNForwardModel(
            input_dim=self.active_dim,
            output_dim=self.output_dim,
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_blocks=int(checkpoint["num_blocks"]),
            dropout=float(checkpoint["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_batch(self, x: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        """批量预测复数频谱。

        参数
        ----
        x:
            形状为 `(N, active_dim)` 的归一化结构参数矩阵。
        batch_size:
            可选的内部推理批大小。通常 PSO 的 swarm 本身不大，可以不传；
            如果以后一次评估几千个点，可以传入这个参数控制显存。

        返回
        ----
        complex_spectra:
            形状为 `(N, frequency_points)` 的复数频谱矩阵。
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.active_dim:
            raise ValueError(f"输入参数维度 {x.shape[1]} 与 active_dim={self.active_dim} 不一致")

        # PNN 训练时输出 real 和 imag 拼接后的向量：
        # [real_1, ..., real_F, imag_1, ..., imag_F]
        # 因此这里预测后再拆回复数频谱。
        outputs: list[np.ndarray] = []
        step = int(batch_size) if batch_size else len(x)
        with torch.no_grad():
            for start in range(0, len(x), step):
                batch = torch.from_numpy(x[start : start + step]).to(self.device)
                outputs.append(self.model(batch).detach().cpu().numpy())

        output = np.concatenate(outputs, axis=0)
        half = self.output_dim // 2
        return (output[:, :half] + 1j * output[:, half:]).astype(np.complex64)

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        """预测单个参数点的复数频谱。"""
        return self.predict_batch(np.asarray(x, dtype=np.float32).reshape(1, -1))[0]
