from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from my_llm_framework.models.inn import INNInverseModel
from my_llm_framework.scripts.train_pnn_forward import spectra_to_target


class INNInverseSurrogate:
    """训练好的 INN direct inverse model 包装器。

    输入目标复数频谱，输出归一化结构参数。后续评估时再把这些参数送入
    PNN forward surrogate，得到频谱并计算 success rate。
    """

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        self.input_dim = int(checkpoint["input_dim"])
        self.active_dim = int(checkpoint["active_dim"])
        self.output_dim = int(checkpoint.get("output_dim", self.active_dim))

        self.model = INNInverseModel(
            input_dim=self.input_dim,
            output_dim=self.active_dim,
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_blocks=int(checkpoint["num_blocks"]),
            dropout=float(checkpoint["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_batch(self, spectra: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """批量预测归一化结构参数。"""
        x = spectra_to_target(spectra)
        if x.shape[1] != self.input_dim:
            raise ValueError(f"输入频谱维度 {x.shape[1]} 与 input_dim={self.input_dim} 不一致")

        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x), int(batch_size)):
                batch = torch.from_numpy(x[start : start + int(batch_size)]).to(self.device)
                pred = self.model(batch).detach().cpu().numpy()
                outputs.append(pred)
        return np.clip(np.concatenate(outputs, axis=0).astype(np.float32), 0.0, 1.0)

    def predict_one(self, spectrum: np.ndarray) -> np.ndarray:
        """预测单条频谱对应的归一化结构参数。"""
        return self.predict_batch(np.asarray(spectrum).reshape(1, -1))[0]
