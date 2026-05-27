from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("训练 INN inverse model 需要安装 PyTorch：pip install torch") from exc

from my_llm_framework.models.pnn import ResidualBlock


class INNInverseModel(nn.Module):
    """直接逆向模型：目标频谱 -> 结构参数。

    这里的 INN 指 inverse neural network（逆向神经网络），不是严格数学意义上的
    invertible neural network。它作为 Phase 2 的 direct inverse baseline：

    输入：
    - real[31] + imag[31] 拼接后的目标复数频谱，维度通常为 62。

    输出：
    - active parameters 的归一化结构参数，维度为 active_param_dim。

    注意：逆向设计天然可能一对多，同一条频谱可能对应多个结构参数。这个模型是
    确定性 direct inverse baseline，因此后续评估时要看它预测出的参数经 PNN forward
    后能否恢复目标频谱，而不能只看参数 MSE。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.blocks(x)
        return self.output(x)
