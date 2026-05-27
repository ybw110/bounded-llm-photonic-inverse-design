from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - 只有训练环境缺 torch 时才触发
    raise ImportError("训练 PNN forward model 需要安装 PyTorch：pip install torch") from exc


class ResidualBlock(nn.Module):
    """简单残差块。

    残差结构的作用是让网络在加深后仍然容易训练。
    这里用的是很朴素的 MLP residual block，输入和输出维度相同。
    """

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class PNNForwardModel(nn.Module):
    """参数到复数频谱的 forward model。

    输入：
    - 归一化结构参数，维度为 active_param_dim

    输出：
    - real[31] + imag[31] 拼接向量，维度为 62

    这个模型对应旧工作里的 PNN 思路，但训练数据来自当前新项目的 processed dataset。
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
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.blocks(x)
        return self.output(x)

