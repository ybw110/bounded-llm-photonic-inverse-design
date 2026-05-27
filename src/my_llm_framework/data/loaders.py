from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_complex_csv(
    csv_path: str | Path,
    param_dim: int,
    response_stride: int = 10,
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """读取旧框架中的归一化 CSV，并还原成“参数 + 复数谱线”。

    旧框架的 CSV 结构通常是：

    参数列 | 谱线实部列 | 谱线虚部列

    例如 Source 数据：

    - 前 6 列是归一化后的参数，其中真实有效参数是前 4 个
    - 后面 301 列是 real
    - 再后面 301 列是 imag

    H-shaped 数据类似，只是 6 个参数都是真实结构参数。

    参数：

    - `csv_path`：CSV 文件路径
    - `param_dim`：参数列数量。这里按旧框架统一读 6 列
    - `response_stride`：谱线下采样步长。旧框架常用 10，所以 301 点变 31 点
    - `max_rows`：只读取前多少行。旧 CSV 兼容检查时会用到

    返回：

    - params：形状为 [N, param_dim] 的 float32 数组
    - spectra：形状为 [N, 31] 的 complex64 复数谱线
    """
    csv_path = Path(csv_path)

    # nrows=max_rows 可以避免兼容检查时一次性读取几十万行大 CSV。
    df = pd.read_csv(csv_path, nrows=max_rows)

    if df.shape[1] <= param_dim:
        raise ValueError(f"CSV 列数不足，无法按 param_dim={param_dim} 读取参数: {csv_path}")

    # 读取参数。这里不做反归一化，因为 retrieval baseline 先在旧框架归一化空间里运行。
    params = df.iloc[:, :param_dim].to_numpy(dtype=np.float32)

    # 除去参数列后，剩下的列必须一半 real、一半 imag。
    response_cols = df.shape[1] - param_dim
    if response_cols % 2 != 0:
        raise ValueError(f"响应列数量不是偶数，无法拆成 real/imag: {csv_path}")

    n_points = response_cols // 2
    real = df.iloc[:, param_dim : param_dim + n_points].to_numpy(dtype=np.float32)
    imag = df.iloc[:, param_dim + n_points : param_dim + 2 * n_points].to_numpy(dtype=np.float32)

    # 合成复数谱线，并按旧框架习惯做 stride=10 下采样。
    spectra = real + 1j * imag
    spectra = spectra[:, ::response_stride]
    return params, spectra.astype(np.complex64)


def pad_params(params: np.ndarray, target_dim: int = 6) -> np.ndarray:
    """把参数数组补齐到统一维度。

    这个函数目前主要是预留接口。
    Source / Self 物理上只有 4 个有效参数，但旧框架里经常补成 6 维，
    以便和 H-shaped 的 6 维参数保持接口一致。
    """
    if params.shape[1] >= target_dim:
        return params[:, :target_dim]

    padded = np.zeros((params.shape[0], target_dim), dtype=params.dtype)
    padded[:, : params.shape[1]] = params
    return padded


def sample_first_rows(
    csv_path: str | Path,
    param_dim: int,
    response_stride: int,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """从 CSV 的前几行快速采样。

    注意：这里不是随机采样，而是读取前 `sample_count` 行。
    这个函数只用于旧 CSV 兼容模式。正式数据主线请使用 raw_loaders.py
    和 prepare_processed_dataset.py。
    """
    return load_complex_csv(
        csv_path=csv_path,
        param_dim=param_dim,
        response_stride=response_stride,
        max_rows=sample_count,
    )
