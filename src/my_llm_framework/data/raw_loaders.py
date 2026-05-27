from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat


@dataclass
class RawDomainData:
    """一个 domain 从原始文件读出来后的统一数据结构。

    这里刻意同时保存 raw params 和 normalized params：

    - raw params：物理单位下的真实结构参数，便于论文解释和后续 CST 使用
    - normalized params：归一化到 0-1 附近的参数，便于模型、检索和优化接口统一

    spectrum 使用原始复数响应，不再沿用旧 split 中可能存在的二次归一化。
    对 retrieval 和 physics summary 来说，真实复数响应更接近物理含义。
    """

    domain: str
    params_raw: np.ndarray
    params_norm: np.ndarray
    spectra: np.ndarray
    frequency: np.ndarray
    source_indices: np.ndarray


def load_raw_domain(
    domain_name: str,
    domain_cfg: dict[str, Any],
    legacy_root: str | Path,
    response_stride: int,
    max_rows: int | None = None,
) -> RawDomainData:
    """按照配置读取一个原始 domain。

    当前 V1 主线支持三类数据：

    - source：`cylinder.mat`
    - target_h：`H.mat`
    - target_self：`target_self.csv`

    `freeform.mat` 暂时不放进 V1 主线，因为它包含 64 x 64 pattern，数据结构和论文问题会复杂很多。
    后续可以为它新增 loader，但不应该在第一版 memory bank 里混进来。
    """
    raw_cfg = domain_cfg["raw_data"]
    raw_type = raw_cfg["type"]

    if raw_type == "mat":
        return _load_mat_domain(domain_name, domain_cfg, legacy_root, response_stride, max_rows)
    if raw_type == "target_self_csv":
        return _load_target_self_csv(domain_name, domain_cfg, legacy_root, response_stride, max_rows)

    raise ValueError(f"未知 raw_data.type: {raw_type}")


def _load_mat_domain(
    domain_name: str,
    domain_cfg: dict[str, Any],
    legacy_root: str | Path,
    response_stride: int,
    max_rows: int | None,
) -> RawDomainData:
    """读取 MATLAB `.mat` 原始数据。

    `cylinder.mat` 和 `H.mat` 都包含：

    - `parameter`：结构参数矩阵
    - `real`：频谱响应实部
    - `imag`：频谱响应虚部

    H 数据的 `parameter` 有 7 列，旧工作只使用第 2-7 列作为
    thickness / refractive_index / Lx / Lx1 / Ly / Ly1。
    这个列选择由配置中的 `param_columns` 控制。
    """
    raw_cfg = domain_cfg["raw_data"]
    path = Path(legacy_root) / raw_cfg["path"]
    mat = loadmat(path)

    param_columns = list(raw_cfg["param_columns"])
    params_raw = np.asarray(mat[raw_cfg.get("param_key", "parameter")], dtype=np.float32)[:, param_columns]
    real = np.asarray(mat[raw_cfg.get("real_key", "real")], dtype=np.float32)
    imag = np.asarray(mat[raw_cfg.get("imag_key", "imag")], dtype=np.float32)

    if max_rows is not None:
        params_raw = params_raw[:max_rows]
        real = real[:max_rows]
        imag = imag[:max_rows]

    spectra_full = real + 1j * imag
    spectra = spectra_full[:, ::response_stride].astype(np.complex64)

    frequency_full = np.linspace(
        float(domain_cfg["frequency_min_thz"]),
        float(domain_cfg["frequency_max_thz"]),
        int(domain_cfg["raw_points"]),
        dtype=np.float32,
    )
    frequency = frequency_full[::response_stride]

    params_norm = normalize_and_pad_params(params_raw, raw_cfg)
    params_raw_padded = pad_raw_params(params_raw, int(raw_cfg.get("pad_to_param_dim", params_norm.shape[1])))
    source_indices = np.arange(len(params_norm), dtype=np.int64)

    return RawDomainData(
        domain=domain_name,
        params_raw=params_raw_padded,
        params_norm=params_norm,
        spectra=spectra,
        frequency=frequency,
        source_indices=source_indices,
    )


def _load_target_self_csv(
    domain_name: str,
    domain_cfg: dict[str, Any],
    legacy_root: str | Path,
    response_stride: int,
    max_rows: int | None,
) -> RawDomainData:
    """读取 `target_self.csv` 原始数据。

    这个 CSV 的前四列是结构参数，后面的每一列形如：

    `(0.5, (-0.5422-0.8394j))`

    也就是“频率 + 该频率处的复数响应”。这里会把频率和复数响应都解析出来。
    """
    raw_cfg = domain_cfg["raw_data"]
    path = Path(legacy_root) / raw_cfg["path"]
    df = pd.read_csv(path, nrows=max_rows)

    param_columns = list(raw_cfg["param_columns"])
    response_start = int(raw_cfg.get("response_start_column", max(param_columns) + 1))
    params_raw = df.iloc[:, param_columns].to_numpy(dtype=np.float32)

    response_values = df.iloc[:, response_start:].to_numpy(dtype=object)
    n_rows, n_points = response_values.shape
    spectra_full = np.empty((n_rows, n_points), dtype=np.complex64)
    frequency_full = np.empty(n_points, dtype=np.float32)

    for col in range(n_points):
        first_freq, first_value = parse_frequency_complex_cell(response_values[0, col])
        frequency_full[col] = first_freq
        spectra_full[0, col] = first_value

        if n_rows > 1:
            spectra_full[1:, col] = np.fromiter(
                (parse_frequency_complex_cell(value)[1] for value in response_values[1:, col]),
                dtype=np.complex64,
                count=n_rows - 1,
            )

    spectra = spectra_full[:, ::response_stride].astype(np.complex64)
    frequency = frequency_full[::response_stride]

    params_norm = normalize_and_pad_params(params_raw, raw_cfg)
    params_raw_padded = pad_raw_params(params_raw, int(raw_cfg.get("pad_to_param_dim", params_norm.shape[1])))
    source_indices = np.arange(len(params_norm), dtype=np.int64)

    return RawDomainData(
        domain=domain_name,
        params_raw=params_raw_padded,
        params_norm=params_norm,
        spectra=spectra,
        frequency=frequency,
        source_indices=source_indices,
    )


def parse_frequency_complex_cell(value: object) -> tuple[float, complex]:
    """解析 target_self CSV 中的单元格。

    使用 `ast.literal_eval` 而不是手写字符串切分，是因为它能安全解析 Python 字面量，
    同时天然支持 complex 数字，例如 `(-0.5+0.2j)`。
    """
    if pd.isna(value):
        raise ValueError("target_self.csv 中出现空响应单元格")

    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, tuple) or len(parsed) != 2:
        raise ValueError(f"无法解析频率-复数响应单元格: {value}")

    frequency = float(parsed[0])
    response = complex(parsed[1])
    return frequency, response


def normalize_and_pad_params(params_raw: np.ndarray, raw_cfg: dict[str, Any]) -> np.ndarray:
    """按物理范围把参数归一化，并补齐到统一维度。

    Source 和 Self 只有 4 个真实参数，但 H-shaped 有 6 个参数。
    为了让 memory bank 中 `params` 可以拼成一个矩阵，统一补齐到 6 维。
    补齐维度的 normalized value 默认是 0.5，表示这个维度在该 domain 中不参与设计。
    """
    ranges = np.asarray(raw_cfg["param_ranges"], dtype=np.float32)
    if ranges.shape[0] != params_raw.shape[1]:
        raise ValueError(f"param_ranges 数量 {ranges.shape[0]} 与参数列数 {params_raw.shape[1]} 不一致")

    low = ranges[:, 0]
    high = ranges[:, 1]
    denom = np.maximum(high - low, 1e-12)
    params_norm_active = (params_raw.astype(np.float32) - low) / denom

    target_dim = int(raw_cfg.get("pad_to_param_dim", params_norm_active.shape[1]))
    pad_value = float(raw_cfg.get("pad_value_norm", 0.5))
    padded = np.full((params_norm_active.shape[0], target_dim), pad_value, dtype=np.float32)
    padded[:, : params_norm_active.shape[1]] = params_norm_active
    return padded


def pad_raw_params(params_raw: np.ndarray, target_dim: int) -> np.ndarray:
    """把物理参数补齐到统一维度。

    raw params 中不存在的维度用 NaN 表示，而不是 0 或 0.5。
    这样后续导出给 CST 或论文表格时，不会误以为这些维度有真实物理含义。
    """
    padded = np.full((params_raw.shape[0], target_dim), np.nan, dtype=np.float32)
    padded[:, : params_raw.shape[1]] = params_raw.astype(np.float32)
    return padded

