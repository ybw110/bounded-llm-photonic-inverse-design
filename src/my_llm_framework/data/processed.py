from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


SPLIT_NAMES = ("train", "val", "test")


def split_indices(
    n_rows: int,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, np.ndarray]:
    """按固定随机种子划分 train / val / test。

    这里默认使用比例划分，而不是旧 LPR 里的固定 100/1000/100。
    这样同一个原始数据可以灵活生成完整训练集、少样本训练集或 upper-bound 训练集。
    """
    total = float(train_fraction + val_fraction + test_fraction)
    if total <= 0:
        raise ValueError("train/val/test fraction 之和必须大于 0")

    train_fraction /= total
    val_fraction /= total
    test_fraction /= total

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_rows)

    train_end = int(round(n_rows * train_fraction))
    val_end = train_end + int(round(n_rows * val_fraction))

    # 最后一个 split 吃掉余数，确保每个样本最多出现一次，并尽量用完整数据。
    return {
        "train": np.sort(shuffled[:train_end]),
        "val": np.sort(shuffled[train_end:val_end]),
        "test": np.sort(shuffled[val_end:]),
    }


def save_processed_split(
    output_root: str | Path,
    domain: str,
    split: str,
    params: np.ndarray,
    params_raw: np.ndarray,
    spectra: np.ndarray,
    frequency: np.ndarray,
    summary_features: np.ndarray,
    summary_names: list[str],
    source_indices: np.ndarray,
) -> Path:
    """保存一个 domain 的一个 split。

    每个 split 都保存为独立 `.npz`，后续构建 memory bank 或做 test 评测时可以按需加载。
    """
    if split not in SPLIT_NAMES:
        raise ValueError(f"未知 split: {split}")

    output_dir = Path(output_root) / domain
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.npz"

    np.savez_compressed(
        output_path,
        domain=np.asarray(domain),
        split=np.asarray(split),
        params=params.astype(np.float32),
        params_raw=params_raw.astype(np.float32),
        spectra_real=np.real(spectra).astype(np.float32),
        spectra_imag=np.imag(spectra).astype(np.float32),
        frequency=frequency.astype(np.float32),
        summary_features=summary_features.astype(np.float32),
        summary_names=np.asarray(summary_names),
        source_indices=source_indices.astype(np.int64),
    )
    return output_path


def load_processed_split(processed_root: str | Path, domain: str, split: str) -> dict[str, Any]:
    """读取一个 processed split，并把 real/imag 还原成 complex spectra。"""
    path = Path(processed_root) / domain / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"processed split 不存在: {path}")

    data = np.load(path, allow_pickle=False)
    return {
        "path": path,
        "domain": str(data["domain"].item()),
        "split": str(data["split"].item()),
        "params": data["params"].astype(np.float32),
        "params_raw": data["params_raw"].astype(np.float32),
        "spectra": (data["spectra_real"] + 1j * data["spectra_imag"]).astype(np.complex64),
        "frequency": data["frequency"].astype(np.float32),
        "summary_features": data["summary_features"].astype(np.float32),
        "summary_names": data["summary_names"].astype(str).tolist(),
        "source_indices": data["source_indices"].astype(np.int64),
    }


def write_manifest(output_root: str | Path, manifest: dict[str, Any]) -> Path:
    """保存 processed dataset 的 manifest。

    manifest 是复现实验时最重要的元数据：它记录原始数据路径、随机种子、划分比例和每个 split 的样本数。
    """
    output_path = Path(output_root) / "manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
