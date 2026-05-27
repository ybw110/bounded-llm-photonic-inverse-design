from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from my_llm_framework.config import load_config
from my_llm_framework.data.processed import load_processed_split


def build_memory_bank_from_processed(
    cfg: dict,
    processed_root: str | Path,
    split: str = "train",
    domains: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """从 processed dataset 构建最终版 memory bank。

    这里不再直接读旧 `PNN_data`，也不再用 `sample_per_domain` 做小样本临时构建。
    正式逻辑是：

    - 原始数据先由 `prepare_processed_dataset.py` 生成可复现的 train / val / test split
    - memory bank 只读取 train split
    - val / test split 留给调参和最终评测，避免数据泄漏
    """
    processed_root = Path(processed_root)
    selected_domains = domains or list(cfg["domains"].keys())

    all_domains = []
    all_params = []
    all_params_raw = []
    all_spectra = []
    all_summary = []
    all_source_indices = []

    manifest = {
        "project_name": cfg["project_name"],
        "processed_root": str(processed_root.resolve()),
        "split": split,
        "domains": {},
    }

    summary_names: list[str] | None = None

    for domain_name in selected_domains:
        split_data = load_processed_split(processed_root, domain_name, split)
        n_rows = len(split_data["params"])

        all_domains.extend([domain_name] * n_rows)
        all_params.append(split_data["params"])
        all_params_raw.append(split_data["params_raw"])
        all_spectra.append(split_data["spectra"])
        all_summary.append(split_data["summary_features"])
        all_source_indices.append(split_data["source_indices"])

        if summary_names is None:
            summary_names = split_data["summary_names"]

        manifest["domains"][domain_name] = {
            "split_path": str(split_data["path"]),
            "rows": int(n_rows),
            "frequency_points": int(len(split_data["frequency"])),
        }

    spectra = np.vstack(all_spectra).astype(np.complex64)

    return {
        "domains": np.asarray(all_domains),
        "params": np.vstack(all_params).astype(np.float32),
        "params_raw": np.vstack(all_params_raw).astype(np.float32),
        "spectra_real": np.real(spectra).astype(np.float32),
        "spectra_imag": np.imag(spectra).astype(np.float32),
        "summary_features": np.vstack(all_summary).astype(np.float32),
        "summary_names": np.asarray(summary_names or []),
        "source_indices": np.concatenate(all_source_indices).astype(np.int64),
        "manifest_json": np.asarray(json.dumps(manifest, ensure_ascii=False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="实验配置 JSON，例如 configs/experiment_v1.json")
    parser.add_argument("--processed-root", default=None, help="processed dataset 目录；默认使用配置中的 processed_root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="用哪个 split 构建 memory bank")
    parser.add_argument("--domains", nargs="*", default=None, help="只纳入指定 domain；默认纳入配置中的全部 domain")
    parser.add_argument("--output", required=True, help="输出 memory bank 的 npz 路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(args.processed_root) if args.processed_root else Path(cfg["_processed_root_abs"])
    bank = build_memory_bank_from_processed(cfg, processed_root, args.split, args.domains)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **bank)

    domains, counts = np.unique(bank["domains"].astype(str), return_counts=True)
    print(f"Saved memory bank: {output.resolve()}")
    print(f"Rows: {len(bank['domains'])}")
    for domain, count in zip(domains, counts):
        print(f"{domain}: {count}")
    print(f"Summary features: {bank['summary_features'].shape}")


if __name__ == "__main__":
    main()
