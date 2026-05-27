from __future__ import annotations

import argparse
from pathlib import Path

from my_llm_framework.config import load_config
from my_llm_framework.data.processed import save_processed_split, split_indices, write_manifest
from my_llm_framework.data.raw_loaders import load_raw_domain
from my_llm_framework.features.physics_summary import SUMMARY_NAMES, summarize_batch


def prepare_processed_dataset(
    cfg: dict,
    output_root: str | Path,
    max_rows_per_domain: int | None = None,
) -> dict:
    """从原始数据生成正式 processed dataset。

    这一步是最终版 memory bank 的上游，不再依赖旧 `PNN_data` split。
    它完成四件事：

    1. 从 `cylinder.mat`、`H.mat`、`target_self.csv` 读取原始数据
    2. 按配置做参数归一化和频谱下采样
    3. 用固定随机种子重新划分 train / val / test
    4. 为每条谱线计算 physics summary，并保存为 `.npz`
    """
    output_root = Path(output_root)
    split_cfg = cfg["split"]
    seed = int(split_cfg["seed"])
    stride = int(cfg["response_stride"])

    manifest = {
        "project_name": cfg["project_name"],
        "legacy_root": cfg["_legacy_root_abs"],
        "processed_root": str(output_root.resolve()),
        "response_stride": stride,
        "split": split_cfg,
        "max_rows_per_domain": max_rows_per_domain,
        "summary_names": SUMMARY_NAMES,
        "domains": {},
    }

    for domain_name, domain_cfg in cfg["domains"].items():
        raw = load_raw_domain(
            domain_name=domain_name,
            domain_cfg=domain_cfg,
            legacy_root=cfg["_legacy_root_abs"],
            response_stride=stride,
            max_rows=max_rows_per_domain,
        )

        _, summary_matrix = summarize_batch(raw.spectra, raw.frequency)
        splits = split_indices(
            n_rows=len(raw.params_norm),
            seed=seed,
            train_fraction=float(split_cfg["train_fraction"]),
            val_fraction=float(split_cfg["val_fraction"]),
            test_fraction=float(split_cfg["test_fraction"]),
        )

        domain_manifest = {
            "raw_path": str(Path(cfg["_legacy_root_abs"]) / domain_cfg["raw_data"]["path"]),
            "raw_rows_loaded": int(len(raw.params_norm)),
            "frequency_points": int(len(raw.frequency)),
            "splits": {},
        }

        for split_name, indices in splits.items():
            save_path = save_processed_split(
                output_root=output_root,
                domain=domain_name,
                split=split_name,
                params=raw.params_norm[indices],
                params_raw=raw.params_raw[indices],
                spectra=raw.spectra[indices],
                frequency=raw.frequency,
                summary_features=summary_matrix[indices],
                summary_names=SUMMARY_NAMES,
                source_indices=raw.source_indices[indices],
            )
            domain_manifest["splits"][split_name] = {
                "rows": int(len(indices)),
                "path": str(save_path),
            }

        manifest["domains"][domain_name] = domain_manifest

    write_manifest(output_root, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="实验配置 JSON，例如 configs/experiment_v1.json")
    parser.add_argument("--output-root", default=None, help="processed dataset 输出目录；默认使用配置中的 processed_root")
    parser.add_argument(
        "--max-rows-per-domain",
        type=int,
        default=None,
        help="开发排错时限制每个 domain 读取多少行；正式实验不要设置这个参数",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(args.output_root) if args.output_root else Path(cfg["_processed_root_abs"])
    manifest = prepare_processed_dataset(cfg, output_root, args.max_rows_per_domain)

    print(f"Saved processed dataset: {output_root.resolve()}")
    for domain_name, domain_info in manifest["domains"].items():
        split_text = ", ".join(f"{name}={info['rows']}" for name, info in domain_info["splits"].items())
        print(f"{domain_name}: {split_text}")


if __name__ == "__main__":
    main()

