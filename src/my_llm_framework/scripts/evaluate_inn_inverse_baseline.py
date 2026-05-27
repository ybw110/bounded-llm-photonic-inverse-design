from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from my_llm_framework.config import load_config, resolve_project_path
from my_llm_framework.data.processed import load_processed_split
from my_llm_framework.evaluation.metrics import is_success, spectrum_metric_bundle
from my_llm_framework.scripts.evaluate_retrieval_optimizer import thresholds_for_domain
from my_llm_framework.surrogates.inn_inverse import INNInverseSurrogate
from my_llm_framework.surrogates.pnn_forward import PNNForwardSurrogate


def evaluate_domain(
    cfg: dict[str, Any],
    processed_root: str | Path,
    domain: str,
    query_split: str,
    inn_model_root: str | Path,
    pnn_model_root: str | Path,
    batch_size: int,
    device: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """评估一个 domain 的 INN direct inverse baseline。

    流程：

    `target spectrum -> INN predicted params -> PNN forward spectrum -> metrics`

    这里不直接用参数 MSE 判断成功，因为逆向设计存在一对多问题；真正关心的是
    预测参数经 forward surrogate 后能不能恢复目标频谱。
    """
    domain_cfg = cfg["domains"][domain]
    active_dim = int(domain_cfg["active_param_dim"])
    thresholds = thresholds_for_domain(cfg, domain)

    data = load_processed_split(processed_root, domain, query_split)
    inn = INNInverseSurrogate(Path(inn_model_root) / domain / "model_best.pt", device=device)
    pnn = PNNForwardSurrogate(Path(pnn_model_root) / domain / "model_best.pt", device=device)

    predicted_params = inn.predict_batch(data["spectra"], batch_size=batch_size)
    predicted_spectra = pnn.predict_batch(predicted_params, batch_size=batch_size)
    true_params = data["params"][:, :active_dim].astype(np.float32)

    rows: list[dict[str, Any]] = []
    success_flags = []
    amplitude_mse_values = []
    amplitude_mae_values = []
    center_error_values = []
    bandwidth_error_values = []
    param_mse_values = []
    param_mae_values = []

    for query_idx in range(len(data["spectra"])):
        metrics = spectrum_metric_bundle(predicted_spectra[query_idx], data["spectra"][query_idx], data["frequency"])
        success = is_success(metrics, thresholds)
        param_diff = predicted_params[query_idx] - true_params[query_idx]
        param_mse = float(np.mean(param_diff * param_diff))
        param_mae = float(np.mean(np.abs(param_diff)))

        success_flags.append(success)
        amplitude_mse_values.append(metrics["amplitude_mse"])
        amplitude_mae_values.append(metrics["amplitude_mae"])
        center_error_values.append(metrics["center_frequency_error_thz"])
        bandwidth_error_values.append(metrics["bandwidth_error_thz"])
        param_mse_values.append(param_mse)
        param_mae_values.append(param_mae)

        row = {
            "domain": domain,
            "query_index": int(query_idx),
            "source_index": int(data["source_indices"][query_idx]),
            "success": bool(success),
            "param_mse": param_mse,
            "param_mae": param_mae,
        }
        for key, value in metrics.items():
            row[key] = value
        for i, value in enumerate(predicted_params[query_idx], start=1):
            row[f"predicted_param_{i}_norm"] = float(value)
        rows.append(row)

    summary = {
        "domain": domain,
        "num_queries": len(rows),
        "baseline_family": "direct_inverse",
        "method": "inn_inverse+pnn_forward",
        "setting": "deterministic",
        "inverse_model": "inn_inverse",
        "forward_surrogate": "pnn_forward",
        "evaluation_source": "pnn_forward",
        "success_rate_final": float(np.mean(success_flags)),
        "mean_final_amplitude_mse": float(np.mean(amplitude_mse_values)),
        "median_final_amplitude_mse": float(np.median(amplitude_mse_values)),
        "mean_final_amplitude_mae": float(np.mean(amplitude_mae_values)),
        "mean_center_frequency_error_thz": float(np.mean(center_error_values)),
        "mean_bandwidth_error_thz": float(np.mean(bandwidth_error_values)),
        "mean_param_mse": float(np.mean(param_mse_values)),
        "median_param_mse": float(np.median(param_mse_values)),
        "mean_param_mae": float(np.mean(param_mae_values)),
        "inn_model_path": str(Path(inn_model_root) / domain / "model_best.pt"),
        "pnn_model_path": str(Path(pnn_model_root) / domain / "model_best.pt"),
    }
    return rows, summary


def evaluate_inn_inverse_baseline(
    cfg: dict[str, Any],
    processed_root: str | Path,
    domains: list[str],
    query_split: str,
    output_dir: str | Path,
    inn_model_root: str | Path,
    pnn_model_root: str | Path,
    batch_size: int,
    device: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """批量评估 INN direct inverse baseline。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summaries = []
    for domain in domains:
        rows, summary = evaluate_domain(
            cfg=cfg,
            processed_root=processed_root,
            domain=domain,
            query_split=query_split,
            inn_model_root=inn_model_root,
            pnn_model_root=pnn_model_root,
            batch_size=batch_size,
            device=device,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    query_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summaries)
    query_df.to_csv(output_dir / "inn_inverse_query_results.csv", index=False)
    summary_df.to_csv(output_dir / "inn_inverse_summary.csv", index=False)
    return query_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="实验配置 JSON")
    parser.add_argument("--processed-root", default=None, help="processed dataset 目录；默认使用配置中的 processed_root")
    parser.add_argument("--domains", nargs="+", default=None, help="要评估的 domain；默认评估全部 domain")
    parser.add_argument("--query-split", default="test", choices=["train", "val", "test"], help="query split")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--inn-model-root", default="outputs/phase2_models/inn_inverse", help="INN 模型根目录")
    parser.add_argument("--pnn-model-root", default=None, help="PNN forward 模型根目录")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default=None, help="例如 cuda、cuda:0 或 cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(args.processed_root) if args.processed_root else Path(cfg["_processed_root_abs"])
    pnn_model_root = (
        Path(args.pnn_model_root)
        if args.pnn_model_root
        else resolve_project_path(cfg, cfg.get("pnn_training", {}).get("output_root", "outputs/phase2_models/pnn_forward"))
    )

    _, summary_df = evaluate_inn_inverse_baseline(
        cfg=cfg,
        processed_root=processed_root,
        domains=args.domains or list(cfg["domains"].keys()),
        query_split=args.query_split,
        output_dir=args.output_dir,
        inn_model_root=Path(args.inn_model_root),
        pnn_model_root=pnn_model_root,
        batch_size=int(args.batch_size),
        device=args.device,
    )

    print(f"Saved INN inverse baseline results: {Path(args.output_dir).resolve()}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
