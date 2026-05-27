"""Train-test split diagnostics for the NMI manuscript.

This script checks whether train and test splits share original source indices
and quantifies how close each test query is to its train-memory neighbourhood.
The Target-Self domain is expected to be retrieval-saturated; these diagnostics
document that this saturation is a calibration-like property of the split rather
than test-query insertion into the train memory bank.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DOMAINS = ("source", "target_h", "target_self")
SUCCESS_THRESHOLD = 0.03


def load_npz(processed_root: Path, domain: str, split: str) -> dict[str, np.ndarray]:
    path = processed_root / domain / f"{split}.npz"
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def amplitude(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.sqrt(data["spectra_real"] ** 2 + data["spectra_imag"] ** 2).astype(np.float32)


def pairwise_nearest_mse(train_amp: np.ndarray, test_amp: np.ndarray, chunk: int = 512) -> np.ndarray:
    out = np.empty(test_amp.shape[0], dtype=np.float64)
    train_norm = np.sum(train_amp.astype(np.float64) ** 2, axis=1)
    for start in range(0, test_amp.shape[0], chunk):
        stop = min(start + chunk, test_amp.shape[0])
        block = test_amp[start:stop].astype(np.float64)
        block_norm = np.sum(block ** 2, axis=1, keepdims=True)
        dist2 = block_norm + train_norm[None, :] - 2.0 * block @ train_amp.T
        dist2 = np.maximum(dist2, 0.0) / train_amp.shape[1]
        out[start:stop] = np.min(dist2, axis=1)
    return out


def rbf_mmd2(x: np.ndarray, y: np.ndarray, max_n: int = 4000) -> float:
    rng = np.random.default_rng(48)
    if x.shape[0] > max_n:
        x = x[rng.choice(x.shape[0], size=max_n, replace=False)]
    if y.shape[0] > max_n:
        y = y[rng.choice(y.shape[0], size=max_n, replace=False)]
    xy = np.vstack([x, y]).astype(np.float64)
    d2_sample = np.sum((xy[: min(500, len(xy)), None, :] - xy[None, : min(500, len(xy)), :]) ** 2, axis=2)
    gamma = 1.0 / (np.median(d2_sample[d2_sample > 0]) + 1e-12)

    def kernel_mean(a: np.ndarray, b: np.ndarray) -> float:
        d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return float(np.exp(-gamma * d2).mean())

    return kernel_mean(x, x) + kernel_mean(y, y) - 2.0 * kernel_mean(x, y)


def latex_table(summary: pd.DataFrame, overlap: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp!]",
        r"\centering",
        r"\caption{\textbf{Train--test split diagnostics and nearest-neighbour similarity.}",
        r"Overlap counts are computed from original source indices. Nearest-neighbour MSE is the amplitude MSE from each held-out test query to its closest train-memory entry in amplitude space. The final column reports the fraction of held-out queries whose nearest amplitude-space train neighbour is below the nominal MSE threshold of 0.03.}",
        r"\label{tab:si_split_similarity}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Domain & Train & Test & Index overlap & NN MSE median & NN MSE 95th pct. & NN below 0.03 \\",
        r"\midrule",
    ]
    overlap_map = dict(zip(overlap["domain"], overlap["source_index_overlap_count"]))
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.domain} & {row.train_n:,} & {row.test_n:,} & {int(overlap_map[row.domain])} & "
            f"${row.top1_mse_median:.3e}$ & ${row.top1_mse_p95:.3e}$ & {row.success_at_1:.4f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nmi_split_similarity_v1"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, float | int | str]] = []
    overlap_rows: list[dict[str, int | str]] = []

    for domain in DOMAINS:
        train = load_npz(args.processed_root, domain, "train")
        test = load_npz(args.processed_root, domain, "test")
        train_amp = amplitude(train)
        test_amp = amplitude(test)
        nn_mse = pairwise_nearest_mse(train_amp, test_amp)
        summary_train = train["summary_features"].astype(np.float64)
        summary_test = test["summary_features"].astype(np.float64)

        overlap = np.intersect1d(train["source_indices"], test["source_indices"]).size
        overlap_rows.append(
            {
                "domain": domain,
                "train_n": int(train_amp.shape[0]),
                "test_n": int(test_amp.shape[0]),
                "source_index_overlap_count": int(overlap),
            }
        )
        summary_rows.append(
            {
                "domain": domain,
                "train_n": int(train_amp.shape[0]),
                "test_n": int(test_amp.shape[0]),
                "summary_mean_distance_l2": float(
                    np.linalg.norm(summary_train.mean(axis=0) - summary_test.mean(axis=0))
                ),
                "summary_rbf_mmd2": rbf_mmd2(summary_train, summary_test),
                "top1_mse_mean": float(nn_mse.mean()),
                "top1_mse_median": float(np.median(nn_mse)),
                "top1_mse_p95": float(np.percentile(nn_mse, 95)),
                "success_at_1": float((nn_mse < SUCCESS_THRESHOLD).mean()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    overlap = pd.DataFrame(overlap_rows)
    summary.to_csv(args.output_dir / "split_similarity_summary.csv", index=False)
    overlap.to_csv(args.output_dir / "split_index_overlap.csv", index=False)
    (args.output_dir / "split_similarity_latex_table.tex").write_text(
        latex_table(summary, overlap),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(overlap.to_string(index=False))


if __name__ == "__main__":
    main()
