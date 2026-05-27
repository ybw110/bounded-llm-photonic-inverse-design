from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_METRICS = (
    "success_at_50",
    "target_domain_success_at_50",
    "top1_target_domain_rate",
    "mean_final_top1_amplitude_mse",
    "mean_final_best_amplitude_mse_at_budget",
    "mean_steps_used",
    "mean_fewshot_cost",
    "oracle_match_rate",
    "fallback_or_invalid_action_rate",
    "oracle_target_domain_success_at_50",
    "oracle_mean_fewshot_cost",
    "oracle_gap_target_domain_success_at_50",
)


def _parse_run_spec(spec: str) -> tuple[str | None, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), Path(path)
    return None, Path(spec)


def _infer_repeat_id(path: Path, fallback: int) -> int:
    text = path.name.lower()
    for pattern in (r"repeat[_-]?(\d+)", r"rep[_-]?(\d+)", r"_r(\d+)", r"run[_-]?(\d+)"):
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return fallback


def _read_run(label: str | None, path: Path, fallback_repeat: int) -> pd.DataFrame:
    summary_path = path / "ood_recovery_planner_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    df = pd.read_csv(summary_path)
    df["run_label"] = label or path.name
    df["repeat_id"] = _infer_repeat_id(path, fallback_repeat)
    df["output_dir"] = str(path)
    if "planner_provider" not in df.columns:
        df["planner_provider"] = df["planner_policy"].astype(str).str.split("_").str[0]
    if "planner_model" not in df.columns:
        df["planner_model"] = ""
    return df


def _mean_std(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    return {
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=1)) if len(numeric.dropna()) > 1 else 0.0,
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def summarize_runs(run_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["run_label", "planner_provider", "planner_model", "target_domain"]
    for keys, group in run_df.groupby(group_cols, dropna=False, sort=False):
        base = dict(zip(group_cols, keys, strict=False))
        base["num_repeats"] = int(group["repeat_id"].nunique())
        base["num_rows"] = int(len(group))
        for metric in metrics:
            if metric not in group.columns:
                continue
            stats = _mean_std(group[metric])
            for stat_name, value in stats.items():
                base[f"{metric}_{stat_name}"] = value
        rows.append(base)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directories, optionally labelled as Label=path.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for idx, spec in enumerate(args.runs, start=1):
        label, path = _parse_run_spec(spec)
        frames.append(_read_run(label, path, idx))
    repeat_df = pd.concat(frames, ignore_index=True)
    aggregate_df = summarize_runs(repeat_df, args.metrics)

    repeat_df.to_csv(output_dir / "phase4b_multimodel_repeat_rows.csv", index=False)
    aggregate_df.to_csv(output_dir / "phase4b_multimodel_repeat_summary.csv", index=False)

    print(f"Saved Phase 4B multimodel repeat summary: {output_dir.resolve()}")
    print("\nRepeat rows:")
    print(repeat_df.to_string(index=False))
    print("\nAggregate:")
    print(aggregate_df.to_string(index=False))


if __name__ == "__main__":
    main()
