from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from my_llm_framework.config import load_config
from my_llm_framework.scripts.design_phase6_llm_metafilter import (
    _build_candidate_pool,
    _llm_select,
    _manual_select,
    _write_outputs,
)


DEFAULT_REQUESTS = [
    {
        "request_id": "mf01_center45_bw5_d90",
        "request_label": "Hero deep notch",
        "center_frequency_thz": 45.0,
        "bandwidth_thz": 5.0,
        "depth": 0.90,
        "role": "Current hero request with a deep central notch.",
    },
    {
        "request_id": "mf02_center42_bw5_d85",
        "request_label": "Low-frequency shift",
        "center_frequency_thz": 42.0,
        "bandwidth_thz": 5.0,
        "depth": 0.85,
        "role": "Move the notch request toward the lower source-cylinder band.",
    },
    {
        "request_id": "mf03_center50_bw5_d85",
        "request_label": "High-frequency shift",
        "center_frequency_thz": 50.0,
        "bandwidth_thz": 5.0,
        "depth": 0.85,
        "role": "Move the notch request toward the higher source-cylinder band.",
    },
    {
        "request_id": "mf04_center45_bw8_d75",
        "request_label": "Broad shallow notch",
        "center_frequency_thz": 45.0,
        "bandwidth_thz": 8.0,
        "depth": 0.75,
        "role": "Change bandwidth and depth while keeping the centre fixed.",
    },
]

POLICY_OUTPUT_LABELS = {
    "manual": "manual",
    "deepseek": "deepseek",
    "glm": "glm51",
}

PLANNER_POLICY_LABELS = {
    "manual": "manual_match",
    "deepseek": "deepseek_designer",
    "glm": "glm_designer",
}


def _safe_id(value: str) -> str:
    allowed = [char.lower() if char.isalnum() else "_" for char in str(value).strip()]
    cleaned = "".join(allowed)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _read_request_table(request_specs_csv: Path | None) -> pd.DataFrame:
    if request_specs_csv is None:
        requests = pd.DataFrame(DEFAULT_REQUESTS)
    else:
        requests = pd.read_csv(request_specs_csv)

    required = {"request_id", "center_frequency_thz", "bandwidth_thz", "depth"}
    missing = sorted(required - set(requests.columns))
    if missing:
        raise ValueError(f"Request spec table is missing columns: {missing}")
    requests = requests.copy()
    requests["request_id"] = requests["request_id"].astype(str).map(_safe_id)
    if requests["request_id"].duplicated().any():
        duplicated = sorted(requests.loc[requests["request_id"].duplicated(), "request_id"].unique())
        raise ValueError(f"Request IDs must be unique: {duplicated}")
    if "request_label" not in requests.columns:
        requests["request_label"] = requests["request_id"]
    if "role" not in requests.columns:
        requests["role"] = ""
    for column in ["center_frequency_thz", "bandwidth_thz", "depth"]:
        requests[column] = pd.to_numeric(requests[column], errors="raise")
    if (requests["bandwidth_thz"] <= 0).any():
        raise ValueError("All bandwidth_thz values must be positive.")
    if ((requests["depth"] <= 0) | (requests["depth"] > 1.0)).any():
        raise ValueError("All depth values must lie in (0, 1].")
    return requests.reset_index(drop=True)


def _select_one_policy(
    policy: str,
    pool: pd.DataFrame,
    evidence: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[pd.Series, dict[str, Any]]:
    if policy == "manual":
        return _manual_select(pool)
    if policy == "deepseek":
        return _llm_select(
            pool,
            evidence,
            provider_label="deepseek",
            api_key_env=args.deepseek_api_key_env,
            model=args.deepseek_model,
            base_url=args.deepseek_base_url,
            temperature=args.deepseek_temperature,
            max_tokens=args.deepseek_max_tokens,
            timeout_seconds=args.deepseek_timeout_seconds,
            use_response_format=True,
        )
    if policy == "glm":
        return _llm_select(
            pool,
            evidence,
            provider_label="glm",
            api_key_env=args.glm_api_key_env,
            model=args.glm_model,
            base_url=args.glm_base_url,
            temperature=args.glm_temperature,
            max_tokens=args.glm_max_tokens,
            timeout_seconds=args.glm_timeout_seconds,
            use_response_format=not args.glm_disable_response_format,
            thinking_type=args.glm_thinking_type,
        )
    raise ValueError(f"Unsupported planner policy: {policy}")


def _load_manifest_row(run_dir: Path, *, audit_row_id: str, request: pd.Series, policy: str) -> pd.Series:
    manifest_path = run_dir / "cst_batch_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != 1:
        raise ValueError(f"Expected one CST row in {manifest_path}, found {len(manifest)}")
    row = manifest.iloc[0].copy()
    row["audit_row_id"] = audit_row_id
    row["audit_request_id"] = str(request["request_id"])
    row["audit_request_label"] = str(request["request_label"])
    row["audit_selector"] = policy
    row["audit_run_dir"] = _relpath(run_dir)
    row["case_id"] = f"phase6_{request['request_id']}_{POLICY_OUTPUT_LABELS[policy]}_{row['target_domain']}"
    return row


def _selection_row(
    *,
    audit_row_id: str,
    request: pd.Series,
    policy: str,
    run_dir: Path,
    spec: dict[str, Any],
    selected: pd.Series,
    decision: dict[str, Any],
) -> dict[str, Any]:
    active_params = {
        key: float(value)
        for key, value in selected.items()
        if key.endswith("_raw") and isinstance(value, (float, int, np.floating, np.integer))
    }
    row: dict[str, Any] = {
        "audit_row_id": audit_row_id,
        "request_id": str(request["request_id"]),
        "request_label": str(request["request_label"]),
        "request_role": str(request.get("role", "")),
        "selector": policy,
        "planner_policy": PLANNER_POLICY_LABELS[policy],
        "provider": str(decision.get("provider", "manual")),
        "model": str(decision.get("model", "")),
        "domain": spec["domain"],
        "split": spec["split"],
        "structure_type": spec["structure_type"],
        "center_frequency_thz": float(spec["center_frequency_thz"]),
        "bandwidth_thz": float(spec["bandwidth_thz"]),
        "depth": float(spec["depth"]),
        "candidate_pool_size": int(spec["candidates"]),
        "selected_rank": int(selected["candidate_rank"]),
        "dataset_index": int(selected["dataset_index"]),
        "source_index": int(selected["source_index"]),
        "offline_target_mse": float(selected["amplitude_mse"]),
        "offline_notch_center_thz": float(selected["candidate_center_frequency_thz"]),
        "offline_center_error_thz": float(abs(selected["candidate_center_frequency_thz"] - spec["center_frequency_thz"])),
        "offline_min_amplitude": float(selected["candidate_min_amplitude"]),
        "offline_mean_amplitude": float(selected["candidate_mean_amplitude"]),
        "decision_reason": str(decision.get("reason", "")),
        "run_dir": _relpath(run_dir),
        "candidate_pool_csv": _relpath(run_dir / "phase6_metafilter_candidate_pool.csv"),
        "selected_candidate_csv": _relpath(run_dir / "phase6_metafilter_selected_candidate.csv"),
        "selected_spectrum_csv": _relpath(run_dir / "phase6_metafilter_spectra.csv"),
        "decision_json": _relpath(run_dir / "phase6_llm_decision.json"),
        **active_params,
    }
    return row


def _candidate_pool_rows(request: pd.Series, pool: pd.DataFrame) -> pd.DataFrame:
    exported = pool.copy()
    exported.insert(0, "request_id", str(request["request_id"]))
    exported.insert(1, "request_label", str(request["request_label"]))
    exported.insert(2, "center_frequency_thz", float(request["center_frequency_thz"]))
    exported.insert(3, "bandwidth_thz", float(request["bandwidth_thz"]))
    exported.insert(4, "depth", float(request["depth"]))
    exported["candidate_center_error_thz"] = (
        exported["candidate_center_frequency_thz"].astype(float) - float(request["center_frequency_thz"])
    ).abs()
    exported["candidate_notch_depth"] = 1.0 - exported["candidate_min_amplitude"].astype(float)
    return exported


def _round_key(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(numeric):
        return "nan"
    return f"{numeric:.7f}"


def _geometry_key(row: pd.Series) -> str:
    fields = [
        "target_domain",
        "structure_type",
        "frequency_min_thz",
        "frequency_max_thz",
        *sorted(column for column in row.index if str(column).startswith("param_")),
    ]
    return "|".join(f"{field}={_round_key(row.get(field, ''))}" for field in fields)


def _build_unique_manifest(
    all_manifest: pd.DataFrame,
    selection_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = all_manifest.copy().reset_index(drop=True)
    manifest["geometry_key"] = manifest.apply(_geometry_key, axis=1)
    manifest["unique_geometry_index"] = manifest.groupby("geometry_key", sort=False).ngroup() + 1
    manifest["unique_geometry_id"] = manifest["unique_geometry_index"].map(lambda idx: f"mf_geom_{int(idx):02d}")

    unique_manifest = manifest.drop_duplicates("geometry_key", keep="first").copy()
    unique_manifest["case_id"] = unique_manifest.apply(
        lambda row: f"phase6_{row['audit_request_id']}_{row['audit_selector']}_{row['unique_geometry_id']}",
        axis=1,
    )
    unique_manifest = unique_manifest.drop(columns=["unique_geometry_index"])

    unique_lookup = unique_manifest[
        ["geometry_key", "unique_geometry_id", "case_id", "expected_result_csv", "audit_row_id"]
    ].rename(
        columns={
            "case_id": "unique_cst_case_id",
            "expected_result_csv": "unique_cst_expected_result_csv",
            "audit_row_id": "canonical_audit_row_id",
        }
    )
    row_lookup = manifest[["audit_row_id", "geometry_key", "unique_geometry_id"]].merge(
        unique_lookup,
        on=["geometry_key", "unique_geometry_id"],
        how="left",
    )
    selection_summary = selection_summary.merge(
        row_lookup.drop(columns=["geometry_key"]),
        on="audit_row_id",
        how="left",
    )
    unique_summary = (
        manifest.groupby(["unique_geometry_id", "geometry_key"], sort=False)
        .agg(
            requests=("audit_request_id", lambda values: ";".join(dict.fromkeys(map(str, values)))),
            selectors=("audit_selector", lambda values: ";".join(dict.fromkeys(map(str, values)))),
            duplicate_selection_count=("audit_row_id", "size"),
        )
        .reset_index()
        .merge(unique_lookup, on=["unique_geometry_id", "geometry_key"], how="left")
    )
    return unique_manifest, selection_summary, unique_summary


def _nonfast_result_csv(path_text: Any) -> str:
    result_path = Path(str(path_text))
    if result_path.parent.name == "cst_results":
        return _relpath(result_path.parent.parent / "cst_results_nonfast" / result_path.name)
    return _relpath(result_path.parent / "cst_results_nonfast" / result_path.name)


def _build_nonfast_manifest(unique_manifest: pd.DataFrame) -> pd.DataFrame:
    """Keep fast screening exports while reserving a formal non-fast result tree."""
    nonfast_manifest = unique_manifest.copy()
    nonfast_manifest["screening_result_csv"] = nonfast_manifest["expected_result_csv"].astype(str)
    nonfast_manifest["expected_result_csv"] = nonfast_manifest["screening_result_csv"].map(_nonfast_result_csv)
    nonfast_manifest["case_id"] = nonfast_manifest["case_id"].astype(str) + "_nonfast"
    nonfast_manifest["cst_status"] = "pending_nonfast"
    nonfast_manifest["cst_fidelity"] = "nonfast_formal"
    return nonfast_manifest


def _write_audit_readme(
    output_dir: Path,
    *,
    requests: pd.DataFrame,
    policies: list[str],
    selection_summary: pd.DataFrame,
    unique_manifest: pd.DataFrame,
) -> None:
    request_lines = [
        f"- `{row.request_id}`: center={row.center_frequency_thz:g} THz, BW={row.bandwidth_thz:g} THz, depth={row.depth:g}; {row.request_label}"
        for row in requests.itertuples(index=False)
    ]
    lines = [
        "# Phase 6 Meta-Filter Four-Request Audit",
        "",
        "This folder stores a bounded on-demand meta-filter audit over four requested Gaussian-notch functions.",
        "",
        "## Requests",
        "",
        *request_lines,
        "",
        "## Selector policies",
        "",
        f"- Policies run: `{', '.join(policies)}`",
        f"- Total selections: `{len(selection_summary)}`",
        f"- Unique CST geometries after parameter deduplication: `{len(unique_manifest)}`",
        "",
        "## Key files",
        "",
        "- `phase6_metafilter_audit_requests.csv`: request definitions.",
        "- `phase6_metafilter_audit_selection_summary.csv`: one selection row per request and selector.",
        "- `phase6_metafilter_audit_candidate_pools.csv`: all exposed candidate pools.",
        "- `phase6_metafilter_audit_cst_manifest_all.csv`: all selected cases before geometry deduplication.",
        "- `phase6_metafilter_audit_cst_manifest_unique.csv`: screening CST manifest for unique selected geometries.",
        "- `phase6_metafilter_audit_cst_manifest_unique_nonfast.csv`: non-fast formal CST manifest with isolated `cst_results_nonfast` outputs.",
        "- `phase6_metafilter_audit_unique_geometry_summary.csv`: duplicate-selection map for CST reuse.",
        "- `requests/<request_id>/<selector>/`: the ordinary single-request Phase 6 outputs.",
        "",
        "Run the screening or non-fast unique manifest unless repeated export of duplicate selections is explicitly required.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_audit_summary_json(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    requests: pd.DataFrame,
    selection_summary: pd.DataFrame,
    unique_manifest: pd.DataFrame,
) -> None:
    payload = {
        "phase": "phase6_meta_filter_four_request_audit",
        "domain": args.domain,
        "split": args.split,
        "candidate_pool_size_per_request": int(args.candidates),
        "planner_policies": list(args.planner_policies),
        "num_requests": int(len(requests)),
        "num_selections": int(len(selection_summary)),
        "num_unique_cst_geometries": int(len(unique_manifest)),
        "request_specs_csv": "phase6_metafilter_audit_requests.csv",
        "selection_summary_csv": "phase6_metafilter_audit_selection_summary.csv",
        "candidate_pools_csv": "phase6_metafilter_audit_candidate_pools.csv",
        "all_cst_manifest_csv": "phase6_metafilter_audit_cst_manifest_all.csv",
        "unique_cst_manifest_csv": "phase6_metafilter_audit_cst_manifest_unique.csv",
        "unique_nonfast_cst_manifest_csv": "phase6_metafilter_audit_cst_manifest_unique_nonfast.csv",
    }
    (output_dir / "phase6_metafilter_audit_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_v1.json")
    parser.add_argument("--request-specs-csv", type=Path, default=None)
    parser.add_argument("--planner-policies", nargs="+", choices=["manual", "deepseek", "glm"], default=["manual", "deepseek", "glm"])
    parser.add_argument("--domain", choices=["source", "target_h", "target_self"], default="source")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase6_metafilter_audit4_v1"))
    parser.add_argument("--deepseek-model", default="deepseek-chat")
    parser.add_argument("--deepseek-base-url", default="https://api.deepseek.com")
    parser.add_argument("--deepseek-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--deepseek-temperature", type=float, default=0.0)
    parser.add_argument("--deepseek-max-tokens", type=int, default=1024)
    parser.add_argument("--deepseek-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--glm-model", default="glm-5.1")
    parser.add_argument("--glm-base-url", default="https://open.bigmodel.cn/api/paas/v4")
    parser.add_argument("--glm-api-key-env", default="GLM_API_KEY")
    parser.add_argument("--glm-temperature", type=float, default=0.0)
    parser.add_argument("--glm-max-tokens", type=int, default=1024)
    parser.add_argument("--glm-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--glm-disable-response-format", action="store_true")
    parser.add_argument("--glm-thinking-type", choices=["disabled", "enabled"], default="disabled")
    args = parser.parse_args()

    if args.candidates <= 0:
        raise ValueError("--candidates must be positive.")

    requests = _read_request_table(args.request_specs_csv)
    cfg = load_config(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    requests.to_csv(output_dir / "phase6_metafilter_audit_requests.csv", index=False)

    selection_rows: list[dict[str, Any]] = []
    candidate_pool_frames: list[pd.DataFrame] = []
    manifest_rows: list[pd.Series] = []

    for request in requests.itertuples(index=False):
        request_series = pd.Series(request._asdict())
        pool, pool_spectra, spec, target, evidence = _build_candidate_pool(
            cfg,
            domain=args.domain,
            split=args.split,
            center_frequency_thz=float(request_series["center_frequency_thz"]),
            bandwidth_thz=float(request_series["bandwidth_thz"]),
            depth=float(request_series["depth"]),
            candidates=int(args.candidates),
        )
        candidate_pool_frames.append(_candidate_pool_rows(request_series, pool))
        for policy in args.planner_policies:
            selected, decision = _select_one_policy(policy, pool, evidence, args)
            run_dir = output_dir / "requests" / str(request_series["request_id"]) / POLICY_OUTPUT_LABELS[policy]
            _write_outputs(
                run_dir,
                pool=pool,
                pool_spectra=pool_spectra,
                spec=spec,
                target=target,
                selected=selected,
                decision=decision,
                planner_policy=PLANNER_POLICY_LABELS[policy],
            )
            audit_row_id = f"{request_series['request_id']}::{policy}"
            selection_rows.append(
                _selection_row(
                    audit_row_id=audit_row_id,
                    request=request_series,
                    policy=policy,
                    run_dir=run_dir,
                    spec=spec,
                    selected=selected,
                    decision=decision,
                )
            )
            manifest_rows.append(_load_manifest_row(run_dir, audit_row_id=audit_row_id, request=request_series, policy=policy))

    selection_summary = pd.DataFrame(selection_rows)
    all_manifest = pd.DataFrame(manifest_rows)
    all_manifest = all_manifest.reset_index(drop=True)
    unique_manifest, selection_summary, unique_summary = _build_unique_manifest(all_manifest, selection_summary)
    nonfast_manifest = _build_nonfast_manifest(unique_manifest)

    candidate_pools = pd.concat(candidate_pool_frames, ignore_index=True)
    selection_summary.to_csv(output_dir / "phase6_metafilter_audit_selection_summary.csv", index=False)
    candidate_pools.to_csv(output_dir / "phase6_metafilter_audit_candidate_pools.csv", index=False)
    all_manifest.to_csv(output_dir / "phase6_metafilter_audit_cst_manifest_all.csv", index=False)
    unique_manifest.to_csv(output_dir / "phase6_metafilter_audit_cst_manifest_unique.csv", index=False)
    nonfast_manifest.to_csv(output_dir / "phase6_metafilter_audit_cst_manifest_unique_nonfast.csv", index=False)
    unique_summary.to_csv(output_dir / "phase6_metafilter_audit_unique_geometry_summary.csv", index=False)
    _write_audit_readme(
        output_dir,
        requests=requests,
        policies=list(args.planner_policies),
        selection_summary=selection_summary,
        unique_manifest=unique_manifest,
    )
    _write_audit_summary_json(
        output_dir,
        args=args,
        requests=requests,
        selection_summary=selection_summary,
        unique_manifest=unique_manifest,
    )

    display_cols = [
        "request_id",
        "selector",
        "selected_rank",
        "offline_target_mse",
        "offline_notch_center_thz",
        "offline_center_error_thz",
        "offline_min_amplitude",
        "unique_geometry_id",
    ]
    print(f"Saved Phase 6 meta-filter audit: {output_dir.resolve()}")
    print(selection_summary[display_cols].to_string(index=False))
    print(f"\nUnique CST geometries: {len(unique_manifest)} from {len(selection_summary)} selections")
    print(f"Run CST on: {(output_dir / 'phase6_metafilter_audit_cst_manifest_unique.csv').resolve()}")


if __name__ == "__main__":
    main()
