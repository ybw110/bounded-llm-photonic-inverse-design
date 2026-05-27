from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from my_llm_framework.config import load_config
from my_llm_framework.data.processed import load_processed_split
from my_llm_framework.features.physics_summary import amplitude


DEFAULT_FILTER_SPECS = {
    "source": {"center_frequency_thz": 45.0, "bandwidth_thz": 5.0},
    "target_h": {"center_frequency_thz": 45.0, "bandwidth_thz": 5.0},
    "target_self": {"center_frequency_thz": 0.75, "bandwidth_thz": 0.08},
}


def _structure_type(domain: str) -> str:
    if domain == "target_h":
        return "h_shape"
    if domain in {"source", "target_self"}:
        return "cylinder"
    raise ValueError(f"Unsupported meta-filter domain: {domain}")


def _gaussian_notch_amplitude(
    frequency: np.ndarray,
    *,
    center_frequency_thz: float,
    bandwidth_thz: float,
    depth: float,
) -> np.ndarray:
    sigma = max(float(bandwidth_thz) / 2.355, np.finfo(float).eps)
    notch = 1.0 - float(depth) * np.exp(-((frequency - float(center_frequency_thz)) ** 2) / (2.0 * sigma**2))
    return np.asarray(np.clip(notch, 0.0, 1.2), dtype=float)


def _candidate_center(frequency: np.ndarray, candidate_amplitude: np.ndarray) -> float:
    return float(frequency[int(np.argmin(candidate_amplitude))])


def _build_candidate_pool(
    cfg: dict[str, Any],
    *,
    domain: str,
    split: str,
    center_frequency_thz: float | None,
    bandwidth_thz: float | None,
    depth: float,
    candidates: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    defaults = DEFAULT_FILTER_SPECS[domain]
    center = float(center_frequency_thz if center_frequency_thz is not None else defaults["center_frequency_thz"])
    bandwidth = float(bandwidth_thz if bandwidth_thz is not None else defaults["bandwidth_thz"])
    if candidates <= 0:
        raise ValueError("candidates must be positive")

    processed_root = Path(cfg["_processed_root_abs"])
    data = load_processed_split(processed_root, domain, split)
    frequency = np.asarray(data["frequency"], dtype=float)
    target_amplitude = _gaussian_notch_amplitude(
        frequency,
        center_frequency_thz=center,
        bandwidth_thz=bandwidth,
        depth=depth,
    )
    candidate_spectra = np.asarray(data["spectra"])
    candidate_amplitudes = amplitude(candidate_spectra)
    amplitude_mse = np.mean((candidate_amplitudes - target_amplitude[None, :]) ** 2, axis=1)
    order = np.argsort(amplitude_mse)[: min(int(candidates), len(amplitude_mse))]

    domain_cfg = cfg["domains"][domain]
    active_dim = int(domain_cfg["active_param_dim"])
    param_names = list(domain_cfg["param_names"][:active_dim])
    rows: list[dict[str, Any]] = []
    spectra_rows: list[dict[str, Any]] = []
    for rank, dataset_idx in enumerate(order):
        amp = np.asarray(candidate_amplitudes[int(dataset_idx)], dtype=float)
        spectrum = np.asarray(candidate_spectra[int(dataset_idx)])
        row: dict[str, Any] = {
            "candidate_rank": int(rank),
            "dataset_index": int(dataset_idx),
            "source_index": int(data["source_indices"][int(dataset_idx)]),
            "amplitude_mse": float(amplitude_mse[int(dataset_idx)]),
            "candidate_center_frequency_thz": _candidate_center(frequency, amp),
            "candidate_min_amplitude": float(np.min(amp)),
            "candidate_mean_amplitude": float(np.mean(amp)),
        }
        for idx, name in enumerate(param_names):
            row[f"{name}_raw"] = float(data["params_raw"][int(dataset_idx), idx])
            row[f"{name}_norm"] = float(data["params"][int(dataset_idx), idx])
        rows.append(row)
        for freq, target_amp, raw_value, cand_amp in zip(frequency, target_amplitude, spectrum, amp):
            spectra_rows.append(
                {
                    "candidate_rank": int(rank),
                    "frequency_thz": float(freq),
                    "target_amplitude": float(target_amp),
                    "candidate_real": float(np.real(raw_value)),
                    "candidate_imag": float(np.imag(raw_value)),
                    "candidate_amplitude": float(cand_amp),
                }
            )

    pool = pd.DataFrame(rows)
    pool_spectra = pd.DataFrame(spectra_rows)
    spec = {
        "device_type": "on_demand_gaussian_notch_meta_filter",
        "domain": domain,
        "split": split,
        "structure_type": _structure_type(domain),
        "frequency_min_thz": float(frequency.min()),
        "frequency_max_thz": float(frequency.max()),
        "frequency_points": int(len(frequency)),
        "center_frequency_thz": center,
        "bandwidth_thz": bandwidth,
        "depth": float(depth),
        "candidates": int(len(pool)),
        "param_names": param_names,
    }
    target = {
        "frequency_thz": frequency.tolist(),
        "target_amplitude": target_amplitude.tolist(),
    }
    evidence = {
        "task": "Select one physically defined meta-filter candidate from the finite candidate pool.",
        "objective": "Prefer the candidate that best matches the requested Gaussian notch amplitude spectrum.",
        "design_spec": spec,
        "selection_schema": {
            "policy": "llm_metafilter_designer",
            "candidate_rank": 0,
            "reason": "short explanation",
        },
        "candidate_pool": _compact_pool_for_llm(pool),
    }
    return pool, pool_spectra, spec, target, evidence


def _compact_pool_for_llm(pool: pd.DataFrame) -> list[dict[str, Any]]:
    preferred = [
        "candidate_rank",
        "amplitude_mse",
        "candidate_center_frequency_thz",
        "candidate_min_amplitude",
        "candidate_mean_amplitude",
        "epsilon_raw",
        "gap_raw",
        "thickness_raw",
        "radius_raw",
        "refractive_index_raw",
        "Lx_raw",
        "Lx1_raw",
        "Ly_raw",
        "Ly1_raw",
    ]
    fields = [field for field in preferred if field in pool.columns]
    compact: list[dict[str, Any]] = []
    for item in pool[fields].to_dict(orient="records"):
        compact.append(
            {
                key: (round(float(value), 7) if isinstance(value, (float, np.floating)) else int(value))
                for key, value in item.items()
            }
        )
    return compact


def _manual_select(pool: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    selected = pool.sort_values(["amplitude_mse", "candidate_rank"], kind="stable").iloc[0].copy()
    return selected, {
        "policy": "manual_meta_filter_match",
        "reason": "Select the lowest-amplitude-MSE candidate from the finite spectral pool.",
    }


def _system_prompt() -> str:
    return (
        "You are a constrained metasurface meta-filter designer. Return exactly one JSON object and no prose. "
        "You are given a requested Gaussian notch response and a finite candidate pool. "
        "Select one candidate_rank already present in the pool. Prefer low amplitude_mse, correct notch center, "
        "and a deep but stable notch response. Do not invent parameters or spectra. "
        'The output must match this schema: {"policy":"llm_metafilter_designer","candidate_rank":0,"reason":"short reason"}.'
    )


def _llm_select(
    pool: pd.DataFrame,
    evidence: dict[str, Any],
    *,
    provider_label: str,
    api_key_env: str,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: float,
    use_response_format: bool,
    thinking_type: str | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"Missing {provider_label} API key. Set {api_key_env}.")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": json.dumps(evidence, ensure_ascii=True, allow_nan=False)},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    if use_response_format:
        payload["response_format"] = {"type": "json_object"}
    if thinking_type:
        payload["thinking"] = {"type": thinking_type}
    content = _post_chat_completion(
        payload,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        provider_label=provider_label,
    )
    decision = _loads_json_object(content)
    rank = _read_candidate_rank(decision)
    available_ranks = set(pool["candidate_rank"].astype(int))
    if rank not in available_ranks:
        fallback, _ = _manual_select(pool)
        rank = int(fallback["candidate_rank"])
        decision.setdefault("metadata", {})["fallback_reason"] = "missing_or_invalid_candidate_rank"
    selected = pool[pool["candidate_rank"].astype(int) == int(rank)].iloc[0].copy()
    decision["candidate_rank"] = int(rank)
    decision["provider"] = provider_label
    decision["model"] = model
    decision["raw_response"] = content
    return selected, decision


def _read_candidate_rank(decision: dict[str, Any]) -> int | None:
    for key in ("candidate_rank", "selected_candidate_rank", "rank"):
        if key in decision:
            try:
                return int(decision[key])
            except (TypeError, ValueError):
                return None
    return None


def _post_chat_completion(
    payload: dict[str, Any],
    *,
    api_key: str,
    base_url: str,
    timeout_seconds: float,
    provider_label: str,
) -> str:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{provider_label} API HTTP {exc.code}: {body[:500]}") from exc
    response_json = json.loads(body)
    choice = response_json["choices"][0]
    message = choice["message"]
    content = message.get("content")
    if content is None or not str(content).strip():
        reasoning = str(message.get("reasoning_content") or "")
        finish_reason = choice.get("finish_reason", "")
        raise RuntimeError(
            f"{provider_label} API returned empty message.content "
            f"(finish_reason={finish_reason!r}, reasoning_chars={len(reasoning)}). "
            "For GLM-5.1 device selection, keep thinking disabled or raise max_tokens."
        )
    return str(content)


def _loads_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


def _selected_spectrum(pool_spectra: pd.DataFrame, rank: int) -> pd.DataFrame:
    selected = pool_spectra[pool_spectra["candidate_rank"].astype(int) == int(rank)].copy()
    return selected.reset_index(drop=True)


def _write_cst_manifest(
    output_dir: Path,
    *,
    spec: dict[str, Any],
    selected: pd.Series,
    planner_policy: str,
) -> tuple[Path, Path]:
    case_id = f"phase6_metafilter_{planner_policy}_{spec['domain']}"
    input_dir = output_dir / "cst_inputs"
    result_dir = output_dir / "cst_results"
    input_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    param_names = list(spec["param_names"])
    candidate_raw = [float(selected[f"{name}_raw"]) for name in param_names]
    param_map = {name: value for name, value in zip(param_names, candidate_raw)}
    expected_result_csv = result_dir / f"{case_id}_cst_spectrum.csv"
    params_json = input_dir / f"{case_id}_params.json"
    params_csv = input_dir / f"{case_id}_params.csv"
    params_payload = {
        "case_id": case_id,
        "target_domain": spec["domain"],
        "structure_type": spec["structure_type"],
        "frequency_min_thz": spec["frequency_min_thz"],
        "frequency_max_thz": spec["frequency_max_thz"],
        "param_names": param_names,
        "candidate_raw": candidate_raw,
        "param_map": param_map,
        "source_case_json": str(output_dir / "phase6_metafilter_design.json"),
        "expected_result_csv": str(expected_result_csv),
    }
    params_json.write_text(json.dumps(params_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([param_map]).to_csv(params_csv, index=False)
    manifest = pd.DataFrame(
        [
            {
                "case_id": case_id,
                "target_domain": spec["domain"],
                "selected_bank_domain": spec["domain"],
                "structure_type": spec["structure_type"],
                "frequency_min_thz": spec["frequency_min_thz"],
                "frequency_max_thz": spec["frequency_max_thz"],
                "params_json": str(params_json),
                "params_csv": str(params_csv),
                "case_json": str(output_dir / "phase6_metafilter_design.json"),
                "expected_result_csv": str(expected_result_csv),
                "cst_status": "pending",
                **{f"param_{name}": value for name, value in param_map.items()},
            }
        ]
    )
    manifest_path = output_dir / "cst_batch_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest_path, params_json


def _write_outputs(
    output_dir: Path,
    *,
    pool: pd.DataFrame,
    pool_spectra: pd.DataFrame,
    spec: dict[str, Any],
    target: dict[str, Any],
    selected: pd.Series,
    decision: dict[str, Any],
    planner_policy: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = output_dir / "phase6_metafilter_candidate_pool.csv"
    selected_path = output_dir / "phase6_metafilter_selected_candidate.csv"
    spectra_path = output_dir / "phase6_metafilter_spectra.csv"
    pool_spectra_path = output_dir / "phase6_metafilter_candidate_spectra.csv"
    metrics_path = output_dir / "phase6_metafilter_metrics.csv"
    pool.to_csv(pool_path, index=False)
    pd.DataFrame([selected]).to_csv(selected_path, index=False)
    pool_spectra.to_csv(pool_spectra_path, index=False)

    selected_rank = int(selected["candidate_rank"])
    selected_spectrum = _selected_spectrum(pool_spectra, selected_rank)
    selected_spectrum.to_csv(spectra_path, index=False)
    manifest_path, params_json = _write_cst_manifest(
        output_dir,
        spec=spec,
        selected=selected,
        planner_policy=planner_policy,
    )
    metrics = {
        "candidate_rank": selected_rank,
        "amplitude_mse": float(selected["amplitude_mse"]),
        "candidate_center_frequency_thz": float(selected["candidate_center_frequency_thz"]),
        "center_frequency_error_thz": float(abs(selected["candidate_center_frequency_thz"] - spec["center_frequency_thz"])),
        "candidate_min_amplitude": float(selected["candidate_min_amplitude"]),
        "candidate_mean_amplitude": float(selected["candidate_mean_amplitude"]),
    }
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    summary = {
        "design_name": f"phase6_{planner_policy}_{spec['domain']}_metafilter",
        "phase": "phase6_device_level_meta_filter_extension",
        "planner_policy": planner_policy,
        **spec,
        **metrics,
        "target": target,
        "candidate_pool_csv": str(pool_path),
        "selected_candidate_csv": str(selected_path),
        "selected_spectrum_csv": str(spectra_path),
        "candidate_spectra_csv": str(pool_spectra_path),
        "metrics_csv": str(metrics_path),
        "cst_manifest_csv": str(manifest_path),
        "cst_params_json": str(params_json),
    }
    (output_dir / "phase6_metafilter_design.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "phase6_llm_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_readme(output_dir, summary, decision)


def _write_readme(output_dir: Path, summary: dict[str, Any], decision: dict[str, Any]) -> None:
    lines = [
        "# Phase 6 On-Demand Meta-Filter Design",
        "",
        "This folder stores one constrained Phase 6 meta-filter design from a finite unit-cell candidate pool.",
        "",
        "## Design",
        "",
        f"- Planner policy: `{summary['planner_policy']}`",
        f"- Domain: `{summary['domain']}`",
        f"- Structure type: `{summary['structure_type']}`",
        f"- Requested notch center: `{summary['center_frequency_thz']}` THz",
        f"- Requested FWHM bandwidth: `{summary['bandwidth_thz']}` THz",
        f"- Candidate rank: `{summary['candidate_rank']}`",
        f"- Candidate amplitude MSE: `{summary['amplitude_mse']:.8g}`",
        "",
        "## Planner decision",
        "",
        f"- Provider: `{decision.get('provider', 'manual')}`",
        f"- Reason: `{decision.get('reason', '')}`",
        "",
        "## Files",
        "",
        "- `phase6_metafilter_candidate_pool.csv`: finite candidate evidence exposed to the designer.",
        "- `phase6_metafilter_selected_candidate.csv`: selected physical geometry and spectral diagnostics.",
        "- `phase6_metafilter_spectra.csv`: requested target spectrum and selected offline candidate spectrum.",
        "- `phase6_metafilter_candidate_spectra.csv`: spectra for every exposed candidate rank.",
        "- `phase6_metafilter_design.json`: design metadata for plotting and collaboration handoff.",
        "- `cst_batch_manifest.csv`: one-case manifest compatible with `phase5_run_fixed_unit_cases.py`.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_v1.json")
    parser.add_argument("--planner-policy", choices=["manual", "deepseek", "glm"], default="manual")
    parser.add_argument("--domain", choices=["source", "target_h", "target_self"], default="source")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--center-frequency-thz", type=float, default=None)
    parser.add_argument("--bandwidth-thz", type=float, default=None)
    parser.add_argument("--depth", type=float, default=0.9)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase6_metafilter_manual_source_v1"))
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

    cfg = load_config(args.config)
    pool, pool_spectra, spec, target, evidence = _build_candidate_pool(
        cfg,
        domain=args.domain,
        split=args.split,
        center_frequency_thz=args.center_frequency_thz,
        bandwidth_thz=args.bandwidth_thz,
        depth=args.depth,
        candidates=args.candidates,
    )
    if args.planner_policy == "manual":
        selected, decision = _manual_select(pool)
        planner_policy = "manual_match"
    elif args.planner_policy == "deepseek":
        selected, decision = _llm_select(
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
        planner_policy = "deepseek_designer"
    else:
        selected, decision = _llm_select(
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
        planner_policy = "glm_designer"

    _write_outputs(
        args.output_dir,
        pool=pool,
        pool_spectra=pool_spectra,
        spec=spec,
        target=target,
        selected=selected,
        decision=decision,
        planner_policy=planner_policy,
    )
    print(f"Saved Phase 6 meta-filter design: {args.output_dir.resolve()}")
    print(pd.DataFrame([selected])[["candidate_rank", "amplitude_mse", "candidate_center_frequency_thz", "candidate_min_amplitude"]].to_string(index=False))


if __name__ == "__main__":
    main()
