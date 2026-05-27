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


C_UM_PER_PS = 299.792458


def _wrap_phase(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def _phase_error(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs(_wrap_phase(a - float(b)))


def _nearest_frequency_index(freq: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(freq - float(value))))


def _ideal_lens_phase(positions_um: np.ndarray, frequency_thz: float, focal_length_um: float) -> tuple[np.ndarray, float]:
    wavelength_um = C_UM_PER_PS / float(frequency_thz)
    k0 = 2.0 * np.pi / wavelength_um
    phase = _wrap_phase(-k0 * (np.sqrt(positions_um**2 + focal_length_um**2) - focal_length_um))
    return np.asarray(phase, dtype=float), float(wavelength_um)


def _build_candidate_pool(
    cfg: dict[str, Any],
    *,
    domain: str,
    split: str,
    design_frequency_thz: float,
    num_units: int,
    lattice_size_um: float,
    focal_length_um: float,
    min_amplitude: float,
    candidates_per_unit: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if num_units % 2 == 0:
        raise ValueError("num_units must be odd so the design has a central unit.")

    processed_root = Path(cfg["_processed_root_abs"])
    data = load_processed_split(processed_root, domain, split)
    freq = data["frequency"]
    freq_idx = _nearest_frequency_index(freq, design_frequency_thz)
    actual_frequency = float(freq[freq_idx])
    spectra_at_f = data["spectra"][:, freq_idx]
    unit_amp = amplitude(spectra_at_f)
    unit_phase = np.angle(spectra_at_f)

    valid = unit_amp >= float(min_amplitude)
    if not np.any(valid):
        raise ValueError(f"No unit cells satisfy min_amplitude={min_amplitude}")

    valid_indices = np.where(valid)[0]
    valid_phase = unit_phase[valid_indices]
    valid_amp = unit_amp[valid_indices]
    positions = (np.arange(num_units) - (num_units - 1) / 2.0) * float(lattice_size_um)
    ideal_phase, wavelength_um = _ideal_lens_phase(positions, actual_frequency, focal_length_um)

    param_names = cfg["domains"][domain]["param_names"][: int(cfg["domains"][domain]["active_param_dim"])]
    rows: list[dict[str, Any]] = []
    for unit_id, (x_um, target_phase) in enumerate(zip(positions, ideal_phase)):
        phase_err = _phase_error(valid_phase, float(target_phase))
        score = phase_err + 0.15 * (1.0 - np.clip(valid_amp, 0.0, 1.5))
        order = np.argsort(score)[: int(candidates_per_unit)]
        mirror_unit_id = int(num_units - 1 - unit_id)
        for rank, local_idx in enumerate(order):
            dataset_idx = int(valid_indices[int(local_idx)])
            row: dict[str, Any] = {
                "unit_id": int(unit_id),
                "mirror_unit_id": mirror_unit_id,
                "x_um": float(x_um),
                "candidate_rank": int(rank),
                "target_phase_rad": float(target_phase),
                "candidate_phase_rad": float(unit_phase[dataset_idx]),
                "phase_error_rad": float(phase_err[int(local_idx)]),
                "selected_amplitude": float(unit_amp[dataset_idx]),
                "candidate_score": float(score[int(local_idx)]),
                "dataset_index": dataset_idx,
                "source_index": int(data["source_indices"][dataset_idx]),
            }
            for idx, name in enumerate(param_names):
                row[f"{name}_raw"] = float(data["params_raw"][dataset_idx, idx])
                row[f"{name}_norm"] = float(data["params"][dataset_idx, idx])
            rows.append(row)

    pool = pd.DataFrame(rows)
    spec = {
        "domain": domain,
        "split": split,
        "design_frequency_thz_requested": float(design_frequency_thz),
        "design_frequency_thz_actual": actual_frequency,
        "num_units": int(num_units),
        "lattice_size_um": float(lattice_size_um),
        "aperture_um": float((num_units - 1) * lattice_size_um),
        "focal_length_um": float(focal_length_um),
        "wavelength_um": wavelength_um,
        "min_amplitude": float(min_amplitude),
        "candidates_per_unit": int(candidates_per_unit),
        "param_names": list(param_names),
    }
    evidence = {
        "task": "Select one candidate_rank for each metalens unit position.",
        "objective": "Minimize phase_error_rad while keeping selected_amplitude high and preserving mirror symmetry.",
        "design_spec": spec,
        "selection_schema": {
            "policy": "llm_device_designer",
            "selections": [{"unit_id": 0, "candidate_rank": 0}],
            "reason": "short explanation",
        },
        "candidate_pool": _compact_pool_for_llm(pool),
    }
    return pool, spec, evidence


def _compact_pool_for_llm(pool: pd.DataFrame) -> list[dict[str, Any]]:
    fields = [
        "unit_id",
        "mirror_unit_id",
        "x_um",
        "candidate_rank",
        "target_phase_rad",
        "candidate_phase_rad",
        "phase_error_rad",
        "selected_amplitude",
        "candidate_score",
        "epsilon_raw",
        "thickness_raw",
        "radius_raw",
    ]
    available = [field for field in fields if field in pool.columns]
    compact: list[dict[str, Any]] = []
    for item in pool[available].to_dict(orient="records"):
        compact.append(
            {
                key: (round(float(value), 5) if isinstance(value, (float, np.floating)) else int(value))
                for key, value in item.items()
            }
        )
    return compact


def _manual_select(pool: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = pool.sort_values(["unit_id", "candidate_score", "phase_error_rad", "selected_amplitude"], ascending=[True, True, True, False])
    selected = selected.groupby("unit_id", as_index=False).head(1).copy()
    return selected, {"policy": "manual_phase_match", "reason": "Fixed numeric score: phase error first, amplitude as tie-breaker."}


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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"Missing {provider_label} API key. Set {api_key_env}.")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": json.dumps(evidence, ensure_ascii=True)},
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
    selected = _apply_llm_selection(pool, decision)
    decision["raw_response"] = content
    decision["model"] = model
    decision["provider"] = provider_label
    return selected, decision


def _system_prompt() -> str:
    return (
        "You are a device-level metasurface planner. Return exactly one JSON object and no prose. "
        "You are given a finite metalens design spec and a candidate pool for each unit position. "
        "Select one candidate_rank for every unit_id. Prefer small phase_error_rad, high selected_amplitude, "
        "and mirror-symmetric choices for unit_id and mirror_unit_id when their target phases are equivalent. "
        "Do not invent new geometry. Only select candidate_rank values present in the provided pool. "
        "The output must match this schema: "
        '{"policy":"llm_device_designer","reason":"short reason","selections":[{"unit_id":0,"candidate_rank":0}]}'
    )


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
    content = content.strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


def _apply_llm_selection(pool: pd.DataFrame, decision: dict[str, Any]) -> pd.DataFrame:
    by_unit_rank = {
        (int(row.unit_id), int(row.candidate_rank)): row.Index
        for row in pool.reset_index().itertuples()
    }
    selected_indices: list[int] = []
    seen_units: set[int] = set()
    for item in decision.get("selections", []):
        try:
            unit_id = int(item["unit_id"])
            rank = int(item["candidate_rank"])
        except Exception:
            continue
        key = (unit_id, rank)
        if key in by_unit_rank and unit_id not in seen_units:
            selected_indices.append(int(by_unit_rank[key]))
            seen_units.add(unit_id)

    # Fill missing units with the deterministic manual selector so malformed or
    # partial LLM output still produces a valid auditable design.
    missing_units = sorted(set(pool["unit_id"].astype(int)) - seen_units)
    if missing_units:
        fallback, _ = _manual_select(pool[pool["unit_id"].isin(missing_units)])
        selected_indices.extend([int(idx) for idx in fallback.index])
        decision.setdefault("metadata", {})["filled_missing_units"] = missing_units

    selected = pool.loc[selected_indices].copy()
    selected = selected.sort_values("unit_id").reset_index(drop=True)
    return selected


def _score_design(selected: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    symmetry_errors: list[float] = []
    unit_to_phase = {int(row.unit_id): float(row.candidate_phase_rad) for row in selected.itertuples()}
    for row in selected.itertuples():
        uid = int(row.unit_id)
        mid = int(row.mirror_unit_id)
        if uid < mid and mid in unit_to_phase:
            symmetry_errors.append(float(abs(_wrap_phase(unit_to_phase[uid] - unit_to_phase[mid]))))
    return {
        "design_frequency_thz": float(spec["design_frequency_thz_actual"]),
        "num_units": int(spec["num_units"]),
        "aperture_um": float(spec["aperture_um"]),
        "focal_length_um": float(spec["focal_length_um"]),
        "mean_phase_error_rad": float(selected["phase_error_rad"].mean()),
        "max_phase_error_rad": float(selected["phase_error_rad"].max()),
        "mean_selected_amplitude": float(selected["selected_amplitude"].mean()),
        "min_selected_amplitude": float(selected["selected_amplitude"].min()),
        "mean_candidate_score": float(selected["candidate_score"].mean()),
        "mirror_phase_mismatch_rad": float(np.mean(symmetry_errors)) if symmetry_errors else 0.0,
    }


def _write_outputs(
    output_dir: Path,
    *,
    pool: pd.DataFrame,
    selected: pd.DataFrame,
    spec: dict[str, Any],
    decision: dict[str, Any],
    planner_policy: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_csv = output_dir / "phase6_device_candidate_pool.csv"
    units_csv = output_dir / "phase6_metalens_units.csv"
    phase_csv = output_dir / "phase6_metalens_phase_profile.csv"
    pool.to_csv(pool_csv, index=False)

    units = selected.rename(columns={"candidate_phase_rad": "selected_phase_rad"}).copy()
    units.to_csv(units_csv, index=False)
    units[
        [
            "unit_id",
            "x_um",
            "target_phase_rad",
            "selected_phase_rad",
            "phase_error_rad",
            "selected_amplitude",
            "candidate_rank",
        ]
    ].to_csv(phase_csv, index=False)

    metrics = _score_design(selected, spec)
    summary = {
        "design_name": f"phase6_{planner_policy}_{spec['domain']}_cylinder_metalens",
        "phase": "phase6_device_level_llm_design",
        "planner_policy": planner_policy,
        **spec,
        **metrics,
        "candidate_pool_csv": str(pool_csv),
        "units_csv": str(units_csv),
        "phase_profile_csv": str(phase_csv),
        "expected_cst_project_dir": str(output_dir / "cst_projects"),
        "expected_cst_field_csv": str(output_dir / "cst_results" / "phase5_hero_metalens_field_export_expected.txt"),
    }
    (output_dir / "phase6_metalens_design.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "phase6_llm_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(output_dir / "phase6_metalens_design_metrics.csv", index=False)
    _write_readme(output_dir, summary, decision)


def _write_readme(output_dir: Path, summary: dict[str, Any], decision: dict[str, Any]) -> None:
    lines = [
        "# Phase 6 Device-Level Metalens Design",
        "",
        "这个目录保存 Phase 6 的器件级设计结果。Phase 6 和 Phase 5 分开：Phase 5 验证 unit-cell CST closed loop，Phase 6 比较人工相位匹配器件设计和 LLM evidence-guided 器件设计。",
        "",
        "## Design",
        "",
        f"- Planner policy: `{summary['planner_policy']}`",
        f"- Domain: `{summary['domain']}`",
        f"- Design frequency: `{summary['design_frequency_thz_actual']:.4f}` THz",
        f"- Number of units: `{summary['num_units']}`",
        f"- Aperture: `{summary['aperture_um']:.4f}` um",
        f"- Focal length: `{summary['focal_length_um']:.4f}` um",
        "",
        "## Metrics",
        "",
        f"- Mean phase error: `{summary['mean_phase_error_rad']:.6f}` rad",
        f"- Max phase error: `{summary['max_phase_error_rad']:.6f}` rad",
        f"- Mean selected amplitude: `{summary['mean_selected_amplitude']:.6f}`",
        f"- Min selected amplitude: `{summary['min_selected_amplitude']:.6f}`",
        f"- Mirror phase mismatch: `{summary['mirror_phase_mismatch_rad']:.6f}` rad",
        "",
        "## LLM / planner decision",
        "",
        f"- Reason: `{decision.get('reason', '')}`",
        "",
        "## Files",
        "",
        "- `phase6_device_candidate_pool.csv`: 每个位置暴露给 planner 的候选 unit pool。",
        "- `phase6_metalens_units.csv`: planner 最终选择的器件级 unit layout，可交给 CST runner。",
        "- `phase6_metalens_design.json`: CST runner 和绘图脚本使用的设计元数据。",
        "- `phase6_llm_decision.json`: LLM 或人工 planner 的结构化决策记录。",
        "- `phase6_metalens_design_metrics.csv`: 设计质量指标。",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_v1.json")
    parser.add_argument("--planner-policy", choices=["manual", "deepseek", "glm"], default="manual")
    parser.add_argument("--domain", default="source", choices=["source", "target_self"])
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--design-frequency-thz", type=float, default=48.0)
    parser.add_argument("--num-units", type=int, default=31)
    parser.add_argument("--lattice-size-um", type=float, default=3.5)
    parser.add_argument("--focal-length-um", type=float, default=80.0)
    parser.add_argument("--min-amplitude", type=float, default=0.20)
    parser.add_argument("--candidates-per-unit", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase6_metalens_manual_source_v1"))
    parser.add_argument("--deepseek-model", default="deepseek-chat")
    parser.add_argument("--deepseek-base-url", default="https://api.deepseek.com")
    parser.add_argument("--deepseek-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--deepseek-temperature", type=float, default=0.0)
    parser.add_argument("--deepseek-max-tokens", type=int, default=4096)
    parser.add_argument("--deepseek-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--glm-model", default="glm-5.1")
    parser.add_argument("--glm-base-url", default="https://open.bigmodel.cn/api/paas/v4")
    parser.add_argument("--glm-api-key-env", default="GLM_API_KEY")
    parser.add_argument("--glm-temperature", type=float, default=0.0)
    parser.add_argument("--glm-max-tokens", type=int, default=4096)
    parser.add_argument("--glm-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--glm-disable-response-format",
        action="store_true",
        help="Disable OpenAI-compatible JSON response_format for GLM models that do not support it.",
    )
    parser.add_argument("--glm-thinking-type", choices=["disabled", "enabled"], default="disabled")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pool, spec, evidence = _build_candidate_pool(
        cfg,
        domain=args.domain,
        split=args.split,
        design_frequency_thz=args.design_frequency_thz,
        num_units=args.num_units,
        lattice_size_um=args.lattice_size_um,
        focal_length_um=args.focal_length_um,
        min_amplitude=args.min_amplitude,
        candidates_per_unit=args.candidates_per_unit,
    )
    if args.planner_policy == "manual":
        selected, decision = _manual_select(pool)
        planner_policy = "manual_phase_match"
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
        planner_policy = "deepseek_device_designer"
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
        planner_policy = "glm_device_designer"

    _write_outputs(args.output_dir, pool=pool, selected=selected, spec=spec, decision=decision, planner_policy=planner_policy)
    print(f"Saved Phase 6 device-level design: {args.output_dir.resolve()}")
    print(json.dumps(_score_design(selected, spec), indent=2))
    print(selected[["unit_id", "x_um", "candidate_rank", "phase_error_rad", "selected_amplitude"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
