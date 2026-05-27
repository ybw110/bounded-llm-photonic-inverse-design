from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _structure_type(domain: str) -> str:
    if domain == "target_h":
        return "h_shape"
    if domain in {"source", "target_self"}:
        return "cylinder"
    raise ValueError(f"Unsupported domain for CST batch preparation: {domain}")


def _load_case(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_cst_batch(case_dir: Path, output_dir: Path, *, include_optional_source_cases: bool) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "cst_inputs"
    result_dir = output_dir / "cst_results"
    input_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for case_path in sorted(case_dir.glob("*.json")):
        payload = _load_case(case_path)
        case_id = str(payload["case_id"])
        target_domain = str(payload["target"]["domain"])
        selected_bank_domain = str(payload["retrieval_candidate"]["selected_bank_domain"])

        if not include_optional_source_cases and target_domain != selected_bank_domain:
            continue

        param_names = list(payload["parameters"]["param_names"])
        candidate_raw = list(payload["parameters"]["candidate_raw"])
        param_map = {name: value for name, value in zip(param_names, candidate_raw)}

        params_json = input_dir / f"{case_id}_params.json"
        params_csv = input_dir / f"{case_id}_params.csv"
        result_csv = result_dir / f"{case_id}_cst_spectrum.csv"

        params_payload = {
            "case_id": case_id,
            "target_domain": target_domain,
            "structure_type": _structure_type(target_domain if target_domain == selected_bank_domain else selected_bank_domain),
            "frequency_min_thz": payload["target"]["frequency_min_thz"],
            "frequency_max_thz": payload["target"]["frequency_max_thz"],
            "param_names": param_names,
            "candidate_raw": candidate_raw,
            "param_map": param_map,
            "source_case_json": str(case_path),
            "expected_result_csv": str(result_csv),
        }
        params_json.write_text(json.dumps(params_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([param_map]).to_csv(params_csv, index=False)

        row = {
            "case_id": case_id,
            "target_domain": target_domain,
            "selected_bank_domain": selected_bank_domain,
            "structure_type": params_payload["structure_type"],
            "frequency_min_thz": params_payload["frequency_min_thz"],
            "frequency_max_thz": params_payload["frequency_max_thz"],
            "params_json": str(params_json),
            "params_csv": str(params_csv),
            "case_json": str(case_path),
            "expected_result_csv": str(result_csv),
            "cst_status": "pending",
        }
        row.update({f"param_{name}": value for name, value in param_map.items()})
        rows.append(row)

    if not rows:
        raise ValueError("No CST batch cases were prepared. Check case_dir or include optional source-domain cases.")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "cst_batch_manifest.csv", index=False)
    write_readme(output_dir, manifest, include_optional_source_cases)
    return manifest


def write_readme(output_dir: Path, manifest: pd.DataFrame, include_optional_source_cases: bool) -> None:
    lines = [
        "# Phase 5 CST Batch Inputs",
        "",
        "本目录把 Phase 5 case JSON 转换成 legacy CST 自动化更容易读取的批处理输入。",
        "",
        "## Files",
        "",
        "- `cst_batch_manifest.csv`: CST 批处理清单。",
        "- `cst_inputs/*_params.json`: 每个 case 的固定参数 JSON。",
        "- `cst_inputs/*_params.csv`: 每个 case 的固定参数 CSV。",
        "- `cst_results/`: 建议保存 CST 输出谱线的位置。",
        "",
        "## Included cases",
        "",
        "```text",
        manifest[["case_id", "target_domain", "selected_bank_domain", "structure_type", "expected_result_csv"]].to_string(index=False),
        "```",
        "",
        "## Notes",
        "",
        "默认只保留 target_domain 与 selected_bank_domain 一致的主文 case。这样 Target-H 的 CST 验证不会误用 source/cylinder 结构。",
        f"`include_optional_source_cases` 当前为 `{include_optional_source_cases}`。",
        "",
        "legacy CST runner 应读取 `params_json`，按 `structure_type` 选择 cylinder 或 h_shape 建模，并将真实仿真谱线保存到 `expected_result_csv`。",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, default=Path("outputs/phase5_cst_case_study_v1/cases"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase5_cst_case_study_v1"))
    parser.add_argument("--include-optional-source-cases", action="store_true")
    args = parser.parse_args()

    manifest = prepare_cst_batch(
        args.case_dir,
        args.output_dir,
        include_optional_source_cases=args.include_optional_source_cases,
    )
    print(f"Saved Phase 5 CST batch manifest: {(args.output_dir / 'cst_batch_manifest.csv').resolve()}")
    print(manifest[["case_id", "target_domain", "selected_bank_domain", "structure_type"]].to_string(index=False))


if __name__ == "__main__":
    main()
