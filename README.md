# Attribution-aware evaluation of bounded language-model agents

This repository contains the code snapshot and lightweight source data for the manuscript:

**An Attribution-Aware Evaluation Framework for Bounded Language-Model Agents in Photonic Inverse Design**

The repository is intended for the maintainable GitHub version of the project. The DOI-linked archival version corresponding to the submitted manuscript should be deposited on Zenodo and cited in the manuscript.

## Repository layout

```text
src/                 Python package for retrieval, surrogates, planners, metrics and plotting
configs/             Experiment configuration files
prompts/             Planner, critic and specification-parser prompts
scripts/             Top-level plotting scripts used during manuscript figure generation
manuscript/          Manuscript LaTeX source and bibliography
source_data/         Lightweight source data currently prepared for manuscript checks
outputs/             Lightweight summary outputs
pyproject.toml       Python package metadata and dependency declaration
requirements.txt     Minimal installation dependencies
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Minimal reproduction order

The main archived data and outputs are provided in the Zenodo package. To regenerate the main processing chain from the processed data and memory bank, use the scripts under `src/my_llm_framework/scripts/` and the configuration in `configs/experiment_v1.json`.

Typical command families are:

```bash
python -m my_llm_framework.scripts.prepare_processed_dataset --config configs/experiment_v1.json
python -m my_llm_framework.scripts.build_memory_bank --config configs/experiment_v1.json
python -m my_llm_framework.scripts.evaluate_retrieval_baselines --config configs/experiment_v1.json
python -m my_llm_framework.scripts.evaluate_retrieval_optimizer --config configs/experiment_v1.json
python -m my_llm_framework.scripts.evaluate_llm_planner_offline --config configs/experiment_v1.json
python -m my_llm_framework.scripts.evaluate_llm_ood_recovery_planner --config configs/experiment_v1.json
```

Exact command lines and complete output tables are preserved in the Zenodo archive.

## Language-model calls

The manuscript evaluates DeepSeek and GLM-5.1 as bounded planners. Private API keys are not included. Prompt templates and planner schemas are provided under `prompts/` and `src/my_llm_framework/`.

## Archive link

Archived version corresponding to the manuscript: **[Zenodo DOI to be inserted]**.

