# Learned Planner Controls

This folder contains the source data used for the NMI learned-control and preview-withheld ablation.

- `learned_planner_summary.csv`: summary metrics for the exposed-preview and preview-withheld learned decision-list controls.
- `learned_planner_query_results.csv`: query-level out-of-fold decisions for the two learned controls.
- `nmi_policy_comparison_summary.csv`: comparison table combining the nominal-threshold replay policies and the learned controls.
- `learned_planner_latex_table.tex`: LaTeX table fragment used to prepare Supplementary Table S34.

The generating script is:

`MY-LLM-Framework/src/my_llm_framework/scripts/analyze_nmi_learned_planner_ablations.py`

The script reads the nominal-threshold few-shot OOD replay pool at $\tau=0.03$.
