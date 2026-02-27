# ML4Env Critical Review — Open Data Package

## Overview

This package contains the complete dataset, statistical summaries, figure source data, LLM extraction pipeline materials, and supporting information tables for a systematic audit of 126 peer-reviewed studies applying machine learning to predict adsorption or degradation performance in environmental remediation.

| Statistic | Value |
|:----------|:------|
| Papers analyzed | 126 |
| Year range | 2018–2025 |
| Research type | Adsorption (110), Degradation (14), Both (2) |
| Data source | Experimental (55), Literature (41), Database (28), Mixed (2) |
| Unique algorithms | 30 |
| Median R² (test set) | 0.978 (n=109) |
| Code available | 16/126 (12.7%) |
| Data available | 40/126 (31.7%) |

## Package Contents

```
├── README.md                              (this file)
├── dataset/
│   ├── ml4env_126_dataset.json            Main dataset (126 records × 49 fields)
│   ├── ml4env_126_dataset.csv             Same dataset in CSV format
│   └── ml4env_126_metadata.json           Field definitions and allowed values
├── statistics/
│   └── statistics_summary.json            Pre-computed statistics used in the manuscript
├── figure_data/
│   ├── fig1_llm_benchmark.json            Fig. 1: LLM extraction benchmark results
│   ├── fig2_bibliometric.json             Fig. 2: Corpus overview and temporal trends
│   ├── fig3_data_bias.json                Fig. 3: Data quality and bias indicators
│   ├── fig4_validation.json               Fig. 4: Validation practice distributions
│   ├── fig5_comparability.json            Fig. 5: Algorithm and metric landscapes
│   └── fig6_deployment.json               Fig. 6: Deployment readiness indicators
├── llm_extraction/
│   ├── extraction_prompt.md               LLM extraction prompt (Text S1)
│   ├── extraction_schema.json             Structured field schema (49 fields)
│   └── gold_standard.json                 Manual annotations for 10 papers (7.9% of corpus)
└── si_tables/
    ├── table_s1_llm_benchmark.csv         Per-field accuracy for 4 LLM models
    ├── table_s2_extraction_schema.csv      Complete field codebook
    ├── table_s3_126_studies.csv            Per-study characteristics summary
    └── table_s4_checklist_scoring.csv      ML4Env-REPORT compliance scores
```

## Dataset Schema

The 49 fields are organized into five analysis pillars:

**Metadata**: `paper_id`, `paper_title`, `doi`, `first_author`, `year`, `journal`, `cited_by_count`, `is_oa`, `research_type`

**Pillar 1 — Data Quality**: `data_source`, `dataset_size`, `n_literature_sources`, `target_variable_name`, `target_variable_name_original`, `target_variable_unit`, `target_variable_range_min`, `target_variable_range_max`, `pollutants`, `materials`, `data_selection_criteria_described`, `data_preprocessing`

**Pillar 2 — Validation**: `validation_method`, `k_fold_k`, `train_test_ratio`, `grouped_splitting`, `best_metric_type`, `best_metric_value`, `best_rmse`, `reports_train_and_test`, `external_validation`

**Pillar 3 — Comparability**: `n_features`, `hyperparameter_tuning`, `best_algorithm`, `interpretability_methods`, `n_features_used`, `feature_selection_method`, `top_3_features`, `mechanistic_discussion`, `ml_algorithms`, `n_algorithms_compared`, `evaluation_metrics`, `software_tools`

**Pillar 4 — Reproducibility**: `code_available`, `data_available`, `compared_with_prior_work`

**Pillar 5 — Deployment**: `water_type`, `discusses_scalability`, `engineering_validation`, `cost_analysis`

See `dataset/ml4env_126_metadata.json` for detailed field descriptions and allowed values.

## LLM Extraction Pipeline

The extraction pipeline used in this study:

1. PDF to Markdown conversion using MinerU (open-source document parser)
2. Structured extraction using Qwen3.5-Plus via DashScope API
3. Schema validation against `llm_extraction/extraction_schema.json`
4. Manual correction of low-accuracy fields (materials, pollutants, feature_selection_method)
5. Algorithm name standardization (26 standard abbreviations)

Four LLMs were benchmarked against a 10-paper gold standard (`llm_extraction/gold_standard.json`):
- Claude Sonnet 4.6: 88.4/100 (highest accuracy, 0% hallucination)
- Qwen3.5-Plus: 83.4/100 (selected for production; $0.002/paper)
- Gemini 3.1 Pro: 82.7/100
- GPT-5.1: 81.1/100

Cross-model pairwise agreement: 82.4%.

## ML4Env-REPORT Checklist

The review proposes a 10-item reporting checklist (manuscript Section 7.1). Per-study compliance scores are in `si_tables/table_s4_checklist_scoring.csv`.

| Item | Category | Criterion |
|:-----|:---------|:----------|
| A1 | Data Transparency | Dataset size AND data source type reported |
| A2 | Data Transparency | For literature data: source count AND selection criteria |
| A3 | Data Transparency | Target variable range (min AND max) reported |
| A4 | Data Transparency | Number of features AND feature selection method reported |
| B1 | Validation Rigor | Validation method reported AND (for literature data, grouped splitting) |
| B2 | Validation Rigor | Both training and test set metrics reported |
| B3 | Validation Rigor | Hyperparameter tuning method reported |
| C1 | Open Science | Code publicly available |
| C2 | Open Science | Data publicly available |
| C3 | Open Science | Software tools reported |

Corpus median compliance: 5/10 (mean 4.9/10).

## Citation

If you use this dataset or materials, please cite:

```
[Citation to be added upon publication]
```

## License

This dataset is released under the **CC-BY-4.0** license.
