# ML4Env Critical Review — Open Data Package

Companion data and materials for:

> **The Illusion of Accuracy: Auditing Machine Learning for Environmental Remediation Across 126 Studies**

## Overview

This package contains the complete dataset, statistical summaries, figure source data, LLM extraction pipeline materials, supporting information tables, and analysis scripts for a systematic audit of 126 peer-reviewed studies applying machine learning to predict adsorption or degradation performance in environmental remediation.

| Statistic | Value |
|:----------|:------|
| Papers analyzed | 126 |
| Year range | 2018–2025 |
| Research type | Adsorption (110), Degradation (14), Both (2) |
| Data source | Experimental (55), Literature (41), Database (28), Mixed (2) |
| Unique algorithms | 30 |
| Median R² (test set) | 0.978 (n=109) |
| Composite rigor score | Median 2.0, Mean 2.6, Range 0–7 |
| Spearman rho (rigor vs R²) | −0.455, p < 0.0001 |
| Code available | 16/126 (12.7%) |
| Data available | 40/126 (31.7%) |

## Package Contents

```
├── README.md                              (this file)
├── dataset/
│   ├── ml4env_126_dataset.json            Main dataset (126 records × 49 fields)
│   ├── ml4env_126_dataset.csv             Same dataset in CSV format
│   ├── ml4env_126_metadata.json           Field definitions and allowed values
│   └── rigor_scores_per_paper.csv         Per-paper composite rigor scores (9 dimensions)
├── statistics/
│   └── statistics_summary.json            Pre-computed statistics used in the manuscript
├── figure_data/
│   ├── fig1_bibliometric.json             Fig. 1: Corpus overview and temporal trends
│   ├── fig2_data_bias.json                Fig. 2: Data quality and bias indicators
│   ├── fig3_validation.json               Fig. 3: Validation practice distributions
│   ├── fig4_comparability.json            Fig. 4: Algorithm and metric landscapes
│   ├── fig5_rigor.json                    Fig. 5: Methodological rigor vs reported performance
│   ├── fig5_rigor_summary.csv             Fig. 5: Per-dimension summary statistics
│   ├── fig6_deployment.json               Fig. 6: Deployment readiness indicators
│   └── figS1_llm_benchmark.json           Fig. S1: LLM extraction benchmark results
├── llm_extraction/
│   ├── extraction_prompt.md               LLM extraction prompt (Text S1)
│   ├── extraction_schema.json             Structured field schema (49 fields)
│   └── gold_standard.json                 Manual annotations for 10 papers (7.9% of corpus)
├── si_tables/
│   ├── table_s1_llm_benchmark.csv         Per-field accuracy for 4 LLM models
│   ├── table_s2_extraction_schema.csv     Complete field codebook
│   ├── table_s3_126_studies.csv           Per-study characteristics summary
│   ├── table_s4_checklist_scoring.csv     ML4Env-REPORT compliance scores
│   └── table_s5_sensitivity_analysis.csv  Leave-one-dimension-out sensitivity analysis
└── scripts/
    ├── plot_config.py                     Shared plotting configuration (colors, fonts, sizes)
    ├── statistical_analysis.py            Generates Fig. 2–4, 6 and prints all statistics
    ├── bibliometric_analysis.py           Generates Fig. 1 (corpus overview)
    ├── plot_fig1_benchmark.py             Generates Fig. S1 (LLM benchmark)
    ├── plot_fig6_rigor.py                 Generates Fig. 5 (rigor analysis, 2×2)
    ├── plot_prisma_flowchart.py           Generates Fig. S2 (PRISMA flow diagram)
    ├── rigor_analysis.py                  Computes composite rigor scores → fig5_rigor.json
    ├── validate_manuscript.py             Data-text consistency validator (90 rules)
    ├── build_statistics.py                Generates statistics_summary.json
    ├── build_public_dataset.py            Prepares public dataset from raw extraction
    └── format_references.py              ACS reference formatting with CrossRef DOI lookup
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

## Rigor Analysis (R1 Revision)

The composite rigor score (0–9) sums nine binary indicators per study: grouped splitting, external validation, reporting both training and test metrics, disclosed hyperparameter tuning method, documented data selection criteria, code availability, data availability, dataset size >= 500, and use of k-fold or more rigorous validation.

Key findings:
- Spearman rho = −0.455 (p < 0.0001, n = 109)
- Low rigor (0–2): median R² = 0.991 (n = 58)
- High rigor (7–9): median R² = 0.910 (n = 3)
- Delta R² = 0.081 (performance inflation estimate)
- Leave-one-dimension-out sensitivity: rho ranged from −0.474 to −0.373 (all p < 0.001)

Per-paper rigor scores are in `dataset/rigor_scores_per_paper.csv`. Sensitivity analysis results are in `si_tables/table_s5_sensitivity_analysis.csv`.

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

## Analysis Scripts

The `scripts/` directory contains all Python scripts used to generate figures and statistics in the manuscript. Scripts read from `dataset/ml4env_126_dataset.json` and `statistics/statistics_summary.json`. Shared plot configuration (colors, fonts, DPI) is defined in `plot_config.py`.

**Requirements**: Python 3.9+, numpy, pandas, matplotlib, seaborn, scipy.

**Note**: Scripts that involve LLM API calls (extraction, benchmarking) are not included as they require API credentials. The extraction prompt and schema are provided in `llm_extraction/` for reproducibility.

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
