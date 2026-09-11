# ResAtt-MLP

Official implementation of the paper **"Predicting Traffic Accident Compensation Support Rates Using a Residual Attention Multi-Layer Perceptron"**.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository predicts the court support rate for traffic-accident compensation claims using a **Residual Attention Multi-Layer Perceptron (ResAtt-MLP)** trained on 50 leakage-free tabular features derived from 1,942 Beijing traffic-accident judgments (2022–2023).

## Abstract

Traffic-accident compensation involves multiple heterogeneous claim items (medical expenses, disability compensation, mental solatium, etc.) with widely varying levels of judicial discretion. ResAtt-MLP introduces three key components for tabular regression:

1. **Feature Attention Gate** — a learnable sigmoid gate that re-weights input features.
2. **Residual Stack** — multiple residual blocks with BatchNorm + GELU + Dropout for stable deep feature transformation.
3. **MSE-SMAPE Joint Loss** — a convex combination of MSE and SMAPE (`α·MSE + (1−α)·SMAPE`) balancing absolute and percentage error.

A hierarchical multi-task extension predicts all 16 item-level support rates and reconstructs the global rate through a parameter-free weighted aggregation over claim ratios, trained with a differentiated masked composite loss.

## Repository Structure

```
ResAtt-MLP-Release/
├── data/
│   ├── compensation_items_summary.xlsx  # De-identified item-level claim/award amounts
│   └── extraction_results.xlsx          # De-identified legal features & raw extracts
├── experiments/                  # End-to-end experiment pipeline
│   ├── config.py                 #   Shared configuration (paths, seeds, hyperparameters)
│   ├── 01_build_dataset.py       #   Dataset construction (50 features, 70/15/15 split)
│   ├── 02_train_baselines.py     #   Baselines (Ridge / RF / XGBoost / LightGBM / MLP / TabNet)
│   ├── 02b_catboost.py           #   CatBoost baseline (supplement)
│   ├── 03_train_resatt_mlp.py    #   ResAtt-MLP single-task training (5 configs × 5 seeds)
│   ├── 04_ablation.py            #   Ablation study (gate / residual / SMAPE / ensemble)
│   ├── 05_scenarios.py           #   Scenario-specific models (A/B/C oracle analysis)
│   ├── 06_full_search.py         #   Full hyperparameter search (82 candidates)
│   ├── 06_hyperparams.py         #   Hyperparameter sensitivity analysis
│   ├── 06_statistical_tests.py   #   Statistical tests (5 seeds + Wilcoxon)
│   ├── 07_enhanced_tests.py      #   Enhanced tests (8 seeds + scenario Wilcoxon)
│   ├── 07_generate_figures.py    #   Paper figure generation
│   ├── 08_train_multitask.py     #   Multi-task ResAtt-MLP training
│   ├── 09_hyperparam_tuning.py   #   Multi-task hyperparameter tuning
│   ├── 09b_combined_ensemble.py  #   Combined-pool ensemble evaluation
│   └── 09c_diverse_tuning.py     #   Diversity tuning
├── models/                       # Model definitions (single source of truth)
│   ├── __init__.py
│   ├── resatt_mlp.py             #   ResAtt-MLP + losses + metrics
│   └── architectures/            #   Convenience re-exports
│       ├── resatt_mlp_net.py      #     ResAttMLP network export
│       └── loss_functions.py      #     Loss exports (MSE-SMAPE / Huber-SMAPE / MaskedComposite)
├── requirements.txt
└── README.md
```

## Data Availability & De-identification

The original corpus contains court judgment records with personally identifiable information (litigant names, case numbers, court identifiers, and free-text extraction fields). For open release, the de-identified data is provided as **two files** under `data/`:

- **`data/compensation_items_summary.xlsx`**
  - `判决书赔偿项目汇总` — item-level `原告主张` / `法院判决` amounts for the 16 compensation items, the totals (`原告主张合计` / `法院判决合计`) and the `数据状态` quality flag. Filtering to `数据状态 == "✅ 正常"` yields the **1,942** samples used in the paper.
  - `数据使用说明` — data provenance, quality tiers and known limitations (2,420 judgments: 1,942 normal, 203 award-missing, 181 claim-missing, 94 fully-missing).
- **`data/extraction_results.xlsx`**
  - `原告责任信息` (`file_name`, `plaintiff_responsibility`) — plaintiff liability;
  - `伤残等级提取` — standardized injury grade (`提取伤残等级(标准化)`);
  - `赔偿差距原因` (`file_name`, `compensation_reason`) — de-identified free-text reason;
  - `claims` / `judgments` / `combined` — raw per-item claim and award extracts;
  - `统计汇总` — per-item aggregate support statistics.

De-identification applied throughout:

- Every judgment filename is anonymized to `case_XXXXXX` (case numbers to `case_no_XXXXXX`);
- All PII columns (litigant/case name, notes) are removed;
- Free-text evidence snippets, document paths and the compensation-reason text are replaced with the placeholder `文本已脱敏`;
- The item-level claim/award amounts, totals, liability ratios and injury grades are preserved **exactly**, so the pipeline reproduces the **1,942** samples and `support_rate` target on the standard 70/15/15 split (1359/291/292).

### ⚠️ Reproducibility caveat

The pipeline runs end-to-end on the de-identified release — `01_build_dataset.py` rebuilds the **same 1,942 samples** with the **same 70/15/15 split (1359/291/292)** — **but the released data is deliberately not identical to the paper's original training data.** During de-identification the `accident_type` and insurance-coverage columns were dropped from `赔偿差距原因` and replaced with the placeholder `文本已脱敏`, so the release can only build a **50-dimensional** feature set instead of the paper's **59-dimensional** one.

The 9 features that can no longer be reconstructed are:

```
has_compulsory_insurance, has_commercial_insurance,
accident_type_motor_vs_motor, accident_type_motor_vs_nonmotor, accident_type_unknown,
insurance_coverage_both, insurance_coverage_commercial_only,
insurance_coverage_compulsory_only, insurance_coverage_none_or_unknown
```

Implications:

- Every script (build → baselines → ResAtt-MLP → ablation → multi-task) executes normally, but trains on **50** features rather than 59;
- The **Main Results** below are the paper's reported values from its original 59-dimensional dataset and are **not expected to reproduce exactly** on the release data — a fresh run reproduces the workflow and yields close, but not identical, metrics;
- Reproducing the paper's exact experiments would require the accident-type / insurance information, which is **not recoverable** from the released files. This trade-off is intentional: retaining the free text would re-introduce the privacy-sensitive content that de-identification is meant to remove.

## Installation

```bash
git clone <your-repository-url>
cd ResAtt-MLP-Release
pip install -r requirements.txt
```

Requirements: Python 3.12+, PyTorch 2.11+, and the packages listed in `requirements.txt`.

## Quick Start

The scripts are numbered in execution order and share configuration from `experiments/config.py`.

```bash
cd experiments

# 1. Build the dataset from the de-identified source
python 01_build_dataset.py

# 2. Train baselines
python 02_train_baselines.py
python 02b_catboost.py

# 3. Train the ResAtt-MLP (single-task)
python 03_train_resatt_mlp.py

# 4. Ablation study
python 04_ablation.py

# 5. Scenario analysis
python 05_scenarios.py

# 6. Hyperparameter search & statistical tests
python 06_full_search.py
python 06_hyperparams.py
python 06_statistical_tests.py
python 07_enhanced_tests.py

# 7. Generate figures
python 07_generate_figures.py

# 8. Multi-task extension
python 08_train_multitask.py
python 09_hyperparam_tuning.py
python 09b_combined_ensemble.py
python 09c_diverse_tuning.py
```

All outputs (processed data, checkpoints, results, figures) are written under `experiments/`.

> **Note**: `06_statistical_tests.py` and `07_enhanced_tests.py` reuse the checkpoints produced by `03_train_resatt_mlp.py` (`checkpoints/resatt_full_cfg*_s*.pt`); run `03` first. The paper's reported single-task result (MAE = 0.24269) is produced by `06_full_search.py`, while `03` is a single-configuration reproduction (MAE ≈ 0.245).

## Main Results

> These are the paper's reported results on its original **59-dimensional** dataset. On the de-identified release (**50** features) the pipeline reproduces the workflow, **not** these exact numbers — see the reproducibility caveat above.

| Model                    | MAE ↓   | SMAPE ↓ | WAPE ↓  |
| ------------------------ | ------- | ------- | ------- |
| Ridge                    | 0.26189 | 0.53814 | 0.42436 |
| Random Forest            | 0.24989 | 0.52237 | 0.40490 |
| XGBoost                  | 0.25455 | 0.52895 | 0.41247 |
| LightGBM                 | 0.25705 | 0.53264 | 0.41652 |
| CatBoost                 | 0.25481 | 0.52755 | 0.41288 |
| Standard MLP             | 0.26509 | 0.54758 | 0.42953 |
| TabNet                   | 0.26348 | 0.53864 | 0.42692 |
| **ResAtt-MLP**           | **0.24269** | **0.49772** | **0.39324** |
| **ResAtt-MLP (Multi-Task)** | **0.24300** | **0.48550** | **0.39370** |

## Citation

If you find this work useful, please cite:

```bibtex
@article{resattmlp2026,
  title  = {Predicting Traffic Accident Compensation Support Rates Using a Residual Attention Multi-Layer Perceptron},
  author = {Liyuan Gu},
  year   = {2026},
}
```

## License

This project is released under the MIT License.
