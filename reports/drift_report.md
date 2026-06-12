# Drift Report

## Summary

| Drift type | Status |
| --- | --- |
| Data drift | не обнаружен |
| Target drift | не обнаружен |
| Concept drift | обнаружен |

## Dataset Windows

| Split | Windows |
| --- | ---: |
| Train | 10189 |
| Validation | 1855 |
| Test | 2124 |

## Model Quality

| Metric | Value |
| --- | ---: |
| PR-AUC | 0.1036 |
| ROC-AUC | 0.7837 |
| F1 | 0.1490 |
| Precision | 0.0981 |
| Recall | 0.3095 |
| Threshold | 0.5000 |
| Test positive rate | 3.95% |

## Data Drift

- Feature count: 77
- Drifted feature count: 0
- Mean PSI: 0.0158
- Max PSI: 0.0864
- PSI threshold: 0.2000

Top features by PSI:

- `smart_198_raw_delta_t`: PSI 0.0864
- `smart_198_raw_delta_t-1`: PSI 0.0862
- `smart_198_raw_delta_t-2`: PSI 0.0830
- `smart_198_raw_delta_t-3`: PSI 0.0817
- `smart_198_raw_delta_t-6`: PSI 0.0792
- `smart_198_raw_delta_t-4`: PSI 0.0786
- `smart_198_raw_delta_t-5`: PSI 0.0782
- `smart_5_raw_delta_t`: PSI 0.0433
- `smart_5_raw_delta_t-6`: PSI 0.0361
- `smart_5_raw_delta_t-1`: PSI 0.0356

## Target Drift

- Reference positive rate: 2.71%
- Current positive rate: 3.95%
- Absolute positive rate change: 1.25%
- PSI: 0.0049

## Concept Drift

- Baseline split: val
- Current split: test
- Primary metric: pr_auc
- Primary metric drop: 0.0553
- Drift threshold: 0.0500
