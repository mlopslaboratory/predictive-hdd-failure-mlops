# Project Structure

## Repository layout

```text
project/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── data.csv.dvc
│   ├── processed/
│   └── raw/
├── docs/
│   ├── DOCKER_COMPOSE.md
│   └── PROJECT_TREE.md
├── legacy/
│   ├── configs/
│   ├── data/
│   ├── features/
│   ├── inference/
│   ├── models/
│   └── split/
├── metrics/
│   ├── drift_metrics.json
│   └── metrics.json
├── models/
│   ├── features.json
│   ├── preprocessing.json
│   ├── rf_model.pkl
│   └── mlflow_run.json
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/provisioning/
├── notebooks/
├── reports/
│   └── drift_report.md
├── src/
│   ├── api/
│   │   └── main.py
│   ├── data/
│   │   ├── backblaze_dataset_builder.py
│   │   └── make_dataset.py
│   ├── models/
│   │   ├── evaluate_baseline.py
│   │   └── train_baseline.py
│   └── monitoring/
│       ├── calculate_drift.py
│       └── generate_drift_report.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Ключевые части

- `src/api/main.py` — FastAPI сервис, OpenAPI, web UI, `/predict`, `/predictions`, `/drift-status`, `/experiments`, `/metrics`.
- `src/data/make_dataset.py` — production data preparation stage для DVC.
- `src/data/backblaze_dataset_builder.py` — вспомогательный builder для notebook-based подготовки Backblaze данных; не входит в текущий DVC production path.
- `src/models/train_baseline.py` — обучение baseline `RandomForestClassifier`, сохранение `models/*`, логирование в MLflow.
- `src/models/evaluate_baseline.py` — расчет test-метрик и регистрация модели в MLflow Model Registry.
- `src/monitoring/calculate_drift.py` — расчет `data_drift`, `target_drift`, `concept_drift`.
- `src/monitoring/generate_drift_report.py` — генерация Markdown-отчета `reports/drift_report.md`.
- `legacy/` — сохраненная ранняя версия training/data/split/inference структуры; не входит в текущий DVC production path.
- `monitoring/` — Prometheus config и Grafana provisioning.
- `data/`, `models/`, `metrics/`, `reports/` — воспроизводимые артефакты пайплайна; данные и модельные артефакты управляются через DVC.
- `notebooks/` — EDA и эксперименты, не часть production serving path.
- `.github/workflows/ci.yml` — lint, tests, Docker build, GHCR publish и GitOps update для Argo CD.

## DVC pipeline

`dvc.yaml` содержит stages:

1. `prepare_data`
2. `train`
3. `evaluate`
4. `drift`
5. `drift_report`

Параметры задаются в `params.yaml`, зафиксированные версии артефактов — в `dvc.lock`.
