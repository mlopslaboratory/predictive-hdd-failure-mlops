# Project Structure

## Repository layout

```text
project/
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── inference/
│   ├── models/
│   ├── split/
│   └── main.py
├── tests/
├── configs/
│   └── config.yaml
├── artifacts/
├── metrics/
├── data/
├── docs/
├── notebooks/
├── models/
├── mlruns/
├── .dvc/
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── .dvcignore
├── .dockerignore
├── .gitignore
├── Dockerfile
└── mlflow.db
```

## Краткое описание ключевых частей

- `src/` — основной код проекта (MLOps pipeline, инференс, API).
  - `src/data/` — подготовка и инженерия признаков на уровне датасета (загрузка, make/prepare-логика).
  - `src/features/` — feature engineering/legacy-пайплайны.
  - `src/models/` — обучение, оценка и запуск тренировочных скриптов.
  - `src/split/` — разбиение данных по правилам.
  - `src/inference/` — логика инференса модели.
  - `src/api/` — сервисная обвязка (FastAPI/точка входа для сервиса).
  - `src/main.py` — корневой запуск/точка входа приложения.

- `api/` — API находится в `src/api/`.
- `training/` — текущие training-скрипты находятся в `src/models/`.

- `artifacts/` — артефакты модели/прогона 
- `tests/` — автоматические тесты (`pytest`), охватывающие подготовку данных, создание признаков и обучение/оценку.
- `configs/` — конфигурация экспериментов/пайплайнов.
- `notebooks/` — исследовательские ноутбуки (EDA/эксперименты), не часть продакшн-пайплайна.
- `.github/workflows/` — CI-конфигурации GitHub Actions (`ci.yml`).
- `docs/` — проектная документация 

- `Dockerfile` — сборка контейнера приложения (Python app + uvicorn).
- `docker-compose` — `docker-compose.yml`

- **DVC-related files**
  - `dvc.yaml` — описание DVC stages (prepare/train/evaluate).
  - `dvc.lock` — зафиксированные версии зависимостей этапов.
  - `params.yaml` — параметры пайплайна/экспериментов.
  - `.dvc/` — служебная папка DVC metadata/cache-конфигурации.
  - `.dvcignore` — исключения для DVC.

