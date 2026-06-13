# Predictive HDD Failure

MLOps-проект для прогнозирования отказов жестких дисков по SMART-метрикам Backblaze Hard Drive Stats.

## Что делает проект

Проект решает бинарную задачу классификации: предсказать, попадет ли диск в риск отказа по временным рядам SMART-показателей.

Основной пайплайн:

1. Подготовка датасета и train/val/test split по `serial_number`.
2. Создание delta-признаков по SMART-метрикам.
3. Обучение модели `RandomForestClassifier`.
4. Оценка качества на test split.
5. Расчет `data_drift`, `target_drift` и `concept_drift`.
6. Публикация FastAPI inference service с Prometheus/Grafana мониторингом.
7. Обновление Kubernetes deployment через GitOps repository и Argo CD.


## Структура проекта

```text
data/                    # raw и processed данные, часть файлов ведется через DVC
docs/                    # дополнительная документация
metrics/                 # JSON-метрики качества и drift
models/                  # обученная модель и metadata preprocessing
notebooks/               # EDA и эксперименты
reports/                 # Markdown-отчеты о drift monitoring
legacy/                  # старые training/data/split модули, не часть production path
src/api/                 # FastAPI inference service
src/data/                # подготовка датасета
src/models/              # обучение и оценка baseline-модели
src/monitoring/          # расчет drift-метрик
tests/                   # unit-тесты
dvc.yaml                 # DVC pipeline
params.yaml              # параметры данных, модели, метрик и drift
```

## Окружение

Установить зависимости

```bash
python -m pip install -r requirements-dev.txt
python -m pip install "dvc[s3]"
```

## Данные и артефакты

Проект использует DVC remote `hdd-mlops`. Если доступ к remote настроен, скачайте данные и артефакты:

```bash
dvc pull
```

Минимально для полного пайплайна нужен файл:

```text
data/data.csv
```
[data.csv](https://drive.google.com/file/d/11-ljkaS62_DQqF-BNVvygIRSjMXMg6V4/view?usp=sharing)

После подготовки данных пайплайн создает:

```text
data/processed/data_prepared.csv
data/processed/train_data_prepared.csv
data/processed/val_data_prepared.csv
data/processed/test_data_prepared.csv
```

После обучения создаются:

```text
models/rf_model.pkl
models/features.json
models/preprocessing.json
models/mlflow_run.json
```

## Полный запуск через DVC

Основной способ воспроизвести проект:

```bash
conda activate <your env>
dvc repro
```

DVC выполнит stages из `dvc.yaml`:

1. `prepare_data`: подготовит processed splits.
2. `train`: обучит baseline-модель и сохранит артефакты.
3. `evaluate`: посчитает test-метрики.
4. `drift`: посчитает data/target/concept drift.
5. `drift_report`: создаст Markdown-отчет о drift monitoring.

Проверить состояние пайплайна:

```bash
dvc status
```

Ожидаемый результат после успешного запуска:

```text
Data and pipelines are up to date.
```

## Ручной запуск по шагам

Если нужно запускать этапы отдельно:

```bash
python -m src.data.make_dataset
python -m src.models.train_baseline
python -m src.models.evaluate_baseline
python -m src.monitoring.calculate_drift
python -m src.monitoring.generate_drift_report
```


## Метрики качества модели

Оценка модели сохраняется в:

```text
metrics/metrics.json
```

Файл содержит:

- `pr_auc`
- `roc_auc`
- `f1`
- `precision`
- `recall`
- `threshold`
- `test_windows`
- `test_positive_rate`

Запустить только оценку:

```bash
python -m src.models.evaluate_baseline
```

## Drift monitoring

Расчет drift-метрик сохраняется в:

```text
metrics/drift_metrics.json
```

Markdown-отчет о дрейфе сохраняется в:

```text
reports/drift_report.md
```

Запустить только drift monitoring:

```bash
python -m src.monitoring.calculate_drift
python -m src.monitoring.generate_drift_report
```

Что считается:

- `data_drift`: PSI по window-признакам модели. Reference split = `train`, current split = `test`.
- `target_drift`: сдвиг доли положительного класса и PSI по target. Reference split = `train`, current split = `test`.
- `concept_drift`: падение качества модели на `test` относительно baseline split = `val`.

Пороги задаются в `params.yaml`:

```yaml
drift:
  psi_bins: 10
  data_drift_threshold: 0.2
  target_drift_threshold: 0.1
  concept_drift_threshold: 0.05
  primary_concept_metric: pr_auc
```

## MLflow

Обучение и оценка используют локальный MLflow tracking URI из `params.yaml`:

```yaml
mlflow:
  tracking_uri: mlflow.db
  experiment_name: hdd_failure
  registered_model_name: hdd_failure_random_forest
```

При запуске `dvc repro` stage `evaluate` регистрирует новую версию модели в локальном MLflow Model Registry.

## Запуск API

API использует артефакты из директории `models/`.

Локальный запуск:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Проверка:

```bash
curl http://localhost:8000/health
```

Swagger UI:

```text
http://localhost:8000/docs
```

Получить информацию о загруженной модели:

```bash
curl http://localhost:8000/model-info
```

Веб UI:

```text
http://localhost:8000/
```

В интерфейсе есть:

- страница инференса;
- таблица последних предсказаний текущего процесса API;
- флаги `data_drift`, `target_drift`, `concept_drift`;
- флаг аномалии для предсказания;
- страница экспериментов `/experiments` с MLflow run info, test metrics и drift summary.

Полезные API endpoints:

```bash
curl http://localhost:8000/predictions
curl http://localhost:8000/drift-status
```

Docker Compose запуск:

```bash
docker compose up --build
```

Prometheus:

```text
http://localhost:9090
```

Grafana:

```text
http://localhost:3000
```

Логин/пароль Grafana для локального compose:

```text
admin / admin
```

Остановить:

```bash
docker compose down
```

Подробнее: `docs/DOCKER_COMPOSE.md`.

## CI/CD и Argo CD

GitHub Actions workflow выполняет:

1. `ruff`;
2. `pytest`;
3. Docker build smoke test;
4. при `push` в `main`: `dvc pull` модельных артефактов, build/push image в GHCR,
   обновление image tag в GitOps repository.

Container image:

```text
ghcr.io/mlopslaboratory/predictive-hdd-failure-api:<git-sha>
```

Для deploy job нужны GitHub Secrets в основном репозитории:

```text
YC_STORAGE_ACCESS_KEY_ID
YC_STORAGE_SECRET_ACCESS_KEY
GITOPS_REPO_TOKEN
```

`GITOPS_REPO_TOKEN` должен иметь `Contents: Read and write` для:

```text
mlopslaboratory/predictive-hdd-failure-gitops
```

Argo CD следит за отдельным GitOps repository:

```text
https://github.com/mlopslaboratory/predictive-hdd-failure-gitops
```

Проверка в Minikube:

```bash
argocd app get predictive-hdd-failure
kubectl get all -n predictive-hdd
minikube service predictive-hdd-api -n predictive-hdd
```


## Проверки качества кода

Запустить тесты:

```bash
python -m pytest
```

Запустить lint:

```bash
python -m ruff check src tests
```

Проверить DVC:

```bash
dvc status
```

Ожидаемый результат в корректном окружении Python 3.10 с установленными зависимостями:

```text
pytest: passed
ruff: All checks passed
dvc status: Data and pipelines are up to date
```

## Ноутбуки

EDA и эксперименты лежат в `notebooks/`.


## Авторы

Даниил Пименов, Ольга Голева
