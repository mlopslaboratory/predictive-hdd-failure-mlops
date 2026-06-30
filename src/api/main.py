from __future__ import annotations

import json
import logging
import os
import pickle
import sqlite3
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "rf_model.pkl"
FEATURES_PATH = MODELS_DIR / "features.json"
PREPROCESSING_PATH = MODELS_DIR / "preprocessing.json"
METRICS_PATH = BASE_DIR / "metrics" / "metrics.json"
DRIFT_METRICS_PATH = BASE_DIR / "metrics" / "drift_metrics.json"
MLFLOW_RUN_INFO_PATH = MODELS_DIR / "mlflow_run.json"
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
PREDICTION_ALERT_THRESHOLD = float(os.getenv("PREDICTION_ALERT_THRESHOLD", "0.5"))
PREDICTIONS_DB_PATH = Path(
    os.getenv("PREDICTIONS_DB_PATH", "/app/storage/predictions.db")
)
MODEL_VERSION = os.getenv("MODEL_VERSION")

model = None
feature_cols: list[str] = []
preprocessing_metadata: dict[str, Any] = {}

HTTP_REQUESTS = Counter(
    "hdd_http_requests_total",
    "Total number of HTTP requests.",
    ["method", "path", "status_code"],
)
HTTP_REQUEST_LATENCY = Histogram(
    "hdd_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
)
PREDICT_REQUESTS = Counter(
    "hdd_predict_requests_total",
    "Total number of requests to the /predict endpoint.",
)
PREDICTIONS_CLASS_1 = Counter(
    "hdd_predictions_class_1_total",
    "Total number of predictions with class 1.",
)
PREDICTIONS_CLASS_0 = Counter(
    "hdd_predictions_class_0_total",
    "Total number of predictions with class 0.",
)
PREDICTION_ERRORS = Counter(
    "hdd_prediction_errors_total",
    "Total number of errors raised by the /predict endpoint.",
)
INFERENCE_LATENCY = Histogram(
    "hdd_inference_latency_seconds",
    "Latency of /predict inference requests in seconds.",
)


def build_prediction_request_example(
    features_path: Path = FEATURES_PATH,
) -> dict[str, Any]:
    try:
        feature_names = json.loads(features_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        feature_names = []

    return {"features": {str(feature): 0 for feature in feature_names}}


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={"example": build_prediction_request_example()},
    )

    features: dict[str, float | int | None] = Field(
        ...,
        description=(
            "Словарь с оконными признаками для модели. "
            "Нужно передать все ключи из models/features.json."
        ),
    )
    disk_id: str | None = Field(
        default=None,
        description="Optional stable disk identifier.",
    )
    serial_number: str | None = Field(
        default=None,
        description="Optional HDD serial number.",
    )
    disk_model: str | None = Field(
        default=None,
        alias="model",
        description="Optional HDD model name used as a fallback identifier.",
    )


class PredictionResponse(BaseModel):
    failure_probability: float
    prediction: int
    anomaly: bool


class PredictionRecord(BaseModel):
    id: int
    created_at: str
    disk_id: str
    failure_probability: float
    prediction: int
    anomaly: bool
    request_payload: str
    model_version: str | None = None
    feature_count: int


class PredictionStatsResponse(BaseModel):
    total_predictions: int
    risky_predictions: int
    latest_disk_id: str | None = None
    latest_failure_probability: float | None = None


class DriftStatusResponse(BaseModel):
    available: bool
    data_drift: bool | None
    target_drift: bool | None
    concept_drift: bool | None
    any_drift: bool | None
    summary: dict[str, Any]


def load_artifacts(
    model_path: Path = MODEL_PATH,
    features_path: Path = FEATURES_PATH,
    preprocessing_path: Path = PREPROCESSING_PATH,
) -> tuple[Any, list[str], dict[str, Any]]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not preprocessing_path.exists():
        raise FileNotFoundError(f"Preprocessing file not found: {preprocessing_path}")

    with model_path.open("rb") as file:
        loaded_model = pickle.load(file)
    loaded_features = json.loads(features_path.read_text(encoding="utf-8"))
    loaded_preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    return loaded_model, loaded_features, loaded_preprocessing


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_predictions_db_path(db_path: Path | None = None) -> Path:
    return db_path or PREDICTIONS_DB_PATH


def init_predictions_db(db_path: Path | None = None) -> None:
    path = resolve_predictions_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                disk_id TEXT NOT NULL,
                failure_probability REAL NOT NULL,
                prediction INTEGER NOT NULL,
                anomaly INTEGER NOT NULL,
                request_payload TEXT NOT NULL,
                model_version TEXT,
                feature_count INTEGER NOT NULL
            )
            """
        )
        connection.commit()


def prediction_request_to_payload(request: PredictionRequest) -> dict[str, Any]:
    return request.model_dump(by_alias=True, exclude_none=True)


def generated_disk_id(record_id: int) -> str:
    return f"Disk-{record_id:06d}"


def extract_disk_id(request: PredictionRequest) -> str | None:
    for value in (request.disk_id, request.serial_number, request.disk_model):
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def record_from_db_row(row: sqlite3.Row) -> PredictionRecord:
    record_id = int(row["id"])
    disk_id = str(row["disk_id"])
    if disk_id == "unknown-disk":
        disk_id = generated_disk_id(record_id)

    return PredictionRecord(
        id=record_id,
        created_at=str(row["created_at"]),
        disk_id=disk_id,
        failure_probability=float(row["failure_probability"]),
        prediction=int(row["prediction"]),
        anomaly=bool(row["anomaly"]),
        request_payload=str(row["request_payload"]),
        model_version=row["model_version"],
        feature_count=int(row["feature_count"]),
    )


def save_prediction_record(
    request: PredictionRequest,
    failure_probability: float,
    prediction: int,
    anomaly: bool,
    db_path: Path | None = None,
) -> PredictionRecord:
    init_predictions_db(db_path)

    path = resolve_predictions_db_path(db_path)
    created_at = datetime.now(timezone.utc).isoformat()
    payload = prediction_request_to_payload(request)
    request_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    explicit_disk_id = extract_disk_id(request)
    disk_id = explicit_disk_id or "Disk-000000"
    feature_count = len(request.features)

    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                created_at,
                disk_id,
                failure_probability,
                prediction,
                anomaly,
                request_payload,
                model_version,
                feature_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                disk_id,
                failure_probability,
                prediction,
                int(anomaly),
                request_payload,
                MODEL_VERSION,
                feature_count,
            ),
        )
        record_id = int(cursor.lastrowid)
        if explicit_disk_id is None:
            disk_id = generated_disk_id(record_id)
            connection.execute(
                "UPDATE predictions SET disk_id = ? WHERE id = ?",
                (disk_id, record_id),
            )
        connection.commit()

    return PredictionRecord(
        id=record_id,
        created_at=created_at,
        disk_id=disk_id,
        failure_probability=failure_probability,
        prediction=prediction,
        anomaly=anomaly,
        request_payload=request_payload,
        model_version=MODEL_VERSION,
        feature_count=feature_count,
    )


def normalize_limit(limit: int) -> int:
    return max(1, min(int(limit), 500))


def fetch_prediction_records(
    limit: int = 20,
    only_anomalies: bool = False,
    db_path: Path | None = None,
) -> list[PredictionRecord]:
    init_predictions_db(db_path)
    path = resolve_predictions_db_path(db_path)
    where_clause = "WHERE anomaly = 1" if only_anomalies else ""

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"""
            SELECT
                id,
                created_at,
                disk_id,
                failure_probability,
                prediction,
                anomaly,
                request_payload,
                model_version,
                feature_count
            FROM predictions
            {where_clause}
            ORDER BY id DESC
            LIMIT ?
            """,
            (normalize_limit(limit),),
        ).fetchall()

    return [record_from_db_row(row) for row in rows]


def fetch_risky_disk_records(
    limit: int = 20,
    db_path: Path | None = None,
) -> list[PredictionRecord]:
    init_predictions_db(db_path)
    path = resolve_predictions_db_path(db_path)

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                id,
                created_at,
                disk_id,
                failure_probability,
                prediction,
                anomaly,
                request_payload,
                model_version,
                feature_count
            FROM predictions
            WHERE prediction = 1 OR anomaly = 1
            ORDER BY failure_probability DESC, id DESC
            LIMIT ?
            """,
            (normalize_limit(limit),),
        ).fetchall()

    return [record_from_db_row(row) for row in rows]


def fetch_prediction_stats(db_path: Path | None = None) -> PredictionStatsResponse:
    init_predictions_db(db_path)
    path = resolve_predictions_db_path(db_path)

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        total_predictions = int(
            connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        )
        risky_predictions = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM predictions
                WHERE prediction = 1 OR anomaly = 1
                """
            ).fetchone()[0]
        )
        latest_row = connection.execute(
            """
            SELECT
                id,
                disk_id,
                failure_probability
            FROM predictions
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

    if latest_row is None:
        return PredictionStatsResponse(
            total_predictions=total_predictions,
            risky_predictions=risky_predictions,
        )

    latest_id = int(latest_row["id"])
    latest_disk_id = str(latest_row["disk_id"])
    if latest_disk_id == "unknown-disk":
        latest_disk_id = generated_disk_id(latest_id)

    return PredictionStatsResponse(
        total_predictions=total_predictions,
        risky_predictions=risky_predictions,
        latest_disk_id=latest_disk_id,
        latest_failure_probability=float(latest_row["failure_probability"]),
    )


def build_drift_status(
    drift_metrics_path: Path = DRIFT_METRICS_PATH,
) -> DriftStatusResponse:
    drift_metrics = load_json_file(drift_metrics_path)
    if not drift_metrics:
        return DriftStatusResponse(
            available=False,
            data_drift=None,
            target_drift=None,
            concept_drift=None,
            any_drift=None,
            summary={},
        )

    data_drift = bool(drift_metrics.get("data_drift", {}).get("drift_detected", False))
    target_drift = bool(
        drift_metrics.get("target_drift", {}).get("drift_detected", False)
    )
    concept_drift = bool(
        drift_metrics.get("concept_drift", {}).get("drift_detected", False)
    )

    return DriftStatusResponse(
        available=True,
        data_drift=data_drift,
        target_drift=target_drift,
        concept_drift=concept_drift,
        any_drift=any([data_drift, target_drift, concept_drift]),
        summary={
            "reference_split": drift_metrics.get("reference_split"),
            "current_split": drift_metrics.get("current_split"),
            "window_counts": drift_metrics.get("window_counts", {}),
            "drifted_feature_count": drift_metrics.get("data_drift", {}).get(
                "drifted_feature_count"
            ),
            "max_psi": drift_metrics.get("data_drift", {}).get("max_psi"),
            "target_positive_rate_current": drift_metrics.get("target_drift", {}).get(
                "current_positive_rate"
            ),
            "concept_primary_metric": drift_metrics.get("concept_drift", {}).get(
                "primary_metric"
            ),
            "concept_primary_metric_drop": drift_metrics.get("concept_drift", {}).get(
                "primary_metric_drop"
            ),
            "top_drifted_features": drift_metrics.get("data_drift", {}).get(
                "top_drifted_features",
                [],
            ),
        },
    )


def format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def render_status_flag(label: str, value: bool | None) -> str:
    if value is None:
        class_name = "flag muted-flag"
        text = "нет данных"
    elif value:
        class_name = "flag bad-flag"
        text = "обнаружен"
    else:
        class_name = "flag good-flag"
        text = "норма"
    return f'<span class="{class_name}">{escape(label)}: {text}</span>'


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global model, feature_cols, preprocessing_metadata

    logger.info("Loading model artifacts...")
    model, feature_cols, preprocessing_metadata = load_artifacts()
    logger.info("Artifacts loaded. Features count: %s", len(feature_cols))
    init_predictions_db()
    logger.info("Predictions DB initialized: %s", PREDICTIONS_DB_PATH)
    yield


app = FastAPI(
    title="Система прогнозирования отказов жестких дисков",
    description="Вывод сервиса для предсказания отказа диска.",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "health", "description": "Service health checks."},
        {"name": "model", "description": "Loaded model metadata."},
        {"name": "inference", "description": "HDD failure inference."},
    ],
)


@app.middleware("http")
async def collect_predict_metrics(request: Request, call_next: Any) -> Response:
    start_time = time.perf_counter()
    path = request.url.path
    method = request.method

    if path == "/predict":
        PREDICT_REQUESTS.inc()

    try:
        response = await call_next(request)
    except Exception:
        elapsed = time.perf_counter() - start_time
        HTTP_REQUESTS.labels(method=method, path=path, status_code="500").inc()
        HTTP_REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed)
        if path == "/predict":
            PREDICTION_ERRORS.inc()
            INFERENCE_LATENCY.observe(elapsed)
        raise

    elapsed = time.perf_counter() - start_time
    HTTP_REQUESTS.labels(
        method=method,
        path=path,
        status_code=str(response.status_code),
    ).inc()
    HTTP_REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed)

    if path == "/predict" and response.status_code >= 400:
        PREDICTION_ERRORS.inc()

    if path == "/predict":
        INFERENCE_LATENCY.observe(elapsed)
    return response


@app.get("/health", tags=["health"], summary="Check service health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    sample_features = {feature: 0 for feature in feature_cols}
    sample_payload = json.dumps({"features": sample_features}, indent=2)
    drift_status = build_drift_status()

    html = f"""
    <!doctype html>
    <html lang="ru">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Predictive HDD Failure MLOps</title>
        <style>
            :root {{
                --bg: #f7f8fb;
                --ink: #16202a;
                --muted: #5d6975;
                --panel: #ffffff;
                --accent: #1f7a5c;
                --accent-dark: #155840;
                --danger: #b42318;
                --warning: #b54708;
                --border: #d8dee8;
                --soft: #edf2f7;
            }}
            body {{
                margin: 0;
                min-height: 100vh;
                font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                color: var(--ink);
                background: var(--bg);
            }}
            main {{
                width: min(1180px, calc(100% - 32px));
                margin: 0 auto;
                padding: 28px 0 40px;
            }}
            header {{
                display: flex;
                justify-content: space-between;
                gap: 18px;
                align-items: flex-start;
                margin-bottom: 18px;
            }}
            h1 {{
                margin: 0 0 6px;
                font-size: 2rem;
                line-height: 1.1;
            }}
            h2 {{
                margin: 0 0 14px;
                font-size: 1.05rem;
            }}
            p {{
                margin: 0;
                color: var(--muted);
                line-height: 1.5;
            }}
            a {{
                color: var(--accent-dark);
                font-weight: 650;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            .top-links {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                justify-content: flex-end;
            }}
            .top-links a {{
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 8px 10px;
                background: var(--panel);
            }}
            .grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 16px;
                align-items: start;
            }}
            .card {{
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 18px;
                box-shadow: 0 10px 28px rgba(22, 32, 42, 0.06);
            }}
            .status-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 12px;
            }}
            .flag {{
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                padding: 6px 10px;
                font-size: 0.86rem;
                font-weight: 700;
                border: 1px solid transparent;
            }}
            .good-flag {{
                color: #05603a;
                background: #ecfdf3;
                border-color: #abefc6;
            }}
            .bad-flag {{
                color: var(--danger);
                background: #fef3f2;
                border-color: #fecdca;
            }}
            .muted-flag {{
                color: var(--muted);
                background: var(--soft);
                border-color: var(--border);
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 10px;
                margin-top: 14px;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 12px;
                margin-bottom: 16px;
            }}
            .metric {{
                background: var(--soft);
                border-radius: 8px;
                padding: 10px;
            }}
            .stat-card {{
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 16px;
                box-shadow: 0 10px 28px rgba(22, 32, 42, 0.06);
            }}
            .stat-card span {{
                display: block;
                color: var(--muted);
                font-size: 0.82rem;
                font-weight: 650;
            }}
            .stat-card strong {{
                display: block;
                margin-top: 7px;
                font-size: 1.55rem;
                line-height: 1.1;
            }}
            .stat-card small {{
                display: block;
                margin-top: 7px;
                color: var(--muted);
                font-size: 0.86rem;
            }}
            .metric strong {{
                display: block;
                margin-top: 4px;
                font-size: 1.1rem;
            }}
            .metric span {{
                color: var(--muted);
                font-size: 0.82rem;
            }}
            textarea {{
                width: 100%;
                min-height: 340px;
                box-sizing: border-box;
                resize: vertical;
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 12px;
                font-family: "Cascadia Mono", Consolas, monospace;
                font-size: 0.9rem;
                background: #fbfcfe;
                color: var(--ink);
            }}
            button {{
                border: 0;
                border-radius: 8px;
                padding: 11px 14px;
                background: var(--accent);
                color: white;
                font-size: 0.95rem;
                font-weight: 750;
                cursor: pointer;
            }}
            button:hover {{
                background: var(--accent-dark);
            }}
            .actions {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 12px;
            }}
            #result {{
                min-height: 72px;
                background: var(--soft);
                color: var(--ink);
                border-radius: 8px;
                padding: 12px;
                margin-top: 12px;
            }}
            .result-card h3 {{
                margin: 0 0 12px;
                font-size: 1rem;
            }}
            .result-grid {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 10px;
            }}
            .result-item {{
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 10px;
            }}
            .result-item span {{
                display: block;
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 700;
            }}
            .result-item strong {{
                display: block;
                margin-top: 5px;
                font-size: 1rem;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9rem;
            }}
            th, td {{
                border-bottom: 1px solid var(--border);
                padding: 10px 8px;
                text-align: left;
                vertical-align: top;
            }}
            th {{
                color: var(--muted);
                font-size: 0.78rem;
            }}
            .full {{
                grid-column: 1 / -1;
            }}
            .empty {{
                color: var(--muted);
                padding: 12px 0;
            }}
            @media (max-width: 860px) {{
                header {{
                    display: block;
                }}
                .top-links {{
                    justify-content: flex-start;
                    margin-top: 12px;
                }}
                .grid, .metric-grid, .stats-grid, .result-grid {{
                    grid-template-columns: 1fr;
                }}
                .full {{
                    grid-column: auto;
                }}
            }}
        </style>
    </head>
    <body>
        <main>
            <header>
                <div>
                    <h1>Система прогнозирования отказов жестких дисков</h1>
                    <p>Инференс, мониторинг дрейфа и эксплуатационные метрики модели отказов дисков.</p>
                </div>
                <nav class="top-links" aria-label="Ссылки мониторинга">
                    <a href="/docs">Документация API</a>
                    <a href="/experiments">Эксперименты</a>
                    <a href="/metrics">Prometheus metrics</a>
                    <a href="{PROMETHEUS_URL}">Prometheus</a>
                    <a href="{GRAFANA_URL}">Grafana</a>
                </nav>
            </header>

            <section class="grid">
                <div class="card">
                    <h2>Статус модели и дрейфа</h2>
                    <p>Загружено признаков: <strong>{len(feature_cols)}</strong>. Порог флага аномалии: <strong>{PREDICTION_ALERT_THRESHOLD:.2f}</strong>.</p>
                    <div class="status-row">
                        {render_status_flag("Data drift", drift_status.data_drift)}
                        {render_status_flag("Target drift", drift_status.target_drift)}
                        {render_status_flag("Concept drift", drift_status.concept_drift)}
                    </div>
                    <div class="metric-grid">
                        <div class="metric">
                            <span>Drifted features</span>
                            <strong>{drift_status.summary.get("drifted_feature_count", "n/a")}</strong>
                        </div>
                        <div class="metric">
                            <span>Max PSI</span>
                            <strong>{format_float(drift_status.summary.get("max_psi"))}</strong>
                        </div>
                        <div class="metric">
                            <span>Target positive rate</span>
                            <strong>{format_percent(drift_status.summary.get("target_positive_rate_current"))}</strong>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h2>Инференс</h2>
                    <form id="predict-form">
                        <label for="payload"><strong>JSON с признаками</strong></label>
                        <textarea id="payload" spellcheck="false">{sample_payload}</textarea>
                        <div class="actions">
                            <button type="submit">Выполнить прогноз</button>
                        </div>
                    </form>
                    <div id="result">Результат прогноза появится здесь.</div>
                </div>

                <div class="stats-grid full" aria-label="Сводка прогнозов">
                    <div class="stat-card">
                        <span>Всего прогнозов</span>
                        <strong id="stat-total">0</strong>
                    </div>
                    <div class="stat-card">
                        <span>Дисков высокого риска</span>
                        <strong id="stat-risky">0</strong>
                    </div>
                    <div class="stat-card">
                        <span>Последний прогноз</span>
                        <strong id="stat-latest-disk">n/a</strong>
                        <small id="stat-latest-probability">Вероятность отказа: n/a</small>
                    </div>
                </div>

                <div class="card full">
                    <h2>Диски с высоким риском отказа</h2>
                    <div id="risky-disks">Загрузка списка...</div>
                </div>

                <div class="card full">
                    <h2>Последние прогнозы</h2>
                    <div id="history">Загрузка истории...</div>
                </div>
            </section>
        </main>
        <script>
            const form = document.getElementById("predict-form");
            const payload = document.getElementById("payload");
            const result = document.getElementById("result");
            const history = document.getElementById("history");
            const riskyDisks = document.getElementById("risky-disks");
            const statTotal = document.getElementById("stat-total");
            const statRisky = document.getElementById("stat-risky");
            const statLatestDisk = document.getElementById("stat-latest-disk");
            const statLatestProbability = document.getElementById("stat-latest-probability");

            function escapeHtml(value) {{
                return String(value)
                    .replaceAll("&", "&amp;")
                    .replaceAll("<", "&lt;")
                    .replaceAll(">", "&gt;")
                    .replaceAll('"', "&quot;")
                    .replaceAll("'", "&#039;");
            }}

            function formatProbability(value) {{
                if (value === null || value === undefined) {{
                    return "n/a";
                }}
                return `${{(Number(value) * 100).toFixed(1)}} %`;
            }}

            function isRisky(record) {{
                return Number(record.prediction) === 1 || Boolean(record.anomaly);
            }}

            function predictionLabel(record) {{
                return Number(record.prediction) === 1 ? "Высокий риск" : "Норма";
            }}

            function anomalyLabel(record) {{
                return record.anomaly ? "Да" : "Нет";
            }}

            function statusLabel(record) {{
                return isRisky(record) ? "Рекомендуется проверка" : "Норма";
            }}

            function statusFlag(record) {{
                const className = isRisky(record) ? "bad-flag" : "good-flag";
                return `<span class="flag ${{className}}">${{statusLabel(record)}}</span>`;
            }}

            function renderPredictionsTable(container, records, emptyText) {{
                if (!records.length) {{
                    container.innerHTML = `<div class="empty">${{emptyText}}</div>`;
                    return;
                }}
                const rows = records.map((record) => `
                    <tr>
                        <td>${{new Date(record.created_at).toLocaleString("ru-RU")}}</td>
                        <td>${{escapeHtml(record.disk_id)}}</td>
                        <td>${{formatProbability(record.failure_probability)}}</td>
                        <td>${{predictionLabel(record)}}</td>
                        <td>${{anomalyLabel(record)}}</td>
                        <td>${{statusFlag(record)}}</td>
                    </tr>
                `).join("");
                container.innerHTML = `
                    <table>
                        <thead>
                            <tr>
                                <th>Время</th>
                                <th>Диск</th>
                                <th>Вероятность отказа</th>
                                <th>Прогноз</th>
                                <th>Аномалия</th>
                                <th>Статус</th>
                            </tr>
                        </thead>
                        <tbody>${{rows}}</tbody>
                    </table>
                `;
            }}

            function renderStats(stats) {{
                statTotal.textContent = stats.total_predictions;
                statRisky.textContent = stats.risky_predictions;
                if (stats.latest_disk_id) {{
                    statLatestDisk.textContent = stats.latest_disk_id;
                    statLatestProbability.textContent = `Вероятность отказа: ${{formatProbability(stats.latest_failure_probability)}}`;
                }} else {{
                    statLatestDisk.textContent = "n/a";
                    statLatestProbability.textContent = "Вероятность отказа: n/a";
                }}
            }}

            function renderPredictionResult(record) {{
                if (!record) {{
                    result.innerHTML = '<div class="empty">Прогноз сохранен. Обновите страницу, если результат не появился в таблице.</div>';
                    return;
                }}
                result.innerHTML = `
                    <div class="result-card">
                        <h3>Результат прогноза</h3>
                        <div class="result-grid">
                            <div class="result-item">
                                <span>Диск</span>
                                <strong>${{escapeHtml(record.disk_id)}}</strong>
                            </div>
                            <div class="result-item">
                                <span>Вероятность отказа</span>
                                <strong>${{formatProbability(record.failure_probability)}}</strong>
                            </div>
                            <div class="result-item">
                                <span>Прогноз</span>
                                <strong>${{predictionLabel(record)}}</strong>
                            </div>
                            <div class="result-item">
                                <span>Статус</span>
                                <strong>${{statusLabel(record)}}</strong>
                            </div>
                            <div class="result-item">
                                <span>Аномалия</span>
                                <strong>${{anomalyLabel(record)}}</strong>
                            </div>
                        </div>
                    </div>
                `;
            }}

            async function refreshTables() {{
                const [statsResponse, riskyResponse, historyResponse] = await Promise.all([
                    fetch("/prediction-stats"),
                    fetch("/risky-disks?limit=20"),
                    fetch("/predictions?limit=100"),
                ]);
                const [stats, riskyRecords, historyRecords] = await Promise.all([
                    statsResponse.json(),
                    riskyResponse.json(),
                    historyResponse.json(),
                ]);
                renderStats(stats);
                renderPredictionsTable(
                    riskyDisks,
                    riskyRecords,
                    "Пока нет дисков с высоким риском отказа."
                );
                renderPredictionsTable(
                    history,
                    historyRecords,
                    "Пока нет сохраненных прогнозов."
                );
                return {{stats, riskyRecords, historyRecords}};
            }}

            form.addEventListener("submit", async (event) => {{
                event.preventDefault();
                result.innerHTML = '<div class="empty">Выполняю прогноз...</div>';

                try {{
                    const response = await fetch("/predict", {{
                        method: "POST",
                        headers: {{"Content-Type": "application/json"}},
                        body: payload.value
                    }});
                    const data = await response.json();
                    if (!response.ok) {{
                        result.textContent = JSON.stringify(data, null, 2);
                        return;
                    }}
                    const refreshed = await refreshTables();
                    renderPredictionResult(refreshed.historyRecords[0]);
                }} catch (error) {{
                    result.textContent = `Запрос не выполнен: ${{error}}`;
                }}
            }});

            refreshTables();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/model-info", tags=["model"], summary="Get loaded model metadata")
def model_info() -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    return {
        "model_class": model.__class__.__name__,
        "features_count": len(feature_cols),
        "feature_cols": feature_cols,
        "preprocessing": {
            "base_features_count": len(preprocessing_metadata.get("base_feature_cols", [])),
            "window_size": preprocessing_metadata.get("window_size"),
            "target_col": preprocessing_metadata.get("target_col"),
        },
        "artifact_paths": {
            "model": MODEL_PATH.as_posix(),
            "features": FEATURES_PATH.as_posix(),
            "preprocessing": PREPROCESSING_PATH.as_posix(),
        },
    }


@app.get(
    "/predictions",
    response_model=list[PredictionRecord],
    tags=["inference"],
    summary="Get recent predictions",
)
def recent_predictions(
    limit: int = 20,
    only_anomalies: bool = False,
) -> list[PredictionRecord]:
    return fetch_prediction_records(limit=limit, only_anomalies=only_anomalies)


@app.get(
    "/risky-disks",
    response_model=list[PredictionRecord],
    tags=["inference"],
    summary="Get disks that need attention",
)
def risky_disks(limit: int = 20) -> list[PredictionRecord]:
    return fetch_risky_disk_records(limit=limit)


@app.get(
    "/prediction-stats",
    response_model=PredictionStatsResponse,
    tags=["inference"],
    summary="Get prediction storage summary",
)
def prediction_stats() -> PredictionStatsResponse:
    return fetch_prediction_stats()


@app.get(
    "/drift-status",
    response_model=DriftStatusResponse,
    tags=["model"],
    summary="Get drift status summary",
)
def drift_status() -> DriftStatusResponse:
    return build_drift_status()


@app.get("/experiments", response_class=HTMLResponse, tags=["model"])
def experiments() -> HTMLResponse:
    metrics_data = load_json_file(METRICS_PATH)
    run_info = load_json_file(MLFLOW_RUN_INFO_PATH)
    drift = build_drift_status()

    metrics_rows = "".join(
        f"<tr><td>{escape(str(key))}</td><td>{escape(format_float(value))}</td></tr>"
        for key, value in metrics_data.items()
        if isinstance(value, (int, float))
    )
    top_features = drift.summary.get("top_drifted_features", [])[:10]
    top_features_html = "".join(
        f"<li>{escape(str(feature))}</li>" for feature in top_features
    )
    run_id = escape(str(run_info.get("run_id", "n/a")))
    model_uri = escape(str(run_info.get("model_uri", "n/a")))
    experiment_name = escape(str(run_info.get("experiment_name", "n/a")))

    html = f"""
    <!doctype html>
    <html lang="ru">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Эксперименты | Predictive HDD Failure</title>
        <style>
            body {{
                margin: 0;
                font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                color: #16202a;
                background: #f7f8fb;
            }}
            main {{
                width: min(1080px, calc(100% - 32px));
                margin: 0 auto;
                padding: 28px 0 40px;
            }}
            header {{
                display: flex;
                justify-content: space-between;
                gap: 16px;
                align-items: center;
                margin-bottom: 18px;
            }}
            h1 {{
                margin: 0;
                font-size: 1.8rem;
            }}
            h2 {{
                margin: 0 0 12px;
                font-size: 1.05rem;
            }}
            a {{
                color: #155840;
                font-weight: 700;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            .grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 16px;
            }}
            .card {{
                background: #ffffff;
                border: 1px solid #d8dee8;
                border-radius: 8px;
                padding: 18px;
                box-shadow: 0 10px 28px rgba(22, 32, 42, 0.06);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            td {{
                border-bottom: 1px solid #d8dee8;
                padding: 9px 6px;
            }}
            td:first-child {{
                color: #5d6975;
                font-weight: 650;
            }}
            .flag {{
                display: inline-flex;
                border-radius: 999px;
                padding: 6px 10px;
                font-size: 0.86rem;
                font-weight: 700;
                margin: 0 6px 6px 0;
            }}
            .good {{
                color: #05603a;
                background: #ecfdf3;
                border: 1px solid #abefc6;
            }}
            .bad {{
                color: #b42318;
                background: #fef3f2;
                border: 1px solid #fecdca;
            }}
            code {{
                word-break: break-all;
            }}
            @media (max-width: 820px) {{
                header, .grid {{
                    display: block;
                }}
                .card {{
                    margin-bottom: 16px;
                }}
            }}
        </style>
    </head>
    <body>
        <main>
            <header>
                <h1>Эксперименты и качество модели</h1>
                <a href="/">Назад к инференсу</a>
            </header>
            <section class="grid">
                <div class="card">
                    <h2>MLflow run</h2>
                    <table>
                        <tbody>
                            <tr><td>experiment</td><td>{experiment_name}</td></tr>
                            <tr><td>run_id</td><td><code>{run_id}</code></td></tr>
                            <tr><td>model_uri</td><td><code>{model_uri}</code></td></tr>
                        </tbody>
                    </table>
                </div>
                <div class="card">
                    <h2>Флаги дрейфа</h2>
                    <span class="flag {'bad' if drift.data_drift else 'good'}">data drift: {'да' if drift.data_drift else 'нет'}</span>
                    <span class="flag {'bad' if drift.target_drift else 'good'}">target drift: {'да' if drift.target_drift else 'нет'}</span>
                    <span class="flag {'bad' if drift.concept_drift else 'good'}">concept drift: {'да' if drift.concept_drift else 'нет'}</span>
                    <table>
                        <tbody>
                            <tr><td>max PSI</td><td>{format_float(drift.summary.get("max_psi"))}</td></tr>
                            <tr><td>drifted features</td><td>{escape(str(drift.summary.get("drifted_feature_count", "n/a")))}</td></tr>
                            <tr><td>concept metric drop</td><td>{format_float(drift.summary.get("concept_primary_metric_drop"))}</td></tr>
                        </tbody>
                    </table>
                </div>
                <div class="card">
                    <h2>Test metrics</h2>
                    <table><tbody>{metrics_rows or '<tr><td colspan="2">Метрики не найдены</td></tr>'}</tbody></table>
                </div>
                <div class="card">
                    <h2>Top drifted features</h2>
                    <ol>{top_features_html or '<li>Нет данных</li>'}</ol>
                </div>
            </section>
        </main>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    summary="Predict HDD failure probability",
)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model is not loaded.")

        missing_features = [feature for feature in feature_cols if feature not in request.features]
        if missing_features:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Missing required features.",
                    "missing_features": missing_features,
                },
            )

        ordered_features = {feature: request.features.get(feature) for feature in feature_cols}
        inference_df = pd.DataFrame([ordered_features], columns=feature_cols)
        inference_df = inference_df.apply(pd.to_numeric, errors="coerce")

        if inference_df.isna().any().any():
            invalid_features = inference_df.columns[inference_df.isna().any()].tolist()
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Some feature values are invalid or null.",
                    "invalid_features": invalid_features,
                },
            )

        logger.info("Running inference for a single request.")
        probability = float(model.predict_proba(inference_df)[0, 1])
        prediction = int(model.predict(inference_df)[0])

        if prediction == 1:
            PREDICTIONS_CLASS_1.inc()
        else:
            PREDICTIONS_CLASS_0.inc()

        anomaly = bool(
            prediction == 1 or probability >= PREDICTION_ALERT_THRESHOLD
        )
        save_prediction_record(
            request=request,
            failure_probability=probability,
            prediction=prediction,
            anomaly=anomaly,
        )

        return PredictionResponse(
            failure_probability=probability,
            prediction=prediction,
            anomaly=anomaly,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
