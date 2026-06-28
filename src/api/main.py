from __future__ import annotations

import json
import logging
import os
import pickle
import time
from collections import deque
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
from pydantic import BaseModel, Field

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
PREDICTION_HISTORY_LIMIT = int(os.getenv("PREDICTION_HISTORY_LIMIT", "20"))

model = None
feature_cols: list[str] = []
preprocessing_metadata: dict[str, Any] = {}
prediction_history: deque[dict[str, Any]] = deque(maxlen=PREDICTION_HISTORY_LIMIT)

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


class PredictionRequest(BaseModel):
    features: dict[str, float | int | None] = Field(
        ...,
        description=(
            "Словарь с оконными признаками для модели. "
            "Нужно передать все ключи из models/features.json."
        ),
    )


class PredictionResponse(BaseModel):
    failure_probability: float
    prediction: int
    anomaly: bool


class PredictionRecord(BaseModel):
    created_at: str
    failure_probability: float
    prediction: int
    anomaly: bool
    feature_count: int


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
            .metric {{
                background: var(--soft);
                border-radius: 8px;
                padding: 10px;
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
                white-space: pre-wrap;
                font-family: "Cascadia Mono", Consolas, monospace;
                background: #111827;
                color: #f9fafb;
                border-radius: 8px;
                padding: 12px;
                overflow-x: auto;
                margin-top: 12px;
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
                text-transform: uppercase;
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
                .grid, .metric-grid {{
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
                    <a href="/docs">OpenAPI</a>
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

                <div class="card">
                    <h2>Последние предсказания</h2>
                    <div id="history">Загрузка истории...</div>
                </div>
            </section>
        </main>
        <script>
            const form = document.getElementById("predict-form");
            const payload = document.getElementById("payload");
            const result = document.getElementById("result");
            const history = document.getElementById("history");

            function renderHistory(records) {{
                if (!records.length) {{
                    history.innerHTML = '<div class="empty">Пока нет предсказаний в текущем процессе API.</div>';
                    return;
                }}
                const rows = records.map((record) => `
                    <tr>
                        <td>${{new Date(record.created_at).toLocaleString("ru-RU")}}</td>
                        <td>${{record.failure_probability.toFixed(4)}}</td>
                        <td>${{record.prediction}}</td>
                        <td>${{record.anomaly ? '<span class="flag bad-flag">аномалия</span>' : '<span class="flag good-flag">норма</span>'}}</td>
                        <td>${{record.feature_count}}</td>
                    </tr>
                `).join("");
                history.innerHTML = `
                    <table>
                        <thead>
                            <tr>
                                <th>Время</th>
                                <th>P(failure)</th>
                                <th>Класс</th>
                                <th>Флаг</th>
                                <th>Признаки</th>
                            </tr>
                        </thead>
                        <tbody>${{rows}}</tbody>
                    </table>
                `;
            }}

            async function refreshHistory() {{
                const response = await fetch("/predictions");
                const records = await response.json();
                renderHistory(records);
            }}

            form.addEventListener("submit", async (event) => {{
                event.preventDefault();
                result.textContent = "Выполняю прогноз...";

                try {{
                    const response = await fetch("/predict", {{
                        method: "POST",
                        headers: {{"Content-Type": "application/json"}},
                        body: payload.value
                    }});
                    const data = await response.json();
                    result.textContent = JSON.stringify(data, null, 2);
                    await refreshHistory();
                }} catch (error) {{
                    result.textContent = `Запрос не выполнен: ${{error}}`;
                }}
            }});

            refreshHistory();
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
def recent_predictions() -> list[PredictionRecord]:
    return [PredictionRecord(**record) for record in reversed(prediction_history)]


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
        prediction_history.append(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "failure_probability": probability,
                "prediction": prediction,
                "anomaly": anomaly,
                "feature_count": len(feature_cols),
            }
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
