from __future__ import annotations

import json
import logging
import pickle
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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

model = None
feature_cols: list[str] = []
preprocessing_metadata: dict[str, Any] = {}

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


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global model, feature_cols, preprocessing_metadata

    logger.info("Loading model artifacts...")
    model, feature_cols, preprocessing_metadata = load_artifacts()
    logger.info("Artifacts loaded. Features count: %s", len(feature_cols))
    yield


app = FastAPI(
    title="Predictive HDD Failure API",
    description="Inference сервис для предсказания отказа диска.",
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
    if request.url.path != "/predict":
        return await call_next(request)

    PREDICT_REQUESTS.inc()
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        PREDICTION_ERRORS.inc()
        INFERENCE_LATENCY.observe(time.perf_counter() - start_time)
        raise

    if response.status_code >= 400:
        PREDICTION_ERRORS.inc()

    INFERENCE_LATENCY.observe(time.perf_counter() - start_time)
    return response


@app.get("/health", tags=["health"], summary="Check service health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    sample_features = {feature: 0 for feature in feature_cols}
    sample_payload = json.dumps({"features": sample_features}, indent=2)

    html = f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Predictive HDD Failure MLOps</title>
        <style>
            :root {{
                --bg: #f4efe6;
                --ink: #18201b;
                --muted: #5d665d;
                --panel: #fffaf0;
                --accent: #d95f32;
                --accent-dark: #9e3f24;
                --border: #d8c8ad;
            }}
            body {{
                margin: 0;
                min-height: 100vh;
                font-family: Georgia, "Times New Roman", serif;
                color: var(--ink);
                background:
                    radial-gradient(circle at 15% 20%, rgba(217, 95, 50, 0.20), transparent 28rem),
                    linear-gradient(135deg, #f4efe6 0%, #e8dcc4 100%);
            }}
            main {{
                width: min(1080px, calc(100% - 32px));
                margin: 0 auto;
                padding: 40px 0;
            }}
            .hero {{
                display: grid;
                grid-template-columns: 1.1fr 0.9fr;
                gap: 24px;
                align-items: stretch;
            }}
            .card {{
                background: rgba(255, 250, 240, 0.92);
                border: 1px solid var(--border);
                border-radius: 24px;
                box-shadow: 0 24px 70px rgba(45, 32, 18, 0.14);
                padding: 28px;
            }}
            h1 {{
                margin: 0 0 12px;
                font-size: clamp(2.1rem, 5vw, 4.5rem);
                line-height: 0.95;
                letter-spacing: -0.04em;
            }}
            p {{
                color: var(--muted);
                font-size: 1.05rem;
                line-height: 1.55;
            }}
            textarea {{
                width: 100%;
                min-height: 430px;
                box-sizing: border-box;
                resize: vertical;
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 16px;
                font-family: "Cascadia Mono", Consolas, monospace;
                font-size: 0.9rem;
                background: #fffdf7;
                color: var(--ink);
            }}
            button {{
                margin-top: 14px;
                width: 100%;
                border: 0;
                border-radius: 999px;
                padding: 14px 18px;
                background: var(--accent);
                color: white;
                font-size: 1rem;
                font-weight: 700;
                cursor: pointer;
            }}
            button:hover {{
                background: var(--accent-dark);
            }}
            #result {{
                min-height: 76px;
                white-space: pre-wrap;
                font-family: "Cascadia Mono", Consolas, monospace;
                background: #18201b;
                color: #f8f0df;
                border-radius: 16px;
                padding: 16px;
                overflow-x: auto;
            }}
            .links {{
                display: grid;
                gap: 12px;
                margin-top: 20px;
            }}
            .links a {{
                color: var(--accent-dark);
                font-weight: 700;
                text-decoration: none;
            }}
            .links a:hover {{
                text-decoration: underline;
            }}
            @media (max-width: 860px) {{
                .hero {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <main>
            <section class="hero">
                <div class="card">
                    <h1>Predictive HDD Failure MLOps</h1>
                    <p>
                        Minimal monitoring and inference UI for the FastAPI service.
                        Paste JSON features, run prediction, then inspect API docs and monitoring.
                    </p>
                    <div id="result">Prediction result will appear here.</div>
                    <div class="links">
                        <a href="/docs">Swagger UI</a>
                        <a href="/metrics">Metrics</a>
                        <a href="http://localhost:9090">Prometheus</a>
                        <a href="http://localhost:3000">Grafana</a>
                    </div>
                </div>
                <div class="card">
                    <form id="predict-form">
                        <label for="payload"><strong>Prediction payload</strong></label>
                        <textarea id="payload" spellcheck="false">{sample_payload}</textarea>
                        <button type="submit">Predict</button>
                    </form>
                </div>
            </section>
        </main>
        <script>
            const form = document.getElementById("predict-form");
            const payload = document.getElementById("payload");
            const result = document.getElementById("result");

            form.addEventListener("submit", async (event) => {{
                event.preventDefault();
                result.textContent = "Running prediction...";

                try {{
                    const response = await fetch("/predict", {{
                        method: "POST",
                        headers: {{"Content-Type": "application/json"}},
                        body: payload.value
                    }});
                    const data = await response.json();
                    result.textContent = JSON.stringify(data, null, 2);
                }} catch (error) {{
                    result.textContent = `Request failed: ${{error}}`;
                }}
            }});
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

        return PredictionResponse(
            failure_probability=probability,
            prediction=prediction,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
