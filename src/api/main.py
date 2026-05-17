from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "features.json"

app = FastAPI(
    title="Predictive HDD Failure API",
    description="Inference сервис для предсказания отказа диска.",
    version="1.0.0",
)

model = None
feature_cols: list[str] = []


class PredictionRequest(BaseModel):
    features: dict[str, float | int | None] = Field(
        ...,
        description="Словарь с признаками для модели.",
    )


class PredictionResponse(BaseModel):
    failure_probability: float
    prediction: int


def load_artifacts() -> tuple[Any, list[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")

    loaded_model = joblib.load(MODEL_PATH)
    loaded_features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    return loaded_model, loaded_features


@app.on_event("startup")
def startup_event() -> None:
    global model, feature_cols

    logger.info("Loading model artifacts...")
    model, feature_cols = load_artifacts()
    logger.info("Artifacts loaded. Features count: %s", len(feature_cols))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    return {
        "model_class": model.__class__.__name__,
        "features_count": len(feature_cols),
        "feature_cols": feature_cols,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
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

    return PredictionResponse(
        failure_probability=probability,
        prediction=prediction,
    )
