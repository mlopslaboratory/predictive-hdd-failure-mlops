from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "rf_model.pkl"
FEATURES_PATH = MODELS_DIR / "features.json"
PREPROCESSING_PATH = MODELS_DIR / "preprocessing.json"

app = FastAPI(
    title="Predictive HDD Failure API",
    description="Inference сервис для предсказания отказа диска.",
    version="1.0.0",
    openapi_tags=[
        {"name": "health", "description": "Service health checks."},
        {"name": "model", "description": "Loaded model metadata."},
        {"name": "inference", "description": "HDD failure inference."},
    ],
)

model = None
feature_cols: list[str] = []
preprocessing_metadata: dict[str, Any] = {}


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


@app.on_event("startup")
def startup_event() -> None:
    global model, feature_cols, preprocessing_metadata

    logger.info("Loading model artifacts...")
    model, feature_cols, preprocessing_metadata = load_artifacts()
    logger.info("Artifacts loaded. Features count: %s", len(feature_cols))


@app.get("/health", tags=["health"], summary="Check service health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    summary="Predict HDD failure probability",
)
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
