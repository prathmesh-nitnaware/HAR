"""MotionSense AI — FastAPI Backend Service.

Provides RESTful endpoints for model metadata, benchmark evaluation metrics,
safe sample inspection, and multi-model activity inference (CNN1D, GRU, LSTM)
over single-window and multi-window inertial sensor streams.
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.inference import (
    ACTIVITIES,
    ACTIVITY_LABELS,
    ModelManager,
    SAMPLE_REGISTRY,
    build_prediction_response,
    load_confusion_matrices_data,
    load_metrics_data,
    load_sample_dataframe,
    resolve_sample_id,
    validate_and_preprocess_sensor_csv,
)
from backend.schemas import (
    ConfusionMatrixResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    ModelMetadata,
    PredictResponse,
    SampleDetailResponse,
    SampleSummary,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("motionsense.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context: initialize and cache deep models once on startup."""
    logger.info("Initializing MotionSense AI Model Manager...")
    mgr = ModelManager.get_instance()
    loaded_count = sum(1 for v in mgr.load_status.values() if v)
    logger.info("Startup complete: %d of %d deep models loaded on %s", loaded_count, len(mgr.load_status), mgr.device)
    yield
    logger.info("Shutting down MotionSense AI API service.")


app = FastAPI(
    title="MotionSense AI API",
    description=(
        "Production-grade REST API serving Deep Learning architectures (1D-CNN, GRU, LSTM) "
        "and classical ML baselines for Human Activity Recognition using smartphone inertial dynamics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for local development (React Vite dev server defaults to localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global Exception Handler ─────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_type": "HTTPException"},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc: Exception):
    logger.error("Unhandled API error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred while processing the request.",
            "error_type": "InternalServerError",
        },
    )


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Service Health and Model Readiness",
    tags=["System"],
)
def get_health():
    """Return API operational status and checkpoint readiness for all architectures."""
    mgr = ModelManager.get_instance()
    loaded_count = sum(1 for loaded in mgr.load_status.values() if loaded)
    total_models = len(mgr.load_status)

    if loaded_count == total_models:
        health_status = "healthy"
    elif loaded_count > 0:
        health_status = "degraded"
    else:
        health_status = "unhealthy"

    return HealthResponse(
        status=health_status,
        device=str(mgr.device),
        models=mgr.load_status,
        models_loaded_count=loaded_count,
        total_models=total_models,
    )


@app.get(
    "/api/models",
    response_model=List[ModelMetadata],
    summary="Model Architecture Registry & Metadata",
    tags=["Models"],
)
def get_models():
    """Return available model architectures, parameter counts, disk sizes, and benchmark metrics."""
    mgr = ModelManager.get_instance()
    specs = mgr.get_model_specs()
    return specs


@app.get(
    "/api/samples",
    response_model=List[SampleSummary],
    summary="List Compatible Benchmark Samples",
    tags=["Samples"],
)
def list_samples():
    """List verified sample recordings from sample_data/ with ground-truth and physical metadata."""
    samples = []
    for s_meta in SAMPLE_REGISTRY.values():
        samples.append(SampleSummary(**s_meta))
    return samples


@app.get(
    "/api/samples/{sample_id}",
    response_model=SampleDetailResponse,
    summary="Retrieve Sample Sensor Stream",
    tags=["Samples"],
)
def get_sample_detail(sample_id: str):
    """Retrieve time-series sensor data and metadata for a verified sample recording."""
    sample_meta = resolve_sample_id(sample_id)
    if not sample_meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample '{sample_id}' not found. Available sample IDs: {list(SAMPLE_REGISTRY.keys())}",
        )

    try:
        df, _ = load_sample_dataframe(sample_meta)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    n_rows = len(df)
    timestamps = [round(i / 50.0, 3) for i in range(n_rows)]
    signals_dict = {col: [round(float(val), 5) for val in df[col].values] for col in df.columns}

    return SampleDetailResponse(
        metadata=SampleSummary(**sample_meta),
        columns=list(df.columns),
        timestamps=timestamps,
        signals=signals_dict,
        rows_count=n_rows,
    )


@app.post(
    "/api/predict/sample/{sample_id}",
    response_model=PredictResponse,
    summary="Run Inference on Pre-packaged Sample",
    tags=["Inference"],
)
def predict_sample(
    sample_id: str,
    stride: int = Query(default=128, description="Window sliding stride (128 for non-overlapping, 64 for 50% overlap)"),
):
    """Execute forward inference on a verified sample using all available models."""
    sample_meta = resolve_sample_id(sample_id)
    if not sample_meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample '{sample_id}' not found. Available sample IDs: {list(SAMPLE_REGISTRY.keys())}",
        )

    mgr = ModelManager.get_instance()
    loaded_models = mgr.get_loaded_models()
    if not loaded_models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No deep learning models are currently loaded. Please check checkpoint availability in out/.",
        )

    df, _ = load_sample_dataframe(sample_meta)
    val_result = validate_and_preprocess_sensor_csv(df)
    if not val_result["valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Sample validation failed: {val_result['error']}",
        )

    # Validate stride
    if stride not in (64, 128):
        stride = 128 if len(df) == 128 else 64

    res = build_prediction_response(
        source_name=sample_meta["filename"],
        val_result=val_result,
        stride=stride,
        models=loaded_models,
        ground_truth=sample_meta.get("ground_truth"),
        ground_truth_label=sample_meta.get("ground_truth_label"),
        device=mgr.device,
    )
    return res


@app.post(
    "/api/predict/csv",
    response_model=PredictResponse,
    summary="Upload Custom Sensor CSV and Predict Activities",
    tags=["Inference"],
)
async def predict_custom_csv(
    file: UploadFile = File(..., description="Compatible CSV file containing 9-channel or 6-channel IMU recordings"),
    stride: int = Form(default=64, description="Window sliding stride: 64 (50% overlap) or 128 (non-overlapping)"),
):
    """Validate, preprocess, segment, and predict activity for an uploaded sensor recording."""
    mgr = ModelManager.get_instance()
    loaded_models = mgr.get_loaded_models()
    if not loaded_models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No deep learning models are currently loaded. Ensure model weights are present in out/.",
        )

    # Security check: verify filename and mime
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only standard comma-separated (.csv) text files are accepted.",
        )

    # Read content with a 10MB safety limit
    MAX_FILE_BYTES = 10 * 1024 * 1024
    content = await file.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum allowed size of {MAX_FILE_BYTES / (1024*1024):.0f} MB.",
        )

    # Parse CSV in-memory
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed CSV: Failed to parse file contents. Details: {str(exc)}",
        )

    # Validate schema, numeric types, missing values, and sequence length
    val_result = validate_and_preprocess_sensor_csv(df)
    if not val_result["valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=val_result["error"],
        )

    if stride not in (32, 64, 128):
        stride = 64

    res = build_prediction_response(
        source_name=file.filename,
        val_result=val_result,
        stride=stride,
        models=loaded_models,
        device=mgr.device,
    )
    return res


@app.get(
    "/api/metrics",
    response_model=MetricsResponse,
    summary="Verified Evaluation Metrics",
    tags=["Evaluation"],
)
def get_metrics():
    """Retrieve verified evaluation metrics (accuracy, macro F1, per-class metrics, loss curves)."""
    data = load_metrics_data()
    return MetricsResponse(**data)


@app.get(
    "/api/confusion-matrices",
    response_model=ConfusionMatrixResponse,
    summary="Confusion Matrices for Evaluated Models",
    tags=["Evaluation"],
)
def get_confusion_matrices():
    """Retrieve confusion matrices (raw counts and normalized percentages) across evaluated models."""
    data = load_confusion_matrices_data()
    return ConfusionMatrixResponse(**data)
