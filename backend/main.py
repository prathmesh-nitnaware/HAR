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

import time
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.inference import (
    ACTIVITIES,
    ACTIVITY_LABELS,
    ModelManager,
    RealTimeStreamBuffer,
    SAMPLE_REGISTRY,
    SIGNAL_NAMES,
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


# ── Live Real-Time Streaming & WebSocket ──────────────────────────────────────

@app.get(
    "/api/stream/simulation-samples",
    summary="Get Continuous Sensor Streams for Simulation",
    tags=["Streaming"],
)
def get_simulation_samples():
    """Returns sample continuous sensor streams across classes for browser live stream simulation."""
    results = {}
    for sample_id, meta in SAMPLE_REGISTRY.items():
        try:
            df, _ = load_sample_dataframe(meta)
            readings = []
            for _, row in df.iterrows():
                r_dict = {}
                for col in df.columns:
                    if col in SIGNAL_NAMES:
                        r_dict[col] = float(row[col])
                r_dict["accel_x"] = float(row.get("total_acc_x", row.get("body_acc_x", 0.0)))
                r_dict["accel_y"] = float(row.get("total_acc_y", row.get("body_acc_y", 0.0)))
                r_dict["accel_z"] = float(row.get("total_acc_z", row.get("body_acc_z", 0.0)))
                r_dict["gyro_x"] = float(row.get("body_gyro_x", 0.0))
                r_dict["gyro_y"] = float(row.get("body_gyro_y", 0.0))
                r_dict["gyro_z"] = float(row.get("body_gyro_z", 0.0))
                readings.append(r_dict)
            results[meta.get("ground_truth", sample_id)] = {
                "name": meta["name"],
                "activity": meta.get("ground_truth", "walking"),
                "readings": readings,
            }
        except Exception as e:
            logger.warning("Could not load simulation sample for %s: %s", sample_id, e)
    return {"samples": results}


@app.websocket("/ws/stream")
async def websocket_sensor_stream(websocket: WebSocket):
    """Real-time bi-directional WebSocket stream for live smartphone or simulated sensor data.
    
    Accepts incoming 50 Hz IMU packets (6-channel raw or 9-channel), buffers rolling 128-sample
    windows, extracts real-time biomechanical telemetry, and pushes continuous model classifications.
    """
    await websocket.accept()
    buffer = RealTimeStreamBuffer(capacity=512, window_size=128, stride=16)
    mgr = ModelManager.get_instance()
    
    selected_model_name = "convlstm" if "convlstm" in mgr.get_loaded_models() else "cnn1d"
    window_count = 0

    try:
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to MotionSense AI Live Sensor Streaming Service",
            "available_models": list(mgr.get_loaded_models().keys()),
            "active_model": selected_model_name,
        })

        while True:
            msg = await websocket.receive_json()
            action = msg.get("action", "")

            if action == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})

            elif action == "configure":
                req_model = str(msg.get("model", "")).lower()
                if req_model in mgr.get_loaded_models():
                    selected_model_name = req_model
                req_stride = int(msg.get("stride", 16))
                if req_stride in (8, 16, 32, 64):
                    buffer.stride = req_stride
                await websocket.send_json({
                    "type": "configured",
                    "active_model": selected_model_name,
                    "stride": buffer.stride,
                })

            elif action == "push":
                reading = msg.get("data", {})
                buffer.add_reading(reading)
                
                if buffer.should_predict():
                    window_count += 1
                    window_arr = buffer.extract_latest_window()
                    model = mgr.get_model(selected_model_name)
                    
                    if model is not None:
                        with torch.no_grad():
                            t_in = torch.from_numpy(window_arr).to(mgr.device)
                            t0 = time.time()
                            logits = model(t_in)
                            latency_ms = round((time.time() - t0) * 1000.0, 2)
                            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                            pred_idx = int(np.argmax(probs))
                            conf = float(probs[pred_idx])
                            pred_key = ACTIVITIES[pred_idx]
                            
                            pred_payload = {
                                "type": "prediction",
                                "timestamp": time.time(),
                                "window_count": window_count,
                                "predicted_activity": pred_key,
                                "predicted_label": ACTIVITY_LABELS.get(pred_key, pred_key),
                                "confidence": round(conf, 4),
                                "probabilities": {
                                    ACTIVITIES[i]: round(float(probs[i]), 4) for i in range(len(ACTIVITIES))
                                },
                                "model": selected_model_name,
                                "latency_ms": latency_ms,
                                "telemetry": buffer.compute_telemetry(),
                            }
                            await websocket.send_json(pred_payload)
                    else:
                        await websocket.send_json({
                            "type": "telemetry",
                            "telemetry": buffer.compute_telemetry(),
                        })
                else:
                    if buffer.total_samples_received % 5 == 0:
                        await websocket.send_json({
                            "type": "telemetry",
                            "telemetry": buffer.compute_telemetry(),
                        })

            elif action == "push_batch":
                batch = msg.get("batch", [])
                for reading in batch:
                    buffer.add_reading(reading)
                    if buffer.should_predict():
                        window_count += 1
                        window_arr = buffer.extract_latest_window()
                        model = mgr.get_model(selected_model_name)
                        if model is not None:
                            with torch.no_grad():
                                t_in = torch.from_numpy(window_arr).to(mgr.device)
                                t0 = time.time()
                                logits = model(t_in)
                                latency_ms = round((time.time() - t0) * 1000.0, 2)
                                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                                pred_idx = int(np.argmax(probs))
                                conf = float(probs[pred_idx])
                                pred_key = ACTIVITIES[pred_idx]
                                
                                await websocket.send_json({
                                    "type": "prediction",
                                    "timestamp": time.time(),
                                    "window_count": window_count,
                                    "predicted_activity": pred_key,
                                    "predicted_label": ACTIVITY_LABELS.get(pred_key, pred_key),
                                    "confidence": round(conf, 4),
                                    "probabilities": {
                                        ACTIVITIES[i]: round(float(probs[i]), 4) for i in range(len(ACTIVITIES))
                                    },
                                    "model": selected_model_name,
                                    "latency_ms": latency_ms,
                                    "telemetry": buffer.compute_telemetry(),
                                })
                await websocket.send_json({
                    "type": "telemetry",
                    "telemetry": buffer.compute_telemetry(),
                })

            elif action == "reset":
                buffer = RealTimeStreamBuffer(capacity=512, window_size=128, stride=buffer.stride)
                window_count = 0
                await websocket.send_json({"type": "reset_ack", "message": "Buffer reset successfully."})

    except WebSocketDisconnect:
        logger.info("Live WebSocket client disconnected.")
    except Exception as exc:
        logger.error("Error in sensor stream WebSocket: %s", exc)

