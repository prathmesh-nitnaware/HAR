"""Pydantic schemas for the MotionSense AI FastAPI backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="'healthy' if all models loaded, 'degraded' if partial")
    device: str = Field(..., description="Compute device ('cpu' or 'cuda')")
    models: Dict[str, bool] = Field(..., description="Load status for each architecture")
    models_loaded_count: int = Field(..., description="Number of successfully loaded models")
    total_models: int = Field(..., description="Total expected deep learning models")


class ModelMetadata(BaseModel):
    id: str = Field(..., description="Model identifier ('cnn1d', 'gru', 'lstm', 'randomforest')")
    name: str = Field(..., description="Display name")
    type: str = Field(..., description="Architecture paradigm")
    architecture: str = Field(..., description="Structural description")
    input_representation: str = Field(..., description="Feature shape and input representation")
    parameters: Optional[int] = Field(None, description="Trainable parameter count")
    disk_size_mb: Optional[float] = Field(None, description="Model checkpoint file size in MB")
    int8_size_mb: Optional[float] = Field(None, description="Quantized INT8 ONNX model size in MB")
    status: str = Field(..., description="'ready' or 'unavailable'")
    test_accuracy: Optional[float] = Field(None, description="Benchmark test accuracy")
    macro_f1: Optional[float] = Field(None, description="Benchmark test macro F1")


class SampleSummary(BaseModel):
    id: str = Field(..., description="Safe unique sample identifier")
    filename: str = Field(..., description="Source CSV filename")
    name: str = Field(..., description="Human-readable title")
    description: str = Field(..., description="Description of the recorded activity")
    ground_truth: Optional[str] = Field(None, description="Standard activity key")
    ground_truth_label: Optional[str] = Field(None, description="Display label for ground truth")
    expected_rows: int = Field(..., description="Number of sensor readings")
    channels: int = Field(..., description="Number of sensor channels")
    duration_sec: float = Field(..., description="Nominal recording duration at 50 Hz")


class SampleDetailResponse(BaseModel):
    metadata: SampleSummary
    columns: List[str]
    timestamps: List[float]
    signals: Dict[str, List[float]]  # channel_name -> list of float readings
    rows_count: int


class WindowModelPrediction(BaseModel):
    predicted_activity: str
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]  # activity_key -> probability float


class WindowPrediction(BaseModel):
    window_id: int
    start_time_sec: float
    end_time_sec: float
    consensus_activity: str
    consensus_label: str
    unanimous_agreement: bool
    models: Dict[str, WindowModelPrediction]


class PredictResponse(BaseModel):
    source_name: str
    ground_truth: Optional[str] = None
    ground_truth_label: Optional[str] = None
    duration_sec: float
    n_rows: int
    n_windows: int
    stride: int
    schema_detected: str
    warnings: List[str] = Field(default_factory=list)
    dominant_activity: str
    dominant_label: str
    model_agreement_rate: float  # Percentage of unanimous windows
    mean_confidence: float
    available_models: List[str]
    windows: List[WindowPrediction]


class PerModelMetrics(BaseModel):
    model: str
    display_name: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    support: Dict[str, int]
    history: Optional[Dict[str, List[float]]] = None


class MetricsResponse(BaseModel):
    models: List[PerModelMetrics]
    activities: List[str]
    activity_labels: Dict[str, str]


class ConfusionMatrixData(BaseModel):
    model: str
    display_name: str
    matrix: List[List[int]]
    matrix_normalized: List[List[float]]
    activities: List[str]
    activity_labels: List[str]


class ConfusionMatrixResponse(BaseModel):
    matrices: Dict[str, ConfusionMatrixData]
    activities: List[str]
    activity_labels: List[str]


class ErrorResponse(BaseModel):
    detail: str
    error_type: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class LiveTelemetry(BaseModel):
    magnitude_g: float
    tilt_angle_deg: float
    state: str
    motion_variance: Optional[float] = None
    sample_count: int
    fall_detected: bool = Field(default=False, description="True if a confirmed fall event is detected")
    fall_state: str = Field(default="normal", description="Current fall engine state ('normal', 'freefall', 'impact', 'immobility', 'confirmed_fall', 'recovered')")
    peak_g: float = Field(default=0.0, description="Peak impact shock acceleration in g")
    jerk_g_per_sec: float = Field(default=0.0, description="Instantaneous jerk rate in g/s")


class FallAlertEvent(BaseModel):
    type: str = "fall_alert"
    timestamp: float
    severity: str = Field(default="critical", description="'warning' or 'critical'")
    peak_g: float
    peak_jerk: float
    posture: str
    tilt_deg: Optional[float] = None
    message: str


class LivePredictionMessage(BaseModel):
    type: str = "prediction"
    timestamp: float
    predicted_activity: str
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    model: str
    telemetry: LiveTelemetry
    window_count: int


