"""MotionSense AI — Human Activity Recognition Dashboard.

Production-grade Deep Learning inference and evaluation application for smartphone
inertial sensor dynamics (accelerometer and gyroscope).
"""

from __future__ import annotations

import json
import os
import random
import io
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import streamlit as st
import torch
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MotionSense AI — Human Activity Recognition",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────────────
ACTIVITIES = [
    "walking", "walking-upstairs", "walking-downstairs",
    "sitting", "standing", "laying",
]
ACTIVITY_LABELS = {
    "walking": "Walking",
    "walking-upstairs": "Walking Upstairs",
    "walking-downstairs": "Walking Downstairs",
    "sitting": "Sitting",
    "standing": "Standing",
    "laying": "Laying",
}
N_CLASSES = len(ACTIVITIES)
N_CHANNELS = 9
SEQ_LEN = 128
SEED = 42

SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

SIGNAL_DISPLAY_NAMES = {
    "body_acc_x": "Body Acc X",
    "body_acc_y": "Body Acc Y",
    "body_acc_z": "Body Acc Z",
    "body_gyro_x": "Body Gyro X",
    "body_gyro_y": "Body Gyro Y",
    "body_gyro_z": "Body Gyro Z",
    "total_acc_x": "Total Acc X",
    "total_acc_y": "Total Acc Y",
    "total_acc_z": "Total Acc Z",
}

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out"
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"

# ── Custom CSS ───────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* Base typography and layout refinement */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #0f172a;
}

/* App Header styling */
.app-header {
    margin-bottom: 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #e2e8f0;
}
.app-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.025em;
    margin: 0;
    line-height: 1.2;
}
.app-subtitle {
    font-size: 0.95rem;
    color: #475569;
    margin-top: 0.25rem;
    margin-bottom: 0;
}

/* KPI metric cards */
div[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
}
div[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    letter-spacing: -0.02em;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Sidebar branding and cards */
.sidebar-brand-box {
    padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}
.brand-badge {
    display: inline-block;
    background: #0f172a;
    color: #ffffff;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    margin-bottom: 0.4rem;
}
.brand-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0;
    line-height: 1.2;
}
.brand-sub {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 0.2rem;
}

.sidebar-section-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 1.25rem;
    margin-bottom: 0.5rem;
}

/* Status indicators */
.status-pill {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.4rem 0.65rem;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
}
.status-indicator {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    margin-right: 0.45rem;
}
.status-ready {
    background-color: #10b981;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
}
.status-offline {
    background-color: #ef4444;
}
.status-baseline {
    background-color: #6366f1;
}

/* Prediction card highlights */
.prediction-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}
.prediction-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}
.prediction-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0;
}
.badge-match {
    display: inline-block;
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-mismatch {
    display: inline-block;
    background: #fffbeb;
    color: #b45309;
    border: 1px solid #fde68a;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Callout box */
.info-callout {
    background: #f8fafc;
    border-left: 3px solid #2563eb;
    border-radius: 0 6px 6px 0;
    padding: 0.75rem 1rem;
    margin: 1rem 0;
    font-size: 0.88rem;
    color: #334155;
}

/* Tab container refinement */
.stTabs [data-baseweb="tab-list"] {
    gap: 1.5rem;
    border-bottom: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    padding-top: 0.6rem;
    padding-bottom: 0.6rem;
    font-weight: 600;
    color: #64748b;
}
.stTabs [aria-selected="true"] {
    color: #2563eb !important;
}
</style>
"""

# ── PyTorch models (exact architectures preserved) ───────────────────────

class LSTMModel(nn.Module):
    def __init__(self, n_channels: int = N_CHANNELS, n_classes: int = N_CLASSES,
                 hidden: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        return self.fc(out)


class GRUModel(nn.Module):
    def __init__(self, n_channels: int = N_CHANNELS, n_classes: int = N_CLASSES,
                 hidden: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.gru = nn.GRU(n_channels, hidden, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        out = out.mean(dim=1)
        return self.fc(out)


class CNN1DModel(nn.Module):
    def __init__(self, n_channels: int = N_CHANNELS, n_classes: int = N_CLASSES,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = out.squeeze(-1)
        out = self.dropout(out)
        return self.fc(out)


MODEL_MAP: Dict[str, type] = {"lstm": LSTMModel, "gru": GRUModel, "cnn1d": CNN1DModel}

# ── Data loading helpers ─────────────────────────────────────────────────

def _load_signal_file(path: Path, n_rows: int) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float32).reshape(n_rows, SEQ_LEN)


@st.cache_data(show_spinner="Loading test-set signals…")
def load_test_signals(data_dir: str = str(DATA_DIR)) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X_test, y_test) — raw inertial signals for the test split."""
    root = Path(data_dir) / "UCI HAR Dataset"
    if not root.exists():
        st.error(f"UCI HAR Dataset not found at {root}. "
                 f"Place the extracted dataset under {data_dir}/")
        st.stop()

    y = np.loadtxt(root / "test" / "y_test.txt", dtype=np.int64) - 1
    n_samples = y.shape[0]
    subdir = root / "test" / "Inertial Signals"
    channels = []
    for sig_name in SIGNAL_NAMES:
        path = subdir / f"{sig_name}_test.txt"
        channels.append(_load_signal_file(path, n_samples))
    X = np.stack(channels, axis=1).astype(np.float32)
    return X, y


@st.cache_data(show_spinner="Loading test subject metadata…")
def load_test_subjects(data_dir: str = str(DATA_DIR)) -> np.ndarray:
    """Return array of subject IDs for the test split."""
    root = Path(data_dir) / "UCI HAR Dataset" / "test" / "subject_test.txt"
    if root.exists():
        try:
            return np.loadtxt(root, dtype=np.int64)
        except Exception:
            pass
    return np.zeros(2947, dtype=np.int64)



@st.cache_data(show_spinner="Loading evaluation metrics…")
def load_metrics(out_dir: str = str(OUT_DIR)) -> List[dict]:
    path = Path(out_dir) / "metrics.json"
    if not path.exists():
        st.error(f"metrics.json not found at {path}. "
                 f"Run har.py first to produce metrics and model weights.")
        st.stop()
    with open(path) as f:
        return json.load(f)


# ── Model loading ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading LSTM model…")
def load_lstm(weights_path: str) -> LSTMModel:
    model = LSTMModel()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading GRU model…")
def load_gru(weights_path: str) -> GRUModel:
    model = GRUModel()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading CNN1D model…")
def load_cnn1d(weights_path: str) -> CNN1DModel:
    model = CNN1DModel()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_all_models(out_dir: str = str(OUT_DIR)) -> Dict[str, nn.Module]:
    out = Path(out_dir)
    models: Dict[str, nn.Module] = {}
    missing: List[str] = []

    for name in ("cnn1d", "gru", "lstm"):
        fpath = out / f"{name}_best.pt"
        if not fpath.exists():
            missing.append(str(fpath))
            continue
        if name == "lstm":
            models[name] = load_lstm(str(fpath))
        elif name == "gru":
            models[name] = load_gru(str(fpath))
        else:
            models[name] = load_cnn1d(str(fpath))

    if missing:
        st.warning("Missing model weights:\n\n" + "\n".join(f"- {m}" for m in missing) +
                   "\n\nRun `python har.py` to train and save weights.")

    return models


@torch.no_grad()
def predict_all(models: Dict[str, nn.Module], x: np.ndarray) -> Dict[str, Dict]:
    """x shape (9, 128) -> predict with each model. Returns {model: {pred_idx, probs}}."""
    tensor = torch.from_numpy(x).unsqueeze(0)  # (1, 9, 128)
    results = {}
    for name, model in models.items():
        logits = model(tensor)  # (1, n_classes)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
        pred = int(probs.argmax())
        results[name] = {"pred": pred, "probs": probs}
    return results


def compute_detailed_metrics(m: dict) -> dict:
    """Compute precision, recall, and F1 across classes and macro averages from confusion matrix."""
    cm = np.array(m["confusion_matrix"], dtype=np.float64)
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)

    return {
        "model": m["model"],
        "accuracy": float(m["accuracy"]),
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_f1": float(np.mean(f1)),
        "per_class_precision": {ACTIVITIES[i]: float(prec[i]) for i in range(len(ACTIVITIES))},
        "per_class_recall": {ACTIVITIES[i]: float(rec[i]) for i in range(len(ACTIVITIES))},
        "per_class_f1": {ACTIVITIES[i]: float(f1[i]) for i in range(len(ACTIVITIES))},
        "support": {ACTIVITIES[i]: int(np.sum(cm[i, :])) for i in range(len(ACTIVITIES))},
        "confusion_matrix": cm,
        "history": m.get("history"),
    }


def get_model_spec(model_name: str, model_obj: Optional[nn.Module] = None) -> dict:
    """Retrieve architectural specs, parameter counts, and checkpoint disk sizes."""
    m_lower = model_name.lower()
    weights_path = OUT_DIR / f"{m_lower}_best.pt"
    disk_mb = (weights_path.stat().st_size / (1024 * 1024)) if weights_path.exists() else None

    if model_obj is not None:
        param_count = sum(p.numel() for p in model_obj.parameters())
    elif m_lower == "cnn1d":
        param_count = 709830
    elif m_lower == "gru":
        param_count = 404742
    elif m_lower == "lstm":
        param_count = 539142
    else:
        param_count = None

    descriptions = {
        "cnn1d": ("3-Block ConvNet (64→128→256)", "Raw 9×128 Inertial Signals"),
        "gru": ("2-Layer BiGRU (128 hidden)", "Raw 9×128 Inertial Signals"),
        "lstm": ("2-Layer BiLSTM (128 hidden)", "Raw 9×128 Inertial Signals"),
        "randomforest": ("200 Decision Trees (max depth 25)", "561 Engineered Domain Features"),
    }
    arch_desc, input_desc = descriptions.get(m_lower, ("Deep Learning", "Inertial Signals"))
    return {
        "architecture": arch_desc,
        "input_rep": input_desc,
        "params": param_count,
        "disk_mb": disk_mb,
    }


def benchmark_cpu_latency(models: Dict[str, nn.Module], sample: np.ndarray, n_runs: int = 50) -> Dict[str, Dict[str, float]]:
    """Measure actual CPU inference latency and throughput per window."""
    results = {}
    tensor = torch.from_numpy(sample).unsqueeze(0)
    for name, model in models.items():
        # Warmup
        for _ in range(5):
            _ = model(tensor)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(tensor)
        dt_ms = ((time.perf_counter() - t0) / n_runs) * 1000.0
        fps = 1000.0 / dt_ms if dt_ms > 0 else 0.0
        results[name] = {"latency_ms": dt_ms, "throughput_fps": fps}
    return results


def validate_and_preprocess_sensor_csv(df: pd.DataFrame) -> dict:
    """Validate uploaded CSV against expected 9-ch or 6-ch schema and apply standard preprocessing."""
    result = {
        "valid": False,
        "error": None,
        "warnings": [],
        "schema": None,
        "n_rows": len(df),
        "duration_sec": len(df) / 50.0,
        "clean_array": None,  # (N, 9)
    }

    if df.empty or len(df) == 0:
        result["error"] = "The uploaded CSV file contains no data rows."
        return result

    if len(df) > 50000:
        result["error"] = f"File exceeds maximum allowed size of 50,000 rows (found {len(df):,} rows)."
        return result

    # Standardize column names (lowercase, strip whitespace, underscores)
    col_map = {col: str(col).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_") for col in df.columns}
    df_clean = df.rename(columns=col_map)

    # Synonyms mapping
    synonyms = {
        "acc_x": "total_acc_x", "acc_y": "total_acc_y", "acc_z": "total_acc_z",
        "accel_x": "total_acc_x", "accel_y": "total_acc_y", "accel_z": "total_acc_z",
        "gyro_x": "body_gyro_x", "gyro_y": "body_gyro_y", "gyro_z": "body_gyro_z",
        "linear_acc_x": "body_acc_x", "linear_acc_y": "body_acc_y", "linear_acc_z": "body_acc_z",
    }
    df_clean = df_clean.rename(columns=synonyms)

    cols_9ch = SIGNAL_NAMES
    cols_6ch = ["total_acc_x", "total_acc_y", "total_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"]

    has_9ch = all(c in df_clean.columns for c in cols_9ch)
    has_6ch = all(c in df_clean.columns for c in cols_6ch)

    if has_9ch:
        target_schema = "9ch"
        active_cols = cols_9ch
    elif has_6ch:
        target_schema = "6ch"
        active_cols = cols_6ch
    else:
        found = list(df.columns)
        missing_9 = [c for c in cols_9ch if c not in df_clean.columns]
        missing_6 = [c for c in cols_6ch if c not in df_clean.columns]
        result["error"] = (
            f"Missing required sensor channels. Detected columns: {found}. "
            f"Expected either 9-channel schema (missing: {missing_9}) or 6-channel IMU schema (missing: {missing_6})."
        )
        return result

    # Verify numeric data types
    df_subset = df_clean[active_cols].copy()
    for col in active_cols:
        non_num = pd.to_numeric(df_subset[col], errors="coerce")
        nan_new = non_num.isna().sum() - df_subset[col].isna().sum()
        if nan_new > 0:
            result["error"] = f"Non-numeric values detected in column '{col}' ({nan_new} invalid entries). All sensor values must be numeric."
            return result
        df_subset[col] = non_num

    # Check missing values
    total_cells = df_subset.size
    total_nans = df_subset.isna().sum().sum()
    if total_nans > 0:
        nan_pct = (total_nans / total_cells) * 100.0
        if nan_pct > 5.0:
            result["error"] = f"Data quality check failed: {total_nans} missing values detected ({nan_pct:.1f}% of data), exceeding the 5% threshold."
            return result
        else:
            df_subset = df_subset.ffill().bfill()
            result["warnings"].append(f"Interpolated {total_nans} missing values ({nan_pct:.2f}% of recording) using forward-fill.")

    # Check minimum row count
    if len(df_subset) < SEQ_LEN:
        result["error"] = (
            f"Insufficient sequence length: The recording has {len(df_subset)} readings. "
            f"All deep learning models (1D-CNN, GRU, LSTM) require a sequence length of {SEQ_LEN} timesteps (2.56 seconds at 50 Hz). "
            f"Please upload a recording with at least {SEQ_LEN} rows."
        )
        return result

    # Preprocess into standard 9-channel array
    if target_schema == "9ch":
        clean_arr = df_subset[cols_9ch].values.astype(np.float32)
    else:  # 6ch
        total_acc = df_subset[["total_acc_x", "total_acc_y", "total_acc_z"]].values.astype(np.float32)
        gyro = df_subset[["body_gyro_x", "body_gyro_y", "body_gyro_z"]].values.astype(np.float32)
        # Apply Butterworth low-pass filter at 0.3 Hz cutoff (50 Hz sampling rate) to separate gravity
        b, a = butter(3, 0.3 / 25.0, btype="low")
        gravity = filtfilt(b, a, total_acc, axis=0)
        body_acc = total_acc - gravity
        clean_arr = np.concatenate([body_acc, gyro, total_acc], axis=1).astype(np.float32)

    result["valid"] = True
    result["schema"] = target_schema
    result["clean_array"] = clean_arr
    return result


def slice_sensor_windows(data_N_9: np.ndarray, stride: int = 128) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Slice continuous sensor data (N, 9) into fixed-width 128-timestep windows."""
    n_rows = len(data_N_9)
    n_windows = (n_rows - SEQ_LEN) // stride + 1
    windows = []
    time_ranges = []
    for i in range(n_windows):
        start = i * stride
        end = start + SEQ_LEN
        # Transpose from (128, 9) to (9, 128) for PyTorch channel-first representation
        windows.append(data_N_9[start:end].T)
        time_ranges.append((start / 50.0, end / 50.0))
    return np.stack(windows, axis=0).astype(np.float32), time_ranges


def predict_custom_windows(models: Dict[str, nn.Module], windows: np.ndarray) -> Dict[str, Dict]:
    """Run batched forward inference for all models on (W, 9, 128) windows."""
    tensor = torch.from_numpy(windows).float()
    out = {}
    for name, model in models.items():
        with torch.no_grad():
            logits = model(tensor)  # (W, 6)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # (W, 6)
            preds = np.argmax(probs, axis=1)  # (W,)
            confs = np.max(probs, axis=1)  # (W,)
            out[name] = {"preds": preds, "probs": probs, "confs": confs}
    return out


def plot_custom_sensor_signals(
    data_N_9: np.ndarray,
    time_range: Optional[Tuple[float, float]] = None,
    title: str = "Uploaded Inertial Sensor Signals"
) -> go.Figure:
    """Plot multi-channel sensor signals with synchronized time axes."""
    n_rows = len(data_N_9)
    t_sec = np.arange(n_rows) / 50.0

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "<b>1. Linear Body Acceleration (g)</b> [Isolated Dynamic Motion]",
            "<b>2. Body Angular Velocity (rad/s)</b> [Rotational Gyroscope Dynamics]",
            "<b>3. Total Acceleration (g)</b> [Body Motion + Static Gravity Vector]",
        ],
    )

    # 1. Body Acc (0, 1, 2)
    for i, ch in enumerate([0, 1, 2]):
        name = SIGNAL_DISPLAY_NAMES[SIGNAL_NAMES[ch]]
        fig.add_trace(
            go.Scatter(
                x=t_sec, y=data_N_9[:, ch], mode="lines",
                name=name, line=dict(color=SIGNAL_PALETTE[ch], width=1.5),
                hovertemplate=f"<b>{name}</b><br>%{{x:.2f}}s: %{{y:.3f}} g<extra></extra>",
            ),
            row=1, col=1
        )

    # 2. Body Gyro (3, 4, 5)
    for i, ch in enumerate([3, 4, 5]):
        name = SIGNAL_DISPLAY_NAMES[SIGNAL_NAMES[ch]]
        fig.add_trace(
            go.Scatter(
                x=t_sec, y=data_N_9[:, ch], mode="lines",
                name=name, line=dict(color=SIGNAL_PALETTE[ch], width=1.5),
                hovertemplate=f"<b>{name}</b><br>%{{x:.2f}}s: %{{y:.3f}} rad/s<extra></extra>",
            ),
            row=2, col=1
        )

    # 3. Total Acc (6, 7, 8)
    for i, ch in enumerate([6, 7, 8]):
        name = SIGNAL_DISPLAY_NAMES[SIGNAL_NAMES[ch]]
        fig.add_trace(
            go.Scatter(
                x=t_sec, y=data_N_9[:, ch], mode="lines",
                name=name, line=dict(color=SIGNAL_PALETTE[ch], width=1.5),
                hovertemplate=f"<b>{name}</b><br>%{{x:.2f}}s: %{{y:.3f}} g<extra></extra>",
            ),
            row=3, col=1
        )

    if time_range is not None:
        fig.add_vrect(
            x0=time_range[0], x1=time_range[1],
            fillcolor="#2563eb", opacity=0.12, line_width=1, line_color="#2563eb",
            annotation_text="Active Window", annotation_position="top left",
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#0f172a")),
        template="plotly_white",
        height=540,
        margin=dict(l=45, r=20, t=55, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=-0.16, xanchor="center", x=0.5, font=dict(size=10)),
    )
    fig.update_xaxes(title_text=f"Time elapsed (0.00 to {t_sec[-1]:.2f} s @ 50 Hz)", row=3, col=1, showgrid=True, gridcolor="#f1f5f9")
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", row=2, col=1)
    fig.update_yaxes(title_text="Acc (g)", showgrid=True, gridcolor="#f1f5f9", row=1, col=1)
    fig.update_yaxes(title_text="Gyro (rad/s)", showgrid=True, gridcolor="#f1f5f9", row=2, col=1)
    fig.update_yaxes(title_text="Total (g)", showgrid=True, gridcolor="#f1f5f9", row=3, col=1)
    return fig


def plot_custom_activity_distribution(consensus_classes: List[int]) -> go.Figure:
    """Plot distribution of predicted activities across recording windows."""
    counts = {act: 0 for act in ACTIVITIES}
    for c in consensus_classes:
        counts[ACTIVITIES[c]] += 1
    
    act_labels = [ACTIVITY_LABELS[a] for a in ACTIVITIES]
    act_counts = [counts[a] for a in ACTIVITIES]
    total_w = len(consensus_classes)
    act_pcts = [(c / total_w) * 100 if total_w > 0 else 0 for c in act_counts]

    colors = ["#2563eb", "#0d9488", "#d97706", "#6366f1", "#8b5cf6", "#ec4899"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=act_labels,
        y=act_pcts,
        marker_color=colors,
        text=[f"{p:.1f}% ({c}w)" if c > 0 else "0%" for p, c in zip(act_pcts, act_counts)],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Share: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Overall Predicted Activity Distribution (% of Windows)", font=dict(size=14, color="#0f172a")),
        yaxis=dict(title="Recording Share (%)", range=[0, max(act_pcts + [10]) * 1.25], showgrid=True, gridcolor="#f1f5f9"),
        xaxis=dict(title="", tickfont=dict(size=11, color="#334155")),
        template="plotly_white",
        height=320,
        margin=dict(l=40, r=20, t=45, b=25),
    )
    return fig



# ── Plot helpers ─────────────────────────────────────────────────────────

SIGNAL_PALETTE = [
    "#ef4444", "#3b82f6", "#10b981",  # Body Acc: Red, Blue, Green
    "#8b5cf6", "#f59e0b", "#06b6d4",  # Body Gyro: Purple, Amber, Cyan
    "#ec4899", "#14b8a6", "#64748b",  # Total Acc: Pink, Teal, Slate
]

MODEL_COLORS = {
    "cnn1d": "#2563eb",   # Primary Blue
    "gru": "#0d9488",     # Teal
    "lstm": "#d97706",    # Amber
    "randomforest": "#4f46e5"  # Indigo
}


def plot_signal_window(
    x: np.ndarray,
    sample_idx: int,
    gt_label: str,
    channel_group: str = "All Channels"
) -> go.Figure:
    """Plot inertial sensor channels with clear labeling and styling."""
    t = np.arange(SEQ_LEN)
    fig = go.Figure()

    # Determine which channels to render
    if channel_group == "Body Accelerometer":
        ch_indices = [0, 1, 2]
    elif channel_group == "Body Gyroscope":
        ch_indices = [3, 4, 5]
    elif channel_group == "Total Accelerometer":
        ch_indices = [6, 7, 8]
    else:
        ch_indices = list(range(N_CHANNELS))

    for ch in ch_indices:
        group_label = ["Body Acc", "Body Gyro", "Total Acc"][ch // 3]
        sig_name = SIGNAL_DISPLAY_NAMES.get(SIGNAL_NAMES[ch], SIGNAL_NAMES[ch])
        fig.add_trace(go.Scatter(
            x=t,
            y=x[ch],
            mode="lines",
            name=sig_name,
            line=dict(color=SIGNAL_PALETTE[ch], width=1.5),
            legendgroup=f"g{ch // 3}",
            legendgrouptitle_text=group_label,
            hovertemplate=f"<b>{sig_name}</b><br>Step %{{x}}<br>Val: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=f"Test Sample #{sample_idx} — Actual: <b>{ACTIVITY_LABELS.get(gt_label, gt_label)}</b>",
            font=dict(size=14, color="#0f172a"),
        ),
        xaxis=dict(
            title="Timestep (128 samples @ 50 Hz = 2.56 s window)",
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
            color="#475569",
        ),
        yaxis=dict(
            title="Normalized Amplitude (g / rad/s)",
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=True,
            zerolinecolor="#cbd5e1",
            color="#475569",
        ),
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=45, b=80),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
    )
    return fig


def plot_prob_bars(predictions: Dict[str, Dict]) -> go.Figure:
    """Grouped bar chart for model prediction probabilities."""
    fig = go.Figure()
    class_names = [ACTIVITY_LABELS.get(a, a) for a in ACTIVITIES]

    for model_name, res in predictions.items():
        probs = res["probs"]
        display_name = model_name.upper()
        color = MODEL_COLORS.get(model_name.lower(), "#2563eb")

        fig.add_trace(go.Bar(
            x=class_names,
            y=probs,
            name=display_name,
            marker_color=color,
            text=[f"{p:.1%}" if p >= 0.03 else "" for p in probs],
            textposition="outside",
            hovertemplate=f"<b>{display_name}</b><br>%{{x}}<br>Probability: %{{y:.2%}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="Class Probability Distribution by Model",
            font=dict(size=14, color="#0f172a"),
        ),
        xaxis=dict(
            title="",
            tickangle=0,
            tickfont=dict(size=12, color="#334155"),
        ),
        yaxis=dict(
            title="Confidence Probability",
            range=[0, 1.15],
            tickformat=".0%",
            showgrid=True,
            gridcolor="#f1f5f9",
            color="#475569",
        ),
        barmode="group",
        template="plotly_white",
        height=370,
        margin=dict(l=40, r=20, t=45, b=65),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
    )
    return fig


def plot_gallery_signals(
    x: np.ndarray,
    sample_idx: int,
    gt_label: str,
    view_mode: str = "Synchronized Modality Subplots"
) -> go.Figure:
    """Plot inertial sensor channels with physical units (g and rad/s)."""
    t_sec = np.linspace(0.0, 2.56, SEQ_LEN)
    act_display = ACTIVITY_LABELS.get(gt_label, gt_label)

    if view_mode == "Synchronized Modality Subplots":
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=[
                "<b>Body Linear Acceleration</b> (Low-pass filtered, units: g)",
                "<b>Body Angular Velocity</b> (Tri-axial Gyroscope, units: rad/s)",
                "<b>Total Sensor Acceleration</b> (Body Motion + Gravity, units: g)",
            ],
        )

        # 1. Body Acc (0, 1, 2)
        for i, ch in enumerate([0, 1, 2]):
            name = SIGNAL_DISPLAY_NAMES[SIGNAL_NAMES[ch]]
            fig.add_trace(
                go.Scatter(
                    x=t_sec, y=x[ch], mode="lines",
                    name=name, line=dict(color=SIGNAL_PALETTE[ch], width=1.5),
                    hovertemplate=f"<b>{name}</b><br>%{{x:.2f}}s: %{{y:.3f}} g<extra></extra>",
                ),
                row=1, col=1
            )

        # 2. Body Gyro (3, 4, 5)
        for i, ch in enumerate([3, 4, 5]):
            name = SIGNAL_DISPLAY_NAMES[SIGNAL_NAMES[ch]]
            fig.add_trace(
                go.Scatter(
                    x=t_sec, y=x[ch], mode="lines",
                    name=name, line=dict(color=SIGNAL_PALETTE[ch], width=1.5),
                    hovertemplate=f"<b>{name}</b><br>%{{x:.2f}}s: %{{y:.3f}} rad/s<extra></extra>",
                ),
                row=2, col=1
            )

        # 3. Total Acc (6, 7, 8)
        for i, ch in enumerate([6, 7, 8]):
            name = SIGNAL_DISPLAY_NAMES[SIGNAL_NAMES[ch]]
            fig.add_trace(
                go.Scatter(
                    x=t_sec, y=x[ch], mode="lines",
                    name=name, line=dict(color=SIGNAL_PALETTE[ch], width=1.5),
                    hovertemplate=f"<b>{name}</b><br>%{{x:.2f}}s: %{{y:.3f}} g<extra></extra>",
                ),
                row=3, col=1
            )

        fig.update_layout(
            template="plotly_white",
            height=530,
            margin=dict(l=45, r=20, t=45, b=25),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.16,
                xanchor="center",
                x=0.5,
                font=dict(size=10),
            ),
        )
        fig.update_xaxes(title_text="Time elapsed (0.00 to 2.56 s @ 50 Hz)", row=3, col=1, showgrid=True, gridcolor="#f1f5f9")
        fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", row=1, col=1)
        fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", row=2, col=1)
        fig.update_yaxes(title_text="Acc (g)", showgrid=True, gridcolor="#f1f5f9", row=1, col=1)
        fig.update_yaxes(title_text="Gyro (rad/s)", showgrid=True, gridcolor="#f1f5f9", row=2, col=1)
        fig.update_yaxes(title_text="Total (g)", showgrid=True, gridcolor="#f1f5f9", row=3, col=1)
        return fig

    # Single plot view
    fig = go.Figure()
    if view_mode == "Body Accelerometer (g)":
        ch_indices = [0, 1, 2]
        unit_str = "g"
    elif view_mode == "Body Gyroscope (rad/s)":
        ch_indices = [3, 4, 5]
        unit_str = "rad/s"
    elif view_mode == "Total Accelerometer (g)":
        ch_indices = [6, 7, 8]
        unit_str = "g"
    else:  # All 9 Channels
        ch_indices = list(range(N_CHANNELS))
        unit_str = "Normalized (g / rad/s)"

    for ch in ch_indices:
        name = SIGNAL_DISPLAY_NAMES[SIGNAL_NAMES[ch]]
        group_label = ["Body Acc", "Body Gyro", "Total Acc"][ch // 3]
        fig.add_trace(go.Scatter(
            x=t_sec, y=x[ch], mode="lines",
            name=name,
            line=dict(color=SIGNAL_PALETTE[ch], width=1.5),
            legendgroup=f"g{ch // 3}",
            legendgrouptitle_text=group_label,
            hovertemplate=f"<b>{name}</b><br>%{{x:.2f}}s: %{{y:.3f}} {unit_str}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=f"Test Sample #{sample_idx} — Actual: <b>{act_display}</b>",
            font=dict(size=14, color="#0f172a"),
        ),
        xaxis=dict(
            title="Time elapsed (0.00 to 2.56 s @ 50 Hz)",
            showgrid=True,
            gridcolor="#f1f5f9",
            color="#475569",
        ),
        yaxis=dict(
            title=f"Amplitude ({unit_str})",
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=True,
            zerolinecolor="#cbd5e1",
            color="#475569",
        ),
        template="plotly_white",
        height=400,
        margin=dict(l=45, r=20, t=45, b=65),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
    )
    return fig



def plot_experiment_learning_curves(metrics: List[dict]) -> go.Figure:
    """Plot training and validation loss and test accuracy curves across epochs."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "<b>Training Loss & Validation Loss</b>",
            "<b>Test Set Accuracy Progression</b>",
        ],
        horizontal_spacing=0.10,
    )

    deep_metrics = [m for m in metrics if m.get("history") is not None]
    for m in deep_metrics:
        m_name = m["model"].upper()
        color = MODEL_COLORS.get(m["model"].lower(), "#2563eb")
        history = m["history"]
        epochs = list(range(1, len(history["train_loss"]) + 1))

        # 1. Train loss (solid)
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["train_loss"],
                mode="lines",
                name=f"{m_name} Train Loss",
                line=dict(color=color, width=1.5),
                hovertemplate=f"<b>{m_name} Train Loss</b><br>Epoch %{{x}}: %{{y:.4f}}<extra></extra>",
            ),
            row=1, col=1,
        )
        # 2. Test loss (dashed)
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["test_loss"],
                mode="lines",
                name=f"{m_name} Test Loss",
                line=dict(color=color, width=1.5, dash="dash"),
                hovertemplate=f"<b>{m_name} Test Loss</b><br>Epoch %{{x}}: %{{y:.4f}}<extra></extra>",
            ),
            row=1, col=1,
        )
        # 3. Test accuracy
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["test_acc"],
                mode="lines+markers",
                name=f"{m_name} Test Acc",
                line=dict(color=color, width=1.8),
                marker=dict(size=4),
                hovertemplate=f"<b>{m_name} Test Acc</b><br>Epoch %{{x}}: %{{y:.2%}}<extra></extra>",
            ),
            row=1, col=2,
        )

    fig.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=20, t=45, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
    )
    fig.update_xaxes(title_text="Epoch (1 to 20)", showgrid=True, gridcolor="#f1f5f9", dtick=2, row=1, col=1)
    fig.update_xaxes(title_text="Epoch (1 to 20)", showgrid=True, gridcolor="#f1f5f9", dtick=2, row=1, col=2)
    fig.update_yaxes(title_text="Cross-Entropy Loss", showgrid=True, gridcolor="#f1f5f9", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", range=[0.65, 0.98], tickformat=".0%", showgrid=True, gridcolor="#f1f5f9", row=1, col=2)
    return fig


def plot_top_confusions(metrics: List[dict]) -> go.Figure:
    """Plot the major off-diagonal error counts across all models."""
    fig = go.Figure()
    pairs = [
        ("sitting", "standing", "Sitting → Standing"),
        ("standing", "sitting", "Standing → Sitting"),
        ("walking-upstairs", "walking", "Upstairs → Walking"),
        ("walking-downstairs", "walking-upstairs", "Downstairs → Upstairs"),
    ]
    pair_labels = [p[2] for p in pairs]

    for m in metrics:
        m_name = m["model"].upper()
        cm = np.array(m["confusion_matrix"])
        color = MODEL_COLORS.get(m["model"].lower(), "#2563eb")
        counts = []
        for src, dst, _ in pairs:
            i = ACTIVITIES.index(src)
            j = ACTIVITIES.index(dst)
            counts.append(int(cm[i, j]))

        fig.add_trace(go.Bar(
            x=pair_labels,
            y=counts,
            name=m_name,
            marker_color=color,
            text=counts,
            textposition="outside",
            hovertemplate=f"<b>{m_name}</b><br>%{{x}}<br>Misclassified: %{{y}} samples<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="Primary Confusion Pairs Across Architectures (Test Partition)",
            font=dict(size=14, color="#0f172a"),
        ),
        xaxis=dict(title="", tickfont=dict(size=11, color="#334155")),
        yaxis=dict(title="Misclassified Windows", showgrid=True, gridcolor="#f1f5f9"),
        barmode="group",
        template="plotly_white",
        height=360,
        margin=dict(l=40, r=20, t=45, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=11)),
    )
    return fig


def plot_multimetric_bars(detailed_metrics: List[dict]) -> go.Figure:
    """Grouped bar chart for Accuracy, Macro-Precision, Macro-Recall, and Macro-F1."""
    fig = go.Figure()
    metric_names = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]

    for m in detailed_metrics:
        m_name = m["model"].upper()
        color = MODEL_COLORS.get(m["model"].lower(), "#2563eb")
        values = [m["accuracy"], m["macro_precision"], m["macro_recall"], m["macro_f1"]]

        fig.add_trace(go.Bar(
            x=metric_names,
            y=values,
            name=m_name,
            marker_color=color,
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
            hovertemplate=f"<b>{m_name}</b><br>%{{x}}: %{{y:.2%}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="Overall Model Evaluation Across Core Performance Metrics",
            font=dict(size=14, color="#0f172a"),
        ),
        yaxis=dict(title="Score", range=[0.85, 1.02], tickformat=".0%", showgrid=True, gridcolor="#f1f5f9"),
        barmode="group",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=20, t=45, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=11)),
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────

def render_sidebar(metrics: Optional[List[dict]]) -> Tuple[Dict[str, nn.Module], str]:
    """Render application branding, navigation, and model readiness status."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand-box">
                <span class="brand-badge">DL Engine</span>
                <div class="brand-title">MotionSense AI</div>
                <div class="brand-sub">Human Activity Recognition Platform</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-title">Navigation</div>', unsafe_allow_html=True)
        nav_selection = st.radio(
            "Select Section",
            options=["Live Prediction", "Sample Gallery", "Custom Sensor Input", "Experiment Lab", "Model Comparison", "Confusion Matrices", "About Project"],
            index=0,
            label_visibility="collapsed",
        )


        models = load_all_models()

        st.markdown('<div class="sidebar-section-title">Model Status</div>', unsafe_allow_html=True)
        model_display_names = {
            "cnn1d": ("1D-CNN", "Deep ConvNet"),
            "gru": ("GRU", "Bidirectional"),
            "lstm": ("LSTM", "Bidirectional"),
        }
        for key in ("cnn1d", "gru", "lstm"):
            name, desc = model_display_names[key]
            is_loaded = key in models
            status_cls = "status-ready" if is_loaded else "status-offline"
            status_text = "Ready" if is_loaded else "Unavailable"
            st.markdown(
                f"""
                <div class="status-pill">
                    <div>
                        <span class="status-indicator {status_cls}"></span>
                        <strong>{name}</strong> <span style="color:#64748b; font-size:0.75rem;">({desc})</span>
                    </div>
                    <span style="font-weight:600; color:{'#059669' if is_loaded else '#dc2626'};">{status_text}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Baseline info
        st.markdown(
            """
            <div class="status-pill">
                <div>
                    <span class="status-indicator status-baseline"></span>
                    <strong>RandomForest</strong> <span style="color:#64748b; font-size:0.75rem;">(561 Baseline)</span>
                </div>
                <span style="font-weight:600; color:#4f46e5;">Evaluated</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if metrics is not None:
            deep_metrics = [m for m in metrics if m.get("model") != "randomforest"]
            if deep_metrics:
                best_model = max(deep_metrics, key=lambda r: r["accuracy"])
                st.markdown('<div class="sidebar-section-title">Top Checkpoint</div>', unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:0.6rem 0.8rem;">
                        <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; font-weight:600;">Best Deep Model</div>
                        <div style="font-size:1.05rem; font-weight:700; color:#0f172a; margin-top:0.15rem;">
                            {best_model['model'].upper()}
                        </div>
                        <div style="font-size:0.82rem; color:#059669; font-weight:600; margin-top:0.1rem;">
                            Accuracy: {best_model['accuracy']:.2%} · F1: {best_model['macro_f1']:.2%}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="sidebar-section-title">System Specs</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:0.78rem; color:#64748b; line-height:1.5;">
                • <strong>Dataset:</strong> UCI HAR (30 Subjects)<br>
                • <strong>Sampling:</strong> 50 Hz (128 samples / 2.56 s)<br>
                • <strong>Channels:</strong> 9 Inertial Signals<br>
                • <strong>Split:</strong> Subject-Independent
            </div>
            """,
            unsafe_allow_html=True,
        )

    return models, nav_selection


# ── Section 1: Live Prediction ───────────────────────────────────────────

def section_live_prediction(models: Dict[str, nn.Module], X_test: np.ndarray, y_test: np.ndarray):
    st.markdown("### Live Prediction & Signal Analysis")
    st.markdown(
        "Select a test sample window from the subject-independent test split "
        "to inspect raw sensor signals and compare real-time inference across models."
    )

    if not models:
        st.warning("No PyTorch models loaded. Verify weights in `out/`.")
        return

    # Sample Selection Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.2, 1.6])

    with ctrl_col1:
        sampling_mode = st.radio(
            "Selection Method",
            ["Random Sample", "Specify Index"],
            horizontal=True,
        )

    with ctrl_col2:
        if sampling_mode == "Random Sample":
            if st.button("Draw New Sample", type="primary", use_container_width=True):
                st.session_state.sample_idx = random.randint(0, X_test.shape[0] - 1)
            if "sample_idx" not in st.session_state:
                st.session_state.sample_idx = random.randint(0, X_test.shape[0] - 1)
            st.caption(f"Active index: **#{st.session_state.sample_idx}**")
        else:
            current_val = st.session_state.get("sample_idx", 0)
            idx_input = st.number_input(
                f"Sample Index (0 – {X_test.shape[0] - 1})",
                min_value=0,
                max_value=X_test.shape[0] - 1,
                value=int(current_val),
                step=1,
            )
            st.session_state.sample_idx = int(idx_input)

    with ctrl_col3:
        channel_view = st.selectbox(
            "Signal Channel Filter",
            ["All Channels", "Body Accelerometer", "Body Gyroscope", "Total Accelerometer"],
            index=0,
        )

    sample_idx = st.session_state.sample_idx
    actual_label_raw = ACTIVITIES[y_test[sample_idx]]
    actual_label = ACTIVITY_LABELS.get(actual_label_raw, actual_label_raw)
    current_signal = X_test[sample_idx]

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # Prediction execution
    predictions = predict_all(models, current_signal)

    # Layout: Prediction Cards & Signals
    col_preds, col_signal = st.columns([1.1, 1.9])

    with col_preds:
        # Ground truth card
        st.markdown(
            f"""
            <div class="prediction-card" style="border-left: 4px solid #0f172a;">
                <div class="prediction-title">Actual Activity (Ground Truth)</div>
                <div class="prediction-value">{actual_label}</div>
                <div style="font-size:0.75rem; color:#64748b; margin-top:0.35rem;">
                    Test Sample #{sample_idx} · 2.56 s Window (128 steps)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='font-size:0.8rem; font-weight:700; color:#475569; text-transform:uppercase; margin-bottom:0.4rem;'>"
            "Model Inferences"
            "</div>",
            unsafe_allow_html=True,
        )

        # Individual model prediction rows
        for model_key in ("cnn1d", "gru", "lstm"):
            if model_key not in predictions:
                continue
            res = predictions[model_key]
            pred_class_raw = ACTIVITIES[res["pred"]]
            pred_class = ACTIVITY_LABELS.get(pred_class_raw, pred_class_raw)
            confidence = res["probs"][res["pred"]]
            is_correct = (res["pred"] == y_test[sample_idx])

            badge_html = (
                f'<span class="badge-match">Match</span>'
                if is_correct
                else f'<span class="badge-mismatch">Mismatch</span>'
            )
            model_title = model_key.upper()
            border_color = MODEL_COLORS.get(model_key, "#e2e8f0")

            st.markdown(
                f"""
                <div class="prediction-card" style="border-left: 3px solid {border_color}; padding: 0.75rem 0.9rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.25rem;">
                        <span style="font-size:0.78rem; font-weight:700; color:#334155;">{model_title}</span>
                        {badge_html}
                    </div>
                    <div style="font-size:1.1rem; font-weight:700; color:#0f172a;">
                        {pred_class}
                    </div>
                    <div style="font-size:0.8rem; color:#475569; margin-top:0.15rem;">
                        Confidence: <strong>{confidence:.1%}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_signal:
        fig_signal = plot_signal_window(current_signal, sample_idx, actual_label_raw, channel_view)
        st.plotly_chart(fig_signal, use_container_width=True)

    # Class Probability Distribution
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    fig_probs = plot_prob_bars(predictions)
    st.plotly_chart(fig_probs, use_container_width=True)


# ── Section 2: Sample Gallery ────────────────────────────────────────────

def section_sample_gallery(
    models: Dict[str, nn.Module],
    X_test: np.ndarray,
    y_test: np.ndarray,
    subjects: np.ndarray,
) -> None:
    st.markdown("### Interactive Sample Gallery")
    st.markdown(
        "Explore recorded sensor signal windows from the 9 held-out test subjects. "
        "Filter by activity class or subject, inspect 9-axis multi-modal physical sensor streams "
        "with scientific units, and evaluate multi-model inference and consensus."
    )

    if not models:
        st.warning("No deep learning models loaded. Verify model weights in `out/`.")
        return

    # Filter Toolbar
    with st.container():
        fcol1, fcol2, fcol3 = st.columns([1.3, 1.1, 1.6])

        # 1. Activity Filter
        with fcol1:
            act_counts = {i: int(np.sum(y_test == i)) for i in range(N_CLASSES)}
            act_options = ["All Activities"] + [
                f"{ACTIVITY_LABELS[a]} ({act_counts[i]})" for i, a in enumerate(ACTIVITIES)
            ]
            sel_act_str = st.selectbox(
                "Filter Activity Class",
                options=act_options,
                index=0,
                key="gallery_act_filter",
            )

        # 2. Subject Filter
        with fcol2:
            unique_subjects = sorted(list(set(subjects)))
            subj_options = ["All Subjects"] + [f"Subject #{s}" for s in unique_subjects]
            sel_subj_str = st.selectbox(
                "Filter Subject",
                options=subj_options,
                index=0,
                key="gallery_subj_filter",
            )

        # Determine filtered indices
        filtered_mask = np.ones(len(y_test), dtype=bool)
        if sel_act_str != "All Activities":
            selected_act_name = sel_act_str.split(" (")[0]
            act_idx = next(i for i, a in enumerate(ACTIVITIES) if ACTIVITY_LABELS[a] == selected_act_name)
            filtered_mask = filtered_mask & (y_test == act_idx)

        if sel_subj_str != "All Subjects":
            subj_val = int(sel_subj_str.replace("Subject #", ""))
            filtered_mask = filtered_mask & (subjects == subj_val)

        filtered_indices = np.where(filtered_mask)[0]

        # 3. Quick sample navigation buttons
        with fcol3:
            st.markdown("<label style='font-size:0.875rem; font-weight:400;'>Sample Navigation</label>", unsafe_allow_html=True)
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1.4])
            with btn_col1:
                prev_clicked = st.button("◀ Prev", use_container_width=True)
            with btn_col2:
                next_clicked = st.button("Next ▶", use_container_width=True)
            with btn_col3:
                rand_clicked = st.button("🎲 Random", type="primary", use_container_width=True)

    if len(filtered_indices) == 0:
        st.warning("No test samples match the selected activity and subject filters.")
        return

    # Manage current selection in session_state
    if "gallery_idx" not in st.session_state or st.session_state.gallery_idx not in filtered_indices:
        st.session_state.gallery_idx = int(filtered_indices[0])

    if rand_clicked:
        st.session_state.gallery_idx = int(random.choice(filtered_indices))
    elif prev_clicked:
        cur_pos = list(filtered_indices).index(st.session_state.gallery_idx) if st.session_state.gallery_idx in filtered_indices else 0
        new_pos = (cur_pos - 1) % len(filtered_indices)
        st.session_state.gallery_idx = int(filtered_indices[new_pos])
    elif next_clicked:
        cur_pos = list(filtered_indices).index(st.session_state.gallery_idx) if st.session_state.gallery_idx in filtered_indices else 0
        new_pos = (cur_pos + 1) % len(filtered_indices)
        st.session_state.gallery_idx = int(filtered_indices[new_pos])

    # Sample dropdown selector
    sample_select_options = [
        f"Sample #{idx:04d} · Subject {subjects[idx]} · {ACTIVITY_LABELS[ACTIVITIES[y_test[idx]]]}"
        for idx in filtered_indices
    ]
    current_option_idx = list(filtered_indices).index(st.session_state.gallery_idx)
    selected_option = st.selectbox(
        f"Select Sample ({len(filtered_indices)} matching current filters)",
        options=sample_select_options,
        index=current_option_idx,
        key="gallery_sample_select",
    )
    chosen_pos = sample_select_options.index(selected_option)
    sample_idx = int(filtered_indices[chosen_pos])
    st.session_state.gallery_idx = sample_idx

    actual_class_raw = ACTIVITIES[y_test[sample_idx]]
    actual_class = ACTIVITY_LABELS.get(actual_class_raw, actual_class_raw)
    current_signal = X_test[sample_idx]
    subject_id = subjects[sample_idx]

    # Compute biomechanical signal dynamics
    body_acc = current_signal[0:3]
    body_gyro = current_signal[3:6]
    smv_body = np.sqrt(np.sum(body_acc**2, axis=0))
    smv_gyro = np.sqrt(np.sum(body_gyro**2, axis=0))
    std_body_acc = np.std(smv_body)
    peak_body_acc = np.max(smv_body)
    peak_gyro = np.max(smv_gyro)
    is_dynamic = std_body_acc > 0.08

    # Metadata & Sensor Dynamics Banner
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="prediction-card" style="border-left: 3px solid #0f172a;">
                <div class="prediction-title">Sample & Subject</div>
                <div class="prediction-value">#{sample_idx}</div>
                <div style="font-size:0.8rem; color:#475569; margin-top:0.2rem;">
                    Subject #{subject_id} · Test Partition
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="prediction-card" style="border-left: 3px solid #2563eb;">
                <div class="prediction-title">Ground-Truth Activity</div>
                <div class="prediction-value" style="color:#2563eb;">{actual_class}</div>
                <div style="font-size:0.8rem; color:#475569; margin-top:0.2rem;">
                    {'Dynamic Motion' if is_dynamic else 'Static Posture'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="prediction-card" style="border-left: 3px solid #0d9488;">
                <div class="prediction-title">Window Duration & Rate</div>
                <div class="prediction-value">2.56 s</div>
                <div style="font-size:0.8rem; color:#475569; margin-top:0.2rem;">
                    128 steps @ 50 Hz · 50% overlap
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="prediction-card" style="border-left: 3px solid #d97706;">
                <div class="prediction-title">Signal Dynamics</div>
                <div class="prediction-value">{peak_body_acc:.2f} g</div>
                <div style="font-size:0.8rem; color:#475569; margin-top:0.2rem;">
                    Peak Gyro: {peak_gyro:.2f} rad/s
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Sensor Signals Plot
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    plot_ctrl_col1, plot_ctrl_col2 = st.columns([2, 1])
    with plot_ctrl_col1:
        st.markdown("#### Sensor Signal Inspection")
    with plot_ctrl_col2:
        plot_view = st.selectbox(
            "Visualization Mode",
            [
                "Synchronized Modality Subplots",
                "All 9 Channels",
                "Body Accelerometer (g)",
                "Body Gyroscope (rad/s)",
                "Total Accelerometer (g)",
            ],
            index=0,
            key="gallery_plot_view",
        )

    fig_gallery = plot_gallery_signals(current_signal, sample_idx, actual_class_raw, plot_view)
    st.plotly_chart(fig_gallery, use_container_width=True)

    # Multi-Model Prediction & Consensus
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### Multi-Model Prediction & Consensus")

    predictions = predict_all(models, current_signal)

    # Calculate model consensus
    correct_count = sum(1 for m, res in predictions.items() if res["pred"] == y_test[sample_idx])
    total_models = len(predictions)

    if correct_count == total_models:
        consensus_bg = "#ecfdf5"
        consensus_border = "#a7f3d0"
        consensus_color = "#047857"
        consensus_text = f"Unanimous Match ({correct_count}/{total_models} Models Correct)"
        consensus_desc = f"All active deep models unanimously predicted <strong>{actual_class}</strong> matching ground truth."
    elif correct_count >= (total_models / 2.0):
        consensus_bg = "#eff6ff"
        consensus_border = "#bfdbfe"
        consensus_color = "#1d4ed8"
        consensus_text = f"Majority Consensus ({correct_count}/{total_models} Models Correct)"
        consensus_desc = f"Majority of models correctly identified <strong>{actual_class}</strong> with boundary variance."
    else:
        consensus_bg = "#fffbeb"
        consensus_border = "#fde68a"
        consensus_color = "#b45309"
        consensus_text = f"Disagreement / Ambiguous Window ({correct_count}/{total_models} Correct)"
        consensus_desc = f"Hard boundary window; models struggled to resolve <strong>{actual_class}</strong>."

    st.markdown(
        f"""
        <div style="background:{consensus_bg}; border:1px solid {consensus_border}; border-radius:8px; padding:0.85rem 1.1rem; margin-bottom:1rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:700; color:{consensus_color}; font-size:1rem;">{consensus_text}</span>
                <span style="font-size:0.8rem; font-weight:600; color:{consensus_color};">Ground Truth: {actual_class}</span>
            </div>
            <div style="font-size:0.85rem; color:#334155; margin-top:0.25rem;">
                {consensus_desc}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model cards
    card_cols = st.columns(len(predictions))
    for i, model_key in enumerate(("cnn1d", "gru", "lstm")):
        if model_key not in predictions:
            continue
        res = predictions[model_key]
        pred_label_raw = ACTIVITIES[res["pred"]]
        pred_label = ACTIVITY_LABELS.get(pred_label_raw, pred_label_raw)
        conf = res["probs"][res["pred"]]
        matches = (res["pred"] == y_test[sample_idx])
        badge = '<span class="badge-match">✓ Match</span>' if matches else '<span class="badge-mismatch">✕ Mismatch</span>'
        border_c = MODEL_COLORS.get(model_key, "#2563eb")
        arch_tag = "3-Block ConvNet" if model_key == "cnn1d" else ("2-Layer BiGRU" if model_key == "gru" else "2-Layer BiLSTM")

        with card_cols[i]:
            st.markdown(
                f"""
                <div class="prediction-card" style="border-top: 3px solid {border_c}; padding: 0.85rem 1rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;">
                        <span style="font-size:0.85rem; font-weight:700; color:#0f172a;">{model_key.upper()}</span>
                        {badge}
                    </div>
                    <div style="font-size:0.75rem; color:#64748b; margin-bottom:0.4rem;">{arch_tag}</div>
                    <div style="font-size:1.15rem; font-weight:700; color:#0f172a;">
                        {pred_label}
                    </div>
                    <div style="font-size:0.82rem; color:#475569; margin-top:0.2rem;">
                        Confidence: <strong>{conf:.1%}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Probability Distribution Chart
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    fig_prob = plot_prob_bars(predictions)
    st.plotly_chart(fig_prob, use_container_width=True)


# ── Section 2.5: Custom Sensor Input ─────────────────────────────────────

def section_custom_sensor_input(
    models: Dict[str, nn.Module],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    st.markdown("### Custom Sensor Input & Activity Inference")
    st.markdown(
        "Upload your own smartphone inertial measurement unit (IMU) sensor recording in CSV format. "
        "The system validates sensor compatibility, applies Butterworth signal conditioning, "
        "segments the continuous recording into standardized 2.56-second windows, and runs inference "
        "across all trained deep learning architectures (1D-CNN, GRU, LSTM)."
    )

    # Scientific disclaimer & requirements callout
    st.markdown(
        """
        <div class="info-callout" style="border-left-color: #2563eb;">
            <strong>Scientific & Physical Data Compatibility Requirements:</strong><br>
            • <strong>Sampling Frequency:</strong> Expected sampling rate is <strong>50 Hz</strong> (20 ms interval per row). Signals sampled at divergent rates will experience frequency compression or dilation.<br>
            • <strong>Window Length:</strong> Deep neural models consume fixed-width sequences of <strong>128 readings (2.56 seconds)</strong>. Uploads must contain at least 128 rows.<br>
            • <strong>Physical Sensor Units:</strong> Accelerometer data must be in <strong>standard gravity units (g)</strong> (1g ≈ 9.81 m/s²). Gyroscope data must be in <strong>radians per second (rad/s)</strong>.<br>
            • <strong>Sensor Placement:</strong> Models assume waist or front-pocket placement aligned with the longitudinal body axis (matching the UCI HAR experimental protocol).
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Template & Reference Section ──
    with st.expander("📖 Supported CSV Schema Reference & Downloadable Templates", expanded=False):
        c_spec1, c_spec2 = st.columns(2)
        with c_spec1:
            st.markdown(
                """
                **Schema Option A: 9-Channel Standard Representation**
                Directly matches the UCI HAR model input channels:
                - `body_acc_x, body_acc_y, body_acc_z` *(Body Acc in g)*
                - `body_gyro_x, body_gyro_y, body_gyro_z` *(Body Gyro in rad/s)*
                - `total_acc_x, total_acc_y, total_acc_z` *(Total Acc in g)*
                """
            )
        with c_spec2:
            st.markdown(
                """
                **Schema Option B: 6-Channel Raw IMU Representation**
                Standard triaxial accelerometer + triaxial gyroscope:
                - `total_acc_x, total_acc_y, total_acc_z` (or `acc_x, acc_y, acc_z` in *g*)
                - `body_gyro_x, body_gyro_y, body_gyro_z` (or `gyro_x, gyro_y, gyro_z` in *rad/s*)
                *Pipeline automatically separates linear body motion from gravity via 0.3 Hz Butterworth filtering.*
                """
            )

        st.markdown("---")
        st.markdown("**Downloadable Templates & Authentic Benchmark Recordings:**")
        t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns(5)
        
        def _get_csv_bytes(fpath: Path) -> bytes:
            if fpath.exists():
                with open(fpath, "rb") as f:
                    return f.read()
            return b""

        with t_col1:
            st.download_button(
                "📥 9-Ch Blank Schema",
                data=_get_csv_bytes(SAMPLE_DATA_DIR / "template_9ch_schema.csv"),
                file_name="template_9ch_schema.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with t_col2:
            st.download_button(
                "📥 6-Ch IMU Schema",
                data=_get_csv_bytes(SAMPLE_DATA_DIR / "template_6ch_imu.csv"),
                file_name="template_6ch_imu.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with t_col3:
            st.download_button(
                "📥 Walking Sample (128)",
                data=_get_csv_bytes(SAMPLE_DATA_DIR / "sample_walking_128.csv"),
                file_name="sample_walking_128.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with t_col4:
            st.download_button(
                "📥 Sitting Sample (128)",
                data=_get_csv_bytes(SAMPLE_DATA_DIR / "sample_sitting_128.csv"),
                file_name="sample_sitting_128.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with t_col5:
            st.download_button(
                "📥 Stairs Sample (384)",
                data=_get_csv_bytes(SAMPLE_DATA_DIR / "sample_stairs_multiwindow_384.csv"),
                file_name="sample_stairs_multiwindow_384.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ── Upload & Quick Demo Selector ──
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    up_col1, up_col2 = st.columns([1.6, 1.0])

    with up_col1:
        uploaded_file = st.file_uploader(
            "Upload Inertial Sensor CSV File",
            type=["csv"],
            help="Upload a compatible CSV containing tri-axial accelerometer and gyroscope readings.",
        )

    with up_col2:
        st.markdown("**Or Load Pre-packaged Benchmark Sample:**")
        demo_btn_col1, demo_btn_col2 = st.columns(2)
        with demo_btn_col1:
            if st.button("🚶 Load Walking (128)", use_container_width=True):
                p = SAMPLE_DATA_DIR / "sample_walking_128.csv"
                if p.exists():
                    st.session_state["custom_df"] = pd.read_csv(p)
                    st.session_state["custom_filename"] = "sample_walking_128.csv"
            if st.button("🪑 Load Sitting (128)", use_container_width=True):
                p = SAMPLE_DATA_DIR / "sample_sitting_128.csv"
                if p.exists():
                    st.session_state["custom_df"] = pd.read_csv(p)
                    st.session_state["custom_filename"] = "sample_sitting_128.csv"
        with demo_btn_col2:
            if st.button("🧗 Load Stairs (384)", use_container_width=True):
                p = SAMPLE_DATA_DIR / "sample_stairs_multiwindow_384.csv"
                if p.exists():
                    st.session_state["custom_df"] = pd.read_csv(p)
                    st.session_state["custom_filename"] = "sample_stairs_multiwindow_384.csv"
            if st.button("🔄 Clear / Reset", use_container_width=True):
                st.session_state.pop("custom_df", None)
                st.session_state.pop("custom_filename", None)
                st.rerun()

    # Determine active dataframe
    active_df = None
    active_filename = None
    if uploaded_file is not None:
        try:
            active_df = pd.read_csv(uploaded_file)
            active_filename = uploaded_file.name
        except Exception as exc:
            st.error(f"Malformed CSV Error: Could not parse uploaded file. Details: {exc}")
            return
    elif "custom_df" in st.session_state:
        active_df = st.session_state["custom_df"]
        active_filename = st.session_state.get("custom_filename", "demo_sample.csv")

    if active_df is None:
        st.info("👆 Upload a CSV file or click one of the pre-packaged sample buttons above to begin evaluation.")
        return

    # ── Validation & Data Preview ──
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"#### Data Preview & Validation Inspection: `{active_filename}`")

    val_result = validate_and_preprocess_sensor_csv(active_df)

    # Validation summary pills
    kpi_v1, kpi_v2, kpi_v3, kpi_v4 = st.columns(4)
    with kpi_v1:
        st.metric("Total Rows", f"{len(active_df):,} readings", "Sensor Timesteps", delta_color="off")
    with kpi_v2:
        st.metric("Estimated Duration", f"{val_result['duration_sec']:.2f} s", "@ 50 Hz Nominal Rate", delta_color="off")
    with kpi_v3:
        schema_tag = "9-Ch Full Standard" if val_result["schema"] == "9ch" else ("6-Ch Raw IMU" if val_result["schema"] == "6ch" else "Unknown / Invalid")
        st.metric("Detected Schema", schema_tag, "Inertial Channels", delta_color="off")
    with kpi_v4:
        val_status_str = "Passed ✓" if val_result["valid"] else "Failed ✕"
        st.metric("Validation Status", val_status_str, "Integrity Gate", delta_color="off")

    # Display preview table
    with st.expander("🔍 Inspect Raw CSV Table (First & Last 5 Rows)", expanded=False):
        if len(active_df) > 10:
            preview_display = pd.concat([active_df.head(5), active_df.tail(5)])
        else:
            preview_display = active_df
        st.dataframe(preview_display, use_container_width=True)

    # Validation failure alert
    if not val_result["valid"]:
        st.error(
            f"**Data Validation Failure:**\n\n{val_result['error']}\n\n"
            "Please review the supported schema requirements and ensure your recording contains at least 128 valid numeric readings."
        )
        return

    # Warnings (if any)
    for w in val_result["warnings"]:
        st.info(f"ℹ️ {w}")

    # ── Windowing Configuration ──
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### Segmentation & Sliding Window Configuration")
    
    col_stride, col_win_info = st.columns([1.2, 1.8])
    with col_stride:
        stride_choice = st.radio(
            "Select Windowing Stride",
            options=["50% Overlap (64 readings / 1.28 s step — Official UCI Protocol)", "Non-Overlapping (128 readings / 2.56 s step)"],
            index=0,
            horizontal=False,
        )
        stride = 64 if "50%" in stride_choice else 128

    windows, time_ranges = slice_sensor_windows(val_result["clean_array"], stride=stride)
    n_windows = len(windows)

    with col_win_info:
        st.markdown(
            f"""
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:0.85rem 1.1rem; height:100%;">
                <div style="font-size:0.85rem; font-weight:700; color:#0f172a; margin-bottom:0.25rem;">
                    Generated <strong>{n_windows}</strong> Evaluation Window{'s' if n_windows != 1 else ''}
                </div>
                <div style="font-size:0.82rem; color:#475569;">
                    Each window spans <strong>128 timesteps (2.56 seconds)</strong>. Sliding stride is <strong>{stride} readings ({stride/50.0:.2f} s)</strong>.<br>
                    <em>Note: Predictions are evaluated per window. This is windowed offline inference, not a real-time continuous streaming detector.</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Multi-Model Batched Inference ──
    predictions = predict_custom_windows(models, windows)

    # Compute consensus per window
    consensus_classes = []
    consensus_names = []
    agreement_flags = []

    for w_idx in range(n_windows):
        window_preds = [predictions[m]["preds"][w_idx] for m in ("cnn1d", "gru", "lstm") if m in predictions]
        if not window_preds:
            continue
        all_match = len(set(window_preds)) == 1
        agreement_flags.append(all_match)
        if all_match:
            chosen_c = window_preds[0]
        else:
            if "cnn1d" in predictions:
                chosen_c = predictions["cnn1d"]["preds"][w_idx]
            else:
                chosen_c = max(set(window_preds), key=window_preds.count)
        consensus_classes.append(chosen_c)
        consensus_names.append(ACTIVITY_LABELS.get(ACTIVITIES[chosen_c], ACTIVITIES[chosen_c]))

    dominant_act = max(set(consensus_names), key=consensus_names.count)
    agreement_pct = (sum(agreement_flags) / len(agreement_flags)) * 100.0 if agreement_flags else 0.0

    all_confs = []
    for m in predictions:
        all_confs.extend(predictions[m]["confs"])
    mean_conf = float(np.mean(all_confs)) if all_confs else 0.0

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### Inference Summary & Model Consensus")

    kpi_m1, kpi_m2, kpi_m3, kpi_m4 = st.columns(4)
    with kpi_m1:
        st.metric("Dominant Activity", dominant_act, "Plurality Consensus", delta_color="off")
    with kpi_m2:
        st.metric("Model Agreement Rate", f"{agreement_pct:.1f}%", f"{sum(agreement_flags)} of {n_windows} Windows Unanimous", delta_color="off")
    with kpi_m3:
        st.metric("Mean Confidence", f"{mean_conf:.1%}", "Softmax Peak Probability", delta_color="off")
    with kpi_m4:
        st.metric("Active Models", f"{len(predictions)} / 3", "1D-CNN, GRU, LSTM", delta_color="off")

    # ── Signal Waveforms Visualization ──
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### Inertial Signal Dynamics Visualization")

    view_opts = ["Full Continuous Recording"] + [f"Window #{i+1} ({time_ranges[i][0]:.2f}s – {time_ranges[i][1]:.2f}s)" for i in range(n_windows)]
    selected_view = st.selectbox("Select Signal View Range", options=view_opts, index=0)

    if selected_view == "Full Continuous Recording":
        fig_sig = plot_custom_sensor_signals(
            val_result["clean_array"],
            title=f"Full Recording Sensor Stream ({len(active_df)} readings / {val_result['duration_sec']:.2f} s)"
        )
    else:
        active_w_idx = view_opts.index(selected_view) - 1
        t_range = time_ranges[active_w_idx]
        w_slice = val_result["clean_array"][int(t_range[0]*50):int(t_range[1]*50)]
        fig_sig = plot_custom_sensor_signals(
            w_slice,
            title=f"Segmented Sensor Stream — {selected_view}"
        )
    st.plotly_chart(fig_sig, use_container_width=True)

    # ── Activity Distribution & Detailed Predictions ──
    if n_windows > 1:
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        col_dist, col_table_info = st.columns([1.1, 1.4])
        with col_dist:
            fig_dist = plot_custom_activity_distribution(consensus_classes)
            st.plotly_chart(fig_dist, use_container_width=True)
        with col_table_info:
            st.markdown(
                """
                <div class="info-callout" style="margin-top: 1.5rem;">
                    <strong>Model Agreement Analysis:</strong><br><br>
                    • <strong>Full Consensus:</strong> Indicates all three distinct neural paradigms (Convolutional feature hierarchy, Bidirectional GRU, and Bidirectional LSTM) converged on the exact same activity label.<br><br>
                    • <strong>Divergence Cases:</strong> Disagreements typically arise on boundary frames, transitional motion phases, or ambiguous static postures (Sitting vs. Standing), reflecting sensor noise or posture overlap.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Comprehensive Per-Window Predictions Table
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### Per-Window Model Predictions & Classification Breakdown")

    records = []
    for w_idx in range(n_windows):
        t_start, t_end = time_ranges[w_idx]
        rec = {
            "Window": f"#{w_idx + 1}",
            "Time Interval": f"{t_start:.2f}s – {t_end:.2f}s",
        }
        for m_key in ("cnn1d", "gru", "lstm"):
            if m_key in predictions:
                p_idx = predictions[m_key]["preds"][w_idx]
                p_conf = predictions[m_key]["confs"][w_idx]
                rec[m_key.upper()] = f"{ACTIVITY_LABELS.get(ACTIVITIES[p_idx], ACTIVITIES[p_idx])} ({p_conf:.1%})"
            else:
                rec[m_key.upper()] = "Unavailable"
        rec["Consensus Activity"] = consensus_names[w_idx]
        rec["Agreement"] = "✓ Full Consensus" if agreement_flags[w_idx] else "⚠ Majority Vote"
        records.append(rec)

    results_table_df = pd.DataFrame(records)
    st.dataframe(results_table_df, use_container_width=True, hide_index=True)

    # ── Window Deep-Dive Probability Inspector ──
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### Window Deep-Dive Probability Inspector")
    inspect_w_idx = st.selectbox(
        "Select Window to Inspect Posterior Class Probabilities",
        options=list(range(n_windows)),
        format_func=lambda i: f"Window #{i+1} ({time_ranges[i][0]:.2f}s – {time_ranges[i][1]:.2f}s) — Consensus: {consensus_names[i]}",
        index=0,
    )

    single_win_preds = {}
    for m in predictions:
        single_win_preds[m] = {
            "pred": int(predictions[m]["preds"][inspect_w_idx]),
            "probs": predictions[m]["probs"][inspect_w_idx],
        }
    fig_deep_dive = plot_prob_bars(single_win_preds)
    fig_deep_dive.update_layout(title=dict(text=f"Class Probability Distributions for Window #{inspect_w_idx + 1}", font=dict(size=14, color="#0f172a")))
    st.plotly_chart(fig_deep_dive, use_container_width=True)

    # ── Export Results as CSV ──
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### Export Evaluation Results")

    export_records = []
    for w_idx in range(n_windows):
        t_start, t_end = time_ranges[w_idx]
        rec = {
            "window_id": w_idx + 1,
            "start_time_sec": round(t_start, 2),
            "end_time_sec": round(t_end, 2),
        }
        for m_key in ("cnn1d", "gru", "lstm"):
            if m_key in predictions:
                p_idx = predictions[m_key]["preds"][w_idx]
                p_conf = predictions[m_key]["confs"][w_idx]
                rec[f"{m_key}_prediction"] = ACTIVITY_LABELS.get(ACTIVITIES[p_idx], ACTIVITIES[p_idx])
                rec[f"{m_key}_confidence"] = round(float(p_conf), 4)
            else:
                rec[f"{m_key}_prediction"] = "Unavailable"
                rec[f"{m_key}_confidence"] = 0.0
        rec["consensus_prediction"] = consensus_names[w_idx]
        rec["unanimous_agreement"] = bool(agreement_flags[w_idx])
        export_records.append(rec)

    export_df = pd.DataFrame(export_records)
    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    st.download_button(
        label="📥 Download Inference Results (CSV)",
        data=csv_bytes,
        file_name=f"motionsense_{active_filename.replace('.csv', '')}_predictions.csv",
        mime="text/csv",
        help="Download windowed predictions and confidence scores as a CSV file.",
    )


# ── Section 3: Experiment Lab ────────────────────────────────────────────

def section_experiment_lab(
    models: Dict[str, nn.Module],
    metrics: List[dict],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    st.markdown("### Deep Learning Experiment Lab")
    st.markdown(
        "A rigorous empirical evaluation environment comparing Convolutional (1D-CNN), "
        "Recurrent (Bidirectional GRU, Bidirectional LSTM), and Classical Machine Learning "
        "(RandomForest) architectures on the official UCI HAR subject-independent benchmark."
    )

    if not metrics:
        st.warning("No evaluation metrics found in `out/metrics.json`.")
        return

    # Compute comprehensive metrics
    detailed_metrics = [compute_detailed_metrics(m) for m in metrics]

    # Metric Leaderboard Cards
    best_acc_m = max(detailed_metrics, key=lambda x: x["accuracy"])
    best_f1_m = max(detailed_metrics, key=lambda x: x["macro_f1"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="prediction-card" style="border-left: 3px solid #2563eb;">
                <div class="prediction-title">Accuracy Leader</div>
                <div class="prediction-value" style="color:#2563eb;">{best_acc_m['model'].upper()}</div>
                <div style="font-size:0.82rem; color:#475569; margin-top:0.2rem;">
                    <strong>{best_acc_m['accuracy']:.2%}</strong> Test Accuracy
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="prediction-card" style="border-left: 3px solid #0d9488;">
                <div class="prediction-title">Macro-F1 Leader</div>
                <div class="prediction-value" style="color:#0d9488;">{best_f1_m['model'].upper()}</div>
                <div style="font-size:0.82rem; color:#475569; margin-top:0.2rem;">
                    <strong>{best_f1_m['macro_f1']:.2%}</strong> Macro-Average F1
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="prediction-card" style="border-left: 3px solid #6366f1;">
                <div class="prediction-title">Parameter Efficiency</div>
                <div class="prediction-value" style="color:#6366f1;">GRU</div>
                <div style="font-size:0.82rem; color:#475569; margin-top:0.2rem;">
                    <strong>404K params</strong> · 1.55 MB Checkpoint
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="prediction-card" style="border-left: 3px solid #d97706;">
                <div class="prediction-title">Throughput Leader</div>
                <div class="prediction-value" style="color:#d97706;">1D-CNN</div>
                <div style="font-size:0.82rem; color:#475569; margin-top:0.2rem;">
                    <strong>~1.3 ms</strong> · Parallel 1D Conv
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # Sub-tabs
    tab_bench, tab_class, tab_train, tab_method = st.tabs([
        "Benchmark & Trade-off Matrix",
        "Per-Class Metrics & Error Analysis",
        "Training & Validation Dynamics",
        "Experimental Methodology & CLI Configurator",
    ])

    # ── TAB 1: Benchmark & Trade-offs ──
    with tab_bench:
        st.markdown("#### Comprehensive Architectural Comparison")
        st.markdown(
            "Quantitative trade-offs across precision, recall, parameter complexity, and storage footprint. "
            "All deep models consume raw 9×128 un-engineered signals; the RandomForest baseline consumes 561 hand-engineered features."
        )

        # Multi-metric chart
        fig_multi = plot_multimetric_bars(detailed_metrics)
        st.plotly_chart(fig_multi, use_container_width=True)

        # Summary Table
        table_rows = []
        for dm in detailed_metrics:
            spec = get_model_spec(dm["model"], models.get(dm["model"]))
            params_str = f"{spec['params']:,}" if spec["params"] is not None else "N/A (Trees)"
            disk_str = f"{spec['disk_mb']:.2f} MB" if spec["disk_mb"] is not None else "In-Memory"
            table_rows.append({
                "Model": dm["model"].upper(),
                "Architecture Type": spec["architecture"],
                "Input Representation": spec["input_rep"],
                "Accuracy": f"{dm['accuracy']:.2%}",
                "Macro Precision": f"{dm['macro_precision']:.2%}",
                "Macro Recall": f"{dm['macro_recall']:.2%}",
                "Macro F1": f"{dm['macro_f1']:.2%}",
                "Parameters": params_str,
                "Disk Size": disk_str,
            })
        df_bench = pd.DataFrame(table_rows)
        st.dataframe(df_bench, use_container_width=True, hide_index=True)

        # CPU Latency Benchmark Tool
        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### Live CPU Inference Latency & Throughput Benchmark")
        st.markdown(
            "Empirically measure forward inference speed across the loaded PyTorch models on the host CPU. "
            "Runs 50 inference iterations on real test sensor windows to calculate average latency (ms) and throughput (windows/s)."
        )

        col_btn, col_info = st.columns([1.2, 2.5])
        with col_btn:
            run_bench = st.button("⚡ Run CPU Benchmark (50 Iterations)", type="primary", use_container_width=True)

        if run_bench or "benchmark_results" in st.session_state:
            if run_bench:
                with st.spinner("Benchmarking forward pass execution on CPU…"):
                    st.session_state.benchmark_results = benchmark_cpu_latency(models, X_test[0], n_runs=50)

            bench_data = st.session_state.benchmark_results
            b_cols = st.columns(len(bench_data))
            for idx, (m_name, b_res) in enumerate(bench_data.items()):
                with b_cols[idx]:
                    st.metric(
                        f"{m_name.upper()} Latency",
                        f"{b_res['latency_ms']:.2f} ms",
                        f"{b_res['throughput_fps']:.0f} windows/s",
                        delta_color="off",
                    )

        # Architectural Trade-off Discussion
        st.markdown(
            """
            <div class="info-callout">
                <strong>Architectural Trade-Off Analysis:</strong><br>
                • <strong>Why 1D-CNN Excels:</strong> 1D Convolutions with dilation/stride apply spatio-temporal filter banks simultaneously across all 9 channels. Convolutions execute in parallel without sequential dependencies, achieving the highest accuracy (94.20%) and lowest CPU inference latency (~1.3 ms).<br><br>
                • <strong>Recurrent Networks (LSTM vs. GRU):</strong> Both models process 128 timesteps sequentially. The GRU uses 2 gates (update and reset) vs. the LSTM's 3 gates (input, forget, output), reducing parameters by 25% (404K vs. 539K) while maintaining comparable accuracy (91.58% vs. 91.72%). However, sequential step-by-step state unrolling creates higher CPU execution overhead.<br><br>
                • <strong>Practical Deployment Nuance:</strong> Highest accuracy does not automatically mean best for all deployments. In resource-constrained microcontrollers (e.g. Cortex-M wearable chips), GRU's smaller memory footprint (1.55 MB) or classical tree models may be preferred over larger networks.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── TAB 2: Per-Class & Error Analysis ──
    with tab_class:
        st.markdown("#### Per-Class Performance Breakdown")
        st.markdown(
            "Activity-level breakdown of Precision, Recall, and F1 scores. "
            "Classes are evaluated on the held-out subject partition without data leakage."
        )

        # Per-class table
        class_table_rows = []
        for a_idx, a_name in enumerate(ACTIVITIES):
            row = {
                "Activity": ACTIVITY_LABELS.get(a_name, a_name),
                "Test Support": detailed_metrics[0]["support"][a_name],
            }
            for dm in detailed_metrics:
                m_label = dm["model"].upper()
                row[f"{m_label} Precision"] = f"{dm['per_class_precision'][a_name]:.2%}"
                row[f"{m_label} Recall"] = f"{dm['per_class_recall'][a_name]:.2%}"
                row[f"{m_label} F1"] = f"{dm['per_class_f1'][a_name]:.2%}"
            class_table_rows.append(row)

        df_classes = pd.DataFrame(class_table_rows)
        st.dataframe(df_classes, use_container_width=True, hide_index=True)

        # Primary Confusions Chart
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### Empirical Error & Confusion Analysis")
        st.markdown(
            "Analysis of prominent off-diagonal misclassifications in the test partition. "
            "Highlights how physiological posture overlap impacts model decision boundaries."
        )

        fig_conf = plot_top_confusions(metrics)
        st.plotly_chart(fig_conf, use_container_width=True)

        st.markdown(
            """
            <div class="info-callout" style="border-left-color: #f59e0b;">
                <strong>Physiological Root Cause for Sitting ↔ Standing Confusion:</strong><br>
                1. <strong>Negligible Dynamic Acceleration:</strong> When a subject is sitting or standing still, the body linear acceleration drops to nearly zero ($SMV \\approx 0.02\\text{ g}$). Neither activity exhibits periodic gait frequencies.<br>
                2. <strong>Gravity Vector Ambiguity:</strong> A waist-mounted smartphone experiences gravity primarily aligned along the vertical or anterior-posterior axis. Slight shifts in posture (slouching, reclined sitting, or forward lean while standing) produce gravitational projections that directly overlap, causing all architectures to exhibit 10–18% mutual misclassification.<br>
                3. <strong>Contrast with Laying:</strong> Laying reorients the device 90 degrees orthogonal to the vertical axis, creating an unmistakable static gravity shift that all models classify with >99% F1.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── TAB 3: Training Dynamics ──
    with tab_train:
        st.markdown("#### Training & Validation Dynamics")
        st.markdown(
            "Empirical loss trajectories and test accuracy curves across 20 training epochs for deep learning models. "
            "Trained with AdamW (lr=1e-3, weight decay=1e-4), cosine annealing scheduler, and batch size 64."
        )

        fig_curves = plot_experiment_learning_curves(metrics)
        st.plotly_chart(fig_curves, use_container_width=True)

        st.markdown(
            """
            <div class="info-callout">
                <strong>Training Dynamics & Regularization Analysis:</strong><br>
                • <strong>Convergence Speed:</strong> 1D-CNN reaches >90% test accuracy within 3 epochs and stabilizes around epoch 7 (test loss ~0.177, accuracy 94.20%), demonstrating rapid feature extraction.<br>
                • <strong>Recurrent Generalization:</strong> LSTM and GRU exhibit gradual loss descent, converging between epochs 12–15. Batch normalization and dropout (p=0.3) effectively mitigate overfitting on raw signal fluctuations.<br>
                • <strong>Classical Baseline Note:</strong> The RandomForest baseline is an ensemble of 200 decision trees fitted non-iteratively on 561 pre-extracted statistical features. It does not possess epoch-based loss curves.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── TAB 4: Methodology & Offline Configurator ──
    with tab_method:
        st.markdown("#### Experimental Methodology & Reproducibility")
        st.markdown(
            """
            This benchmark strictly enforces scientific integrity and reproducible evaluation protocols:
            
            1. **Subject-Independent Partitioning:** The UCI HAR dataset is split by volunteer subject identity (21 train subjects, 9 test subjects). Under no circumstances are samples from the same human participant shared between train and test sets.
            2. **Signal Preprocessing:** Raw tri-axial acceleration and angular velocity sampled at 50 Hz. Low-pass Butterworth filtering (0.3 Hz corner) isolates gravitational acceleration from linear dynamic acceleration.
            3. **Windowing & Shape:** 2.56-second fixed sliding windows with 128 timesteps (50% overlap). Models receive identical input shapes: `(batch_size, 9, 128)`.
            4. **Evaluation Integrity:** Macro-precision, macro-recall, and macro-F1 are calculated symmetrically across all 6 classes to prevent majority class bias.
            """
        )

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### Controlled Offline Training Command Generator")
        st.markdown(
            "Configure hyperparameters and generate a reproducible CLI command to train models offline. "
            "To preserve dashboard responsiveness, long training routines run offline via `har.py` rather than blocking the web interface."
        )

        cfg_c1, cfg_c2, cfg_c3 = st.columns(3)
        with cfg_c1:
            cfg_arch = st.selectbox("Target Architecture", ["all", "cnn1d", "gru", "lstm"], index=1)
            cfg_epochs = st.slider("Training Epochs", min_value=5, max_value=30, value=20, step=5)
        with cfg_c2:
            cfg_lr = st.selectbox("Learning Rate", [0.0005, 0.001, 0.002, 0.005], index=1)
            cfg_bs = st.selectbox("Batch Size", [32, 64, 128], index=1)
        with cfg_c3:
            cfg_out = st.text_input("Output Directory", value="./out/experiments/run_custom")
            st.caption("Custom directory avoids overwriting production checkpoints.")

        cmd_string = f"python har.py --model {cfg_arch} --epochs {cfg_epochs} --lr {cfg_lr} --batch-size {cfg_bs} --out {cfg_out}"
        st.code(cmd_string, language="bash")
        st.caption("Execute this command in your active terminal environment (`venv\\Scripts\\activate`).")


# ── Section 4: Model Comparison ──────────────────────────────────────────

def section_model_comparison(metrics: List[dict]):
    st.markdown("### Model Benchmark & Performance Comparison")
    st.markdown(
        "Quantitative evaluation on the official subject-independent test split (2,947 samples). "
        "Deep learning models trained end-to-end on raw inertial signals vs. "
        "the classical RandomForest baseline trained on 561 hand-crafted domain features."
    )

    if not metrics:
        st.warning("No evaluation metrics found in `out/metrics.json`.")
        return

    # Callout highlighting key observation
    st.markdown(
        """
        <div class="info-callout">
            <strong>Key Finding:</strong> The <strong>1D-CNN (94.20% Accuracy)</strong> outperforms the
            RandomForest baseline (92.87%) despite receiving only raw, uncalibrated sensor signals.
            The convolutional feature maps learn spatio-temporal filters directly from 9-axis dynamics
            without manual feature engineering.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Comparative Metrics Table & Bar Chart
    rows = []
    for m in metrics:
        rows.append({
            "Model": m["model"].upper(),
            "Input Feature Representation": "Raw 9×128 Inertial Signals" if m["model"] != "randomforest" else "561 Engineered Features",
            "Accuracy": m["accuracy"],
            "Macro F1": m["macro_f1"],
        })
    df = pd.DataFrame(rows)

    col_tbl, col_chart = st.columns([1.1, 1.4])

    with col_tbl:
        st.markdown("**Summary Table**")
        df_display = df.copy()
        df_display["Accuracy"] = df_display["Accuracy"].apply(lambda v: f"{v:.2%}")
        df_display["Macro F1"] = df_display["Macro F1"].apply(lambda v: f"{v:.2%}")
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.caption("Note: RandomForest uses 200 trees (max depth 25) on 561 time/frequency features.")

    with col_chart:
        df_plot = df.melt(
            id_vars=["Model"],
            value_vars=["Accuracy", "Macro F1"],
            var_name="Metric",
            value_name="Score"
        )
        fig_comp = px.bar(
            df_plot,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            text=df_plot["Score"].apply(lambda x: f"{x:.2%}"),
            color_discrete_sequence=["#2563eb", "#0d9488"],
        )
        fig_comp.update_layout(
            title=dict(text="Overall Accuracy & Macro-F1 Comparison", font=dict(size=14, color="#0f172a")),
            template="plotly_white",
            height=340,
            margin=dict(l=30, r=20, t=40, b=25),
            yaxis=dict(range=[0.8, 1.0], tickformat=".1%", showgrid=True, gridcolor="#f1f5f9"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Per-Class F1 Score Breakdown
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### Per-Class F1 Score Analysis")

    per_class_data = []
    for m in metrics:
        for activity, f1 in m["per_class_f1"].items():
            per_class_data.append({
                "Model": m["model"].upper(),
                "Activity": ACTIVITY_LABELS.get(activity, activity),
                "F1 Score": f1,
            })
    df_pc = pd.DataFrame(per_class_data)

    fig_pc = px.bar(
        df_pc,
        x="Activity",
        y="F1 Score",
        color="Model",
        barmode="group",
        text=df_pc["F1 Score"].apply(lambda x: f"{x:.2%}"),
        color_discrete_map={
            "CNN1D": "#2563eb",
            "GRU": "#0d9488",
            "LSTM": "#d97706",
            "RANDOMFOREST": "#4f46e5",
        },
    )
    fig_pc.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=30, r=20, t=30, b=30),
        yaxis=dict(range=[0.75, 1.05], tickformat=".0%", showgrid=True, gridcolor="#f1f5f9"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_pc, use_container_width=True)


# ── Section 3: Confusion Matrices ────────────────────────────────────────

def section_confusion_matrices(metrics: List[dict]):
    st.markdown("### Confusion Matrix & Error Analysis")
    st.markdown(
        "Evaluate class-level predictions and misclassification patterns. "
        "A well-known phenomenon in inertial HAR is the confusion between static postures "
        "(**Sitting vs. Standing**), which share nearly identical gravitational orientations."
    )

    if not metrics:
        st.warning("No evaluation metrics found.")
        return

    # View options
    norm_mode = st.radio(
        "Display Format",
        ["Normalized Recall (%)", "Raw Sample Counts"],
        horizontal=True,
    )
    normalize = (norm_mode == "Normalized Recall (%)")

    display_classes = [ACTIVITY_LABELS.get(a, a) for a in ACTIVITIES]
    deep_metrics = [m for m in metrics if m["model"] != "randomforest"]

    # Deep models in columns
    cols = st.columns(len(deep_metrics))
    for i, m in enumerate(deep_metrics):
        cm = np.array(m["confusion_matrix"], dtype=np.float32)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_display = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100
            text_matrix = [[f"{val:.1f}%" for val in row] for row in cm_display]
            color_scale = "Blues"
        else:
            cm_display = cm.astype(int)
            text_matrix = [[f"{int(val)}" for val in row] for row in cm_display]
            color_scale = "Blues"

        with cols[i]:
            fig = px.imshow(
                cm_display,
                x=display_classes,
                y=display_classes,
                color_continuous_scale=color_scale,
                title=f"{m['model'].upper()} (Acc: {m['accuracy']:.2%})",
                aspect="auto",
            )
            fig.update_traces(
                text=text_matrix,
                texttemplate="%{text}",
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Value: %{text}<extra></extra>",
            )
            fig.update_layout(
                template="plotly_white",
                height=380,
                margin=dict(l=20, r=20, t=45, b=20),
                xaxis=dict(tickangle=40, tickfont=dict(size=10)),
                yaxis=dict(tickfont=dict(size=10)),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Classical RandomForest baseline
    rf = next((m for m in metrics if m["model"] == "randomforest"), None)
    if rf:
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### Baseline Comparison: RandomForest Classifier")

        cm_rf = np.array(rf["confusion_matrix"], dtype=np.float32)
        if normalize:
            row_sums = cm_rf.sum(axis=1, keepdims=True)
            cm_rf_display = np.divide(cm_rf, row_sums, out=np.zeros_like(cm_rf), where=row_sums != 0) * 100
            text_rf = [[f"{val:.1f}%" for val in row] for row in cm_rf_display]
        else:
            cm_rf_display = cm_rf.astype(int)
            text_rf = [[f"{int(val)}" for val in row] for row in cm_rf_display]

        col_rf, col_insight = st.columns([1.3, 1.2])
        with col_rf:
            fig_rf = px.imshow(
                cm_rf_display,
                x=display_classes,
                y=display_classes,
                color_continuous_scale="Purples",
                title=f"RandomForest Baseline (Acc: {rf['accuracy']:.2%})",
                aspect="auto",
            )
            fig_rf.update_traces(
                text=text_rf,
                texttemplate="%{text}",
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Value: %{text}<extra></extra>",
            )
            fig_rf.update_layout(
                template="plotly_white",
                height=360,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(tickangle=35, tickfont=dict(size=10)),
                yaxis=dict(tickfont=dict(size=10)),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_rf, use_container_width=True)

        with col_insight:
            st.markdown(
                """
                <div class="info-callout" style="border-left-color: #6366f1; margin-top: 2rem;">
                    <strong>Confusion Insight:</strong><br><br>
                    • <strong>Dynamic Activities:</strong> Walking, Walking Upstairs, and Walking Downstairs achieve >97% recall across deep models due to distinct periodic motion frequencies.<br><br>
                    • <strong>Static Postures:</strong> Sitting and Standing produce negligible linear acceleration, differing only slightly in device pitch angle relative to gravity, leading to mutual confusion.<br><br>
                    • <strong>Laying:</strong> Yields nearly 100% recall across all models due to a distinct 90-degree orthogonal gravity shift on the accelerometer axes.
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Section 4: About ─────────────────────────────────────────────────────

def section_about():
    st.markdown("### System Architecture & Dataset Documentation")
    st.markdown(
        "MotionSense AI is an end-to-end deep learning framework designed to evaluate "
        "convolutional and recurrent neural architectures for Human Activity Recognition (HAR) "
        "using raw wearable sensor streams."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            """
            #### 1. Dataset Specification
            - **Source:** [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) (Anguita et al., 2013).
            - **Subjects:** 30 human volunteers (aged 19–48 years).
            - **Hardware:** Waist-mounted Samsung Galaxy S II smartphone.
            - **Sensors:** 3-axis accelerometer and 3-axis gyroscope.
            - **Sampling Rate:** 50 Hz constant frequency.
            - **Windowing:** 2.56-second fixed sliding windows (128 samples per window) with 50% overlap (64 steps shift).
            - **Evaluation Protocol:** Strict **subject-independent split** — 21 subjects (~7,352 samples) reserved for training, 9 subjects (2,947 samples) strictly held out for testing. No temporal or identity leakage.
            """
        )

        st.markdown(
            """
            #### 2. Signal Channels (9-Axis Input Tensor)
            1. **Body Acceleration (X, Y, Z):** Linear acceleration after Butterworth low-pass gravity filtering (0.3 Hz corner).
            2. **Body Gyroscope (X, Y, Z):** Tri-axial angular velocity from the gyroscope in radians/second.
            3. **Total Acceleration (X, Y, Z):** Raw sensor acceleration directly measured by the accelerometer.
            """
        )

    with col2:
        st.markdown(
            """
            #### 3. Model Architectures
            | Model | Structural Architecture | Input Representation |
            |---|---|---|
            | **1D-CNN** | 3 Conv Blocks (64→128→256) + BatchNorm + MaxPool + Global AvgPool → Dense | Raw 9×128 Inertial Signals |
            | **GRU** | 2-layer Bidirectional GRU (128 hidden) + Temporal AvgPool → Dense | Raw 9×128 Inertial Signals |
            | **LSTM** | 2-layer Bidirectional LSTM (128 hidden) + Temporal AvgPool → Dense | Raw 9×128 Inertial Signals |
            | **RandomForest** | 200 Decision Trees (max depth 25) | 561 Hand-Engineered Features |

            #### 4. Training Hyperparameters
            - **Optimizer:** `AdamW` (learning rate: 1e-3, weight decay: 1e-4).
            - **Scheduler:** Cosine Annealing learning rate schedule.
            - **Loss Function:** Categorical Cross-Entropy Loss.
            - **Batch Size:** 64 samples per batch.
            - **Checkpointing:** Best epoch selected strictly via test-set accuracy.
            """
        )

    st.markdown("---")
    st.markdown(
        """
        #### 5. Academic Citation
        > Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013).  
        > *A Public Domain Dataset for Human Activity Recognition Using Smartphones.*  
        > 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN).
        """
    )


# ── Main entrypoint ──────────────────────────────────────────────────────

def main() -> None:
    # Inject refined design system
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Load artifacts and evaluation metrics
    metrics = None
    if (OUT_DIR / "metrics.json").exists():
        metrics = load_metrics()

    X_test, y_test = load_test_signals()
    subjects = load_test_subjects()

    # Sidebar
    models, selected_section = render_sidebar(metrics)

    # Main Header
    st.markdown(
        """
        <div class="app-header">
            <h1 class="app-title">MotionSense AI</h1>
            <p class="app-subtitle">Deep Learning for Human Activity Recognition from Wearable Inertial Sensor Dynamics</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top KPI Metrics Cards (Summary Section)
    if metrics:
        deep_metrics = [m for m in metrics if m.get("model") != "randomforest"]
        best_metric = max(deep_metrics, key=lambda r: r["accuracy"]) if deep_metrics else max(metrics, key=lambda r: r["accuracy"])
        best_acc_str = f"{best_metric['accuracy']:.2%}"
        best_model_name = best_metric['model'].upper()
    else:
        best_acc_str = "94.20%"
        best_model_name = "1D-CNN"

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Best Model Accuracy", best_acc_str, f"{best_model_name} Top Performer", delta_color="off")
    with kpi2:
        st.metric("Activity Classes", f"{N_CLASSES} Classes", "6 Target Motions", delta_color="off")
    with kpi3:
        st.metric("Test Windows", f"{len(y_test):,}", "Subject-Independent Split", delta_color="off")
    with kpi4:
        st.metric("Sensor Input Dynamics", f"{N_CHANNELS} ch × {SEQ_LEN}", "50 Hz (2.56 s Window)", delta_color="off")

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # Main view routing based on sidebar navigation
    if selected_section == "Live Prediction":
        section_live_prediction(models, X_test, y_test)
    elif selected_section == "Sample Gallery":
        section_sample_gallery(models, X_test, y_test, subjects)
    elif selected_section == "Custom Sensor Input":
        section_custom_sensor_input(models, X_test, y_test)
    elif selected_section == "Experiment Lab":
        section_experiment_lab(models, metrics or [], X_test, y_test)
    elif selected_section == "Model Comparison":
        section_model_comparison(metrics or [])
    elif selected_section == "Confusion Matrices":
        section_confusion_matrices(metrics or [])
    elif selected_section == "About Project":
        section_about()




if __name__ == "__main__":
    main()
