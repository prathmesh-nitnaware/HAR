"""Inference engine and signal processing pipeline for MotionSense AI.

Extracts and unifies model loading, validation, Butterworth filtering,
windowing, and multi-model inference for CNN1D, GRU, and LSTM architectures.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from har import (
    ACTIVITIES,
    CNN1DModel,
    DeepConvLSTMModel,
    GRUModel,
    LSTMModel,
    MODEL_REGISTRY,
    N_CLASSES,
    N_CHANNELS,
    SEQ_LEN,
    SIGNAL_NAMES,
)

logger = logging.getLogger("motionsense.inference")

OUT_DIR = PROJECT_ROOT / "out"
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"

ACTIVITY_LABELS: Dict[str, str] = {
    "walking": "Walking",
    "walking-upstairs": "Walking Upstairs",
    "walking-downstairs": "Walking Downstairs",
    "sitting": "Sitting",
    "standing": "Standing",
    "laying": "Laying",
}

SIGNAL_DISPLAY_NAMES: Dict[str, str] = {
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

# Safe registry of verified sample files — prevents arbitrary filesystem traversal
SAMPLE_REGISTRY: Dict[str, dict] = {
    "walking_128": {
        "id": "walking_128",
        "filename": "sample_walking_128.csv",
        "name": "Walking (Single Window)",
        "description": "2.56-second single-window recording of dynamic walking gait at 50 Hz.",
        "ground_truth": "walking",
        "ground_truth_label": "Walking",
        "expected_rows": 128,
        "channels": 9,
        "duration_sec": 2.56,
    },
    "sitting_128": {
        "id": "sitting_128",
        "filename": "sample_sitting_128.csv",
        "name": "Sitting (Single Window)",
        "description": "2.56-second single-window recording of static seated posture at 50 Hz.",
        "ground_truth": "sitting",
        "ground_truth_label": "Sitting",
        "expected_rows": 128,
        "channels": 9,
        "duration_sec": 2.56,
    },
    "stairs_384": {
        "id": "stairs_384",
        "filename": "sample_stairs_multiwindow_384.csv",
        "name": "Walking Upstairs (Multi-Window)",
        "description": "7.68-second multi-window recording of stair climbing at 50 Hz.",
        "ground_truth": "walking-upstairs",
        "ground_truth_label": "Walking Upstairs",
        "expected_rows": 384,
        "channels": 9,
        "duration_sec": 7.68,
    },
}


class ModelManager:
    """Singleton manager for loading, caching, and serving PyTorch HAR models."""

    _instance: Optional[ModelManager] = None

    def __init__(self, out_dir: Path = OUT_DIR) -> None:
        self.out_dir = Path(out_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, nn.Module] = {}
        self.load_status: Dict[str, bool] = {"convlstm": False, "cnn1d": False, "gru": False, "lstm": False}
        self.load_errors: Dict[str, str] = {}
        self.load_all()

    @classmethod
    def get_instance(cls, out_dir: Path = OUT_DIR) -> ModelManager:
        if cls._instance is None:
            cls._instance = cls(out_dir=out_dir)
        return cls._instance

    def load_all(self) -> None:
        """Load all deep learning model checkpoints once at initialization."""
        for m_name, model_cls in [("convlstm", DeepConvLSTMModel), ("cnn1d", CNN1DModel), ("gru", GRUModel), ("lstm", LSTMModel)]:
            checkpoint_path = self.out_dir / f"{m_name}_best.pt"
            if not checkpoint_path.exists():
                self.load_status[m_name] = False
                self.load_errors[m_name] = f"Checkpoint not found at {checkpoint_path}"
                logger.warning("Model checkpoint not found: %s", checkpoint_path)
                continue

            try:
                model = model_cls()
                state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.models[m_name] = model
                self.load_status[m_name] = True
                logger.info("Loaded %s model checkpoint from %s on %s", m_name.upper(), checkpoint_path, self.device)
            except Exception as exc:
                self.load_status[m_name] = False
                self.load_errors[m_name] = str(exc)
                logger.error("Failed to load checkpoint for %s: %s", m_name, exc)

    def get_model(self, name: str) -> Optional[nn.Module]:
        return self.models.get(name.lower())

    def get_loaded_models(self) -> Dict[str, nn.Module]:
        return self.models

    def get_model_specs(self) -> List[dict]:
        """Return structural specifications, parameter counts, and disk sizes."""
        specs = []
        descriptions = {
            "convlstm": ("Deep ConvLSTM + Attention", "3 Conv Blocks + 2-Layer BiLSTM (96 hidden) + Temporal Self-Attention + Physics Postural Tilt Fusion", "Raw 9×128 Inertial Signals"),
            "cnn1d": ("Deep ConvNet", "3 Conv Blocks (64→128→256) + BatchNorm + MaxPool + Global AvgPool → Dense", "Raw 9×128 Inertial Signals"),
            "gru": ("Bidirectional Recurrent", "2-Layer BiGRU (128 hidden) + Temporal AvgPool → Dense", "Raw 9×128 Inertial Signals"),
            "lstm": ("Bidirectional Recurrent", "2-Layer BiLSTM (128 hidden) + Temporal AvgPool → Dense", "Raw 9×128 Inertial Signals"),
            "randomforest": ("Classical Ensemble", "200 Decision Trees (max depth 25)", "561 Engineered Domain Features"),
        }

        # Check metrics for benchmark scores
        metrics_dict = {}
        try:
            metrics_file = self.out_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    data = json.load(f)
                    for item in data:
                        metrics_dict[item["model"].lower()] = item
        except Exception as e:
            logger.warning("Could not read metrics.json: %s", e)

        for m_key in ("convlstm", "cnn1d", "gru", "lstm", "randomforest"):
            m_type, arch_desc, input_desc = descriptions[m_key]
            weights_path = self.out_dir / f"{m_key}_best.pt"
            disk_mb = round(weights_path.stat().st_size / (1024 * 1024), 2) if weights_path.exists() else None

            model_obj = self.models.get(m_key)
            if model_obj is not None:
                params = sum(p.numel() for p in model_obj.parameters())
            elif m_key == "convlstm":
                params = 517735
            elif m_key == "cnn1d":
                params = 709830
            elif m_key == "gru":
                params = 404742
            elif m_key == "lstm":
                params = 539142
            else:
                params = None

            is_ready = self.load_status.get(m_key, False) if m_key != "randomforest" else True
            m_metric = metrics_dict.get(m_key, {})

            display_name = {
                "convlstm": "DeepConvLSTM (Advanced)",
                "cnn1d": "1D-CNN",
                "gru": "GRU",
                "lstm": "LSTM",
                "randomforest": "RandomForest Baseline",
            }[m_key]

            specs.append({
                "id": m_key,
                "name": display_name,
                "type": m_type,
                "architecture": arch_desc,
                "input_representation": input_desc,
                "parameters": params,
                "disk_size_mb": disk_mb,
                "status": "ready" if is_ready else "unavailable",
                "test_accuracy": m_metric.get("accuracy"),
                "macro_f1": m_metric.get("macro_f1"),
            })
        return specs


# ── Signal Validation & Preprocessing ────────────────────────────────────────

def validate_and_preprocess_sensor_csv(df: pd.DataFrame) -> dict:
    """Validate uploaded CSV against expected 9-ch or 6-ch schema and apply standard preprocessing."""
    result = {
        "valid": False,
        "error": None,
        "warnings": [],
        "schema": None,
        "n_rows": len(df),
        "duration_sec": round(len(df) / 50.0, 2),
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

    # Preprocess into standard 9-channel array (samples, 9)
    if target_schema == "9ch":
        clean_arr = df_subset[cols_9ch].values.astype(np.float32)
    else:  # 6ch
        total_acc = df_subset[["total_acc_x", "total_acc_y", "total_acc_z"]].values.astype(np.float32)
        gyro = df_subset[["body_gyro_x", "body_gyro_y", "body_gyro_z"]].values.astype(np.float32)
        # Apply 3rd-order Butterworth low-pass filter at 0.3 Hz (50 Hz rate -> Nyquist 25 Hz) to separate gravity
        b, a = butter(3, 0.3 / 25.0, btype="low")
        gravity = filtfilt(b, a, total_acc, axis=0)
        body_acc = total_acc - gravity
        clean_arr = np.concatenate([body_acc, gyro, total_acc], axis=1).astype(np.float32)

    result["valid"] = True
    result["schema"] = target_schema
    result["clean_array"] = clean_arr
    return result


def slice_sensor_windows(data_N_9: np.ndarray, stride: int = 128) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Slice continuous sensor data (N, 9) into fixed-width 128-timestep windows (W, 9, 128)."""
    n_rows = len(data_N_9)
    n_windows = (n_rows - SEQ_LEN) // stride + 1
    windows = []
    time_ranges = []
    for i in range(n_windows):
        start = i * stride
        end = start + SEQ_LEN
        # Transpose from (128, 9) to (9, 128) for PyTorch channel-first representation
        windows.append(data_N_9[start:end].T)
        time_ranges.append((round(start / 50.0, 2), round(end / 50.0, 2)))
    return np.stack(windows, axis=0).astype(np.float32), time_ranges


@torch.no_grad()
def predict_windows(models: Dict[str, nn.Module], windows: np.ndarray, device: Optional[torch.device] = None) -> Dict[str, Dict]:
    """Run batched forward inference for all loaded models on (W, 9, 128) windows."""
    dev = device or torch.device("cpu")
    tensor = torch.from_numpy(windows).float().to(dev)
    results = {}

    for name, model in models.items():
        logits = model(tensor)  # (W, 6)
        probs = torch.softmax(logits, dim=1).cpu().numpy()  # (W, 6)
        preds = np.argmax(probs, axis=1)  # (W,)
        confs = np.max(probs, axis=1)  # (W,)
        results[name] = {
            "preds": preds,
            "probs": probs,
            "confs": confs,
        }
    return results


def build_prediction_response(
    source_name: str,
    val_result: dict,
    stride: int,
    models: Dict[str, nn.Module],
    ground_truth: Optional[str] = None,
    ground_truth_label: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Orchestrate window slicing, inference, and consensus aggregation into structured output."""
    clean_array = val_result["clean_array"]
    windows, time_ranges = slice_sensor_windows(clean_array, stride=stride)
    n_windows = len(windows)

    predictions = predict_windows(models, windows, device=device)
    available_model_keys = [k for k in ("convlstm", "cnn1d", "gru", "lstm") if k in predictions]

    consensus_activities = []
    consensus_labels = []
    agreement_flags = []
    window_objects = []

    for w_idx in range(n_windows):
        t_start, t_end = time_ranges[w_idx]
        w_preds = [predictions[m]["preds"][w_idx] for m in available_model_keys]

        all_match = (len(set(w_preds)) == 1) if w_preds else False
        agreement_flags.append(all_match)

        # Consensus determination: priority to DeepConvLSTM or CNN1D if tied, else majority vote
        if all_match:
            chosen_c = w_preds[0]
        else:
            if "convlstm" in predictions:
                chosen_c = predictions["convlstm"]["preds"][w_idx]
            elif "cnn1d" in predictions:
                chosen_c = predictions["cnn1d"]["preds"][w_idx]
            else:
                chosen_c = max(set(w_preds), key=w_preds.count) if w_preds else 0

        act_key = ACTIVITIES[chosen_c]
        act_label = ACTIVITY_LABELS.get(act_key, act_key)
        consensus_activities.append(act_key)
        consensus_labels.append(act_label)

        # Per-model details for this window
        model_details = {}
        for m_key in available_model_keys:
            p_idx = int(predictions[m_key]["preds"][w_idx])
            p_conf = float(predictions[m_key]["confs"][w_idx])
            p_probs = predictions[m_key]["probs"][w_idx]
            m_act_key = ACTIVITIES[p_idx]
            model_details[m_key] = {
                "predicted_activity": m_act_key,
                "predicted_label": ACTIVITY_LABELS.get(m_act_key, m_act_key),
                "confidence": round(p_conf, 4),
                "probabilities": {
                    ACTIVITIES[i]: round(float(p_probs[i]), 4) for i in range(len(ACTIVITIES))
                },
            }

        window_objects.append({
            "window_id": w_idx + 1,
            "start_time_sec": t_start,
            "end_time_sec": t_end,
            "consensus_activity": act_key,
            "consensus_label": act_label,
            "unanimous_agreement": all_match,
            "models": model_details,
        })

    # Summary metrics across the recording
    dominant_act = max(set(consensus_activities), key=consensus_activities.count) if consensus_activities else "unknown"
    dominant_lbl = ACTIVITY_LABELS.get(dominant_act, dominant_act)
    agreement_rate = round((sum(agreement_flags) / len(agreement_flags)) * 100.0, 2) if agreement_flags else 0.0

    all_confs = []
    for m in available_model_keys:
        all_confs.extend(predictions[m]["confs"])
    mean_confidence = round(float(np.mean(all_confs)), 4) if all_confs else 0.0

    return {
        "source_name": source_name,
        "ground_truth": ground_truth,
        "ground_truth_label": ground_truth_label,
        "duration_sec": val_result["duration_sec"],
        "n_rows": val_result["n_rows"],
        "n_windows": n_windows,
        "stride": stride,
        "schema_detected": "9-Channel Full Standard" if val_result["schema"] == "9ch" else "6-Channel Raw IMU",
        "warnings": val_result["warnings"],
        "dominant_activity": dominant_act,
        "dominant_label": dominant_lbl,
        "model_agreement_rate": agreement_rate,
        "mean_confidence": mean_confidence,
        "available_models": available_model_keys,
        "windows": window_objects,
    }


# ── Sample Registry Helpers ──────────────────────────────────────────────────

def resolve_sample_id(sample_id: str) -> Optional[dict]:
    """Resolve sample_id against the safe allowlist, normalizing common prefixes/extensions."""
    clean_id = sample_id.strip().lower()
    if clean_id in SAMPLE_REGISTRY:
        return SAMPLE_REGISTRY[clean_id]

    # Normalize 'sample_walking_128.csv' -> 'walking_128'
    norm = clean_id.replace("sample_", "").replace(".csv", "")
    if norm in SAMPLE_REGISTRY:
        return SAMPLE_REGISTRY[norm]

    return None


def load_sample_dataframe(sample_meta: dict) -> Tuple[pd.DataFrame, Path]:
    """Load sample CSV file safely from sample_data/."""
    file_path = SAMPLE_DATA_DIR / sample_meta["filename"]
    if not file_path.exists():
        raise FileNotFoundError(f"Sample file not found at {file_path}")
    df = pd.read_csv(file_path)
    return df, file_path


# ── Metrics & Confusion Matrices ─────────────────────────────────────────────

def load_metrics_data(out_dir: Path = OUT_DIR) -> dict:
    """Load metrics.json and compute detailed performance figures."""
    metrics_path = Path(out_dir) / "metrics.json"
    if not metrics_path.exists():
        return {"models": [], "activities": ACTIVITIES, "activity_labels": ACTIVITY_LABELS}

    with open(metrics_path) as f:
        raw_metrics = json.load(f)

    detailed_models = []
    display_names = {
        "convlstm": "DeepConvLSTM",
        "cnn1d": "1D-CNN",
        "gru": "GRU",
        "lstm": "LSTM",
        "randomforest": "RandomForest Baseline",
    }

    for m in raw_metrics:
        cm = np.array(m["confusion_matrix"], dtype=np.float64)
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp

        with np.errstate(divide="ignore", invalid="ignore"):
            prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
            f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)

        detailed_models.append({
            "model": m["model"],
            "display_name": display_names.get(m["model"].lower(), m["model"].upper()),
            "accuracy": float(m["accuracy"]),
            "macro_precision": round(float(np.mean(prec)), 4),
            "macro_recall": round(float(np.mean(rec)), 4),
            "macro_f1": float(m["macro_f1"]),
            "per_class_precision": {ACTIVITIES[i]: round(float(prec[i]), 4) for i in range(len(ACTIVITIES))},
            "per_class_recall": {ACTIVITIES[i]: round(float(rec[i]), 4) for i in range(len(ACTIVITIES))},
            "per_class_f1": {ACTIVITIES[i]: round(float(f1[i]), 4) for i in range(len(ACTIVITIES))},
            "support": {ACTIVITIES[i]: int(np.sum(cm[i, :])) for i in range(len(ACTIVITIES))},
            "history": m.get("history"),
        })

    return {
        "models": detailed_models,
        "activities": ACTIVITIES,
        "activity_labels": ACTIVITY_LABELS,
    }


def load_confusion_matrices_data(out_dir: Path = OUT_DIR) -> dict:
    """Load confusion matrix figures and normalize by true class rows."""
    metrics_path = Path(out_dir) / "metrics.json"
    if not metrics_path.exists():
        return {"matrices": {}, "activities": ACTIVITIES, "activity_labels": [ACTIVITY_LABELS[a] for a in ACTIVITIES]}

    with open(metrics_path) as f:
        raw_metrics = json.load(f)

    display_names = {
        "convlstm": "DeepConvLSTM",
        "cnn1d": "1D-CNN",
        "gru": "GRU",
        "lstm": "LSTM",
        "randomforest": "RandomForest Baseline",
    }
    act_labels = [ACTIVITY_LABELS[a] for a in ACTIVITIES]
    matrices_dict = {}

    for m in raw_metrics:
        m_name = m["model"].lower()
        cm = np.array(m["confusion_matrix"], dtype=np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm_cm = np.where(row_sums > 0, cm / row_sums, 0.0)

        matrices_dict[m_name] = {
            "model": m["model"],
            "display_name": display_names.get(m_name, m_name.upper()),
            "matrix": [[int(val) for val in row] for row in cm],
            "matrix_normalized": [[round(float(val), 4) for val in row] for row in norm_cm],
            "activities": ACTIVITIES,
            "activity_labels": act_labels,
        }

    return {
        "matrices": matrices_dict,
        "activities": ACTIVITIES,
        "activity_labels": act_labels,
    }


# ── Live Real-Time Stream Buffer ─────────────────────────────────────────────

from collections import deque


class RealTimeStreamBuffer:
    """Manages continuous high-frequency inertial sensor readings for live WebSocket streaming.
    
    Accepts 6-channel raw readings (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z) or
    9-channel standard readings, performs real-time window buffering, low-pass gravity filtering,
    instant inference, and biomechanical posture/motion telemetry extraction.
    """

    def __init__(self, capacity: int = 512, window_size: int = 128, stride: int = 16) -> None:
        self.capacity = capacity
        self.window_size = window_size
        self.stride = stride
        self.samples_since_last_pred = 0
        self.total_samples_received = 0
        
        # Buffer of 9-channel samples
        self.raw_accel_history: deque = deque(maxlen=capacity)
        self.raw_gyro_history: deque = deque(maxlen=capacity)
        self.buffer_9ch: deque = deque(maxlen=capacity)

    def add_reading(self, reading: dict) -> None:
        """Add a single reading dict. Supports 6-channel or 9-channel keys."""
        has_9ch = all(k in reading for k in SIGNAL_NAMES)
        if has_9ch:
            sample_9 = [float(reading[k]) for k in SIGNAL_NAMES]
            self.buffer_9ch.append(sample_9)
        else:
            ax = float(reading.get("accel_x", reading.get("ax", 0.0)))
            ay = float(reading.get("accel_y", reading.get("ay", 0.0)))
            az = float(reading.get("accel_z", reading.get("az", 0.0)))
            gx = float(reading.get("gyro_x", reading.get("gx", 0.0)))
            gy = float(reading.get("gyro_y", reading.get("gy", 0.0)))
            gz = float(reading.get("gyro_z", reading.get("gz", 0.0)))
            
            self.raw_accel_history.append([ax, ay, az])
            self.raw_gyro_history.append([gx, gy, gz])
            
            # Online low-pass filter approximation over recent 25 samples (~0.5s at 50Hz)
            recent_accel = list(self.raw_accel_history)[-min(25, len(self.raw_accel_history)):]
            g_vec = np.mean(recent_accel, axis=0) if recent_accel else np.array([0.0, 0.0, 1.0])
            
            body_acc = [ax - float(g_vec[0]), ay - float(g_vec[1]), az - float(g_vec[2])]
            body_gyro = [gx, gy, gz]
            total_acc = [ax, ay, az]
            
            sample_9 = body_acc + body_gyro + total_acc
            self.buffer_9ch.append(sample_9)

        self.total_samples_received += 1
        self.samples_since_last_pred += 1

    def should_predict(self) -> bool:
        """Returns True if enough samples have accumulated for a new window prediction."""
        return (len(self.buffer_9ch) >= self.window_size) and (self.samples_since_last_pred >= self.stride)

    def extract_latest_window(self) -> np.ndarray:
        """Return (1, 9, 128) array for model input."""
        recent = list(self.buffer_9ch)[-self.window_size:]
        arr = np.array(recent, dtype=np.float32).T  # (9, 128)
        self.samples_since_last_pred = 0
        return arr[np.newaxis, ...]  # (1, 9, 128)

    def compute_telemetry(self) -> dict:
        """Compute real-time biomechanical telemetry from the most recent readings."""
        if not self.buffer_9ch:
            return {
                "magnitude_g": 0.0,
                "tilt_angle_deg": 0.0,
                "state": "unknown",
                "sample_count": 0,
            }
        latest = self.buffer_9ch[-1]
        tot_x, tot_y, tot_z = latest[6], latest[7], latest[8]
        norm_g = np.sqrt(tot_x**2 + tot_y**2 + tot_z**2)
        
        tilt_deg = float(np.arccos(np.clip(tot_y / max(norm_g, 1e-6), -1.0, 1.0)) * (180.0 / np.pi))
        
        recent = list(self.buffer_9ch)[-min(32, len(self.buffer_9ch)):]
        body_acc = np.array([s[0:3] for s in recent])
        motion_var = float(np.mean(np.var(body_acc, axis=0)))
        state = "dynamic" if motion_var > 0.015 else "static"
        
        return {
            "magnitude_g": round(float(norm_g), 3),
            "tilt_angle_deg": round(tilt_deg, 1),
            "state": state,
            "motion_variance": round(motion_var, 4),
            "sample_count": self.total_samples_received,
        }

