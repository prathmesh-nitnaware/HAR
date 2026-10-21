/**
 * Typed API interfaces for MotionSense AI
 * Matches FastAPI backend Pydantic schemas exactly.
 */

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy' | string;
  device: 'cpu' | 'cuda' | string;
  models: Record<string, boolean>;
  models_loaded_count: number;
  total_models: number;
}

export interface ModelMetadata {
  id: 'convlstm' | 'cnn1d' | 'gru' | 'lstm' | 'randomforest' | string;
  name: string;
  type: string;
  architecture: string;
  input_representation: string;
  parameters: number | null;
  disk_size_mb: number | null;
  int8_size_mb?: number | null;
  status: 'ready' | 'unavailable' | string;
  test_accuracy: number | null;
  macro_f1: number | null;
}

export interface LiveTelemetry {
  magnitude_g: number;
  tilt_angle_deg: number;
  state: 'dynamic' | 'static' | string;
  motion_variance?: number;
  sample_count: number;
  fall_detected?: boolean;
  fall_state?: 'normal' | 'freefall' | 'impact' | 'immobility' | 'confirmed_fall' | 'recovered' | string;
  peak_g?: number;
  jerk_g_per_sec?: number;
}

export interface FallAlertEvent {
  type: 'fall_alert';
  timestamp: number;
  severity: 'warning' | 'critical' | string;
  peak_g: number;
  peak_jerk: number;
  posture: string;
  tilt_deg?: number;
  message: string;
}

export interface LivePredictionMessage {
  type: 'prediction';
  timestamp: number;
  window_count: number;
  predicted_activity: string;
  predicted_label: string;
  confidence: number;
  probabilities: Record<string, number>;
  model: string;
  latency_ms: number;
  telemetry: LiveTelemetry;
}


export interface SampleSummary {
  id: string;
  filename: string;
  name: string;
  description: string;
  ground_truth: string | null;
  ground_truth_label: string | null;
  expected_rows: number;
  channels: number;
  duration_sec: number;
}

export interface SampleDetailResponse {
  metadata: SampleSummary;
  columns: string[];
  timestamps: number[];
  signals: Record<string, number[]>;
  rows_count: number;
}

export interface WindowModelPrediction {
  predicted_activity: string;
  predicted_label: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface WindowPrediction {
  window_id: number;
  start_time_sec: number;
  end_time_sec: number;
  consensus_activity: string;
  consensus_label: string;
  unanimous_agreement: boolean;
  models: Record<string, WindowModelPrediction>;
}

export interface PredictResponse {
  source_name: string;
  ground_truth?: string | null;
  ground_truth_label?: string | null;
  duration_sec: number;
  n_rows: number;
  n_windows: number;
  stride: number;
  schema_detected: string;
  warnings: string[];
  dominant_activity: string;
  dominant_label: string;
  model_agreement_rate: number;
  mean_confidence: number;
  available_models: string[];
  windows: WindowPrediction[];
}

export interface TrainingHistory {
  train_loss: number[];
  test_loss: number[];
  test_acc: number[];
}

export interface PerModelMetrics {
  model: string;
  display_name: string;
  accuracy: number;
  macro_precision: number;
  macro_recall: number;
  macro_f1: number;
  per_class_precision: Record<string, number>;
  per_class_recall: Record<string, number>;
  per_class_f1: Record<string, number>;
  support: Record<string, number>;
  history: TrainingHistory | null;
}

export interface MetricsResponse {
  models: PerModelMetrics[];
  activities: string[];
  activity_labels: Record<string, string>;
}

export interface ConfusionMatrixData {
  model: string;
  display_name: string;
  matrix: number[][];
  matrix_normalized: number[][];
  activities: string[];
  activity_labels: string[];
}

export interface ConfusionMatrixResponse {
  matrices: Record<string, ConfusionMatrixData>;
  activities: string[];
  activity_labels: string[];
}

export interface ApiError {
  detail: string;
  error_type?: string;
  status?: number;
}
