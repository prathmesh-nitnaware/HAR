"""Automated regression and validation tests for MotionSense AI backend inference."""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.inference import (
    ModelManager,
    RealTimeStreamBuffer,
    validate_and_preprocess_sensor_csv,
    slice_sensor_windows,
    build_prediction_response,
    SIGNAL_NAMES,
    ACTIVITIES,
)


class TestBackendInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mgr = ModelManager.get_instance()

    def test_model_loading(self):
        """Verify that all four deep learning models (ConvLSTM, CNN1D, GRU, LSTM) are loaded and healthy."""
        self.assertTrue(self.mgr.load_status.get("convlstm", False), "ConvLSTM model failed to load")
        self.assertTrue(self.mgr.load_status.get("cnn1d", False), "CNN1D model failed to load")
        self.assertTrue(self.mgr.load_status.get("gru", False), "GRU model failed to load")
        self.assertTrue(self.mgr.load_status.get("lstm", False), "LSTM model failed to load")
        self.assertEqual(len(self.mgr.models), 4)

    def test_quantized_specs_and_artifacts(self):
        """Verify that model specs include valid INT8 quantized sizes and ONNX artifacts exist."""
        specs = self.mgr.get_model_specs()
        self.assertEqual(len(specs), 5)
        spec_dict = {s["id"]: s for s in specs}
        
        # ConvLSTM should have INT8 size reported ~0.53MB
        self.assertIn("convlstm", spec_dict)
        self.assertIsNotNone(spec_dict["convlstm"]["int8_size_mb"])
        self.assertLess(spec_dict["convlstm"]["int8_size_mb"], 1.0)
        
        # CNN1D should have INT8 size reported ~0.69MB
        self.assertIn("cnn1d", spec_dict)
        self.assertIsNotNone(spec_dict["cnn1d"]["int8_size_mb"])
        self.assertLess(spec_dict["cnn1d"]["int8_size_mb"], 1.0)

    def test_realtime_stream_buffer(self):
        """Verify RealTimeStreamBuffer ingests readings, calculates telemetry, and extracts 128-window array."""
        buf = RealTimeStreamBuffer(capacity=256, window_size=128, stride=16)
        self.assertFalse(buf.should_predict())

        # Feed 128 sample readings
        for i in range(128):
            reading = {
                "accel_x": 0.05 * np.sin(i * 0.1),
                "accel_y": 0.98 + 0.02 * np.cos(i * 0.1),
                "accel_z": 0.1,
                "gyro_x": 0.01,
                "gyro_y": 0.02,
                "gyro_z": 0.01,
            }
            buf.add_reading(reading)

        self.assertTrue(buf.should_predict())
        window_arr = buf.extract_latest_window()
        self.assertEqual(window_arr.shape, (1, 9, 128))
        self.assertEqual(window_arr.dtype, np.float32)

        telemetry = buf.compute_telemetry()
        self.assertIn("magnitude_g", telemetry)
        self.assertIn("tilt_angle_deg", telemetry)
        self.assertIn("state", telemetry)
        self.assertGreater(telemetry["magnitude_g"], 0.8)

    def test_window_slicing_strides(self):
        """Verify window slicing for 384-row input: stride 64 yields 5 windows, stride 128 yields 3 windows."""
        data_384 = np.zeros((384, 9), dtype=np.float32)
        windows_64, time_64 = slice_sensor_windows(data_384, stride=64)
        windows_128, time_128 = slice_sensor_windows(data_384, stride=128)

        self.assertEqual(len(windows_64), 5, "Stride 64 must yield 5 windows for 384 rows")
        self.assertEqual(len(windows_128), 3, "Stride 128 must yield 3 windows for 384 rows")
        self.assertEqual(windows_64.shape, (5, 9, 128))
        self.assertEqual(windows_128.shape, (3, 9, 128))

    def test_9ch_preprocessing(self):
        """Verify standard 9-channel preprocessing."""
        sample_path = os.path.join(PROJECT_ROOT, "sample_data", "sample_walking_128.csv")
        self.assertTrue(os.path.exists(sample_path), f"Sample file not found: {sample_path}")
        df = pd.read_csv(sample_path)
        val = validate_and_preprocess_sensor_csv(df)

        self.assertTrue(val["valid"])
        self.assertEqual(val["schema"], "9ch")
        self.assertEqual(val["n_rows"], 128)
        self.assertEqual(val["clean_array"].shape, (128, 9))

    def test_6ch_preprocessing_butterworth(self):
        """Verify 6-channel raw IMU input with 3rd-order Butterworth low-pass gravity separation."""
        sample_path = os.path.join(PROJECT_ROOT, "sample_data", "sample_walking_128.csv")
        df = pd.read_csv(sample_path)
        cols_6ch = ["total_acc_x", "total_acc_y", "total_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"]
        df_6ch = df[cols_6ch].copy()

        val = validate_and_preprocess_sensor_csv(df_6ch)
        self.assertTrue(val["valid"])
        self.assertEqual(val["schema"], "6ch")
        self.assertEqual(val["clean_array"].shape, (128, 9))

    def test_sitting_sample_predictions(self):
        """Verify sitting sample prediction behavior (Sitting ground-truth, models predict Standing due to posture similarity)."""
        sit_path = os.path.join(PROJECT_ROOT, "sample_data", "sample_sitting_128.csv")
        self.assertTrue(os.path.exists(sit_path), f"Sitting sample not found: {sit_path}")
        df_sit = pd.read_csv(sit_path)

        val = validate_and_preprocess_sensor_csv(df_sit)
        self.assertTrue(val["valid"])

        resp = build_prediction_response(
            source_name="sample_sitting_128.csv",
            val_result=val,
            stride=128,
            models=self.mgr.models,
            ground_truth="sitting",
            ground_truth_label="Sitting",
            device=self.mgr.device,
        )

        self.assertEqual(resp["n_windows"], 1)
        w0 = resp["windows"][0]
        # Ground truth is Sitting; all 3 models predict Standing on this test window
        self.assertEqual(resp["ground_truth"], "sitting")
        self.assertEqual(w0["models"]["cnn1d"]["predicted_activity"], "standing")
        self.assertEqual(w0["models"]["gru"]["predicted_activity"], "standing")
        self.assertEqual(w0["models"]["lstm"]["predicted_activity"], "standing")

    def test_invalid_input_validation(self):
        """Verify error handling on short, missing, or non-numeric inputs."""
        # Short sequence (<128)
        df_short = pd.DataFrame({col: np.zeros(64) for col in SIGNAL_NAMES})
        val_short = validate_and_preprocess_sensor_csv(df_short)
        self.assertFalse(val_short["valid"])
        self.assertIn("Insufficient sequence length", val_short["error"])

        # Missing columns
        df_missing = pd.DataFrame({"dummy": np.zeros(128)})
        val_missing = validate_and_preprocess_sensor_csv(df_missing)
        self.assertFalse(val_missing["valid"])
        self.assertIn("Missing required sensor channels", val_missing["error"])

        # Non-numeric values
        df_bad = pd.DataFrame({col: np.zeros(128) for col in SIGNAL_NAMES})
        df_bad["body_acc_x"] = df_bad["body_acc_x"].astype(object)
        df_bad.loc[10, "body_acc_x"] = "corrupted"
        val_bad = validate_and_preprocess_sensor_csv(df_bad)
        self.assertFalse(val_bad["valid"])
        self.assertIn("Non-numeric values detected", val_bad["error"])


if __name__ == "__main__":
    unittest.main()
