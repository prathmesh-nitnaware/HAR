"""Unit tests for VisionPoseClassifier and optical posture recognition fallback."""
import os
import unittest
from backend.vision import get_vision_classifier, VISION_ACTIVITIES


class TestVisionPoseClassifier(unittest.TestCase):
    def setUp(self):
        self.clf = get_vision_classifier()
        self.sample_dir = os.path.join(os.path.dirname(__file__), "..", "backend", "vision_samples")

    def test_sample_files_exist(self):
        for act in ["walking", "sitting", "standing", "laying", "fall"]:
            p = os.path.join(self.sample_dir, f"demo_{act}.jpg")
            self.assertTrue(os.path.exists(p), f"Demo image {p} must exist")

    def test_standing_classification(self):
        p = os.path.join(self.sample_dir, "demo_standing.jpg")
        with open(p, "rb") as f:
            res = self.clf.classify_bytes(f.read())
        self.assertEqual(res["predicted_activity"], "standing")
        self.assertGreaterEqual(res["confidence"], 0.80)
        self.assertTrue(res["annotated_image_base64"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(len(res["keypoints"]), 13)

    def test_walking_classification(self):
        p = os.path.join(self.sample_dir, "demo_walking.jpg")
        with open(p, "rb") as f:
            res = self.clf.classify_bytes(f.read())
        self.assertEqual(res["predicted_activity"], "walking")
        self.assertGreaterEqual(res["confidence"], 0.80)
        self.assertGreater(res["biomechanics"]["stride_separation"], 0.10)

    def test_sitting_classification(self):
        p = os.path.join(self.sample_dir, "demo_sitting.jpg")
        with open(p, "rb") as f:
            res = self.clf.classify_bytes(f.read())
        self.assertEqual(res["predicted_activity"], "sitting")
        self.assertGreaterEqual(res["confidence"], 0.80)
        self.assertLess(res["biomechanics"]["knee_angle_deg"], 135.0)

    def test_laying_classification(self):
        p = os.path.join(self.sample_dir, "demo_laying.jpg")
        with open(p, "rb") as f:
            res = self.clf.classify_bytes(f.read())
        self.assertEqual(res["predicted_activity"], "laying")
        self.assertGreaterEqual(res["confidence"], 0.80)
        self.assertEqual(res["biomechanics"]["orientation"], "horizontal")

    def test_fall_classification(self):
        p = os.path.join(self.sample_dir, "demo_fall.jpg")
        with open(p, "rb") as f:
            res = self.clf.classify_bytes(f.read())
        self.assertEqual(res["predicted_activity"], "fall")
        self.assertGreaterEqual(res["confidence"], 0.80)
        self.assertGreaterEqual(res["biomechanics"]["ground_proximity"], 0.70)


class TestVisionEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from backend.main import app
        cls.client = TestClient(app)
        cls.sample_dir = os.path.join(os.path.dirname(__file__), "..", "backend", "vision_samples")

    def test_get_demo_samples(self):
        res = self.client.get("/api/vision/demo-samples")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("samples", data)
        self.assertEqual(len(data["samples"]), 5)
        for s in data["samples"]:
            self.assertTrue(s["image_base64"].startswith("data:image/jpeg;base64,"))
            self.assertIn(s["activity"], VISION_ACTIVITIES)

    def test_predict_upload_endpoint(self):
        p = os.path.join(self.sample_dir, "demo_walking.jpg")
        with open(p, "rb") as f:
            res = self.client.post("/api/vision/predict", files={"file": ("walking.jpg", f, "image/jpeg")})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["predicted_activity"], "walking")
        self.assertTrue(data["annotated_image_base64"].startswith("data:image/jpeg;base64,"))

    def test_predict_base64_endpoint(self):
        p = os.path.join(self.sample_dir, "demo_sitting.jpg")
        import base64
        with open(p, "rb") as f:
            b64_str = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"
        res = self.client.post("/api/vision/predict-base64", json={"image_base64": b64_str})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["predicted_activity"], "sitting")


if __name__ == "__main__":
    unittest.main()

