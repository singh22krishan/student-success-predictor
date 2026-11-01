from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


class StudentSuccessModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.metadata = None

    @property
    def configured(self) -> bool:
        return all([
            (MODEL_DIR / "student_success_model.keras").exists(),
            (MODEL_DIR / "preprocessor.joblib").exists(),
            (MODEL_DIR / "label_encoder.joblib").exists(),
            (MODEL_DIR / "model_metadata.json").exists(),
        ])

    def _load(self):
        if self.model is not None:
            return
        if not self.configured:
            raise FileNotFoundError("Trained model artifacts are not available")

        import tensorflow as tf

        self.model = tf.keras.models.load_model(MODEL_DIR / "student_success_model.keras")
        self.preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
        self.label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
        self.metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))

    def predict(self, features: dict) -> dict:
        self._load()
        expected = self.metadata["enrollment_features"]
        missing = [name for name in expected if name not in features]
        if missing:
            raise ValueError(f"Missing {len(missing)} required enrollment features")

        frame = pd.DataFrame([{name: features[name] for name in expected}])
        transformed = self.preprocessor.transform(frame)
        probabilities = self.model.predict(transformed, verbose=0)[0]
        index = int(probabilities.argmax())
        label = str(self.label_encoder.inverse_transform([index])[0])
        return {
            "prediction": label,
            "confidence": round(float(probabilities[index]) * 100, 2),
            "probabilities": {
                str(name): round(float(value) * 100, 2)
                for name, value in zip(self.label_encoder.classes_, probabilities)
            },
        }
