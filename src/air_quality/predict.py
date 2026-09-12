from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.pipeline import Pipeline

from air_quality.config import MODEL_PATH, ONNX_MODEL_PATH


class AirQualityPredictor:
    """Load preprocessing and ONNX model for PM2.5 prediction."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        onnx_model_path: Path = ONNX_MODEL_PATH,
    ) -> None:
        self.model_path = model_path
        self.onnx_model_path = onnx_model_path

        self.model = self._load_pipeline()
        self.session = self._load_onnx_model()

        self.input_name = (
            self.session.get_inputs()[0].name
        )

    def _load_pipeline(self) -> Pipeline:
        """Load the sklearn pipeline containing preprocessing."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        return joblib.load(self.model_path)

    def _load_onnx_model(self) -> ort.InferenceSession:
        """Load the ONNX Random Forest inference session."""
        if not self.onnx_model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self.onnx_model_path}"
            )

        return ort.InferenceSession(
            str(self.onnx_model_path)
        )

    def _transform(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """Apply the sklearn preprocessing pipeline."""
        preprocessor = self.model.named_steps[
            "preprocessor"
        ]

        transformed = preprocessor.transform(features)

        return np.asarray(
            transformed,
            dtype=np.float32,
        )

    def _predict(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """Run ONNX inference on preprocessed features."""
        transformed = self._transform(features)

        predictions = self.session.run(
            None,
            {
                self.input_name: transformed,
            },
        )[0]

        return predictions.ravel()

    def predict_one(
        self,
        features: dict,
    ) -> float:
        """Predict PM2.5 for a single observation."""
        data = pd.DataFrame([features])

        prediction = self._predict(data)

        return float(prediction[0])

    def predict_batch(
        self,
        features: pd.DataFrame,
    ) -> list[float]:
        """Predict PM2.5 for multiple observations."""
        predictions = self._predict(features)

        return predictions.tolist()