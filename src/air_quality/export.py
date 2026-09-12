import joblib
import logging
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestRegressor

from air_quality.config import MODEL_PATH, ONNX_MODEL_PATH

logger = logging.getLogger(__name__)

def export_random_forest() -> None:
    """Export only the trained Random Forest to ONNX."""
    pipeline = joblib.load(MODEL_PATH)

    forest = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if not isinstance(forest, RandomForestRegressor):
        raise TypeError(
            "Expected RandomForestRegressor in trained pipeline."
        )

    feature_count = len(
        preprocessor.get_feature_names_out()
    )

    initial_types = [
        (
            "features",
            FloatTensorType([None, feature_count]),
        )
    ]

    onnx_model = convert_sklearn(
        forest,
        initial_types=initial_types,
        target_opset=17,
    )

    ONNX_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(ONNX_MODEL_PATH, "wb") as file:
        file.write(onnx_model.SerializeToString())

    logger.info(
        "Random Forest ONNX model saved to: %s",
        ONNX_MODEL_PATH,
    )


if __name__ == "__main__":
    export_random_forest()
    