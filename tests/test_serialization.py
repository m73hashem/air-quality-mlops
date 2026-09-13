import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

from air_quality.config import (
    CUTOFF_DATETIME,
    MODEL_PATH,
    ONNX_MODEL_PATH,
    RAW_DATA_DIR,
)
from air_quality.data import load_station_data
from air_quality.features import (
    create_features,
    create_target,
    prepare_datetime,
    split_data,
)


DROP_COLUMNS = [
    "No",
    "datetime",
    "PM2.5_next",
]


def prepare_validation_data() -> pd.DataFrame:
    """Prepare 500 validation rows for serialization comparison."""
    df = load_station_data(RAW_DATA_DIR)
    df = prepare_datetime(df)
    df = create_target(df)
    df = create_features(df)

    _, test_df = split_data(
        df,
        pd.Timestamp(CUTOFF_DATETIME),
    )

    return test_df.head(500).drop(
        columns=DROP_COLUMNS
    )


def test_onnx_predictions_match_pickle() -> None:
    """ONNX Random Forest must match the sklearn Random Forest."""
    features = prepare_validation_data()

    # Load the original sklearn pipeline.
    pipeline = joblib.load(MODEL_PATH)

    # Separate preprocessing from the Random Forest.
    preprocessor = pipeline.named_steps["preprocessor"]
    forest = pipeline.named_steps["model"]

    # sklearn preprocessing.
    transformed = preprocessor.transform(
        features
    )

    # Original sklearn Random Forest prediction.
    pickle_predictions = forest.predict(
        transformed
    )

    # Load ONNX Random Forest.
    session = ort.InferenceSession(
        str(ONNX_MODEL_PATH)
    )

    input_name = session.get_inputs()[0].name

    # ONNX Runtime expects float32 input.
    onnx_inputs = {
        input_name: np.asarray(
            transformed,
            dtype=np.float32,
        )
    }

    # ONNX prediction.
    onnx_predictions = session.run(
        None,
        onnx_inputs,
    )[0].ravel()

    assert len(pickle_predictions) == 500
    assert len(onnx_predictions) == 500

    absolute_diff = np.abs(
        pickle_predictions - onnx_predictions
    )

    print(
        f"\nMax absolute difference: "
        f"{absolute_diff.max():.8f}"
    )

    print(
        f"Mean absolute difference: "
        f"{absolute_diff.mean():.8f}"
    )

    print(
        f"Differences > 1e-4: "
        f"{np.sum(absolute_diff > 1e-4)}"
    )

    assert np.allclose(
        pickle_predictions,
        onnx_predictions,
        atol=1e-4,
    )
    