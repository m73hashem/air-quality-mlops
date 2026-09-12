import pytest

from air_quality.config import RAW_DATA_DIR
from air_quality.data import load_station_data
from air_quality.features import (
    create_features,
    create_target,
    prepare_datetime,
)
from air_quality.predict import AirQualityPredictor


@pytest.fixture
def prepared_data():
    """Load and prepare data for prediction tests."""
    df = load_station_data(RAW_DATA_DIR)
    df = prepare_datetime(df)
    df = create_target(df)
    df = create_features(df)

    return df


@pytest.fixture
def predictor():
    """Create a predictor using the trained model."""
    return AirQualityPredictor()


def test_model_is_loaded(predictor):
    """The trained model should be loaded successfully."""
    assert predictor.model is not None


def test_predict_one(predictor, prepared_data):
    """The predictor should return one numeric prediction."""
    row = (
        prepared_data
        .dropna(subset=["PM2.5_next"])
        .iloc[100]
    )

    features = (
        row
        .drop(labels=["No", "datetime", "PM2.5_next"])
        .to_dict()
    )

    prediction = predictor.predict_one(features)

    assert isinstance(prediction, float)


def test_predict_batch(predictor, prepared_data):
    """The predictor should return one prediction per input row."""
    features = (
        prepared_data
        .drop(columns=["No", "datetime", "PM2.5_next"])
        .dropna()
        .head(5)
    )

    predictions = predictor.predict_batch(features)

    assert isinstance(predictions, list)
    assert len(predictions) == 5


def test_predict_batch_is_deterministic(
    predictor,
    prepared_data,
):
    """The same input should produce the same predictions."""
    features = (
        prepared_data
        .drop(columns=["No", "datetime", "PM2.5_next"])
        .dropna()
        .head(5)
    )

    predictions_1 = predictor.predict_batch(features)
    predictions_2 = predictor.predict_batch(features)

    assert predictions_1 == pytest.approx(
        predictions_2,
        rel=1e-12,
        abs=1e-12,
    )

