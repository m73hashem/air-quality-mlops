import pytest
from fastapi.testclient import TestClient

from air_quality.main import app

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

SAMPLE_FEATURES = {
    "year": 2016,
    "month": 5,
    "day": 12,
    "hour": 20,
    "pm25": 26.0,
    "pm10": 52.0,
    "so2": 2.0,
    "no2": 30.0,
    "co": 400.0,
    "o3": 66.0,
    "temp": 17.4,
    "pres": 1012.6,
    "dewp": 0.1,
    "rain": 0.0,
    "wspm": 0.6,
    "wd": "NNW",
    "station": "Aotizhongxin",
    "pm25_lag_1": 10.0,
    "pm25_lag_3": 17.0,
    "pm25_lag_24": 151.0,
    "day_of_week": 3,
    "is_weekend": 0,
    "hour_sin": -0.8660254037844386,
    "hour_cos": 0.5000000000000001,
    "month_sin": 0.49999999999999994,
    "month_cos": -0.8660254037844387,
}


def test_health(client):
    """Health endpoint should return a successful response."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metadata(client):
    """Metadata endpoint should return model information."""
    response = client.get("/metadata")

    assert response.status_code == 200

    data = response.json()

    assert data["model_version"] == "0.1.0"
    assert data["target"] == "PM2.5_next"
    assert data["model_type"] == "RandomForestRegressor"
    assert data["feature_count"] == 26


def test_predict(client):
    """Predict endpoint should return one prediction."""
    response = client.post(
        "/predict",
        json=SAMPLE_FEATURES,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert isinstance(data["prediction"], float)


def test_predict_is_deterministic(client):
    """The same input should produce the same prediction."""
    response_1 = client.post(
        "/predict",
        json=SAMPLE_FEATURES,
    )

    response_2 = client.post(
        "/predict",
        json=SAMPLE_FEATURES,
    )

    assert response_1.status_code == 200
    assert response_2.status_code == 200

    assert response_1.json()["prediction"] == pytest.approx(
        response_2.json()["prediction"],
        rel=1e-12,
        abs=1e-12,
    )


def test_predict_batch(client):
    """Batch endpoint should return one prediction per input item."""
    response = client.post(
        "/predict/batch",
        json={
            "items": [
                SAMPLE_FEATURES,
                SAMPLE_FEATURES,
            ]
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 2

    assert all(
        isinstance(prediction, float)
        for prediction in data["predictions"]
    )


def test_predict_batch_matches_single_prediction(client):
    """Batch prediction should match single prediction."""
    single_response = client.post(
        "/predict",
        json=SAMPLE_FEATURES,
    )

    batch_response = client.post(
        "/predict/batch",
        json={
            "items": [SAMPLE_FEATURES]
        },
    )

    assert single_response.status_code == 200
    assert batch_response.status_code == 200

    single_prediction = single_response.json()["prediction"]
    batch_prediction = batch_response.json()["predictions"][0]

    assert batch_prediction == pytest.approx(
        single_prediction,
        rel=1e-12,
        abs=1e-12,
    )


def test_predict_validation_error(client):
    """Invalid input should return HTTP 422."""
    invalid_features = SAMPLE_FEATURES.copy()
    invalid_features["month"] = 13

    response = client.post(
        "/predict",
        json=invalid_features,
    )

    assert response.status_code == 422


def test_predict_missing_field(client):
    """Missing required input should return HTTP 422."""
    invalid_features = SAMPLE_FEATURES.copy()
    del invalid_features["pm25"]

    response = client.post(
        "/predict",
        json=invalid_features,
    )

    assert response.status_code == 422


def test_request_id_is_returned(client):
    """The API should return the request correlation ID."""
    request_id = "test-request-123"

    response = client.get(
        "/health",
        headers={
            "X-Request-ID": request_id,
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == request_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("month", 0),
        ("month", 13),
        ("hour", -1),
        ("hour", 24),
        ("day_of_week", -1),
        ("day_of_week", 7),
        ("is_weekend", -1),
        ("is_weekend", 2),
    ],
)
def test_predict_field_validation(field, value, client):
    """Invalid bounded fields should return HTTP 422."""
    invalid_features = SAMPLE_FEATURES.copy()
    invalid_features[field] = value

    response = client.post(
        "/predict",
        json=invalid_features,
    )

    assert response.status_code == 422

def test_predict_model_failure(client):
    """Model prediction failure should be raised by the API layer."""

    def mock_predict_one(features):
        raise RuntimeError("Model prediction failed")

    original_predict_one = (
        client.app.state.predictor.predict_one
    )

    client.app.state.predictor.predict_one = (
        mock_predict_one
    )

    try:
        with pytest.raises(RuntimeError):
            client.post(
                "/predict",
                json=SAMPLE_FEATURES,
            )
    finally:
        client.app.state.predictor.predict_one = (
            original_predict_one
        )


def test_predict_logs_warning_for_out_of_range_input(
    client,
    caplog,
):
    payload = {
        "year": 2016,
        "month": 5,
        "day": 13,
        "hour": 12,
        "pm25": 5000.0,
        "pm10": 100.0,
        "so2": 20.0,
        "no2": 30.0,
        "co": 1000.0,
        "o3": 50.0,
        "temp": 20.0,
        "pres": 1000.0,
        "dewp": 10.0,
        "rain": 0.0,
        "wspm": 2.0,
        "wd": "NW",
        "station": "Aotizhongxin",
        "pm25_lag_1": 50.0,
        "pm25_lag_3": 50.0,
        "pm25_lag_24": 50.0,
        "day_of_week": 4,
        "is_weekend": 0,
        "hour_sin": 0.0,
        "hour_cos": 1.0,
        "month_sin": 0.0,
        "month_cos": 1.0,
    }

    with caplog.at_level("WARNING"):
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "Input outside training range" in caplog.text
    assert "PM2.5" in caplog.text

