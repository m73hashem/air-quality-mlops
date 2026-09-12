import logging
import time
import uuid

import pandas as pd
from fastapi import Depends, FastAPI, Request
from contextlib import asynccontextmanager

from air_quality import features
from air_quality.config import LOG_LEVEL
from air_quality.logging_config import (
    configure_logging,
    set_correlation_id,
)
from air_quality.predict import AirQualityPredictor
from air_quality.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
)


configure_logging(LOG_LEVEL)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the prediction model when the application starts."""
    app.state.predictor = AirQualityPredictor()
    yield

def get_predictor(request: Request) -> AirQualityPredictor:
    """Return the application-wide predictor."""
    return request.app.state.predictor

app = FastAPI(
    title="Air Quality Prediction API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def correlation_id_middleware(
    request: Request,
    call_next,
):
    """Attach a correlation ID to each HTTP request."""
    request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4()),
    )

    set_correlation_id(request_id)

    response = await call_next(request)

    response.headers["X-Request-ID"] = request_id

    return response


def request_to_features(request: PredictionRequest) -> dict:
    """Convert an API request into model features."""
    return {
        "year": request.year,
        "month": request.month,
        "day": request.day,
        "hour": request.hour,
        "PM2.5": request.pm25,
        "PM10": request.pm10,
        "SO2": request.so2,
        "NO2": request.no2,
        "CO": request.co,
        "O3": request.o3,
        "TEMP": request.temp,
        "PRES": request.pres,
        "DEWP": request.dewp,
        "RAIN": request.rain,
        "wd": request.wd,
        "WSPM": request.wspm,
        "station": request.station,
        "PM2.5_lag_1": request.pm25_lag_1,
        "PM2.5_lag_3": request.pm25_lag_3,
        "PM2.5_lag_24": request.pm25_lag_24,
        "day_of_week": request.day_of_week,
        "is_weekend": request.is_weekend,
        "hour_sin": request.hour_sin,
        "hour_cos": request.hour_cos,
        "month_sin": request.month_sin,
        "month_cos": request.month_cos,
    }

def warn_if_outside_training_range(
    features: dict,
) -> None:
    """Log warnings for input values outside training ranges."""
    from air_quality.config import TRAINING_FEATURE_RANGES

    for feature_name, (minimum, maximum) in TRAINING_FEATURE_RANGES.items():
        value = features.get(feature_name)

        if value is None:
            continue

        if value < minimum or value > maximum:
            logger.warning(
                "Input outside training range: "
                "feature=%s value=%.4f expected_range=[%.4f, %.4f]",
                feature_name,
                value,
                minimum,
                maximum,
            )

@app.get("/health")
def health() -> dict[str, str]:
    """Return the API health status."""
    return {"status": "ok"}


@app.get(
    "/metadata",
    response_model=MetadataResponse,
)
def metadata() -> MetadataResponse:
    """Return information about the prediction model."""
    return MetadataResponse(
        model_version="0.1.0",
        target="PM2.5_next",
        model_type="RandomForestRegressor",
        feature_count=26,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
    predictor: AirQualityPredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Predict PM2.5 for one observation."""
    start = time.perf_counter()

    features = request_to_features(request)
    logger.debug(
        "Prediction features: %s",
        features,
    )

    warn_if_outside_training_range(features)

    prediction = predictor.predict_one(features)

    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "Prediction completed: prediction=%.4f latency_ms=%.2f",
        prediction,
        latency_ms,
    )

    return PredictionResponse(prediction=prediction)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
def predict_batch(
    request: BatchPredictionRequest,
    predictor: AirQualityPredictor = Depends(get_predictor),
) -> BatchPredictionResponse:
    """Predict PM2.5 for multiple observations."""
    features = [
        request_to_features(item)
        for item in request.items
    ]

    predictions = predictor.predict_batch(
        pd.DataFrame(features)
    )

    return BatchPredictionResponse(
        predictions=predictions,
    )

