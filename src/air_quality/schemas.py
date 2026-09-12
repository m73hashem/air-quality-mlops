from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    year: int = Field(..., ge=2013)
    month: int = Field(..., ge=1, le=12)
    day: int = Field(..., ge=1, le=31)
    hour: int = Field(..., ge=0, le=23)

    pm25: float
    pm10: float
    so2: float
    no2: float
    co: float
    o3: float
    temp: float
    pres: float
    dewp: float
    rain: float
    wspm: float

    wd: str
    station: str

    pm25_lag_1: float
    pm25_lag_3: float
    pm25_lag_24: float

    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)

    hour_sin: float
    hour_cos: float
    month_sin: float
    month_cos: float


class PredictionResponse(BaseModel):
    prediction: float


class BatchPredictionRequest(BaseModel):
    """Request containing multiple prediction observations."""

    items: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Response containing multiple PM2.5 predictions."""

    predictions: list[float]


class MetadataResponse(BaseModel):
    """Model metadata returned by the API."""

    model_version: str
    target: str
    model_type: str
    feature_count: int

