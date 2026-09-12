from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    model_dir: Path = PROJECT_ROOT / "models"

    baseline_model_path: Path = (
        PROJECT_ROOT / "models" / "baseline.pkl"
    )
    model_path: Path = (
        PROJECT_ROOT / "models" / "model.pkl"
    )

    onnx_model_path: Path = (
        PROJECT_ROOT / "models" / "model.onnx"
    )

    cutoff_datetime: str = "2016-05-12 19:00:00"

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="AIR_",
        env_file=".env",
        extra="ignore",
    )


settings = Settings()

RAW_DATA_DIR = settings.raw_data_dir
MODEL_DIR = settings.model_dir

BASELINE_MODEL_PATH = settings.baseline_model_path
MODEL_PATH = settings.model_path
ONNX_MODEL_PATH = settings.onnx_model_path

CUTOFF_DATETIME = settings.cutoff_datetime
LOG_LEVEL = settings.log_level

TRAINING_FEATURE_RANGES = {
    "PM2.5": (2.0, 999.0),
    "PM10": (2.0, 999.0),
    "TEMP": (-19.9, 41.6),
    "PRES": (982.4, 1042.8),
    "CO": (100.0, 10000.0),
    "NO2": (1.0265, 290.0),
    "SO2": (0.2856, 500.0),
    "O3": (0.2142, 1071.0),
}
