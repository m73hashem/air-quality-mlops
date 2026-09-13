import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from air_quality.utils import timed
import logging
from air_quality.logging_config import configure_logging

from air_quality.config import (
    CUTOFF_DATETIME,
    MODEL_DIR,
    MODEL_PATH,
    RAW_DATA_DIR,
)
from air_quality.data import load_station_data
from air_quality.features import (
    build_preprocessor,
    create_features,
    create_target,
    prepare_datetime,
    split_data,
)

configure_logging()
logger = logging.getLogger(__name__)

DROP_COLUMNS = [
    "No",
    "datetime",
    "PM2.5_next",
]

CATEGORICAL_FEATURES = [
    "wd",
    "station",
]


def prepare_train_test_data():
    """Load, prepare, and split the dataset."""
    df = load_station_data(RAW_DATA_DIR)

    df = prepare_datetime(df)
    df = create_target(df)
    df = create_features(df)

    train_df, test_df = split_data(
        df,
        pd.Timestamp(CUTOFF_DATETIME),
    )

    X_train = train_df.drop(columns=DROP_COLUMNS)
    y_train = train_df["PM2.5_next"]

    X_test = test_df.drop(columns=DROP_COLUMNS)
    y_test = test_df["PM2.5_next"]

    return X_train, y_train, X_test, y_test


def build_model(numeric_features: list[str]) -> Pipeline:
    """Build the preprocessing and Random Forest pipeline."""
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=CATEGORICAL_FEATURES,
    )

    model = RandomForestRegressor(
        n_estimators=30,
        max_depth=15,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


@timed
def train_model() -> Pipeline:
    """Train and save the baseline model."""
    X_train, y_train, X_test, y_test = prepare_train_test_data()

    numeric_features = [
        column
        for column in X_train.columns
        if column not in CATEGORICAL_FEATURES
    ]

    pipeline = build_model(numeric_features)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        MODEL_PATH,
        compress=3,
    )

    logger.info("Validation MAE: %.4f", mae)
    logger.info("Validation RMSE: %.4f", rmse)
    logger.info("Model saved to: %s", MODEL_PATH)

    return pipeline


if __name__ == "__main__":
    train_model()
