import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Create datetime and sort observations chronologically per station."""
    df = df.copy()

    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]]
    )

    return df.sort_values(
        ["station", "datetime"]
    ).reset_index(drop=True)


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the PM2.5 one-hour-ahead prediction target."""
    df = df.copy()

    next_pm25 = (
        df.groupby("station")["PM2.5"]
        .shift(-1)
    )

    next_time = (
        df.groupby("station")["datetime"]
        .shift(-1)
    )

    df["PM2.5_next"] = next_pm25.where(
        next_time.eq(
            df["datetime"] + pd.Timedelta(hours=1)
        )
    )

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create historical and time-based prediction features."""
    df = df.copy()

    df["PM2.5_lag_1"] = (
        df.groupby("station")["PM2.5"].shift(1)
    )

    df["PM2.5_lag_3"] = (
        df.groupby("station")["PM2.5"].shift(3)
    )

    df["PM2.5_lag_24"] = (
        df.groupby("station")["PM2.5"].shift(24)
    )

    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    return df


def split_data(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset chronologically."""
    train_df = df[
        (df["datetime"] <= cutoff)
        & (df["PM2.5_next"].notna())
    ].copy()

    test_df = df[
        (df["datetime"] > cutoff)
        & (df["PM2.5_next"].notna())
    ].copy()

    return train_df, test_df


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build the preprocessing pipeline for model training."""
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )