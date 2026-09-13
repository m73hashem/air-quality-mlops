from pathlib import Path

import pandas as pd


def load_station_data(data_dir: Path) -> pd.DataFrame:
    """Load and combine all station CSV files."""
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in: {data_dir}"
        )

    dataframes = [
        pd.read_csv(file)
        for file in csv_files
    ]

    return pd.concat(dataframes, ignore_index=True)
