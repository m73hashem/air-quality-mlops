# air-quality-mlops

An end-to-end MLOps pipeline and production service for Beijing Air Quality forecasting, implementing clean package structure, structured logging, ONNX runtime, FastAPI, Pytest, and multi-stage Docker builds.

## Run the production API

The published Docker image can be run without cloning this repository.

### 3 commands

```bash
docker pull m73hashem/air-quality-mlops:0.1.0
```

```bash
docker run -d --name air-quality-mlops -p 8000:8000 m73hashem/air-quality-mlops:0.1.0
```

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## API example

### Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2016,
    "month": 5,
    "day": 13,
    "hour": 12,
    "pm25": 80,
    "pm10": 100,
    "so2": 10,
    "no2": 40,
    "co": 800,
    "o3": 60,
    "temp": 20,
    "pres": 1010,
    "dewp": 10,
    "rain": 0,
    "wspm": 2,
    "wd": "NW",
    "station": "Aotizhongxin",
    "pm25_lag_1": 75,
    "pm25_lag_3": 70,
    "pm25_lag_24": 90,
    "day_of_week": 4,
    "is_weekend": 0,
    "hour_sin": 0,
    "hour_cos": -1,
    "month_sin": 0.5,
    "month_cos": -0.866
  }'
```

Example response:

```json
{
  "prediction": 84.3072509765625
}
```

## API endpoints

* `GET /health` — service health check
* `GET /metadata` — model metadata
* `POST /predict` — single PM2.5 prediction
* `POST /predict/batch` — batch PM2.5 predictions

## Repository structure

```text
air-quality-mlops/
├── data/
│   └── raw/
├── models/
│   ├── model.pkl
│   └── model.onnx
├── notebooks/
│   └── 00-baseline.ipynb
├── src/
│   └── air_quality/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── export.py
│       ├── features.py
│       ├── logging_config.py
│       ├── main.py
│       ├── predict.py
│       ├── schemas.py
│       ├── train.py
│       └── utils.py
├── tests/
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.single-stage
│   └── docker-compose.yml
├── pyproject.toml
├── .gitignore
└── README.md
```
