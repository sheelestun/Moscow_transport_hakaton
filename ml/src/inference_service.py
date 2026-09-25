"""FastAPI-инференс: POST /predict и /predict/batch.

Модель — ансамбль CatBoost из `ml/artifacts/catboost_seed*.cbm` (train + eval через `train_catboost.py`).
Онлайн-фичи строятся тем же кодом, что и в батче (`features/tabular.build_features` через
`features/from_stream.build_features_online`) — так исключаем расхождение train/prod.

Контракт вход/выход — см. `ARCHITECTURE_AND_ROLES.md` §7.1. Здесь схема упрощённая: вместо
предвычисленных фичей клиент шлёт **сырой контекст** — буфер телеметрии и слайс расписания.
Все фичи вычисляются сервисом, чтобы фиче-логика жила в одном месте.

Запуск::

    uvicorn inference_service:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features.from_stream import build_features_online  # noqa: E402
from features.tabular import CAT_FEATURES  # noqa: E402

ARTIFACTS_DIR = Path(os.environ.get("ML_ARTIFACTS", Path(__file__).resolve().parents[1] / "artifacts"))
MODEL_VERSION = os.environ.get("ML_MODEL_VERSION", "catboost-ensemble-v1")
RISK_MID_SEC = float(os.environ.get("ML_RISK_MID_SEC", 120.0))
RISK_SLOPE_SEC = float(os.environ.get("ML_RISK_SLOPE_SEC", 60.0))


# ------------------------------------------------------------------ Pydantic-схемы


class TelemetryPing(BaseModel):
    tr_id: int
    event_time: str
    lon: Optional[float] = None
    lat: Optional[float] = None
    speed: Optional[float] = None
    location_valid: bool | str = "true"
    is_hist_data: int | bool = 0


class ScheduleStop(BaseModel):
    tr_id: int
    tt_action_item_id: int
    time_begin: str
    geom: str
    manual_fill: bool | str = "false"


class PredictRequest(BaseModel):
    sample_id: str
    tr_id: int
    T: str = Field(..., description="Момент прогноза, ISO-8601")
    target_stop_id: int
    target_time_begin: str = Field(..., description="Плановое время целевой остановки, ISO-8601")
    cur_dev_s: float = Field(..., description="Задержка на последней уже пройденной остановке (сек)")
    telemetry: list[TelemetryPing] = Field(..., description="Пинги NDTP с event_time ≤ T")
    schedule: list[ScheduleStop] = Field(..., description="Плановое расписание рейса ТС")


class BatchRequest(BaseModel):
    requests: list[PredictRequest]


class TopFeature(BaseModel):
    name: str
    value: float


class PredictResponse(BaseModel):
    sample_id: str
    delay_pred_sec: float
    risk_score: float
    confidence: float
    top_features: list[TopFeature]
    model_version: str


class BatchResponse(BaseModel):
    responses: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    n_models: int
    n_features: int
    model_version: str


# ------------------------------------------------------------------ загрузка моделей


def _load_models() -> tuple[list[CatBoostRegressor], dict]:
    meta_path = ARTIFACTS_DIR / "catboost_meta.json"
    if not meta_path.exists():
        raise RuntimeError(f"meta не найден: {meta_path}. Сначала запусти train_catboost.py --fit")
    meta = json.loads(meta_path.read_text())
    models: list[CatBoostRegressor] = []
    for i in range(meta["n_models"]):
        m = CatBoostRegressor()
        m.load_model(str(ARTIFACTS_DIR / f"catboost_seed{i}.cbm"))
        models.append(m)
    return models, meta


MODELS: list[CatBoostRegressor] = []
META: dict = {}


app = FastAPI(
    title="Delay predictor",
    version=MODEL_VERSION,
    description="Предсказание задержки прибытия ТС на остановку через 10-15 мин.",
)


@app.on_event("startup")
def _load_on_startup() -> None:
    global MODELS, META
    MODELS, META = _load_models()


# ------------------------------------------------------------------ утилиты


def _risk(delay: float) -> float:
    """Логистическая свёртка задержки в [0,1]. delay=RISK_MID_SEC → 0.5."""
    z = (delay - RISK_MID_SEC) / max(RISK_SLOPE_SEC, 1e-6)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _confidence(residuals: np.ndarray, scale_sec: float = 60.0) -> float:
    """1 − std(residuals)/scale, в [0, 1]."""
    if len(residuals) < 2:
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - residuals.std(ddof=0) / scale_sec)))


def _predict_one(req: PredictRequest) -> PredictResponse:
    sample = {
        "sample_id": req.sample_id,
        "tr_id": req.tr_id,
        "T": req.T,
        "target_stop_id": req.target_stop_id,
        "target_time_begin": req.target_time_begin,
        "cur_dev_s": req.cur_dev_s,
    }
    try:
        X = build_features_online(sample, [p.model_dump() for p in req.telemetry],
                                  [s.model_dump() for s in req.schedule])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"features: {e}") from e

    feats = META["features"]
    cats = [c for c in CAT_FEATURES if c in feats]
    pool = Pool(X[feats], cat_features=cats)
    residuals = np.array([m.predict(pool)[0] for m in MODELS], dtype=float)
    residual_mean = float(residuals.mean())
    delay = residual_mean + req.cur_dev_s

    top_names = ["cur_dev_s", "gps_med5", "n_trip_breaks", "spd15", "eta_dev_s"]
    top = []
    for name in top_names:
        if name in X.columns:
            val = X[name].iloc[0]
            if pd.notna(val):
                top.append(TopFeature(name=name, value=float(val)))

    return PredictResponse(
        sample_id=req.sample_id,
        delay_pred_sec=round(delay, 1),
        risk_score=round(_risk(delay), 3),
        confidence=round(_confidence(residuals), 3),
        top_features=top,
        model_version=MODEL_VERSION,
    )


# ------------------------------------------------------------------ endpoints


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if MODELS else "cold",
        n_models=len(MODELS),
        n_features=len(META.get("features", [])),
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    t0 = time.perf_counter()
    resp = _predict_one(req)
    latency_ms = (time.perf_counter() - t0) * 1000
    app.state.last_latency_ms = latency_ms
    return resp


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest) -> BatchResponse:
    if not batch.requests:
        return BatchResponse(responses=[])
    return BatchResponse(responses=[_predict_one(r) for r in batch.requests])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_service:app", host="0.0.0.0", port=8001)
