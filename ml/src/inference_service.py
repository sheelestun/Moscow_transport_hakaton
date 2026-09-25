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
STATS_DIR = Path(os.environ.get("ML_STATS_DIR", Path(__file__).resolve().parents[2] / "statistics" / "tables"))
MODEL_VERSION = os.environ.get("ML_MODEL_VERSION", "catboost-ensemble-v1")
RISK_MID_SEC = float(os.environ.get("ML_RISK_MID_SEC", 120.0))
RISK_SLOPE_SEC = float(os.environ.get("ML_RISK_SLOPE_SEC", 60.0))
MAE_TARGET = 78.0

WHATIF_DELTA_MAP: dict[str, float] = {
    "add_reserve": -60.0,
    "adjust_interval": -30.0,
    "detour": -90.0,
    "signal_priority": -45.0,
    "hold_at_stop": 30.0,
}


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
    reason_pattern: str
    recommendation: str
    model_version: str


class BatchResponse(BaseModel):
    responses: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    n_models: int
    n_features: int
    model_version: str


class MetricsResponse(BaseModel):
    mae_train_s: Optional[float] = None
    mae_test_s: Optional[float] = None
    mae_baseline_train_s: Optional[float] = None
    mae_baseline_test_s: Optional[float] = None
    score_estimate: Optional[float] = None
    latency_ms_p50: Optional[float] = None
    n_models: int
    n_features: int
    model_version: str


class WhatIfRequest(PredictRequest):
    scenario: str = Field(..., description=f"Один из: {sorted(WHATIF_DELTA_MAP)}")


class WhatIfResponse(BaseModel):
    sample_id: str
    scenario: str
    delay_baseline_sec: float
    delay_scenario_sec: float
    delta_sec: float
    risk_baseline: float
    risk_scenario: float
    recommendation_still_applies: bool
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


def _read_mae_cohort_all(csv_path: Path, mae_col: str) -> dict[str, Optional[float]]:
    """Читает `split,dimension,group,points,<mae_col>` и возвращает MAE для cohort=all."""
    out: dict[str, Optional[float]] = {"train": None, "test": None}
    if not csv_path.exists():
        return out
    try:
        df = pd.read_csv(csv_path)
        if mae_col not in df.columns:
            return out
        sel = df[(df["dimension"] == "cohort") & (df["group"] == "all")]
        for split in ("train", "test"):
            row = sel[sel["split"] == split]
            if not row.empty:
                out[split] = float(row[mae_col].iloc[0])
    except Exception:
        pass
    return out


def _load_metrics() -> dict:
    """Собирает витринные метрики из CSV/JSON. Отсутствующие источники → None-поля."""
    metrics: dict = {
        "mae_train_s": None,
        "mae_test_s": None,
        "mae_baseline_train_s": None,
        "mae_baseline_test_s": None,
        "score_estimate": None,
        "latency_ms_p50": None,
    }

    model_mae = _read_mae_cohort_all(STATS_DIR / "model_metrics.csv", "mae_model_s")
    metrics["mae_train_s"] = model_mae["train"]
    metrics["mae_test_s"] = model_mae["test"]

    baseline_mae = _read_mae_cohort_all(STATS_DIR / "baseline_metrics.csv", "mae_zero_s")
    metrics["mae_baseline_train_s"] = baseline_mae["train"]
    metrics["mae_baseline_test_s"] = baseline_mae["test"]

    bench_path = ARTIFACTS_DIR / "bench_latency.json"
    if bench_path.exists():
        try:
            bench = json.loads(bench_path.read_text())
            lat = bench.get("latency_ms_per_sample", {}) or {}
            val = lat.get("onnx_fp32")
            if val is None:
                val = lat.get("catboost")
            if val is not None:
                metrics["latency_ms_p50"] = float(val)
        except Exception:
            pass

    mae = metrics["mae_test_s"]
    mae_zero = metrics["mae_baseline_test_s"]
    if mae is not None and mae_zero is not None and mae_zero > MAE_TARGET:
        raw = (mae_zero - mae) / (mae_zero - MAE_TARGET)
        metrics["score_estimate"] = float(max(0.0, min(1.0, raw)))

    return metrics


MODELS: list[CatBoostRegressor] = []
META: dict = {}
METRICS: dict = {}


app = FastAPI(
    title="Delay predictor",
    version=MODEL_VERSION,
    description="Предсказание задержки прибытия ТС на остановку через 10-15 мин.",
)


@app.on_event("startup")
def _load_on_startup() -> None:
    global MODELS, META, METRICS
    MODELS, META = _load_models()
    METRICS = _load_metrics()


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


# Соответствие reason → recommendation зафиксировано в фронте (frontend/js/mock.js REC_FOR_REASON).
REASON_TO_RECOMMENDATION = {
    "traffic_jam_ahead": "detour",
    "long_dwell": "adjust_interval",
    "speed_drop": "signal_priority",
    "accumulated_delay": "release_reserve",
    "on_track": "monitor",
}


def _reason_pattern(delay_pred: float, x: pd.Series) -> str:
    """Rule-based метка причины задержки поверх фичей `tabular.py`.

    Правила порядковые: первая сработавшая — победила. Порядок:
      1. delay < 30 сек                              → on_track
      2. manual_fill (конечная) или cur_dev_s > 120  → accumulated_delay
      3. spd15 < 5 м/с и route_left_m > 500          → traffic_jam_ahead
      4. stop_p95 > 30 сек (долгие простои)          → long_dwell
      5. spd15 < 10 м/с                              → speed_drop
      6. fallback                                    → accumulated_delay

    bunching (сбой интервала) в правилах нет — у нас в фичах нет headway
    к соседним ТС, так что честнее его не выдавать.
    """
    def g(name: str, default: float = 0.0) -> float:
        v = x.get(name)
        return default if v is None or pd.isna(v) else float(v)

    if delay_pred < 30:
        return "on_track"
    if g("manual_fill") >= 0.5 or g("cur_dev_s") > 120:
        return "accumulated_delay"
    spd15, route_left = g("spd15"), g("route_left_m")
    if spd15 < 5 and route_left > 500:
        return "traffic_jam_ahead"
    if g("stop_p95") > 30:
        return "long_dwell"
    if spd15 < 10:
        return "speed_drop"
    return "accumulated_delay"


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

    reason = _reason_pattern(delay, X.iloc[0])
    return PredictResponse(
        sample_id=req.sample_id,
        delay_pred_sec=round(delay, 1),
        risk_score=round(_risk(delay), 3),
        confidence=round(_confidence(residuals), 3),
        top_features=top,
        reason_pattern=reason,
        recommendation=REASON_TO_RECOMMENDATION.get(reason, "monitor"),
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


@app.get("/metrics/model", response_model=MetricsResponse)
def metrics_model() -> MetricsResponse:
    """Витринные метрики модели: MAE, latency, score-оценка."""
    return MetricsResponse(
        mae_train_s=METRICS.get("mae_train_s"),
        mae_test_s=METRICS.get("mae_test_s"),
        mae_baseline_train_s=METRICS.get("mae_baseline_train_s"),
        mae_baseline_test_s=METRICS.get("mae_baseline_test_s"),
        score_estimate=METRICS.get("score_estimate"),
        latency_ms_p50=METRICS.get("latency_ms_p50"),
        n_models=len(MODELS),
        n_features=len(META.get("features", [])),
        model_version=MODEL_VERSION,
    )


@app.post("/whatif/predict", response_model=WhatIfResponse)
def whatif_predict(req: WhatIfRequest) -> WhatIfResponse:
    """Пересчёт delay для «что если бы применили сценарий» — rule-based модификатор."""
    if req.scenario not in WHATIF_DELTA_MAP:
        raise HTTPException(
            status_code=422,
            detail=f"unknown scenario, allowed: {sorted(WHATIF_DELTA_MAP)}",
        )
    base_req = PredictRequest(**{k: v for k, v in req.model_dump().items() if k != "scenario"})
    base = _predict_one(base_req)
    delta = WHATIF_DELTA_MAP[req.scenario]
    new_delay = base.delay_pred_sec + delta
    risk_scenario = _risk(new_delay)
    return WhatIfResponse(
        sample_id=base.sample_id,
        scenario=req.scenario,
        delay_baseline_sec=round(base.delay_pred_sec, 1),
        delay_scenario_sec=round(new_delay, 1),
        delta_sec=round(delta, 1),
        risk_baseline=round(base.risk_score, 3),
        risk_scenario=round(risk_scenario, 3),
        recommendation_still_applies=bool(risk_scenario >= 0.5),
        model_version=MODEL_VERSION,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_service:app", host="0.0.0.0", port=8001)
