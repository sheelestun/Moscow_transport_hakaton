"""ML-сервис прогноза задержек (FastAPI). Контракт — ARCHITECTURE_AND_ROLES.md §7.1, расширенный.

Backend присылает по ТС: момент ``T``, целевую остановку, ``cur_dev_s`` и телеметрию за последние ~90 минут
(``telemetry`` или ``features.history`` из контракта). Сервис считает признаки той же функцией
``features.tabular.point_features``, что и при обучении, поэтому онлайн-прогноз совпадает с батчем (сабмитом).

Ответ — готовая карточка для дашборда: задержка, интервал 10–90% (конформно откалиброван на 80% покрытия),
вероятности early/ontime/late, светофор, причины (SHAP), рекомендация, статус данных.

Запуск::

    uvicorn inference_service:app --app-dir ml/src --port 8001
    # Swagger: http://localhost:8001/docs

Переменные окружения: ``ML_ARTIFACTS`` (папка с моделями), ``SCHEDULE_PATH`` (плановое расписание CSV).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explain import explain, recommendation, risk_level  # noqa: E402
from features.tabular import Stops, index_schedule, make_tele, point_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ART = Path(os.environ.get("ML_ARTIFACTS", ROOT / "ml" / "artifacts"))
SCHEDULE_PATH = Path(os.environ.get("SCHEDULE_PATH", ROOT / "dataset" / "validate" / "schedule_plan.csv"))
STALE_AFTER_S = 180       # нет свежих координат дольше — данные «устарели», уверенность снижаем
EPOCH = datetime(1970, 1, 1)


# ----------------------------------------------------------------------------- схемы API


class TelemetryPoint(BaseModel):
    """Один навигационный пакет. Время — ``ts`` (абсолютное) или ``t`` (секунды относительно T, <= 0)."""
    t: Optional[float] = Field(None, description="секунды относительно T (отрицательные — в прошлом)")
    ts: Optional[datetime] = Field(None, description="абсолютное время пакета (локальное, как в датасете)")
    lat: Optional[float] = None
    lon: Optional[float] = None
    speed: Optional[float] = Field(None, description="км/ч")
    location_valid: bool = True


class PlannedStop(BaseModel):
    """Плановая остановка ТС — если сервису не загружено расписание этого ТС."""
    stop_id: Union[int, str]
    time_begin: datetime
    lat: float
    lon: float
    manual_fill: bool = False
    address: Optional[str] = None


class PredictRequest(BaseModel):
    sample_id: Optional[str] = None
    vehicle_id: Union[int, str]
    T: datetime = Field(..., description="момент прогноза; используется только телеметрия <= T")
    target_stop_id: Union[int, str]
    target_time_begin: datetime
    cur_dev_s: Optional[float] = Field(None, description="задержка на последней пройденной остановке, с")
    telemetry: list[TelemetryPoint] = Field(default_factory=list)
    planned_stops: Optional[list[PlannedStop]] = None
    features: Optional[dict[str, Any]] = Field(None, description="совместимость с §7.1: cur_dev_s, history")

    model_config = {"json_schema_extra": {"example": {
        "sample_id": "131672_1767670500", "vehicle_id": 131672, "T": "2026-01-06T03:35:00",
        "target_stop_id": 53700172828, "target_time_begin": "2026-01-06T03:50:00", "cur_dev_s": 274.0,
        "telemetry": [{"t": -30, "lat": 55.7512, "lon": 37.6101, "speed": 18.0},
                      {"t": -15, "lat": 55.7515, "lon": 37.6112, "speed": 21.0}]}}}


class Cause(BaseModel):
    code: str
    text: str
    contribution_sec: float


class TopFeature(BaseModel):
    name: str
    contribution: float


class StopRef(BaseModel):
    stop_id: Optional[Union[int, str]] = None
    plan_time: Optional[datetime] = None
    address: Optional[str] = None


class PredictResponse(BaseModel):
    sample_id: Optional[str]
    vehicle_id: Union[int, str]
    T: datetime
    lead_min: float = Field(..., description="минут от T до планового прибытия на целевую остановку")
    delay_pred_sec: float = Field(..., description="прогноз задержки (факт − план), с; MAE считается по нему")
    delay_interval_sec: list[float] = Field(..., description="интервал, в который факт попадает в ~80% случаев")
    p_early: float
    p_ontime: float
    p_late: float = Field(..., description="вероятность опоздания > +120 с")
    risk_level: str = Field(..., description="green / yellow / red")
    risk_score: float = Field(..., description="= p_late, для сортировки алертов")
    confidence: float
    causes: list[Cause]
    top_features: list[TopFeature]
    recommendation: str
    segment_from: StopRef
    target_stop: StopRef
    data_status: str = Field(..., description="live / stale / no_telemetry / fallback")
    model_version: str
    latency_ms: float


# ----------------------------------------------------------------------------- модели и состояние


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.loaded = False
        self.latencies: deque = deque(maxlen=2000)
        self.n_requests = 0

    def load(self) -> None:
        meta = json.loads((ART / "catboost_meta.json").read_text(encoding="utf-8"))
        regs = [CatBoostRegressor().load_model(str(ART / f"catboost_seed{i}.cbm")) for i in range(meta["n_models"])]
        q = CatBoostRegressor().load_model(str(ART / "catboost_quantiles.cbm"))
        clf = CatBoostClassifier().load_model(str(ART / "catboost_classes.cbm"))
        unc = json.loads((ART / "catboost_uncertainty.json").read_text())
        metrics_path = ART / "catboost_metrics.json"
        sched = pd.read_csv(SCHEDULE_PATH, parse_dates=["time_begin"])
        with self.lock:
            self.meta, self.regs, self.q, self.clf, self.unc = meta, regs, q, clf, unc
            self.metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
            self.feats, self.cats = meta["features"], meta["cat_features"]
            self.class_order = [list(clf.classes_).index(c) for c in meta.get("classes", ["early", "ontime", "late"])]
            self.stops = index_schedule(sched)
            self.address = dict(zip(sched["tt_action_item_id"], sched["building_address"].where(sched["building_address"].notna(), None)))
            self.loaded = True


STATE = State()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    STATE.load()
    yield


app = FastAPI(title="Delay predictor — ML module", version="1.0", lifespan=lifespan,
              description="Прогноз задержки ТС на остановке в горизонте 10–15 минут (CatBoost).")


# ----------------------------------------------------------------------------- подготовка признаков


def _sec(dt: datetime) -> int:
    """Секунды как ``features.tabular.to_sec``: время без часового пояса (как в датасете)."""
    return int(np.floor((dt.replace(tzinfo=None) - EPOCH).total_seconds()))


def _as_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return x


def _stops_for(req: PredictRequest) -> Stops:
    if req.planned_stops:
        df = pd.DataFrame([{"tt_action_item_id": _as_int(s.stop_id), "time_begin": s.time_begin.replace(tzinfo=None),
                            "geom": f"POINT ({s.lon} {s.lat})", "manual_fill": s.manual_fill,
                            "tr_id": _as_int(req.vehicle_id)} for s in req.planned_stops])
        return index_schedule(df)[_as_int(req.vehicle_id)]
    sg = STATE.stops.get(_as_int(req.vehicle_id))
    if sg is None:
        raise HTTPException(422, f"нет планового расписания для ТС {req.vehicle_id}: передайте planned_stops")
    return sg


def _prepare(req: PredictRequest) -> tuple[dict, float, Stops, int, bool]:
    T = _sec(req.T)
    feats_in = req.features or {}
    cur = req.cur_dev_s if req.cur_dev_s is not None else feats_in.get("cur_dev_s", 0.0)
    cur = float(cur) if cur is not None and not pd.isna(cur) else 0.0
    pts = list(req.telemetry) or [TelemetryPoint(**h) for h in feats_in.get("history", [])]
    rows = []
    for p in pts:
        ts = ((p.ts.replace(tzinfo=None) - EPOCH).total_seconds() if p.ts is not None
              else (T + p.t if p.t is not None else None))
        if ts is None or ts > T:  # анти-утечка: пакеты позже T отбрасываем
            continue
        rows.append((ts, np.nan if p.lon is None else p.lon, np.nan if p.lat is None else p.lat,
                     np.nan if p.speed is None else p.speed, p.location_valid))
    tele = make_tele(*zip(*rows)) if rows else None
    has_fix = tele is not None and bool(np.any(~np.isnan(tele.lat)))
    sg = _stops_for(req)
    f = point_features(T, _as_int(req.target_stop_id), _sec(req.target_time_begin), cur, tele, sg)
    f["route"] = str(_as_int(req.vehicle_id))
    return f, cur, sg, T, has_fix


def _predict(reqs: list[PredictRequest]) -> list[PredictResponse]:
    t0 = time.perf_counter()
    prepared = [_prepare(r) for r in reqs]
    X = pd.DataFrame([p[0] for p in prepared])
    for c in STATE.feats:
        if c not in X:
            X[c] = np.nan
    X = X[STATE.feats]
    cur = np.array([p[1] for p in prepared])
    pool = Pool(X, cat_features=STATE.cats)
    resid = np.mean([m.predict(pool) for m in STATE.regs], axis=0)
    delay = cur + resid
    qs = np.sort(STATE.q.predict(pool), axis=1) + cur[:, None]
    margin = STATE.unc["conformal_margin_s"]
    lo, hi = np.minimum(qs[:, 0] - margin, delay), np.maximum(qs[:, 2] + margin, delay)
    proba = STATE.clf.predict_proba(pool)[:, STATE.class_order]
    # причины: приближённый SHAP одной модели ансамбля (~35 мс на точку вместо ~130 на каждую из 5 моделей);
    # сумма вкладов сдвигается к среднему ансамбля, чтобы причины объясняли именно выданный прогноз
    shap = STATE.regs[0].get_feature_importance(data=pool, type="ShapValues", shap_calc_type="Approximate")
    shap[:, -1] += resid - shap.sum(axis=1)
    elapsed = (time.perf_counter() - t0) * 1000 / len(reqs)

    out = []
    for i, (req, (f, _, sg, T, has_fix)) in enumerate(zip(reqs, prepared)):
        age = f.get("last_fix_age_s", np.nan)
        status = "no_telemetry" if not has_fix else ("stale" if (np.isnan(age) or age > STALE_AFTER_S) else "live")
        conf = float(np.clip(1 - (hi[i] - lo[i]) / 600, 0.05, 0.99)) * (1.0 if status == "live" else 0.5)
        ex = explain(shap[i], STATE.feats, f, float(delay[i]))
        level = risk_level(float(delay[i]), float(proba[i, 0]), float(proba[i, 2]))
        before = np.where(sg.plan <= T)[0]
        from_id = sg.id[before[-1]] if len(before) else None
        tgt = _as_int(req.target_stop_id)
        out.append(PredictResponse(
            sample_id=req.sample_id, vehicle_id=req.vehicle_id, T=req.T,
            lead_min=round((_sec(req.target_time_begin) - T) / 60, 1),
            delay_pred_sec=round(float(delay[i]), 1),
            delay_interval_sec=[round(float(lo[i]), 1), round(float(hi[i]), 1)],
            p_early=round(float(proba[i, 0]), 3), p_ontime=round(float(proba[i, 1]), 3),
            p_late=round(float(proba[i, 2]), 3), risk_level=level, risk_score=round(float(proba[i, 2]), 3),
            confidence=round(conf, 3), causes=ex["causes"], top_features=ex["top_features"],
            recommendation=recommendation(level, float(delay[i]), ex["causes"]),
            segment_from=StopRef(stop_id=_as_int(from_id) if from_id is not None else None,
                                 plan_time=EPOCH + timedelta(seconds=int(sg.plan[before[-1]])) if len(before) else None,
                                 address=STATE.address.get(from_id)),
            target_stop=StopRef(stop_id=tgt, plan_time=req.target_time_begin, address=STATE.address.get(tgt)),
            data_status=status, model_version=STATE.meta.get("model_version", "cb-tab"),
            latency_ms=round(elapsed, 2)))
    with STATE.lock:
        STATE.latencies.extend([elapsed] * len(reqs))
        STATE.n_requests += len(reqs)
    return out


def _fallback(req: PredictRequest, err: Exception) -> PredictResponse:
    """Деградация: модель недоступна / ошибка признаков -> прогноз = cur_dev_s (бейзлайн), сервис не падает."""
    cur = req.cur_dev_s if req.cur_dev_s is not None else (req.features or {}).get("cur_dev_s", 0.0) or 0.0
    level = risk_level(float(cur), 0.0, 1.0 if cur >= 120 else 0.0)
    return PredictResponse(
        sample_id=req.sample_id, vehicle_id=req.vehicle_id, T=req.T,
        lead_min=round((_sec(req.target_time_begin) - _sec(req.T)) / 60, 1), delay_pred_sec=float(cur),
        delay_interval_sec=[float(cur) - 150, float(cur) + 150], p_early=0.0, p_ontime=0.0, p_late=0.0,
        risk_level=level, risk_score=0.0, confidence=0.05,
        causes=[Cause(code="fallback", text=f"модель недоступна ({type(err).__name__}); прогноз по последнему отклонению",
                      contribution_sec=float(cur))],
        top_features=[], recommendation=recommendation(level, float(cur), []),
        segment_from=StopRef(), target_stop=StopRef(stop_id=_as_int(req.target_stop_id), plan_time=req.target_time_begin),
        data_status="fallback", model_version="baseline-cur_dev", latency_ms=0.0)


# ----------------------------------------------------------------------------- эндпоинты


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if STATE.loaded else "loading", "models_loaded": STATE.loaded,
            "vehicles_in_schedule": len(STATE.stops) if STATE.loaded else 0,
            "model_version": STATE.meta.get("model_version") if STATE.loaded else None}


@app.get("/model/info")
def model_info() -> dict:
    return {"model_version": STATE.meta.get("model_version"), "n_regressors": len(STATE.regs),
            "target": STATE.meta.get("target"), "features": STATE.feats, "params": STATE.meta.get("params"),
            "validation": STATE.metrics, "uncertainty": STATE.unc}


@app.get("/metrics")
def metrics() -> dict:
    lat = np.array(STATE.latencies) if STATE.latencies else np.array([np.nan])
    return {"requests": STATE.n_requests, "latency_ms_p50": round(float(np.nanpercentile(lat, 50)), 2),
            "latency_ms_p95": round(float(np.nanpercentile(lat, 95)), 2), "validation": STATE.metrics}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        return _predict([req])[0]
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 — деградация вместо 500
        return _fallback(req, e)


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(reqs: list[PredictRequest]) -> list[PredictResponse]:
    try:
        return _predict(reqs)
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 — одна плохая точка не валит пачку
        return [predict(r) for r in reqs]


@app.post("/reload")
def reload() -> dict:
    """Подхватить переобученные модели из ML_ARTIFACTS без перезапуска контейнера."""
    STATE.load()
    return health()
