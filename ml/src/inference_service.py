"""FastAPI-инференс ML-модуля: POST /predict, /predict/batch, /whatif/predict; GET /health, /metrics/model,
/model/info; POST /reload.

Модель — ансамбль CatBoost из ``ml/artifacts/catboost_seed*.cbm`` (``train_catboost.py --fit``) плюс модели
неопределённости: ``catboost_quantiles.cbm`` (интервал 10–90%, конформно откалиброван на 80% покрытия) и
``catboost_classes.cbm`` (вероятности early / ontime / late). Если моделей неопределённости нет — сервис
работает на одном ансамбле, риск считается сигмоидой от задержки (как в контракте §7.1).

Онлайн-фичи строятся тем же кодом, что и в батче (``features/tabular.build_features`` через
``point_features``; быстрый путь без DataFrame, см. ``_features``) — онлайн-прогноз совпадает с ``submission.csv``.

Клиент шлёт **сырой контекст**: буфер телеметрии (пакеты с ``event_time <= T``; более поздние сервис
отбрасывает сам) и слайс планового расписания ТС. Если ``schedule`` не передан, берётся план ТС из
``SCHEDULE_PATH`` (если файл есть).

Запуск::

    uvicorn inference_service:app --app-dir ml/src --host 0.0.0.0 --port 8001   # Swagger: /docs
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explain import REC_CODE, explain, reason_pattern, recommendation, risk_level  # noqa: E402
from features.tabular import CAT_FEATURES, FEATURES, index_schedule, make_tele, point_features, to_sec  # noqa: E402

ARTIFACTS_DIR = Path(os.environ.get("ML_ARTIFACTS", Path(__file__).resolve().parents[1] / "artifacts"))
STATS_DIR = Path(os.environ.get("ML_STATS_DIR", Path(__file__).resolve().parents[2] / "statistics" / "tables"))
SCHEDULE_PATH = Path(os.environ.get("SCHEDULE_PATH",
                                    Path(__file__).resolve().parents[2] / "dataset" / "validate" / "schedule_plan.csv"))
MODEL_VERSION_ENV = os.environ.get("ML_MODEL_VERSION")
RISK_MID_SEC = float(os.environ.get("ML_RISK_MID_SEC", 120.0))
RISK_SLOPE_SEC = float(os.environ.get("ML_RISK_SLOPE_SEC", 60.0))
STALE_AFTER_S = 180  # нет свежих координат дольше — данные «устарели», уверенность снижаем
MAE_TARGET = 78.0    # оценка по условию «бейзлайн даёт score 0.40» (см. train_catboost.mae_target_estimate)

# What-if — эвристические сдвиги задержки (секунды), не выученные моделью; для дашборда это «оценка сценария».
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
    building_address: Optional[str] = None


class PredictRequest(BaseModel):
    sample_id: str
    tr_id: int
    T: str = Field(..., description="Момент прогноза, ISO-8601 (локальное время, как в датасете)")
    target_stop_id: int
    target_time_begin: str = Field(..., description="Плановое время целевой остановки, ISO-8601")
    cur_dev_s: float = Field(..., description="Задержка на последней уже пройденной остановке (сек)")
    telemetry: list[TelemetryPing] = Field(default_factory=list,
                                           description="Пинги NDTP; пакеты позже T сервис отбрасывает")
    schedule: list[ScheduleStop] = Field(default_factory=list,
                                         description="Плановое расписание ТС; пусто — из SCHEDULE_PATH")


class BatchRequest(BaseModel):
    requests: list[PredictRequest]


class TopFeature(BaseModel):
    name: str
    value: Optional[float] = Field(None, description="значение признака")
    contribution: float = Field(0.0, description="доля во вкладе в прогноз (|SHAP| / сумма |SHAP| топа), 0..1")
    contribution_sec: float = Field(0.0, description="вклад в прогноз задержки, сек (со знаком)")


class Cause(BaseModel):
    code: str
    text: str
    contribution_sec: float


class PredictResponse(BaseModel):
    sample_id: str
    delay_pred_sec: float = Field(..., description="прогноз задержки (факт − план), с; MAE считается по нему")
    risk_score: float = Field(..., description="P(опоздание > +120 с); светофор дашборда: ≥0.7 красный, ≥0.35 жёлтый")
    confidence: float
    top_features: list[TopFeature]
    reason_pattern: str
    recommendation: str = Field(..., description="код рекомендации для дашборда")
    model_version: str
    # --- расширение контракта
    lead_min: Optional[float] = None
    delay_interval_sec: list[float] = Field(default_factory=list, description="факт попадает сюда в ~80% случаев")
    p_early: Optional[float] = None
    p_ontime: Optional[float] = None
    p_late: Optional[float] = None
    risk_level: str = "green"
    causes: list[Cause] = Field(default_factory=list)
    recommendation_text: str = ""
    data_status: str = Field("live", description="live / stale / no_telemetry / fallback")
    latency_ms: float = 0.0


class BatchResponse(BaseModel):
    responses: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    n_models: int
    n_features: int
    model_version: str
    uncertainty_models: bool = False
    vehicles_in_schedule: int = 0


class MetricsResponse(BaseModel):
    mae_train_s: Optional[float] = None
    mae_test_s: Optional[float] = None
    mae_baseline_train_s: Optional[float] = None
    mae_baseline_test_s: Optional[float] = None
    score_estimate: Optional[float] = None
    latency_ms_p50: Optional[float] = None
    latency_ms_p95: Optional[float] = None
    requests_served: int = 0
    validation: dict = Field(default_factory=dict, description="holdout / proxy / LOVO из train_catboost --eval")
    uncertainty: dict = Field(default_factory=dict, description="покрытие интервала, AUC вероятностей")
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


# ------------------------------------------------------------------ модели и метрики


class State:
    """Всё, что сервис держит в памяти. ``load`` можно вызывать повторно (POST /reload)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.models: list[CatBoostRegressor] = []
        self.meta: dict = {}
        self.q: Optional[CatBoostRegressor] = None
        self.clf: Optional[CatBoostClassifier] = None
        self.unc: dict = {}
        self.metrics: dict = {}
        self.plan_by_tr: dict[int, list[dict]] = {}
        self.stops_by_tr: dict = {}          # разобранный план (features.tabular.Stops) — кеш на весь день
        self.latencies: deque = deque(maxlen=2000)
        self.n_requests = 0

    def load(self) -> None:
        meta_path = ARTIFACTS_DIR / "catboost_meta.json"
        if not meta_path.exists():
            raise RuntimeError(f"meta не найден: {meta_path}. Сначала запусти train_catboost.py --fit")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        models = [CatBoostRegressor().load_model(str(ARTIFACTS_DIR / f"catboost_seed{i}.cbm"))
                  for i in range(meta["n_models"])]
        q = clf = None
        unc: dict = {}
        if all((ARTIFACTS_DIR / f).exists() for f in ("catboost_quantiles.cbm", "catboost_classes.cbm",
                                                       "catboost_uncertainty.json")):
            q = CatBoostRegressor().load_model(str(ARTIFACTS_DIR / "catboost_quantiles.cbm"))
            clf = CatBoostClassifier().load_model(str(ARTIFACTS_DIR / "catboost_classes.cbm"))
            unc = json.loads((ARTIFACTS_DIR / "catboost_uncertainty.json").read_text())
        plan: dict[int, list[dict]] = {}
        if SCHEDULE_PATH.exists():
            s = pd.read_csv(SCHEDULE_PATH)
            cols = [c for c in ("tr_id", "tt_action_item_id", "time_begin", "geom", "manual_fill", "building_address")
                    if c in s]
            s = s[cols].astype(object).where(s[cols].notna(), None)
            plan = {int(k): g.to_dict("records") for k, g in s.groupby("tr_id")}
            stops = index_schedule(pd.read_csv(SCHEDULE_PATH, parse_dates=["time_begin"]))
        else:
            stops = {}
        with self.lock:
            self.models, self.meta, self.q, self.clf, self.unc, self.plan_by_tr = models, meta, q, clf, unc, plan
            self.stops_by_tr = stops
            self.metrics = _load_metrics()

    @property
    def version(self) -> str:
        return MODEL_VERSION_ENV or self.meta.get("model_version", "catboost-ensemble-v1")


STATE = State()


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
    except Exception:  # noqa: BLE001
        pass
    return out


def _load_metrics() -> dict:
    """Витринные метрики: свежие из ``train_catboost --eval`` поверх CSV из statistics/tables."""
    m: dict = {"validation": {}}
    model_mae = _read_mae_cohort_all(STATS_DIR / "model_metrics.csv", "mae_model_s")
    base_mae = _read_mae_cohort_all(STATS_DIR / "baseline_metrics.csv", "mae_zero_s")
    m.update(mae_train_s=model_mae["train"], mae_test_s=model_mae["test"],
             mae_baseline_train_s=base_mae["train"], mae_baseline_test_s=base_mae["test"], score_estimate=None)
    fresh = ARTIFACTS_DIR / "catboost_metrics.json"
    if fresh.exists():
        v = json.loads(fresh.read_text())
        m["validation"] = v
        m["mae_test_s"] = v.get("holdout_mae", m["mae_test_s"])
    mae, mae_zero = m["mae_test_s"], m["mae_baseline_test_s"]
    if mae is not None and mae_zero is not None and mae_zero > MAE_TARGET:
        m["score_estimate"] = float(max(0.0, min(1.0, (mae_zero - mae) / (mae_zero - MAE_TARGET))))
    return m


app = FastAPI(title="Delay predictor — ML module", version="1.1",
              description="Прогноз задержки ТС на остановке в горизонте 10–15 минут (CatBoost-ансамбль).")


@app.on_event("startup")
def _load_on_startup() -> None:
    STATE.load()


# ------------------------------------------------------------------ утилиты


def _sigmoid_risk(delay: float) -> float:
    """Риск из контракта §7.1 — запасной вариант, если нет классификатора. delay=RISK_MID_SEC → 0.5."""
    z = (delay - RISK_MID_SEC) / max(RISK_SLOPE_SEC, 1e-6)
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def _p_late_shifted(delay: float, lo: float, hi: float, delta: float) -> float:
    """P(задержка + delta > 120) при нормальном приближении по 80%-интервалу — для what-if."""
    sigma = max((hi - lo) / (2 * 1.2816), 1.0)
    return float(0.5 * math.erfc((120.0 - (delay + delta)) / (sigma * math.sqrt(2))))


def _naive(ts) -> pd.Timestamp:
    t = pd.to_datetime(ts)
    return t.tz_localize(None) if t.tzinfo is not None else t


def _features(req: PredictRequest) -> pd.DataFrame:
    """Признаки точки той же ``point_features``, что в батче (``tabular.build_features``).

    Быстрый путь: план ТС разобран один раз при старте (``STATE.stops_by_tr``), телеметрия собирается прямо
    в массивы с теми же правилами очистки (``make_tele``). ~10 мс вместо ~180 мс через DataFrame-адаптер
    ``from_stream.build_features_online`` (он остаётся для verify_streaming и как эталон).
    """
    T = _naive(req.T)
    t_sec = int(to_sec([T.to_datetime64()])[0])
    if req.schedule:
        df = pd.DataFrame([s.model_dump() for s in req.schedule])
        df["time_begin"] = pd.to_datetime(df["time_begin"], format="ISO8601")
        sg = index_schedule(df).get(req.tr_id)
    else:
        sg = STATE.stops_by_tr.get(req.tr_id)
    if sg is None:
        raise HTTPException(status_code=422, detail=f"нет планового расписания для ТС {req.tr_id}: передайте schedule")
    tele = None
    if req.telemetry:
        # один векторный разбор времени на весь буфер (поштучный pd.to_datetime — сотни мс на запрос)
        ev = pd.to_datetime(pd.Series([p.event_time for p in req.telemetry]), format="ISO8601")
        if ev.dt.tz is not None:
            ev = ev.dt.tz_localize(None)
        ts = (ev.values.astype("datetime64[us]").astype(np.int64)) / 1e6
        keep = ev.values <= T.to_datetime64()  # анти-утечка: только event_time <= T
        if keep.any():
            hist = np.array([int(bool(p.is_hist_data)) for p in req.telemetry])
            order = np.lexsort((hist, ts))  # как clean_traffic: по времени, при равенстве — не-исторический первым
            order = order[keep[order]]
            arr = lambda attr: np.array([np.nan if getattr(p, attr) is None else getattr(p, attr)
                                         for p in req.telemetry], dtype=float)[order]
            valid = np.array([str(p.location_valid).lower() == "true" for p in req.telemetry])[order]
            tele = make_tele(ts[order], arr("lon"), arr("lat"), arr("speed"), valid)
    tgt_plan = int(to_sec([_naive(req.target_time_begin).to_datetime64()])[0])
    f = point_features(t_sec, req.target_stop_id, tgt_plan, float(req.cur_dev_s), tele, sg)
    f["route"] = str(req.tr_id)
    X = pd.DataFrame([f], index=[req.sample_id])
    for c in FEATURES:
        if c not in X:
            X[c] = np.nan
    return X[FEATURES]


def _num(x) -> Optional[float]:
    if x is None or isinstance(x, str):
        return None
    return None if pd.isna(x) else float(x)


def _predict(reqs: list[PredictRequest]) -> list[PredictResponse]:
    t0 = time.perf_counter()
    X = pd.concat([_features(r) for r in reqs])
    feats = STATE.meta["features"]
    cats = [c for c in CAT_FEATURES if c in feats]
    pool = Pool(X[feats], cat_features=cats)
    cur = np.array([r.cur_dev_s for r in reqs], dtype=float)
    per_model = np.array([m.predict(pool) for m in STATE.models])          # (n_models, n)
    resid = per_model.mean(axis=0)
    delay = cur + resid

    if STATE.q is not None:
        qs = np.sort(STATE.q.predict(pool), axis=1) + cur[:, None]
        margin = STATE.unc.get("conformal_margin_s", 0.0)
        lo, hi = np.minimum(qs[:, 0] - margin, delay), np.maximum(qs[:, 2] + margin, delay)
        classes = STATE.meta.get("classes", ["early", "ontime", "late"])
        order = [list(STATE.clf.classes_).index(c) for c in classes]
        proba = STATE.clf.predict_proba(pool)[:, order]
    else:  # без моделей неопределённости: разброс сидов и сигмоида из контракта
        spread = per_model.std(axis=0)
        lo, hi = delay - 150 - 2 * spread, delay + 150 + 2 * spread
        p_late = np.array([_sigmoid_risk(d) for d in delay])
        proba = np.column_stack([np.zeros_like(p_late), 1 - p_late, p_late])

    # причины: приближённый SHAP одной модели (~35 мс/точка); база сдвигается к среднему ансамбля
    shap = STATE.models[0].get_feature_importance(data=pool, type="ShapValues", shap_calc_type="Approximate")
    shap[:, -1] += resid - shap.sum(axis=1)
    elapsed = (time.perf_counter() - t0) * 1000 / len(reqs)

    out = []
    for i, req in enumerate(reqs):
        f = X.iloc[i].to_dict()
        age = _num(f.get("last_fix_age_s"))
        status = "no_telemetry" if age is None else ("stale" if age > STALE_AFTER_S else "live")
        conf = float(np.clip(1 - (hi[i] - lo[i]) / 600, 0.05, 0.99)) * (1.0 if status == "live" else 0.5)
        d, pe, pl = float(delay[i]), float(proba[i, 0]), float(proba[i, 2])
        ex = explain(shap[i], feats, f, d)
        level = risk_level(d, pe, pl)
        reason = reason_pattern(level, d, ex["causes"], f)
        total = sum(abs(t["contribution"]) for t in ex["top_features"]) or 1.0
        top = [TopFeature(name=t["name"], value=_num(f.get(t["name"])),
                          contribution=round(abs(t["contribution"]) / total, 3), contribution_sec=t["contribution"])
               for t in ex["top_features"]]
        out.append(PredictResponse(
            sample_id=req.sample_id, delay_pred_sec=round(d, 1), risk_score=round(pl, 3), confidence=round(conf, 3),
            top_features=top, reason_pattern=reason, recommendation=REC_CODE.get(reason, "monitor"),
            model_version=STATE.version,
            lead_min=round((_naive(req.target_time_begin) - _naive(req.T)).total_seconds() / 60, 1),
            delay_interval_sec=[round(float(lo[i]), 1), round(float(hi[i]), 1)],
            p_early=round(pe, 3), p_ontime=round(float(proba[i, 1]), 3), p_late=round(pl, 3), risk_level=level,
            causes=[Cause(**c) for c in ex["causes"]], recommendation_text=recommendation(level, d, ex["causes"]),
            data_status=status, latency_ms=round(elapsed, 2)))
    with STATE.lock:
        STATE.latencies.extend([elapsed] * len(reqs))
        STATE.n_requests += len(reqs)
    return out


def _fallback(req: PredictRequest, err: Exception) -> PredictResponse:
    """Деградация: ошибка модели/фичей -> прогноз = cur_dev_s (бейзлайн), сервис отвечает, а не падает."""
    d = float(req.cur_dev_s)
    risk = _sigmoid_risk(d)
    level = risk_level(d, 0.0, risk)
    return PredictResponse(
        sample_id=req.sample_id, delay_pred_sec=round(d, 1), risk_score=round(risk, 3), confidence=0.05,
        top_features=[TopFeature(name="cur_dev_s", value=d, contribution=1.0, contribution_sec=d)],
        reason_pattern="accumulated_delay" if level != "green" else "on_track",
        recommendation="release_reserve" if level == "red" else "monitor", model_version="baseline-cur_dev",
        delay_interval_sec=[d - 150, d + 150], risk_level=level,
        causes=[Cause(code="fallback", text=f"модель недоступна ({type(err).__name__}): прогноз по последнему отклонению",
                      contribution_sec=d)],
        recommendation_text=recommendation(level, d, []), data_status="fallback")


# ------------------------------------------------------------------ endpoints


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok" if STATE.models else "cold", n_models=len(STATE.models),
                          n_features=len(STATE.meta.get("features", [])), model_version=STATE.version,
                          uncertainty_models=STATE.q is not None, vehicles_in_schedule=len(STATE.plan_by_tr))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        return _predict([req])[0]
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 — деградация вместо HTTP 500
        return _fallback(req, e)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest) -> BatchResponse:
    if not batch.requests:
        return BatchResponse(responses=[])
    try:
        return BatchResponse(responses=_predict(batch.requests))
    except Exception:  # noqa: BLE001 — одна плохая точка не валит пачку: считаем поштучно
        return BatchResponse(responses=[predict(r) for r in batch.requests])


@app.get("/metrics/model", response_model=MetricsResponse)
def metrics_model() -> MetricsResponse:
    """Витринные метрики: MAE по схемам валидации, live-latency, покрытие интервала."""
    lat = np.array(STATE.latencies) if STATE.latencies else None
    return MetricsResponse(
        mae_train_s=STATE.metrics.get("mae_train_s"), mae_test_s=STATE.metrics.get("mae_test_s"),
        mae_baseline_train_s=STATE.metrics.get("mae_baseline_train_s"),
        mae_baseline_test_s=STATE.metrics.get("mae_baseline_test_s"),
        score_estimate=STATE.metrics.get("score_estimate"),
        latency_ms_p50=None if lat is None else round(float(np.percentile(lat, 50)), 2),
        latency_ms_p95=None if lat is None else round(float(np.percentile(lat, 95)), 2),
        requests_served=STATE.n_requests, validation=STATE.metrics.get("validation", {}), uncertainty=STATE.unc,
        n_models=len(STATE.models), n_features=len(STATE.meta.get("features", [])), model_version=STATE.version)


@app.get("/model/info")
def model_info() -> dict:
    return {"model_version": STATE.version, "n_models": len(STATE.models), "target": STATE.meta.get("target"),
            "features": STATE.meta.get("features"), "params": STATE.meta.get("params"),
            "uncertainty": STATE.unc, "validation": STATE.metrics.get("validation", {})}


@app.post("/reload", response_model=HealthResponse)
def reload() -> HealthResponse:
    """Подхватить переобученные модели из ML_ARTIFACTS без перезапуска контейнера."""
    STATE.load()
    return health()


@app.post("/whatif/predict", response_model=WhatIfResponse)
def whatif_predict(req: WhatIfRequest) -> WhatIfResponse:
    """«Что если применить сценарий»: эвристический сдвиг задержки, риск пересчитывается по интервалу модели."""
    if req.scenario not in WHATIF_DELTA_MAP:
        raise HTTPException(status_code=422, detail=f"unknown scenario, allowed: {sorted(WHATIF_DELTA_MAP)}")
    base = predict(PredictRequest(**{k: v for k, v in req.model_dump().items() if k != "scenario"}))
    delta = WHATIF_DELTA_MAP[req.scenario]
    lo, hi = (base.delay_interval_sec or [base.delay_pred_sec - 150, base.delay_pred_sec + 150])[:2]
    risk_scn = _p_late_shifted(base.delay_pred_sec, lo, hi, delta)
    return WhatIfResponse(
        sample_id=base.sample_id, scenario=req.scenario, delay_baseline_sec=round(base.delay_pred_sec, 1),
        delay_scenario_sec=round(base.delay_pred_sec + delta, 1), delta_sec=round(delta, 1),
        risk_baseline=round(base.risk_score, 3), risk_scenario=round(risk_scn, 3),
        recommendation_still_applies=bool(risk_scn >= 0.5), model_version=base.model_version)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_service:app", host="0.0.0.0", port=8001)
