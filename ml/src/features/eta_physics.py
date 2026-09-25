"""«Физическая» модель проезда: за сколько секунд ТС проедет следующие D метров (без расписания).

Обучается на GPS-треках: в якорный момент t берём признаки движения **до t** (скорость, простои, время суток)
и дистанцию D; ответ — через сколько секунд путь ТС по его же треку после t превысит D. Будущее трека
используется только как разметка при обучении, признаки — строго прошлое.

Чтобы не было утечки через «тот же автобус в тот же день», модели обучаются **перекрёстно по машинам**
(cross-fitting): прогноз для ТС V делает модель, обученная на треках **других** ТС (V и его клоны исключены).
В основную модель идёт признак ``eta_phys_dev_s`` = (T + прогноз проезда оставшегося пути) − план прибытия.

Машины без разметки в датасете почти все стоят в парке (средняя скорость ~0), поэтому их одних для обучения
физики мало — они просто добавляются в общий пул треков.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .tabular import Tele, dist_m

ANCHOR_STEP_S = 60                      # якорь каждую минуту трека
DISTANCES_M = (500, 1000, 2000, 3000, 5000)
MAX_HORIZON_S = 3600                    # не ищем дальше часа вперёд
ETA_FEATURES = ["dist_m", "spd1", "spd5", "spd15", "stop5", "stop15", "moving_spd15", "hour_sin", "hour_cos"]
ETA_PARAMS = dict(iterations=400, learning_rate=0.08, depth=6, loss_function="RMSE", verbose=0,
                  allow_writing_files=False, thread_count=-1)


def motion_features(tg: Tele, t: int) -> dict:
    """Признаки движения на момент t — те же определения, что spd*/stop* в tabular.point_features."""
    b = np.searchsorted(tg.ts, t, side="right")
    f = {}
    for w in (1, 5, 15):
        a = np.searchsorted(tg.ts, t - 60 * w)
        sp = tg.speed[a:b]
        sp = sp[~np.isnan(sp)]
        f[f"spd{w}"] = sp.mean() if len(sp) else np.nan
        if w in (5, 15):
            f[f"stop{w}"] = (sp < 3).mean() if len(sp) else np.nan
        if w == 15:
            mv = sp[sp >= 3]
            f["moving_spd15"] = mv.mean() if len(mv) else np.nan
    hh = (t % 86400) / 3600.0
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hh / 24), np.cos(2 * np.pi * hh / 24)
    return f


def track_samples(tg: Tele) -> pd.DataFrame:
    """Обучающие пары (признаки на t, D) -> секунды до прохождения D метров по треку после t."""
    ok = ~np.isnan(tg.lat)
    ts, lon, lat = tg.ts[ok], tg.lon[ok], tg.lat[ok]
    if len(ts) < 10:
        return pd.DataFrame()
    step = dist_m(lon[1:], lat[1:], lon[:-1], lat[:-1])
    dt = np.diff(ts)
    step[(dt > 300) | (step / np.maximum(dt, 1) > 30)] = 0.0  # разрывы связи и GPS-скачки (> 108 км/ч) не считаем путём
    cum = np.r_[0.0, np.cumsum(step)]
    rows = []
    for t in range(int(ts[0]) + 900, int(ts[-1]), ANCHOR_STEP_S):
        i = np.searchsorted(ts, t, side="right") - 1
        if i < 0 or t - ts[i] > 60:       # на t нет свежей координаты
            continue
        f = motion_features(tg, t)
        if np.isnan(f["spd15"]):
            continue
        for D in DISTANCES_M:
            j = np.searchsorted(cum, cum[i] + D)
            if j >= len(ts) or ts[j] - t > MAX_HORIZON_S:
                continue
            rows.append({**f, "dist_m": float(D), "y_s": float(ts[j] - t)})
    return pd.DataFrame(rows)


def fit_eta(samples: pd.DataFrame, seed: int = 0) -> CatBoostRegressor:
    """Учим log(время проезда): ошибка в процентах, а не в секундах, — одинаково важны 1 и 5 км."""
    m = CatBoostRegressor(**ETA_PARAMS, random_seed=seed)
    m.fit(samples[ETA_FEATURES], np.log(samples["y_s"].clip(lower=5)))
    return m


def predict_eta(model: CatBoostRegressor, X: pd.DataFrame) -> np.ndarray:
    return np.exp(model.predict(X[ETA_FEATURES]))


def eta_features_for_points(model: CatBoostRegressor, X: pd.DataFrame) -> pd.DataFrame:
    """Признаки для основной модели из уже посчитанных признаков точки (``tabular.point_features``).

    ``eta_phys_s`` — прогноз времени проезда оставшегося пути ``route_left_m``;
    ``eta_phys_dev_s`` — во сколько относительно плана ТС приедет при такой «физике» (lead_s = план − T).
    """
    Z = X.rename(columns={"route_left_m": "dist_m"})
    out = pd.DataFrame(index=X.index)
    has = Z["dist_m"].notna() & Z["spd15"].notna()
    eta = np.full(len(X), np.nan)
    if has.any():
        eta[has.to_numpy()] = predict_eta(model, Z.loc[has].assign(dist_m=Z.loc[has, "dist_m"].clip(50, 8000)))
    out["eta_phys_s"] = eta
    out["eta_phys_dev_s"] = eta - X["lead_s"].to_numpy()
    return out
