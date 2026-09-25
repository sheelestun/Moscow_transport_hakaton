"""Табличные фичи для CatBoost-трека (батч и онлайн — одна функция на точку).

Точка прогноза — (tr_id, T). Все признаки строятся строго по данным, доступным на момент T:

* телеметрия с ``event_time <= T`` (своя), очищенная ``clean_traffic``;
* **плановое** расписание (план известен заранее; факт ``time_fact_begin`` не используется вообще —
  в validate и на живом потоке его нет);
* подсказка ``cur_dev_s``.

Ключевая идея — **фактические прибытия на пройденные остановки восстанавливаются из GPS**
(``gps_history``): ТС ближе всего к точке остановки, потом отъехало. Так получаем честную историю
отклонений на момент T, одинаковую для train / test / validate / NDTP-потока.

Вторая идея — **рейсы**. Разрыв планового расписания > ``TRIP_GAP_S`` = конечная. Внутри рейса задержка
соседних остановок коррелирует на ~0.96, через конечную — на ~0.02. Поэтому признаки «сколько конечных
между T и целью» и «где цель внутри рейса» решают, можно ли доверять текущему отклонению.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRIP_GAP_S = 300          # плановый разрыв между остановками больше этого — конечная / новый рейс
ARR_RADIUS_M = 60         # «на остановке», м
LEAVE_M = 30              # остановка считается пройденной, если ТС отъехало на столько от минимума
WIN_EARLY_S = 420         # окно поиска прибытия: от −7 мин ...
WIN_LATE_S = 720          # ... до +12 мин от планового времени
HIST_BACK_S = 3600        # на сколько назад смотрим историю прибытий

CAT_FEATURES = ["route"]

FEATURES = [
    # подсказка
    "cur_dev_s",
    # геометрия прогноза по плану
    "lead_s", "n_stops_between", "plan_gap_prev_s", "plan_gap_next_s", "tgt_plan_gap_prev_s",
    "plan_route_dist_m", "plan_speed_kmh",
    # ручное заполнение факта
    "tgt_manual_fill", "prev_manual_fill", "mf_share_next", "mf_share_trip",
    # рейсы и конечные
    "n_trip_breaks", "tgt_first_in_trip", "tgt_idx_in_trip", "tgt_left_in_trip",
    "cur_idx_in_trip", "break_gap_s", "plan_gap_break_to_tgt_s",
    # история отклонений по GPS
    "gps_n", "gps_last_dev", "gps_med3", "gps_med5", "gps_max5", "gps_min5", "gps_slope",
    "gps_age_s", "gps_last_dwell_s", "gps_dev_minus_cur", "overdue_s", "overdue_n",
    "gps_n_trip", "gps_trip_first_dev",
    # движение
    "spd1", "spd5", "spd15", "spd_std5", "stop5", "stop15", "spd_trend", "moving_spd15",
    "disp5_m", "disp15_m", "n_pkt15", "last_fix_age_s", "last_speed",
    # положение и «физический» прогноз
    "dist_tgt_m", "dist_next_stop_m", "route_left_m", "eta_dev_s", "eta_dev_moving_s", "req_speed_kmh",
    # время
    "hour", "hour_sin", "hour_cos",
    # категориальные
    "route",
]


# ----------------------------------------------------------------------------- загрузка и очистка


def load_split(dataset_dir: Path | str, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Точки, телеметрия и **плановое** расписание сплита (факт расписания отбрасывается)."""
    d = Path(dataset_dir)
    if split == "validate":
        points = pd.read_csv(d / "validate" / "points.csv")
        traffic = pd.read_csv(d / "validate" / "traffic.csv", low_memory=False)
        schedule = pd.read_csv(d / "validate" / "schedule_plan.csv")
    elif split in ("train", "test"):
        points = pd.read_csv(d / "labels" / f"labels_{split}.csv")
        traffic = pd.read_csv(d / split / "traffic.csv", low_memory=False)
        schedule = pd.read_csv(d / split / "schedule.csv")
    else:
        raise ValueError(split)
    schedule = schedule.drop(columns=[c for c in ("time_fact_begin",) if c in schedule])  # анти-утечка
    points["T"] = pd.to_datetime(points["T"])
    points["target_time_begin"] = pd.to_datetime(points["target_time_begin"])
    traffic["event_time"] = pd.to_datetime(traffic["event_time"])
    schedule["time_begin"] = pd.to_datetime(schedule["time_begin"])
    return points, traffic, schedule


def clean_traffic(t: pd.DataFrame) -> pd.DataFrame:
    """Невалидные навигационные ячейки NDTP -> NaN; служебные значения speed/alt -> NaN; дубли пакетов убираем."""
    t = t.copy()
    valid = t["location_valid"].astype(str).str.lower().eq("true")
    bad_xy = ~valid | t["lat"].isna() | (t["lat"].abs() < 1) | (t["lon"].abs() < 1)
    t.loc[bad_xy, ["lon", "lat"]] = np.nan
    t.loc[t["speed"] > 120, "speed"] = np.nan
    t = t.sort_values(["tr_id", "event_time", "is_hist_data"])
    t = t.drop_duplicates(["tr_id", "event_time"], keep="first")
    return t.reset_index(drop=True)


def to_sec(values) -> np.ndarray:
    return np.asarray(values).astype("datetime64[s]").astype(np.int64)


def tele_sec(values) -> np.ndarray:
    """Время пакета в целых секундах с округлением **вверх**: пакет 03:35:00.5 позже T = 03:35:00 и в прогноз
    на этот момент не попадает (строгое ``event_time <= T``)."""
    us = np.asarray(values).astype("datetime64[us]").astype(np.int64)
    return -((-us) // 1_000_000)


def dist_m(lon1, lat1, lon2, lat2):
    k = np.pi / 180
    return 6371000.0 * np.hypot((np.asarray(lon1) - lon2) * k * np.cos(lat2 * k), (np.asarray(lat1) - lat2) * k)


# ----------------------------------------------------------------------------- индексы по ТС


@dataclass
class Tele:
    ts: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    speed: np.ndarray


@dataclass
class Stops:
    id: np.ndarray
    plan: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    mf: np.ndarray
    trip: np.ndarray        # номер рейса
    idx_in_trip: np.ndarray
    trip_len: np.ndarray
    gap_prev: np.ndarray    # плановый интервал до предыдущей остановки (с)
    cum_dist: np.ndarray    # накопленное расстояние по прямой между остановками (м)


def make_tele(ts, lon, lat, speed, valid=None) -> Tele:
    """Телеметрия одного ТС из массивов (онлайн-путь). Те же правила очистки, что в ``clean_traffic``."""
    ts = np.asarray(ts, dtype=float)  # секунды с дробной частью; дубли — только точно совпадающие метки
    lon, lat = np.asarray(lon, float).copy(), np.asarray(lat, float).copy()
    speed = np.asarray(speed, float).copy()
    valid = np.ones(len(ts), bool) if valid is None else np.asarray(valid, bool)
    bad = ~valid | np.isnan(lat) | (np.abs(lat) < 1) | (np.abs(lon) < 1)
    lon[bad], lat[bad] = np.nan, np.nan
    speed[speed > 120] = np.nan
    order = np.argsort(ts, kind="stable")
    ts, lon, lat, speed = ts[order], lon[order], lat[order], speed[order]
    keep = np.r_[True, np.diff(ts) != 0]
    return Tele(np.ceil(ts[keep]).astype(np.int64), lon[keep], lat[keep], speed[keep])


def index_traffic(traffic: pd.DataFrame) -> dict[int, Tele]:
    t = clean_traffic(traffic)
    return {int(k): Tele(tele_sec(g["event_time"]), g["lon"].to_numpy(float), g["lat"].to_numpy(float),
                         g["speed"].to_numpy(float)) for k, g in t.groupby("tr_id")}


def index_schedule(schedule: pd.DataFrame) -> dict[int, Stops]:
    s = schedule.sort_values(["tr_id", "time_begin"]).copy()
    xy = s["geom"].str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)").astype(float)
    s["lon"], s["lat"] = xy[0].to_numpy(), xy[1].to_numpy()
    out = {}
    for k, g in s.groupby("tr_id"):
        plan = to_sec(g["time_begin"])
        gap = np.diff(plan, prepend=plan[0] - 10**6)
        trip = np.cumsum(gap > TRIP_GAP_S)
        idx = g.assign(_t=trip).groupby("_t").cumcount().to_numpy()
        tlen = pd.Series(trip).map(pd.Series(trip).value_counts()).to_numpy()
        lon, lat = g["lon"].to_numpy(), g["lat"].to_numpy()
        step = np.r_[0.0, dist_m(lon[1:], lat[1:], lon[:-1], lat[:-1])]
        step[gap > TRIP_GAP_S] = 0.0
        out[int(k)] = Stops(g["tt_action_item_id"].to_numpy(), plan, lon, lat,
                            g["manual_fill"].astype(str).str.lower().eq("true").to_numpy(int),
                            trip, idx, tlen, np.where(gap > 10**5, np.nan, gap), np.cumsum(step))
    return out


# ----------------------------------------------------------------------------- GPS-история


def gps_history(tg: Tele, sg: Stops, T: int) -> np.ndarray:
    """Прибытия на пройденные к моменту T «автоматические» остановки.

    Идём по остановкам в порядке маршрута; прибытие = минимум расстояния (< ARR_RADIUS_M) в окне
    [plan − WIN_EARLY_S, plan + WIN_LATE_S] ∩ (prev_arrival, T], после которого ТС отъехало на LEAVE_M.
    Возвращает массив строк (stop_pos, plan, delay, arrival, dwell).
    """
    cand = np.where((sg.plan >= T - HIST_BACK_S) & (sg.plan <= T + 300) & (sg.mf == 0))[0]
    rows, prev = [], -np.inf
    for i in cand:
        lo, hi = max(sg.plan[i] - WIN_EARLY_S, prev + 1), min(sg.plan[i] + WIN_LATE_S, T)
        a, b = np.searchsorted(tg.ts, lo), np.searchsorted(tg.ts, hi, side="right")
        if b - a < 2:
            continue
        d = dist_m(tg.lon[a:b], tg.lat[a:b], sg.lon[i], sg.lat[i])
        if np.all(np.isnan(d)):
            continue
        j = int(np.nanargmin(d))
        after = d[j + 1:]
        after = after[~np.isnan(after)]
        if d[j] < ARR_RADIUS_M and len(after) and after.max() > d[j] + LEAVE_M:
            near = np.where(d < ARR_RADIUS_M)[0]
            dwell = float(tg.ts[a + near.max()] - tg.ts[a + near.min()]) if len(near) else 0.0
            prev = tg.ts[a + j]
            rows.append((i, sg.plan[i], prev - sg.plan[i], prev, dwell))
    return np.array(rows, dtype=float).reshape(-1, 5)


# ----------------------------------------------------------------------------- фичи одной точки


def point_features(T: int, tgt_id, tgt_plan: int, cur_dev: float, tg: Tele | None, sg: Stops) -> dict:
    """Все признаки одной прогнозной точки. Используется и в батче, и в онлайн-сервисе."""
    f: dict = {"cur_dev_s": cur_dev, "lead_s": tgt_plan - T}
    hh = (T % 86400) / 3600.0  # время в датасете локальное (без tz), так что это час по Москве
    f.update(hour=hh, hour_sin=np.sin(2 * np.pi * hh / 24), hour_cos=np.cos(2 * np.pi * hh / 24))

    k = np.where(sg.id == tgt_id)[0]
    tk = int(k[0]) if len(k) else int(np.searchsorted(sg.plan, tgt_plan))
    tk = min(tk, len(sg.plan) - 1)
    before = np.where(sg.plan <= T)[0]
    ck = int(before[-1]) if len(before) else None           # последняя остановка с планом <= T
    nk = ck + 1 if ck is not None else 0                     # следующая по плану
    between = np.arange(nk, tk)

    f["n_stops_between"] = len(between)
    f["plan_gap_prev_s"] = T - sg.plan[ck] if ck is not None else np.nan
    f["plan_gap_next_s"] = sg.plan[nk] - T if nk < len(sg.plan) else np.nan
    f["tgt_plan_gap_prev_s"] = sg.gap_prev[tk]
    f["tgt_manual_fill"] = sg.mf[tk]
    f["prev_manual_fill"] = sg.mf[ck] if ck is not None else np.nan
    f["mf_share_next"] = sg.mf[nk:tk + 1].mean() if tk >= nk else np.nan
    f["mf_share_trip"] = sg.mf[sg.trip == sg.trip[tk]].mean()

    cur_trip = sg.trip[ck] if ck is not None else sg.trip[0] - 1
    f["n_trip_breaks"] = int(sg.trip[tk] - cur_trip)
    f["tgt_first_in_trip"] = int(sg.idx_in_trip[tk] == 0)
    f["tgt_idx_in_trip"] = sg.idx_in_trip[tk]
    f["tgt_left_in_trip"] = sg.trip_len[tk] - sg.idx_in_trip[tk] - 1
    f["cur_idx_in_trip"] = sg.idx_in_trip[ck] if ck is not None else np.nan
    if f["n_trip_breaks"] > 0:
        first_new = int(np.where(sg.trip == sg.trip[tk])[0][0])
        f["break_gap_s"] = sg.gap_prev[first_new]
        f["plan_gap_break_to_tgt_s"] = tgt_plan - sg.plan[first_new]
    else:
        f["break_gap_s"] = 0.0
        f["plan_gap_break_to_tgt_s"] = np.nan
    same_trip = sg.trip[tk] == cur_trip
    route_plan = sg.cum_dist[tk] - (sg.cum_dist[ck] if ck is not None and same_trip else sg.cum_dist[np.where(sg.trip == sg.trip[tk])[0][0]])
    f["plan_route_dist_m"] = route_plan
    f["plan_speed_kmh"] = route_plan / max(tgt_plan - (sg.plan[ck] if ck is not None else T), 60) * 3.6

    # ---- GPS-история
    h = gps_history(tg, sg, T) if tg is not None else np.empty((0, 5))
    d = h[:, 2]
    n = len(d)
    f.update(gps_n=n,
             gps_last_dev=d[-1] if n else np.nan,
             gps_med3=np.median(d[-3:]) if n else np.nan,
             gps_med5=np.median(d[-5:]) if n else np.nan,
             gps_max5=d[-5:].max() if n else np.nan,
             gps_min5=d[-5:].min() if n else np.nan,
             gps_slope=np.polyfit(h[-5:, 1] / 60, d[-5:], 1)[0] if n >= 3 else np.nan,
             gps_age_s=T - h[-1, 3] if n else np.nan,
             gps_last_dwell_s=h[-1, 4] if n else np.nan)
    f["gps_dev_minus_cur"] = f["gps_med3"] - cur_dev if n else np.nan
    in_trip = h[sg.trip[h[:, 0].astype(int)] == cur_trip] if n else h
    f["gps_n_trip"] = len(in_trip)
    f["gps_trip_first_dev"] = in_trip[0, 2] if len(in_trip) else np.nan

    # «просрочка»: остановки после последнего подтверждённого прибытия, чей план уже прошёл, а ТС их не прошло
    last_pos = int(h[-1, 0]) if n else (ck if ck is not None else -1)
    overdue = np.where((np.arange(len(sg.plan)) > last_pos) & (sg.plan <= T) & (sg.plan >= T - HIST_BACK_S)
                       & (sg.trip == cur_trip))[0]
    f["overdue_n"] = len(overdue)
    f["overdue_s"] = T - sg.plan[overdue[0]] if len(overdue) else 0.0

    # ---- движение и положение
    lead = max(tgt_plan - T, 60)
    if tg is not None:
        b = np.searchsorted(tg.ts, T, side="right")
        for w in (1, 5, 15):
            a = np.searchsorted(tg.ts, T - 60 * w)
            sp = tg.speed[a:b]
            sp = sp[~np.isnan(sp)]
            f[f"spd{w}"] = sp.mean() if len(sp) else np.nan
            if w == 5:
                f["spd_std5"] = sp.std() if len(sp) > 1 else np.nan
                f["stop5"] = (sp < 3).mean() if len(sp) else np.nan
            if w == 15:
                f["stop15"] = (sp < 3).mean() if len(sp) else np.nan
                f["n_pkt15"] = b - a
                mv = sp[sp >= 3]
                f["moving_spd15"] = mv.mean() if len(mv) else np.nan
            if w in (5, 15):
                seg = np.where(~np.isnan(tg.lat[a:b]))[0]
                f[f"disp{w}_m"] = (dist_m(tg.lon[a + seg[-1]], tg.lat[a + seg[-1]], tg.lon[a + seg[0]], tg.lat[a + seg[0]])
                                   if len(seg) > 1 else np.nan)
        f["spd_trend"] = f["spd5"] - f["spd15"] if not np.isnan(f.get("spd5", np.nan)) else np.nan
        sp_last = tg.speed[:b][~np.isnan(tg.speed[:b])]
        f["last_speed"] = sp_last[-1] if len(sp_last) else np.nan
        valid = np.where(~np.isnan(tg.lat[:b]))[0]
        if len(valid):
            j = valid[-1]
            f["last_fix_age_s"] = T - tg.ts[j]
            lon0, lat0 = tg.lon[j], tg.lat[j]
            f["dist_tgt_m"] = dist_m(lon0, lat0, sg.lon[tk], sg.lat[tk])
            # оставшийся путь по остановкам: до ближайшей непройденной + по плану до цели
            nxt = last_pos + 1 if last_pos + 1 <= tk else tk
            f["dist_next_stop_m"] = dist_m(lon0, lat0, sg.lon[nxt], sg.lat[nxt])
            route_left = f["dist_next_stop_m"] + (sg.cum_dist[tk] - sg.cum_dist[nxt])
            f["route_left_m"] = route_left
            eff = f.get("spd15", np.nan)
            f["eta_dev_s"] = route_left / (max(eff, 3) / 3.6) - (tgt_plan - T) if not np.isnan(eff) else np.nan
            mv = f.get("moving_spd15", np.nan)
            n_left = max(tk - nxt + 1, 1)
            dwell = f["gps_last_dwell_s"] if not np.isnan(f["gps_last_dwell_s"]) else 20.0
            f["eta_dev_moving_s"] = (route_left / (max(mv, 5) / 3.6) + n_left * min(dwell, 90) - (tgt_plan - T)
                                     if not np.isnan(mv) else np.nan)
            f["req_speed_kmh"] = route_left / lead * 3.6
    return f


def build_features(points: pd.DataFrame, traffic: pd.DataFrame, schedule: pd.DataFrame,
                   route_of: dict | None = None) -> pd.DataFrame:
    """DataFrame признаков (index = sample_id) для набора точек; ``route_of`` — tr_id -> маршрут (для клонов)."""
    TG, SG = index_traffic(traffic), index_schedule(schedule)
    rows = []
    for r in points.itertuples(index=False):
        tid = int(r.tr_id)
        f = point_features(int(to_sec([r.T])[0]), r.target_stop_id, int(to_sec([r.target_time_begin])[0]),
                           float(r.cur_dev_s) if pd.notna(r.cur_dev_s) else 0.0, TG.get(tid), SG[tid])
        f["route"] = str((route_of or {}).get(tid, tid))
        f["sample_id"] = r.sample_id
        rows.append(f)
    X = pd.DataFrame(rows).set_index("sample_id")
    for c in FEATURES:
        if c not in X:
            X[c] = np.nan
    return X[FEATURES]


def clone_sources(schedule_train: pd.DataFrame) -> dict[int, int]:
    """Синтетические ТС (id >= 9_000_000) -> реальный прототип по совпадению последовательности адресов."""
    s = schedule_train.sort_values(["tr_id", "time_begin"])
    seq = {k: tuple(g["building_address"].fillna("")) + (len(g),) for k, g in s.groupby("tr_id")}
    real = {v: k for k, v in seq.items() if k < 9_000_000}
    return {k: (k if k < 9_000_000 else real.get(v, k)) for k, v in seq.items()}
