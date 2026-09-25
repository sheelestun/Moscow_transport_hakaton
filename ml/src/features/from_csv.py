"""Общий feature-модуль (батч + онлайн — один контракт).

Из точек (tr_id, T) + traffic.csv + schedule.csv собирает два артефакта:
- static_df: DataFrame (index=sample_id) со списком STATIC_FEATURES для CatBoost и статик-головы torch-модели.
- seq_array: np.ndarray (N, SEQ_LEN, len(SEQ_FEATURES)) с последними GPS-пингами до T — для GRU/Transformer.

Антиутечка: для точки T используются только строки traffic с event_time ≤ T.
cur_dev_s даётся в точке (пред-посчитанная задержка на последней пройденной остановке).

Таргет для обучения — residual = target_delay_s − cur_dev_s. На инференсе прибавляем cur_dev_s обратно.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

STATIC_FEATURES = [
    "cur_dev_s",
    "speed_avg_5min",
    "speed_avg_15min",
    "std_speed_5min",
    "last_speed",
    "last_heading",
    "n_valid_gps_5min",
    "dist_to_target_m",
    "planned_time_to_target_sec",
    "planned_avg_speed_ms",
    "speed_trend",
    "speed_vs_planned",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    # Target-encoding по tr_id и часу — восстановлены из train
    # (для test/validate передаются извне; при отсутствии — глобальное среднее).
    "tr_hist_delay_mean",
    "tr_hist_delay_std",
    "tr_hist_cur_dev_mean",
    "hour_hist_delay_mean",
]

SEQ_FEATURES = ["dt_from_prev_sec", "speed", "heading", "dist_to_target_m", "valid_flag"]
SEQ_LEN = 30  # ~5–6 мин при 12–15 сек между пингами


def load_split(dataset_dir: Path | str, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """split ∈ {'train','test','validate'}.

    Возвращает (points, traffic, schedule) со сконвертированными datetime-колонками.
    Для validate: points = validate/points.csv (без target_delay_s), schedule = schedule_plan.csv (без факта).
    """
    dataset_dir = Path(dataset_dir)
    if split == "validate":
        points = pd.read_csv(dataset_dir / "validate" / "points.csv")
        traffic = pd.read_csv(dataset_dir / "validate" / "traffic.csv")
        schedule = pd.read_csv(dataset_dir / "validate" / "schedule_plan.csv")
    elif split in {"train", "test"}:
        points = pd.read_csv(dataset_dir / "labels" / f"labels_{split}.csv")
        traffic = pd.read_csv(dataset_dir / split / "traffic.csv")
        schedule = pd.read_csv(dataset_dir / split / "schedule.csv")
    else:
        raise ValueError(f"Unknown split: {split!r}")

    points["T"] = pd.to_datetime(points["T"])
    points["target_time_begin"] = pd.to_datetime(points["target_time_begin"])
    traffic["event_time"] = pd.to_datetime(traffic["event_time"])
    schedule["time_begin"] = pd.to_datetime(schedule["time_begin"])
    if "time_fact_begin" in schedule.columns:
        schedule["time_fact_begin"] = pd.to_datetime(schedule["time_fact_begin"])

    return points, traffic, schedule


def _parse_geom_wkt(s: str) -> tuple[float, float]:
    """'POINT (lon lat)' → (lat, lon). NaN если распарсить не удалось."""
    if not isinstance(s, str):
        return np.nan, np.nan
    try:
        inside = s[s.index("(") + 1 : s.index(")")]
        lon, lat = map(float, inside.split())
        return lat, lon
    except Exception:
        return np.nan, np.nan


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray | float:
    """Расстояние в метрах между парами (lat, lon). Работает и на скалярах, и на массивах."""
    r = 6371000.0
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlon = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _stop_coords_map(schedule: pd.DataFrame) -> dict[int, tuple[float, float]]:
    coords: dict[int, tuple[float, float]] = {}
    for stop_id, g in schedule.groupby("tt_action_item_id"):
        coords[stop_id] = _parse_geom_wkt(g["geom"].iloc[0])
    return coords


def _group_traffic_sorted(traffic: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Группировка traffic по tr_id с сортировкой event_time и сбросом индекса.

    Возвращает dict {tr_id: df}, где каждый df готов к бинарному поиску через searchsorted.
    """
    traffic = traffic.sort_values(["tr_id", "event_time"])
    return {tr_id: g.reset_index(drop=True) for tr_id, g in traffic.groupby("tr_id")}


def compute_history_stats(train_points: pd.DataFrame) -> dict:
    """Считает target-encoding статистики по train (для test/validate).

    Возвращает dict со словарями по tr_id / hour + глобальные средние (fallback).
    Использовать только на train, чтобы не было утечки в test.
    """
    if "target_delay_s" not in train_points.columns:
        raise ValueError("history можно считать только из train с target_delay_s")

    tr_grp = train_points.groupby("tr_id")
    tr_delay_mean = tr_grp["target_delay_s"].mean().to_dict()
    tr_delay_std = tr_grp["target_delay_s"].std().fillna(0.0).to_dict()
    tr_curdev_mean = tr_grp["cur_dev_s"].mean().to_dict()

    hours = pd.to_datetime(train_points["T"]).dt.hour
    hour_grp = train_points.assign(_h=hours.values).groupby("_h")
    hour_delay_mean = hour_grp["target_delay_s"].mean().to_dict()

    return {
        "tr_delay_mean": tr_delay_mean,
        "tr_delay_std": tr_delay_std,
        "tr_curdev_mean": tr_curdev_mean,
        "hour_delay_mean": hour_delay_mean,
        "global_delay_mean": float(train_points["target_delay_s"].mean()),
        "global_delay_std": float(train_points["target_delay_s"].std()),
        "global_curdev_mean": float(train_points["cur_dev_s"].mean()),
    }


def build_features(
    points: pd.DataFrame,
    traffic: pd.DataFrame,
    schedule: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    history: Optional[dict] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Основная функция: собирает static-фичи и sequence-тензор.

    static_df.index = sample_id, порядок строк совпадает с axis=0 seq_array.

    history — результат compute_history_stats(train_points). Если None,
    tr/hour-статистики считаются на self (для train — это OK, для test/validate — утечка).
    """
    traffic_by_tr = _group_traffic_sorted(traffic)
    stop_coords = _stop_coords_map(schedule)
    if history is None and "target_delay_s" in points.columns:
        history = compute_history_stats(points)
    if history is None:
        # validate без train-history: используем нули как fallback (нужно передать history)
        history = {
            "tr_delay_mean": {}, "tr_delay_std": {}, "tr_curdev_mean": {},
            "hour_delay_mean": {},
            "global_delay_mean": 0.0, "global_delay_std": 0.0, "global_curdev_mean": 0.0,
        }

    n = len(points)
    seq_array = np.zeros((n, seq_len, len(SEQ_FEATURES)), dtype=np.float32)
    static_rows = []

    for i, (_, row) in enumerate(points.reset_index(drop=True).iterrows()):
        tr_id = row["tr_id"]
        t_ts = row["T"]
        target_lat, target_lon = stop_coords.get(row["target_stop_id"], (np.nan, np.nan))
        cur_dev = float(row["cur_dev_s"]) if pd.notna(row["cur_dev_s"]) else 0.0

        tr_df = traffic_by_tr.get(tr_id)
        if tr_df is not None and len(tr_df):
            end_idx = int(np.searchsorted(tr_df["event_time"].values, np.datetime64(t_ts), side="right"))
            window = tr_df.iloc[:end_idx]
        else:
            window = traffic.iloc[0:0]

        static_rows.append(
            _compute_static(row, window, t_ts, target_lat, target_lon, cur_dev, history)
        )
        if len(window):
            seq_array[i] = _compute_sequence(window.tail(seq_len), seq_len, target_lat, target_lon)

    static_df = pd.DataFrame(static_rows).set_index("sample_id")[STATIC_FEATURES]
    return static_df, seq_array


def _compute_static(
    row: pd.Series,
    window: pd.DataFrame,
    t_ts: pd.Timestamp,
    target_lat: float,
    target_lon: float,
    cur_dev: float,
    history: dict,
) -> dict:
    last_5 = window[window["event_time"] > t_ts - pd.Timedelta("5min")]
    last_15 = window[window["event_time"] > t_ts - pd.Timedelta("15min")]
    valid_5 = last_5[last_5["location_valid"] == True]  # noqa: E712 (обычно bool, оставляем явно)
    valid_15 = last_15[last_15["location_valid"] == True]  # noqa: E712

    speeds_5 = valid_5["speed"].dropna()
    speeds_15 = valid_15["speed"].dropna()

    last_valid = window[window["location_valid"] == True].tail(1)  # noqa: E712
    if len(last_valid):
        last_lat = float(last_valid["lat"].iloc[0])
        last_lon = float(last_valid["lon"].iloc[0])
        last_speed = float(last_valid["speed"].iloc[0]) if pd.notna(last_valid["speed"].iloc[0]) else 0.0
        last_heading = float(last_valid["heading"].iloc[0]) if pd.notna(last_valid["heading"].iloc[0]) else 0.0
    else:
        last_lat, last_lon, last_speed, last_heading = np.nan, np.nan, 0.0, 0.0

    if pd.notna(last_lat) and pd.notna(target_lat):
        dist_to_target = float(haversine_m(last_lat, last_lon, target_lat, target_lon))
    else:
        dist_to_target = 0.0

    planned_time = float((row["target_time_begin"] - t_ts).total_seconds())
    planned_speed = (dist_to_target / planned_time) if planned_time > 0 else 0.0

    speed_avg_5 = float(speeds_5.mean()) if len(speeds_5) else 0.0
    speed_avg_15 = float(speeds_15.mean()) if len(speeds_15) else 0.0
    # Тренд: положительный = ускоряется (за последние 5 мин быстрее, чем за 15),
    # отрицательный = замедляется — сильный сигнал для residual.
    speed_trend = speed_avg_5 - speed_avg_15
    # Отклонение мгновенной скорости от плановой (dist/time до цели).
    speed_vs_planned = last_speed - float(planned_speed)

    # Циклическое кодирование времени — модели не приходится учить,
    # что 23 и 0 близки (для дня недели: 6 и 0).
    hour_rad = 2 * np.pi * t_ts.hour / 24.0
    wday_rad = 2 * np.pi * t_ts.weekday() / 7.0

    tr_id = row["tr_id"]
    hour = int(t_ts.hour)
    tr_delay_mean = history["tr_delay_mean"].get(tr_id, history["global_delay_mean"])
    tr_delay_std = history["tr_delay_std"].get(tr_id, history["global_delay_std"])
    tr_curdev_mean = history["tr_curdev_mean"].get(tr_id, history["global_curdev_mean"])
    hour_delay_mean = history["hour_delay_mean"].get(hour, history["global_delay_mean"])

    return {
        "sample_id": row["sample_id"],
        "cur_dev_s": cur_dev,
        "speed_avg_5min": speed_avg_5,
        "speed_avg_15min": speed_avg_15,
        "std_speed_5min": float(speeds_5.std()) if len(speeds_5) > 1 else 0.0,
        "last_speed": last_speed,
        "last_heading": last_heading,
        "n_valid_gps_5min": float(len(valid_5)),
        "dist_to_target_m": dist_to_target,
        "planned_time_to_target_sec": planned_time,
        "planned_avg_speed_ms": float(planned_speed),
        "speed_trend": speed_trend,
        "speed_vs_planned": speed_vs_planned,
        "hour_sin": float(np.sin(hour_rad)),
        "hour_cos": float(np.cos(hour_rad)),
        "weekday_sin": float(np.sin(wday_rad)),
        "weekday_cos": float(np.cos(wday_rad)),
        "tr_hist_delay_mean": float(tr_delay_mean),
        "tr_hist_delay_std": float(tr_delay_std),
        "tr_hist_cur_dev_mean": float(tr_curdev_mean),
        "hour_hist_delay_mean": float(hour_delay_mean),
    }


def _compute_sequence(
    window: pd.DataFrame,
    seq_len: int,
    target_lat: float,
    target_lon: float,
) -> np.ndarray:
    """window уже tail(seq_len). Возвращает (seq_len, len(SEQ_FEATURES)) с zero-padding в начало."""
    n_real = len(window)
    out = np.zeros((seq_len, len(SEQ_FEATURES)), dtype=np.float32)

    times = window["event_time"].values.astype("datetime64[s]").astype(np.int64)
    dts = np.diff(times, prepend=times[0]).astype(np.float32)

    speeds = window["speed"].fillna(0.0).to_numpy(dtype=np.float32)
    headings = window["heading"].fillna(0.0).to_numpy(dtype=np.float32)
    valid = (window["location_valid"] == True).to_numpy(dtype=np.float32)  # noqa: E712

    lats = window["lat"].ffill().fillna(0.0).to_numpy(dtype=np.float64)
    lons = window["lon"].ffill().fillna(0.0).to_numpy(dtype=np.float64)
    if pd.notna(target_lat):
        dist = haversine_m(lats, lons, target_lat, target_lon).astype(np.float32)
    else:
        dist = np.zeros(n_real, dtype=np.float32)

    seq = np.stack([dts, speeds, headings, dist, valid], axis=1)
    out[seq_len - n_real :] = seq
    return out


def build_target_residual(points: pd.DataFrame) -> Optional[np.ndarray]:
    """target_delay_s − cur_dev_s. None если у points нет колонки target_delay_s (это validate)."""
    if "target_delay_s" not in points.columns:
        return None
    return (points["target_delay_s"].astype(float) - points["cur_dev_s"].astype(float)).to_numpy(dtype=np.float32)


if __name__ == "__main__":
    import sys
    import time

    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[3] / "dataset"
    split = sys.argv[2] if len(sys.argv) > 2 else "test"

    t0 = time.time()
    points, traffic, schedule = load_split(dataset_dir, split)
    print(f"[{split}] points={len(points)}, traffic={len(traffic)}, schedule={len(schedule)}")
    static_df, seq_array = build_features(points, traffic, schedule)
    print(f"static_df shape: {static_df.shape}, seq_array shape: {seq_array.shape}")
    print(f"elapsed: {time.time() - t0:.1f} s")
    print(static_df.describe().round(2))
    y = build_target_residual(points)
    if y is not None:
        print(f"residual: mean={y.mean():.1f}, std={y.std():.1f}, |mean|={np.abs(y).mean():.1f}")
