"""Онлайн-обёртка над `tabular.build_features` для инференс-сервиса.

Батч (`predict_submission.py`, `train_catboost.py`) и онлайн (`inference_service.py`) используют
**один и тот же** feature-код из `tabular.py` — так гарантируется, что распределение фичей
на train и в проде совпадает. Здесь только адаптер: приняли список пингов и слайс расписания →
собрали одноточечный DataFrame → вернули dict фичей.

Использование::

    from features.from_stream import build_features_online

    feats = build_features_online(
        sample={"sample_id": ..., "tr_id": ..., "T": ..., "target_stop_id": ...,
                "target_time_begin": ..., "cur_dev_s": ...},
        telemetry=[{"event_time": "...", "lon": ..., "lat": ..., "speed": ...,
                    "location_valid": "true", "is_hist_data": 0, "tr_id": ...}, ...],
        schedule=[{"tt_action_item_id": ..., "time_begin": "...", "geom": "POINT (lon lat)",
                   "manual_fill": "false", "tr_id": ...}, ...],
        route_of=None,      # маппинг клон→прототип (для батча), в проде обычно None
    )
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .tabular import FEATURES, build_features


REQUIRED_TELEMETRY_COLS = ("tr_id", "event_time", "lon", "lat", "speed", "location_valid", "is_hist_data")
REQUIRED_SCHEDULE_COLS = ("tr_id", "tt_action_item_id", "time_begin", "geom", "manual_fill")
REQUIRED_SAMPLE_KEYS = ("sample_id", "tr_id", "T", "target_stop_id", "target_time_begin", "cur_dev_s")


def build_features_online(
    sample: dict,
    telemetry: Iterable[dict],
    schedule: Iterable[dict],
    route_of: dict | None = None,
) -> pd.DataFrame:
    """Собрать фичи для одной точки прогноза из потоковых данных.

    Возвращает DataFrame с одной строкой (index = sample_id) и колонками FEATURES.
    Дальше он передаётся в модель как есть.
    """
    for k in REQUIRED_SAMPLE_KEYS:
        if k not in sample:
            raise ValueError(f"sample: обязательное поле '{k}' отсутствует")

    points_df = pd.DataFrame([{k: sample[k] for k in REQUIRED_SAMPLE_KEYS}])
    points_df["T"] = pd.to_datetime(points_df["T"], format="ISO8601")
    points_df["target_time_begin"] = pd.to_datetime(points_df["target_time_begin"], format="ISO8601")

    traffic_df = pd.DataFrame(list(telemetry))
    if traffic_df.empty:
        traffic_df = pd.DataFrame(columns=list(REQUIRED_TELEMETRY_COLS))
    # ISO8601: в одном буфере бывают метки с микросекундами и без — без format pandas падает
    traffic_df["event_time"] = pd.to_datetime(traffic_df["event_time"], format="ISO8601")
    for col in REQUIRED_TELEMETRY_COLS:
        if col not in traffic_df.columns:
            traffic_df[col] = None

    schedule_df = pd.DataFrame(list(schedule))
    if schedule_df.empty:
        raise ValueError("schedule пустой — features/from_stream не может построить план")
    schedule_df["time_begin"] = pd.to_datetime(schedule_df["time_begin"], format="ISO8601")
    for col in REQUIRED_SCHEDULE_COLS:
        if col not in schedule_df.columns:
            schedule_df[col] = None

    X = build_features(points_df, traffic_df, schedule_df, route_of=route_of)
    for c in FEATURES:
        if c not in X.columns:
            X[c] = None
    return X[FEATURES]
