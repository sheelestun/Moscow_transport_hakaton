"""Карточка инцидента для диспетчера: уровень риска, причины, рекомендация.

Причины строятся из SHAP-вкладов CatBoost (``get_feature_importance(type="ShapValues")``): прогноз задержки
раскладывается на сумму вкладов признаков. Вклады группируются в понятные человеку причины («долгая стоянка»,
«медленное движение», «разворот на конечной» ...). ``cur_dev_s`` сама по себе — база прогноза (модель учит
поправку к ней), поэтому её значение целиком относится к причине «накопленное отклонение».
"""

from __future__ import annotations

import numpy as np

# группа признаков -> (код причины, шаблон текста)
CAUSE_GROUPS: dict[str, list[str]] = {
    "accumulated": ["cur_dev_s", "gps_last_dev", "gps_med3", "gps_med5", "gps_max5", "gps_min5",
                    "gps_dev_minus_cur", "gps_trip_first_dev"],
    "trend": ["gps_slope"],
    "dwell": ["stop5", "stop15", "gps_last_dwell_s", "spd1", "last_speed"],
    "slow": ["spd5", "spd15", "moving_spd15", "spd_std5", "spd_trend", "disp5_m", "disp15_m"],
    "distance": ["dist_tgt_m", "dist_next_stop_m", "route_left_m", "eta_dev_s", "eta_dev_moving_s", "req_speed_kmh"],
    "overdue": ["overdue_s", "overdue_n"],
    "terminal": ["n_trip_breaks", "tgt_first_in_trip", "tgt_idx_in_trip", "tgt_left_in_trip", "cur_idx_in_trip",
                 "break_gap_s", "plan_gap_break_to_tgt_s"],
    "manual": ["tgt_manual_fill", "prev_manual_fill", "mf_share_next", "mf_share_trip"],
    "context": ["route", "hour", "hour_sin", "hour_cos", "lead_s", "n_stops_between", "tgt_plan_gap_prev_s",
                "plan_gap_prev_s", "plan_gap_next_s", "plan_route_dist_m", "plan_speed_kmh", "gps_n", "gps_age_s",
                "gps_n_trip", "n_pkt15", "last_fix_age_s"],
}
FEATURE_GROUP = {f: g for g, fs in CAUSE_GROUPS.items() for f in fs}


def _v(f: dict, k: str, default=np.nan) -> float:
    x = f.get(k, default)
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def cause_text(group: str, f: dict) -> str:
    """Человеческая формулировка причины по значениям признаков."""
    if group == "accumulated":
        cur, gps = _v(f, "cur_dev_s", 0.0), _v(f, "gps_med3")
        if abs(cur) < 1 and not np.isnan(gps):  # подсказка пустая — вклад дала GPS-история прибытий
            return f"отклонение по GPS на последних пройденных остановках: {gps:+.0f} с"
        return f"накопленное отклонение: {cur:+.0f} с на последней пройденной остановке"
    if group == "trend":
        return f"отклонение меняется на {_v(f, 'gps_slope'):+.0f} с за минуту"
    if group == "dwell":
        s = _v(f, "stop15")
        return f"долгая стоянка: {s * 100:.0f}% последних 15 мин без движения" if not np.isnan(s) else "долгая стоянка"
    if group == "slow":
        v = _v(f, "spd15")
        return f"медленное движение: в среднем {v:.0f} км/ч за 15 мин" if not np.isnan(v) else "медленное движение"
    if group == "distance":
        eta, left = _v(f, "eta_dev_s"), _v(f, "route_left_m")
        if not np.isnan(eta) and not np.isnan(left):
            return f"до остановки {left / 1000:.1f} км; с текущей скоростью ТС приедет на {eta / 60:+.0f} мин к плану"
        return "расстояние до остановки"
    if group == "overdue":
        return f"плановое время остановки прошло {_v(f, 'overdue_s', 0):.0f} с назад, а ТС её ещё не проехало"
    if group == "terminal":
        if _v(f, "n_trip_breaks", 0) > 0:
            return f"перед остановкой разворот на конечной (стоянка по плану {_v(f, 'break_gap_s', 0) / 60:.0f} мин)"
        return f"позиция в рейсе: до конца рейса {_v(f, 'tgt_left_in_trip', 0):.0f} остановок"
    if group == "manual":
        return "у остановки ручная отметка факта (обычно ставится по плану)"
    return f"типичная картина для маршрута в {int(_v(f, 'hour', 0)):02d}:00"


def explain(shap_row: np.ndarray, feats: list[str], f: dict, delay_pred: float, top_k: int = 3) -> dict:
    """SHAP-вклады одной точки (len(feats)+1, последний — базовое значение) -> причины и топ-признаки.

    Вклады в секундах задержки. ``cur_dev_s`` добавляется к своей группе целиком (база прогноза).
    """
    contrib = dict(zip(feats, shap_row[:-1]))
    contrib["cur_dev_s"] = contrib.get("cur_dev_s", 0.0) + _v(f, "cur_dev_s", 0.0)
    by_group: dict[str, float] = {}
    for name, c in contrib.items():
        g = FEATURE_GROUP.get(name, "context")
        by_group[g] = by_group.get(g, 0.0) + float(c)
    sign = 1.0 if delay_pred >= 0 else -1.0
    # причины — группы, которые толкают прогноз в ту же сторону, что и итог (к опозданию или к опережению)
    pushing = sorted(((g, c) for g, c in by_group.items() if c * sign > 5), key=lambda x: -abs(x[1]))
    causes = [{"code": g, "text": cause_text(g, f), "contribution_sec": round(c, 1)} for g, c in pushing[:top_k]]
    top = sorted(contrib.items(), key=lambda x: -abs(x[1]))[:5]
    return {"causes": causes,
            "top_features": [{"name": n, "contribution": round(float(c), 1)} for n, c in top]}


# Коды причин и рекомендаций — те, что понимает дашборд (frontend/js/labels.js). Новые коды:
# terminal_turnaround (разворот на конечной), ahead_of_schedule (опережение), on_track (по графику).
REASON_CODE = {"accumulated": "accumulated_delay", "dwell": "long_dwell", "slow": "speed_drop", "trend": "speed_drop",
               "distance": "traffic_jam_ahead", "overdue": "traffic_jam_ahead", "terminal": "terminal_turnaround",
               "manual": "on_track", "context": "accumulated_delay"}
REC_CODE = {"traffic_jam_ahead": "detour", "long_dwell": "adjust_interval", "speed_drop": "signal_priority",
            "accumulated_delay": "release_reserve", "terminal_turnaround": "release_reserve",
            "ahead_of_schedule": "hold_at_stop", "on_track": "monitor"}
RISK_RED, RISK_YELLOW = 0.7, 0.35  # как в frontend/js/config.js


def risk_level(delay_pred: float, p_early: float, p_late: float) -> str:
    """Светофор по вероятности опоздания > +120 с (пороги дашборда); сильное опережение — жёлтый."""
    if p_late >= RISK_RED:
        return "red"
    if p_late >= RISK_YELLOW or p_early >= 0.5 or delay_pred <= -60:
        return "yellow"
    return "green"


def reason_pattern(level: str, delay_pred: float, causes: list[dict], f: dict) -> str:
    """Код главной причины для дашборда (из SHAP-групп, а не из порогов по отдельным признакам)."""
    if level == "green":
        return "on_track"
    if delay_pred <= -60:
        return "ahead_of_schedule"
    if not causes:
        return "accumulated_delay"
    code = REASON_CODE.get(causes[0]["code"], "accumulated_delay")
    if code == "speed_drop" and _v(f, "spd15", 99) < 5 and _v(f, "route_left_m", 0) > 500:
        return "traffic_jam_ahead"
    return code


def recommendation(level: str, delay_pred: float, causes: list[dict]) -> str:
    """Правила поверх прогноза: что диспетчеру стоит сделать (текст для карточки)."""
    codes = {c["code"] for c in causes}
    if level == "green":
        return "действий не требуется"
    if delay_pred <= -60:
        return "ТС опережает график: придержать на ближайшей остановке"
    if "terminal" in codes:
        return "проконтролировать отправление с конечной; при сохранении опоздания — выпустить резервное ТС"
    if "dwell" in codes:
        return "выяснить причину длительной стоянки (посадка, ДТП, неисправность), предупредить следующее ТС"
    if "slow" in codes or "distance" in codes or "overdue" in codes:
        return "затруднённое движение на участке: рассмотреть корректировку интервалов или объезд"
    if level == "red":
        return "риск нарушения интервала: рассмотреть выпуск резервного ТС"
    return "наблюдать: риск опоздания"
