"""Симулятор движения ТС по маршрутам — Python-порт логики frontend/js/mock.js.

Держит N ТС на каждом маршруте, двигает их по polyline с плановой скоростью
+ случайные проблемы (пробки, долгие простои, обрыв интервала). Каждый тик
обновляет позиции, `delay_now_sec` и производные `delay_pred_sec` / `risk_score`.

Прогнозы формируем локально по эвристике; когда доступен ML-сервис — вызывающая
сторона может подменить `delay_pred_sec` реальным ответом от /predict.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from routes_data import ROUTES


PLAN_SPEED_MPS = 18 / 3.6  # 5 м/с — средняя плановая скорость с учётом остановок
HORIZON_S = 12.5 * 60      # середина окна прогноза 10–15 мин
RISK_MID_SEC = 120.0
RISK_SLOPE_SEC = 60.0
RISK_RED = 0.7

REASONS = ["traffic_jam_ahead", "long_dwell", "speed_drop", "bunching", "accumulated_delay"]
REC_FOR_REASON = {
    "traffic_jam_ahead": "detour",
    "long_dwell": "adjust_interval",
    "speed_drop": "signal_priority",
    "bunching": "hold_at_stop",
    "accumulated_delay": "add_reserve",
}
FEATURES_FOR_REASON = {
    "traffic_jam_ahead": ["traffic_score", "speed_avg_5min", "cur_dev_s", "hour_of_day"],
    "long_dwell": ["dwell_last_stop_sec", "cur_dev_s", "hour_of_day", "headway_to_prev_sec"],
    "speed_drop": ["speed_avg_5min", "speed_avg_15min", "distance_to_target_m", "cur_dev_s"],
    "bunching": ["headway_to_next_sec", "headway_to_prev_sec", "cur_dev_s", "speed_avg_5min"],
    "accumulated_delay": ["cur_dev_s", "current_delay_sec", "planned_time_to_target_sec", "traffic_score"],
}


def risk_from_delay(delay: float) -> float:
    return 1.0 / (1.0 + math.exp(-(delay - RISK_MID_SEC) / RISK_SLOPE_SEC))


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


@dataclass
class RouteRuntime:
    """Маршрут с посчитанными кумулятивными расстояниями и остановками с pos_m."""
    route_id: str
    transport_type: str
    name: str
    line: list[list[float]]
    stops: list[dict]
    cum_m: list[float]
    length_m: float

    @classmethod
    def build(cls, raw: dict) -> "RouteRuntime":
        line = raw["line"]
        cum = [0.0]
        for i in range(1, len(line)):
            cum.append(cum[-1] + haversine_m(line[i - 1], line[i]))
        # проецируем каждую остановку на ближайший сегмент линии → pos_m
        stops_out: list[dict] = []
        for idx, s in enumerate(raw["stops"]):
            name, lat, lon = s[0], s[1], s[2]
            pos_m = _project(line, cum, (lat, lon))
            stops_out.append({
                "stop_id": f"{raw['route_id']}-{idx + 1}",
                "name": name,
                "lat": lat,
                "lon": lon,
                "pos_m": pos_m,
            })
        stops_out.sort(key=lambda x: x["pos_m"])
        return cls(
            route_id=raw["route_id"], transport_type=raw["transport_type"], name=raw["name"],
            line=line, stops=stops_out, cum_m=cum, length_m=cum[-1],
        )

    def point_at(self, pos_m: float) -> tuple[float, float]:
        pos_m = max(0.0, min(pos_m, self.length_m))
        for i in range(len(self.cum_m) - 1):
            if self.cum_m[i + 1] >= pos_m:
                seg = self.cum_m[i + 1] - self.cum_m[i]
                t = 0.0 if seg <= 0 else (pos_m - self.cum_m[i]) / seg
                a, b = self.line[i], self.line[i + 1]
                return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        last = self.line[-1]
        return last[0], last[1]


def _project(line: list[list[float]], cum: list[float], pt: tuple[float, float]) -> float:
    """Проекция точки на polyline: возвращает pos_m ближайшей точки."""
    best_pos = 0.0
    best_d = float("inf")
    for i in range(len(line) - 1):
        a, b = line[i], line[i + 1]
        ax, ay = a[1], a[0]  # lon, lat
        bx, by = b[1], b[0]
        px, py = pt[1], pt[0]
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
        proj = (ay + dy * t, ax + dx * t)  # lat, lon
        d = haversine_m(proj, pt)
        if d < best_d:
            best_d = d
            seg_len = cum[i + 1] - cum[i]
            best_pos = cum[i] + seg_len * t
    return best_pos


@dataclass
class Vehicle:
    vehicle_id: str
    route_id: str
    direction: int          # +1 или -1
    pos_m: float
    speed_kmh: float
    speed_avg: float
    anchor_ms: int          # ms эпохи для начала рейса (плановое время в точке 0)
    trip_id: int
    trouble_left_s: float = 0.0
    trouble_speed_kmh: float = 12.0
    reason: str = "accumulated_delay"
    features: list[dict] = field(default_factory=list)
    confidence: float = 0.75
    noise: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    delay_now_s: float = 0.0
    delay_pred_s: int = 0
    risk_score: float = 0.0
    updated_at: str = ""
    is_reserve: bool = False
    recover_left_s: float = 0.0
    facts: dict[str, float] = field(default_factory=dict)  # stop_id → ms фактического прохождения
    last_trip: Optional[dict] = None
    alerted_stops: set[str] = field(default_factory=set)
    alerted_trip: int = -1


def _along(v: Vehicle, r: RouteRuntime, pos_m: float) -> float:
    return pos_m if v.direction > 0 else r.length_m - pos_m


def _plan_time_ms(v: Vehicle, along_m: float) -> float:
    return v.anchor_ms + (along_m / PLAN_SPEED_MPS) * 1000


def _make_features(reason: str) -> list[dict]:
    weights = [random.uniform(0.30, 0.45), random.uniform(0.15, 0.25),
               random.uniform(0.08, 0.15), random.uniform(0.03, 0.08)]
    return [{"name": n, "contribution": round(w, 2)}
            for n, w in zip(FEATURES_FOR_REASON[reason], weights)]


class Simulator:
    """Один инстанс на процесс. `step(dt)` — сдвиг симуляции на `dt` секунд симуляции."""

    def __init__(self, speed_factor: float = 5.0, seed: int = 42) -> None:
        random.seed(seed)
        self.routes: dict[str, RouteRuntime] = {}
        for raw in ROUTES:
            r = RouteRuntime.build(raw)
            self.routes[r.route_id] = r
        self.vehicles: list[Vehicle] = []
        self.by_id: dict[str, Vehicle] = {}
        self._trip_seq = 0
        self._reserve_seq = 1
        self._sim_now_ms = int(time.time() * 1000)
        self.speed_factor = float(speed_factor)
        self.offline = False
        self._init_fleet()

    # ---------- инициализация ----------

    def _init_fleet(self) -> None:
        for r in self.routes.values():
            for i in range(4):
                pos = r.length_m * (i + random.uniform(0.15, 0.7)) / 4
                v = Vehicle(
                    vehicle_id=str(random.randint(120000, 139999)),
                    route_id=r.route_id,
                    direction=-1 if i % 2 else 1,
                    pos_m=pos,
                    speed_kmh=random.uniform(17, 22),
                    speed_avg=18.0,
                    anchor_ms=0,
                    trip_id=0,
                    features=_make_features("accumulated_delay"),
                    confidence=round(random.uniform(0.60, 0.85), 2),
                )
                self._new_trip(v, delay_s=random.uniform(-40, 70))
                self.vehicles.append(v)
                self.by_id[v.vehicle_id] = v
        # несколько ТС «в проблеме» с первой секунды
        for idx, k in enumerate((1, 5, 9, 13)):
            if k < len(self.vehicles):
                v = self.vehicles[k]
                self._new_trip(v, delay_s=random.uniform(110, 170))
                self._start_trouble(v, strength=0.7 if idx == 3 else 1.0)
                v.trouble_left_s = random.uniform(300, 700)

    def _new_trip(self, v: Vehicle, delay_s: float) -> None:
        r = self.routes[v.route_id]
        if v.facts:
            v.last_trip = {"trip_id": v.trip_id, "facts": dict(v.facts),
                           "anchor_ms": v.anchor_ms, "direction": -v.direction}
        self._trip_seq += 1
        v.trip_id = self._trip_seq
        v.noise = random.uniform(-40, 40)
        v.facts = {}
        v.anchor_ms = int(self._sim_now_ms - delay_s * 1000 - (_along(v, r, v.pos_m) / PLAN_SPEED_MPS) * 1000)

    def _start_trouble(self, v: Vehicle, strength: float = 1.0) -> None:
        v.trouble_left_s = random.uniform(240, 720)
        v.reason = random.choice([r for r in REASONS if r != "accumulated_delay"])
        v.trouble_speed_kmh = random.uniform(7, 12) / max(strength, 0.1)
        v.noise = random.uniform(-70, 70)
        v.features = _make_features(v.reason)
        v.confidence = round(random.uniform(0.62, 0.90), 2)

    # ---------- симуляция ----------

    def step(self, dt_s: float) -> None:
        self._sim_now_ms += int(dt_s * 1000)
        now_ms = self._sim_now_ms
        for v in self.vehicles:
            r = self.routes[v.route_id]
            if v.trouble_left_s > 0:
                v.trouble_left_s -= dt_s
                v.speed_kmh = max(2.0, min(14.0, v.trouble_speed_kmh + random.uniform(-1.5, 1.5)))
            else:
                if v.reason != "accumulated_delay":
                    v.reason = "accumulated_delay"
                    v.features = _make_features(v.reason)
                # шанс новой проблемы (нормированный на реальное время)
                p = 0.001 * math.sqrt(self.speed_factor / 5.0) * dt_s / max(self.speed_factor, 0.5)
                if random.random() < p:
                    self._start_trouble(v)
                target = 22 if v.delay_now_s > 20 else (14 if v.delay_now_s < -20 else 18)
                v.speed_kmh = max(10.0, min(28.0,
                                            v.speed_kmh + (target - v.speed_kmh) * 0.3 + random.uniform(-1, 1)))
            v.speed_avg += (v.speed_kmh - v.speed_avg) * 0.08 * dt_s

            if v.recover_left_s > 0:
                d = min(v.recover_left_s, 1.2 * dt_s)
                v.anchor_ms += int(d * 1000)
                v.recover_left_s -= d

            before = _along(v, r, v.pos_m)
            v.pos_m = max(0.0, min(r.length_m, v.pos_m + v.direction * (v.speed_kmh / 3.6) * dt_s))
            after = _along(v, r, v.pos_m)
            for s in r.stops:
                d = _along(v, r, s["pos_m"])
                if before < d <= after + 0.5:
                    v.facts[s["stop_id"]] = now_ms
            if v.pos_m >= r.length_m or v.pos_m <= 0:
                v.direction *= -1
                self._new_trip(v, delay_s=random.uniform(-30, 60))

            v.lat, v.lon = r.point_at(v.pos_m)
            v.delay_now_s = (now_ms - _plan_time_ms(v, _along(v, r, v.pos_m))) / 1000
            v.delay_pred_s = int(max(-300, min(900, self._predict_delay(v, HORIZON_S))))
            v.risk_score = round(risk_from_delay(v.delay_pred_s), 3)
            v.updated_at = _iso_ms(now_ms)

    def _future_delay(self, v: Vehicle, t: float) -> float:
        d = v.delay_now_s
        tl = max(0.0, v.trouble_left_s)
        tr = min(t, tl)
        if tr > 0:
            d += (1 - (v.trouble_speed_kmh / 3.6) / PLAN_SPEED_MPS) * tr
        rest = t - tr
        if v.recover_left_s > 0:
            d -= min(v.recover_left_s, 1.2 * t)
        catch_up = 1 - (22 / 3.6) / PLAN_SPEED_MPS
        if d > 0:
            d = max(0, d + catch_up * rest)
        else:
            d = min(0, d - catch_up * rest)
        return d

    def _predict_delay(self, v: Vehicle, t: float) -> float:
        """Локальный прогноз с шумом (как у настоящей модели MAE~40-60 с)."""
        return self._future_delay(v, t) + v.noise * min(1.0, t / HORIZON_S)

    # ---------- публичные вьюхи ----------

    def get_routes(self) -> list[dict]:
        return [{
            "route_id": r.route_id, "name": r.name, "transport_type": r.transport_type,
            "geometry": r.line,
            "stops": [{"stop_id": s["stop_id"], "name": s["name"], "lat": s["lat"], "lon": s["lon"]}
                      for s in r.stops],
        } for r in self.routes.values()]

    def pub_vehicle(self, v: Vehicle) -> dict:
        out = {
            "vehicle_id": v.vehicle_id, "route_id": v.route_id, "lat": v.lat, "lon": v.lon,
            "is_reserve": v.is_reserve,
            "speed": round(v.speed_kmh), "heading": None,
            "delay_now_sec": round(v.delay_now_s), "delay_pred_sec": v.delay_pred_s,
            "risk_score": v.risk_score, "updated_at": v.updated_at,
        }
        if v.risk_score >= 0.35:  # yellow+red — фронт ждёт reason/rec для не-зелёных
            out.update({
                "reason_pattern": v.reason,
                "recommendation": REC_FOR_REASON.get(v.reason, "monitor"),
                "top_features": v.features,
                "confidence": v.confidence,
            })
        return out

    def get_vehicles(self) -> list[dict]:
        return [self.pub_vehicle(v) for v in self.vehicles]

    def get_schedule(self, vehicle_id: str) -> dict:
        v = self.by_id.get(vehicle_id)
        if v is None:
            raise KeyError(vehicle_id)
        r = self.routes[v.route_id]
        now_ms = self._sim_now_ms
        d_cur = _along(v, r, v.pos_m)
        ordered = list(r.stops) if v.direction > 0 else list(reversed(r.stops))
        rows: list[dict] = []
        next_found = False
        for s in ordered:
            d = _along(v, r, s["pos_m"])
            plan_ms = _plan_time_ms(v, d)
            row: dict = {"stop_id": s["stop_id"], "name": s["name"], "lat": s["lat"], "lon": s["lon"],
                         "time_plan": _iso_ms(int(plan_ms))}
            if d <= d_cur:
                fact_ms = v.facts.get(s["stop_id"])
                if fact_ms is None:
                    # реконструкция: доля пройденного × текущее отклонение
                    frac = 0.4 + 0.6 * (d / max(d_cur, 1))
                    fact_ms = plan_ms + v.delay_now_s * 1000 * frac
                row.update({
                    "status": "passed",
                    "time_fact": _iso_ms(int(fact_ms)),
                    "delay_sec": round((fact_ms - plan_ms) / 1000),
                })
            else:
                ahead = (d - d_cur) / PLAN_SPEED_MPS
                delay = self._predict_delay(v, ahead)
                row.update({
                    "status": "upcoming" if next_found else "next",
                    "time_pred": _iso_ms(int(plan_ms + delay * 1000)),
                    "delay_sec": round(delay),
                    "_ahead": ahead,
                })
                next_found = True
            rows.append(row)
        upcoming = [x for x in rows if x["status"] != "passed"]
        target = None
        for x in upcoming:
            if 600 < x["_ahead"] <= 900:
                target = x
                break
        if target is None and upcoming:
            target = min(upcoming, key=lambda x: abs(x["_ahead"] - HORIZON_S))
        if target is not None:
            target["is_target"] = True
        for x in rows:
            x.pop("_ahead", None)
        return {
            "vehicle_id": v.vehicle_id, "route_id": v.route_id,
            "direction": f"{ordered[0]['name']} → {ordered[-1]['name']}",
            "stops": rows,
        }

    # ---------- what-if ----------

    def measure_effect(self, v: Vehicle, scenario: str) -> int:
        """Локальный сценарный сдвиг (MEASURE_EFFECT из mock.js)."""
        effect = {"add_reserve": 0.30, "adjust_interval": 0.25, "detour": 0.25,
                  "signal_priority": 0.25, "hold_at_stop": 0.20}
        fits = {"add_reserve": ["accumulated_delay", "long_dwell"],
                "adjust_interval": ["bunching", "long_dwell"],
                "detour": ["traffic_jam_ahead"],
                "signal_priority": ["speed_drop", "traffic_jam_ahead"],
                "hold_at_stop": ["bunching"]}
        before = v.delay_pred_s
        if before <= 30:
            return before
        k = effect.get(scenario, 0.0)
        if REC_FOR_REASON.get(v.reason) == scenario:
            k += 0.45
        elif v.reason in fits.get(scenario, ()):
            k += 0.20
        h = sum(ord(c) for c in v.vehicle_id) % 10
        k = min(0.90, k * (0.90 + h / 50))
        return round(before * (1 - k))

    def apply_measure(self, route_id: str, scenario: str) -> dict:
        for v in [x for x in self.vehicles if x.route_id == route_id]:
            gain = v.delay_pred_s - self.measure_effect(v, scenario)
            if gain > 0:
                v.recover_left_s += max(0.0, v.delay_now_s) * (gain / max(v.delay_pred_s, 1))
            fits = {"detour": ["traffic_jam_ahead"], "signal_priority": ["speed_drop", "traffic_jam_ahead"],
                    "hold_at_stop": ["bunching"], "adjust_interval": ["bunching", "long_dwell"],
                    "add_reserve": ["accumulated_delay", "long_dwell"]}
            if v.reason in fits.get(scenario, ()):
                v.trouble_left_s = 0
        if scenario == "add_reserve":
            self._add_reserve(route_id)
        return {"ok": True}

    def _add_reserve(self, route_id: str) -> Vehicle:
        r = self.routes[route_id]
        v = Vehicle(
            vehicle_id=f"Р{self._reserve_seq}-{route_id}",
            route_id=route_id,
            direction=1,
            pos_m=0.0,
            speed_kmh=22.0,
            speed_avg=20.0,
            anchor_ms=0,
            trip_id=0,
            features=_make_features("accumulated_delay"),
            confidence=0.80,
            is_reserve=True,
        )
        self._reserve_seq += 1
        self._new_trip(v, delay_s=-20)
        self.vehicles.append(v)
        self.by_id[v.vehicle_id] = v
        # первый шаг чтобы координаты обновились
        self.step(0.0)
        return v

    def now_ms(self) -> int:
        return self._sim_now_ms


def _iso_ms(ms: int) -> str:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(ms / 1000, tz=_dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
