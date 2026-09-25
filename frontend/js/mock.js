// Фейковый источник данных — имитирует бэкенд прямо в браузере.
// Отдаёт данные в ТОЧНО таком же формате, как настоящий бэкенд
// (раздел 7.2 ARCHITECTURE_AND_ROLES.md + уточнения в frontend/README.md),
// поэтому остальной код не знает, мок это или живой API.
//
// Модель простая и честная: у каждого рейса есть плановое расписание
// (ТС должно ехать со средней скоростью PLAN_SPEED). Если ТС едет медленнее —
// копится опоздание. «Прогноз модели» экстраполирует тренд на 10–15 минут вперёд.
window.App = window.App || {};

(function (App) {
  const G = App.geo;
  const PLAN_SPEED = 18 / 3.6; // м/с — плановая средняя скорость с учётом остановок
  const HORIZON_S = 12.5 * 60; // середина окна прогноза 10–15 мин

  const REASONS = ["traffic_jam_ahead", "long_dwell", "speed_drop", "bunching", "accumulated_delay"];
  const REC_FOR_REASON = {
    traffic_jam_ahead: "detour",
    long_dwell: "adjust_interval",
    speed_drop: "signal_priority",
    bunching: "hold_at_stop",
    accumulated_delay: "release_reserve",
  };
  const FEATURES_FOR_REASON = {
    traffic_jam_ahead: ["traffic_score", "speed_avg_5min", "cur_dev_s", "hour_of_day"],
    long_dwell: ["dwell_last_stop_sec", "cur_dev_s", "hour_of_day", "headway_to_prev_sec"],
    speed_drop: ["speed_avg_5min", "speed_avg_15min", "distance_to_target_m", "cur_dev_s"],
    bunching: ["headway_to_next_sec", "headway_to_prev_sec", "cur_dev_s", "speed_avg_5min"],
    accumulated_delay: ["cur_dev_s", "current_delay_sec", "planned_time_to_target_sec", "traffic_score"],
  };

  const rnd = (a, b) => a + Math.random() * (b - a);
  const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  // Та же формула, что в контракте ML: risk = sigmoid((delay_pred - 120) / 60)
  const riskFromDelay = (d) => 1 / (1 + Math.exp(-(d - 120) / 60));

  function buildRoute(raw) {
    const line = raw.line;
    const cum = G.cumulative(line);
    const stops = raw.stops
      .map((s, i) => {
        const p = G.project(line, cum, [s[1], s[2]]);
        return { stop_id: `${raw.route_id}-${i + 1}`, name: s[0], lat: p.point[0], lon: p.point[1], pos_m: p.pos_m };
      })
      .sort((a, b) => a.pos_m - b.pos_m);
    return { route_id: raw.route_id, name: raw.name, geometry: line, stops, _cum: cum, length_m: cum[cum.length - 1] };
  }

  App.createMockSource = function () {
    // Часы симуляции: могут идти быстрее реальных (кнопки ×1 / ×5 / ×20 на карте)
    let simNow = Date.now();
    let speedFactor = 5;
    App.now = () => simNow;
    const routes = App.MOCK_ROUTES.map(buildRoute);
    const routeById = Object.fromEntries(routes.map((r) => [r.route_id, r]));
    const vehicles = [];
    const byId = new Map();
    const alerts = new Map();
    let alertSeq = 91000;
    let timer = null;

    // ---------- ТС и рейсы ----------
    // Расстояние, пройденное от начала рейса (с учётом направления)
    const along = (v, pos) => (v._dir > 0 ? pos : routeById[v.route_id].length_m - pos);
    // Плановое время (мс), когда ТС должно быть в точке d (метры от начала рейса)
    const planAt = (v, d) => v._anchor + (d / PLAN_SPEED) * 1000;

    function newTrip(v, delaySec) {
      v._facts = {};
      v._anchor = simNow - delaySec * 1000 - (along(v, v._pos) / PLAN_SPEED) * 1000;
    }

    function startTrouble(v, strength = 1) {
      v._trouble = Math.floor(rnd(70, 160));
      v._reason = pick(REASONS.filter((r) => r !== "accumulated_delay"));
      v._troubleSpeed = rnd(4, 9) / strength;
      v._feats = makeFeatures(v._reason);
      v._conf = +rnd(0.62, 0.9).toFixed(2);
    }

    function makeFeatures(reason) {
      const w = [rnd(0.3, 0.45), rnd(0.15, 0.25), rnd(0.08, 0.15), rnd(0.03, 0.08)];
      return FEATURES_FOR_REASON[reason].map((name, i) => ({ name, contribution: +w[i].toFixed(2) }));
    }

    routes.forEach((r) => {
      for (let i = 0; i < 4; i++) {
        const v = {
          vehicle_id: String(Math.floor(rnd(120000, 139999))),
          route_id: r.route_id,
          _pos: (r.length_m * (i + rnd(0.15, 0.7))) / 4,
          _dir: i % 2 ? -1 : 1,
          _trouble: 0,
          _reason: "accumulated_delay",
          _feats: makeFeatures("accumulated_delay"),
          _conf: +rnd(0.6, 0.85).toFixed(2),
          speed: rnd(17, 22),
          _speedAvg: 18,
        };
        newTrip(v, rnd(-40, 70));
        vehicles.push(v);
        byId.set(v.vehicle_id, v);
      }
    });
    // Несколько ТС сразу «в проблеме», чтобы на демо было что показать с первой секунды
    [vehicles[1], vehicles[9], vehicles[14], vehicles[21]].forEach((v, i) => {
      newTrip(v, rnd(110, 170));
      startTrouble(v, i === 3 ? 0.7 : 1);
      v._speedAvg = v._troubleSpeed * 1.6;
    });

    // dt — секунды симуляции
    function step(dt) {
      simNow += dt * 1000;
      const now = simNow;
      for (const v of vehicles) {
        const r = routeById[v.route_id];
        if (v._trouble > 0) {
          v._trouble -= dt;
          v.speed = clamp(v._troubleSpeed + rnd(-1.5, 1.5), 2, 14);
        } else {
          if (v._reason !== "accumulated_delay") {
            // проблема закончилась, но опоздание ещё не отыграно
            v._reason = "accumulated_delay";
            v._feats = makeFeatures(v._reason);
          }
          if (Math.random() < (0.0015 * dt) / speedFactor) startTrouble(v); // частота — в реальном времени
          v.speed = clamp(v.speed + rnd(-1.5, 1.5), 17, 27);
        }
        v._speedAvg += (v.speed - v._speedAvg) * 0.08 * dt; // сглаженная скорость

        // Движение по линии маршрута (реальное время)
        const before = along(v, v._pos);
        v._pos += v._dir * (v.speed / 3.6) * dt;
        if (v._pos >= r.length_m || v._pos <= 0) {
          v._pos = clamp(v._pos, 0, r.length_m);
          v._dir *= -1;
          newTrip(v, rnd(-30, 60)); // новый рейс в обратную сторону
        } else {
          // Фиксируем фактическое время прохождения остановок
          const after = along(v, v._pos);
          for (const s of r.stops) {
            const d = along(v, s.pos_m);
            if (d > before && d <= after) v._facts[s.stop_id] = now;
          }
        }
        const [lat, lon] = G.pointAt(r.geometry, r._cum, v._pos);
        v.lat = lat;
        v.lon = lon;

        // Текущее отклонение и прогноз
        v.delay_now_sec = (now - planAt(v, along(v, v._pos))) / 1000;
        v.delay_pred_sec = Math.round(clamp(predDelay(v, HORIZON_S), -300, 900));
        v.risk_score = +riskFromDelay(v.delay_pred_sec).toFixed(3);
        v.updated_at = new Date(now).toISOString();
      }
    }

    // Прогноз опоздания через t секунд: текущее + тренд (насколько медленнее плана едем)
    function predDelay(v, t) {
      const rate = 1 - (v._speedAvg / 3.6) / PLAN_SPEED; // сек опоздания за сек пути
      return v.delay_now_sec + rate * Math.min(t, 900) * 0.6;
    }

    const pub = (v) => {
      const level = App.riskLevel(v.risk_score);
      const out = {
        vehicle_id: v.vehicle_id, route_id: v.route_id, lat: v.lat, lon: v.lon,
        speed: Math.round(v.speed), heading: null,
        delay_now_sec: Math.round(v.delay_now_sec), delay_pred_sec: v.delay_pred_sec, risk_score: v.risk_score,
        updated_at: v.updated_at,
      };
      if (level !== "green") {
        const reason = v._reason || "accumulated_delay";
        Object.assign(out, {
          reason_pattern: reason,
          recommendation: REC_FOR_REASON[reason],
          top_features: v._feats,
          confidence: v._conf,
        });
      }
      return out;
    };

    // ---------- Расписание рейса ----------
    function schedule(v) {
      const r = routeById[v.route_id];
      const now = simNow;
      const dCur = along(v, v._pos);
      const ordered = v._dir > 0 ? r.stops : [...r.stops].reverse();
      let nextFound = false;
      const rows = ordered.map((s) => {
        const d = along(v, s.pos_m);
        const plan = planAt(v, d);
        const row = { stop_id: s.stop_id, name: s.name, lat: s.lat, lon: s.lon, time_plan: new Date(plan).toISOString() };
        if (d <= dCur) {
          // Пройдена. Если прошли её до старта симуляции — «восстанавливаем» факт
          const fact = v._facts[s.stop_id] || plan + v.delay_now_sec * 1000 * (0.4 + 0.6 * (d / Math.max(dCur, 1)));
          Object.assign(row, { status: "passed", time_fact: new Date(fact).toISOString(), delay_sec: Math.round((fact - plan) / 1000) });
        } else {
          const ahead = (d - dCur) / PLAN_SPEED;
          const delay = predDelay(v, ahead);
          Object.assign(row, {
            status: nextFound ? "upcoming" : "next",
            time_pred: new Date(plan + delay * 1000).toISOString(),
            delay_sec: Math.round(delay),
            _ahead: ahead,
          });
          nextFound = true;
        }
        return row;
      });
      // Целевая остановка — первая, чьё плановое время попадает в окно T+10…15 мин
      const upcoming = rows.filter((x) => x.status !== "passed");
      const target = upcoming.find((x) => x._ahead > 600 && x._ahead <= 900)
        || upcoming.slice().sort((a, b) => Math.abs(a._ahead - HORIZON_S) - Math.abs(b._ahead - HORIZON_S))[0];
      if (target) target.is_target = true;
      rows.forEach((x) => delete x._ahead);
      return {
        vehicle_id: v.vehicle_id, route_id: v.route_id,
        direction: `${ordered[0].name} → ${ordered[ordered.length - 1].name}`,
        stops: rows,
      };
    }

    function makeAlert(v) {
      const sch = schedule(v);
      const t = sch.stops.find((s) => s.is_target) || sch.stops[sch.stops.length - 1];
      const p = pub(v);
      return {
        type: "alert.new",
        alert_id: `a-${++alertSeq}`,
        vehicle_id: v.vehicle_id,
        route_id: v.route_id,
        target_stop_id: t.stop_id,
        target_stop_name: t.name,
        delay_pred_sec: v.delay_pred_sec,
        risk_score: v.risk_score,
        confidence: p.confidence,
        eta_incident: t.time_pred || t.time_plan,
        reason_pattern: p.reason_pattern,
        recommendation: p.recommendation,
        top_features: p.top_features,
        model_version: "ens-0.3.1 (mock)",
        created_at: new Date(simNow).toISOString(),
      };
    }

    let last = Date.now();
    function tick(h) {
      const real = Date.now();
      let left = Math.min(5, (real - last) / 1000) * speedFactor; // сколько секунд симуляции прошло
      last = real;
      while (left > 0) { const d = Math.min(2, left); step(d); left -= d; } // мелкими шагами, чтобы не «перепрыгивать» остановки
      h.onVehicles && h.onVehicles(vehicles.map(pub));

      for (const v of vehicles) {
        const has = [...alerts.values()].find((a) => a.vehicle_id === v.vehicle_id);
        if (!has && v.risk_score >= App.config.RISK_RED) {
          const a = makeAlert(v);
          alerts.set(a.alert_id, a);
          h.onAlertNew && h.onAlertNew(a);
        }
        if (has && v.risk_score < App.config.RISK_YELLOW) {
          alerts.delete(has.alert_id);
          h.onAlertResolved && h.onAlertResolved(has.alert_id);
        }
      }
    }

    return {
      name: "mock",
      async getRoutes() {
        return routes.map((r) => ({
          route_id: r.route_id, name: r.name, geometry: r.geometry,
          stops: r.stops.map(({ pos_m, ...s }) => s),
        }));
      },
      async getVehicles() {
        step(0);
        return vehicles.map(pub);
      },
      async getAlerts() {
        return [...alerts.values()];
      },
      async getSchedule(vehicleId) {
        const v = byId.get(vehicleId);
        if (!v) throw new Error("ТС не найдено");
        return schedule(v);
      },
      async getMetrics() {
        return { mae_sec: 61.4, score: 0.58, p50_latency_ms: 18, p95_latency_ms: 42, model_version: "ens-0.3.1 (mock)" };
      },
      async whatif({ scenario, route_id, at_stop_id }) {
        await new Promise((res) => setTimeout(res, 600));
        const k = scenario === "add_reserve" ? [0.3, 0.55] : [0.55, 0.8];
        const list = vehicles.filter((v) => v.route_id === route_id).map((v) => {
          const before = v.delay_pred_sec;
          const after = Math.round(before > 0 ? before * rnd(k[0], k[1]) - 15 : before);
          return {
            vehicle_id: v.vehicle_id,
            delay_before_sec: before, delay_after_sec: after,
            risk_before: v.risk_score, risk_after: +riskFromDelay(after).toFixed(3),
          };
        });
        const avg = (key) => Math.round(list.reduce((s, x) => s + x[key], 0) / (list.length || 1));
        const red = (key) => list.filter((x) => x[key] >= App.config.RISK_RED).length;
        return {
          type: "whatif.result", scenario, route_id, at_stop_id,
          summary: {
            avg_delay_before_sec: avg("delay_before_sec"), avg_delay_after_sec: avg("delay_after_sec"),
            red_before: red("risk_before"), red_after: red("risk_after"),
          },
          vehicles: list,
        };
      },
      // Скорость симуляции (только у мока)
      setSpeed(x) { speedFactor = x; },
      getSpeed() { return speedFactor; },
      start(h) {
        h.onStatus && h.onStatus("mock");
        last = Date.now();
        timer = setInterval(() => tick(h), 1000);
      },
      stop() {
        clearInterval(timer);
      },
    };
  };
})(window.App);
