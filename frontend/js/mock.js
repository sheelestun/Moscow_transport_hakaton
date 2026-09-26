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
    accumulated_delay: "add_reserve",
  };
  // Эффект мер в демо: какую долю опоздания мера «снимает» у опаздывающих ТС маршрута.
  // Если мера бьёт в причину опоздания — эффект сильнее (поэтому рекомендация модели обычно лучшая).
  const MEASURE_EFFECT = { add_reserve: 0.3, adjust_interval: 0.25, detour: 0.25, signal_priority: 0.25, hold_at_stop: 0.2 };
  const MEASURE_FITS = {
    add_reserve: ["accumulated_delay", "long_dwell"],
    adjust_interval: ["bunching", "long_dwell"],
    detour: ["traffic_jam_ahead"],
    signal_priority: ["speed_drop", "traffic_jam_ahead"],
    hold_at_stop: ["bunching"],
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

  // Одно направление маршрута: своя линия, свои остановки и светофоры (с позицией вдоль линии)
  function buildDir(raw, dirRaw, dirIdx) {
    const line = dirRaw.line;
    const cum = G.cumulative(line);
    // остановки — по порядку движения (как в реальном маршруте), проецируем без «прыжков назад»
    let prev = 0;
    const stops = dirRaw.stops.map((s, i) => {
      const p = G.projectAfter(line, cum, [s[1], s[2]], prev);
      prev = p.pos_m;
      return { stop_id: `${raw.route_id}-${dirIdx}-${i + 1}`, name: s[0], lat: p.point[0], lon: p.point[1], pos_m: p.pos_m };
    });
    // Светофоры: реальные координаты (OpenStreetMap), фазы — симуляция.
    // Цикл 80–100 с, зелёный 45–60% цикла, у каждого перекрёстка свой сдвиг фазы.
    const signals = ((App.MOCK_SIGNALS || {})[raw.route_id] || [])
      .map(([lat, lon], i) => {
        const p = G.project(line, cum, [lat, lon]);
        const seed = Math.abs(Math.sin((lat * 1e4 + lon * 1e4) * 12.9898)) * 1000;
        const cycle = 80 + (seed % 20);
        return {
          id: `${raw.route_id}-s${i + 1}`, lat, lon, pos_m: p.pos_m, off_m: p.off_m,
          cycle, green: cycle * (0.6 + (seed % 12) / 100), offset: seed % cycle, // на магистралях зелёный длиннее
        };
      })
      .filter((sg) => sg.off_m < 30) // светофор стоит на этой стороне дороги
      .sort((a, b) => a.pos_m - b.pos_m);
    const length = cum[cum.length - 1];
    // средняя задержка на светофорах на метр пути (для «прогноза модели»): P(красный) × средний остаток красного
    const avgWait = signals.reduce((s, sg) => { const red = sg.cycle - sg.green; return s + (red / sg.cycle) * (red / 2); }, 0);
    const waitPerM = length ? avgWait / length : 0;
    const name = `${stops[0].name} → ${stops[stops.length - 1].name}`;
    return { line, cum, length, stops, signals, name, waitPerM };
  }

  function buildRoute(raw) {
    const dirs = raw.dirs.map((d, i) => buildDir(raw, d, i));
    if (raw.loop) dirs[0].name = `по кругу от «${dirs[0].stops[0].name}»`;
    return { route_id: raw.route_id, name: raw.name, transport_type: raw.transport_type, loop: !!raw.loop, dirs };
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
    // Направление, по которому сейчас едет ТС (0 — «туда», 1 — «обратно»)
    const dirOf = (v) => routeById[v.route_id].dirs[v._d];
    // Расстояние от начала рейса: у каждого направления своя линия, позиция считается вдоль неё
    const along = (v, pos) => pos;
    // Плановое время (мс), когда ТС должно быть в точке d (метры от начала рейса)
    const planAt = (v, d) => v._anchor + (d / PLAN_SPEED) * 1000;

    let tripSeq = 0;
    function newTrip(v, delaySec) {
      // прошлый рейс запоминаем — по нему сверяем прогнозы, если рейс уже закончился
      if (v._facts) v._lastTrip = { id: v._tripId, facts: v._facts, anchor: v._anchor, d: v._d };
      v._tripId = ++tripSeq;
      v._noise = rnd(-40, 40);
      v._facts = {};
      v._anchor = simNow - delaySec * 1000 - (along(v, v._pos) / PLAN_SPEED) * 1000;
    }

    function startTrouble(v, strength = 1) {
      v._trouble = Math.floor(rnd(240, 720)); // проблема длится 4–12 минут
      v._reason = pick(REASONS.filter((r) => r !== "accumulated_delay"));
      v._troubleSpeed = rnd(7, 12) / strength;
      v._noise = rnd(-70, 70); // ошибка «модели» для этого случая (как у настоящей: MAE ~40–60 с)
      v._feats = makeFeatures(v._reason);
      v._conf = +rnd(0.62, 0.9).toFixed(2);
    }

    function makeFeatures(reason) {
      const w = [rnd(0.3, 0.45), rnd(0.15, 0.25), rnd(0.08, 0.15), rnd(0.03, 0.08)];
      return FEATURES_FOR_REASON[reason].map((name, i) => ({ name, contribution: +w[i].toFixed(2) }));
    }

    routes.forEach((r) => {
      // сколько ТС на линии — по длине маршрута (примерно одно на 3 км трассы)
      const total = r.dirs.reduce((s, d) => s + d.length, 0);
      const n = Math.max(4, Math.min(9, Math.round(total / 3000)));
      for (let i = 0; i < n; i++) {
        const v = {
          vehicle_id: String(Math.floor(rnd(120000, 139999))),
          route_id: r.route_id,
          _d: i % r.dirs.length,
          _pos: r.dirs[i % r.dirs.length].length * ((Math.floor(i / r.dirs.length) + rnd(0.1, 0.8)) / Math.ceil(n / r.dirs.length)),
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
    [1, 9, 14, 21].map((i) => vehicles[i % vehicles.length]).forEach((v, i) => {
      newTrip(v, rnd(110, 170));
      startTrouble(v, i === 3 ? 0.7 : 1);
      v._trouble = rnd(300, 700);
    });

    // dt — секунды симуляции
    function step(dt) {
      simNow += dt * 1000;
      const now = simNow;
      for (const v of vehicles) {
        const r = routeById[v.route_id];
        const D = r.dirs[v._d];
        if (v._trouble > 0) {
          v._trouble -= dt;
          v.speed = clamp(v._troubleSpeed + rnd(-1.5, 1.5), 2, 14);
          // «Резкое падение скорости перед перекрёстком»: в демо — дольше стоит на красном

        } else {
          if (v._reason !== "accumulated_delay") {
            // проблема закончилась, но опоздание ещё не отыграно
            v._reason = "accumulated_delay";
            v._feats = makeFeatures(v._reason);
          }
          // частота новых проблем — в реальном времени, чтобы при любой скорости симуляции было что показать
          if (Math.random() < (0.001 * (24 / vehicles.length) * Math.sqrt(speedFactor / 5) * dt) / speedFactor) startTrouble(v);
          // Водитель догоняет график, если опаздывает, и придерживается, если идёт с опережением
          const target = v.delay_now_sec > 20 ? 22 : v.delay_now_sec < -20 ? 14 : 18;
          if (v.speed < 8) v.speed = 12; // тронулся после светофора
          v.speed = clamp(v.speed + (target - v.speed) * 0.3 + rnd(-1, 1), 10, 28);
        }
        v._speedAvg += (v.speed - v._speedAvg) * 0.08 * dt; // сглаженная скорость
        // Применённая мера: опоздание постепенно «отыгрывается» (сдвигаем плановое время)
        if (v._recover > 0) {
          const d = Math.min(v._recover, 1.2 * dt);
          v._anchor += d * 1000;
          v._recover -= d;
        }

        // Движение по линии маршрута (реальное время)
        const before = along(v, v._pos);
        let move = (v.speed / 3.6) * dt;
        // Светофор впереди горит красным — останавливаемся перед стоп-линией (за 8 м)
        v._wait = null;
        for (const sg of D.signals) {
          const ds = along(v, sg.pos_m);
          if (ds <= before - 1 || ds > before + move + 8) continue;
          const st = signalState(r, sg);
          if (st.state !== "red") continue;
          move = Math.max(0, Math.min(move, ds - 8 - before));
          v._wait = { sig: sg, left: st.left };
          break;
        }
        // «Стоит на красном» — только когда реально остановился (а не подъезжает)
        if (v._wait && move < 0.5) {
          v._waitSince = v._waitSince || now;
          v.speed = 0;
        } else {
          v._wait = null;
          v._waitSince = null;
        }
        v._pos = clamp(v._pos + move, 0, D.length);
        // Фиксируем фактическое время прохождения остановок (включая конечную)
        const after = along(v, v._pos);
        for (const s of D.stops) {
          const d = along(v, s.pos_m);
          if (d > before && d <= after + 0.5) v._facts[s.stop_id] = now;
        }
        if (v._pos >= D.length) {
          // конечная: разворот, новый рейс в обратную сторону — по своей стороне дороги
          v._d = (v._d + 1) % r.dirs.length; // у кольцевого одно направление — едет дальше по кругу
          v._pos = 0;
          newTrip(v, rnd(-30, 60));
        }
        const DN = r.dirs[v._d];
        const [lat, lon] = G.pointAt(DN.line, DN.cum, v._pos);
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
    // Прогноз опоздания через t секунд. В демо «модель» знает, сколько продлится проблема,
    // и ошибается на v._noise — как настоящая модель с MAE около 40–60 секунд.
    function predDelay(v, t) {
      return futureDelay(v, t) + (v._noise || 0) * Math.min(1, t / HORIZON_S);
    }
    // «Истинное» будущее опоздание при текущей обстановке
    function futureDelay(v, t) {
      let d = v.delay_now_sec;
      const tl = Math.max(0, v._trouble || 0);
      const tr = Math.min(t, tl);
      if (tr > 0) d += (1 - (v._troubleSpeed / 3.6) / PLAN_SPEED) * tr; // во время проблемы копится
      const rest = t - tr;
      if (v._recover > 0) d -= Math.min(v._recover, 1.2 * t);        // применённая мера
      // ожидаемые ожидания на светофорах впереди (если на маршруте не включён приоритет)
      const r = routeById[v.route_id];
      if (!(r._priorityUntil > simNow)) d += r.dirs[v._d].waitPerM * PLAN_SPEED * t;
      const catchUp = 1 - (22 / 3.6) / PLAN_SPEED;                     // < 0: догоняет график
      d = d > 0 ? Math.max(0, d + catchUp * rest) : Math.min(0, d - catchUp * rest);
      return d;
    }

    const pub = (v) => {
      const level = App.riskLevel(v.risk_score);
      const out = {
        vehicle_id: v.vehicle_id, route_id: v.route_id, direction_id: v._d, lat: v.lat, lon: v.lon, is_reserve: !!v._reserve,
        speed: Math.round(v.speed), heading: null,
        delay_now_sec: Math.round(v.delay_now_sec), delay_pred_sec: v.delay_pred_sec, risk_score: v.risk_score,
        updated_at: v.updated_at,
      };
      // Стоит на красном — диспетчер видит, почему ТС не едет
      if (v._wait) {
        out.waiting_signal = {
          signal_id: v._wait.sig.id, lat: v._wait.sig.lat, lon: v._wait.sig.lon,
          waited_sec: Math.round((simNow - (v._waitSince || simNow)) / 1000), left_sec: v._wait.left,
        };
      }
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
      const ordered = r.dirs[v._d].stops;
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
        delay_pred_sec: t.delay_sec != null ? t.delay_sec : v.delay_pred_sec, // прогноз именно для целевой остановки
        risk_score: v.risk_score,
        confidence: p.confidence,
        eta_incident: t.time_pred || t.time_plan,
        reason_pattern: p.reason_pattern,
        recommendation: p.recommendation,
        top_features: p.top_features,
        model_version: "ens-0.3.1 (mock)",
        created_at: new Date(simNow).toISOString(),
        _trip: v._tripId, _plan: new Date(t.time_plan).getTime(),
      };
    }

    // Состояние светофора сейчас: "green" | "red" | "priority" (включён приоритет для ОТ на маршруте)
    function signalState(route, sig) {
      if (route._priorityUntil > simNow) return { state: "priority", left: Math.round((route._priorityUntil - simNow) / 1000) };
      const t = (simNow / 1000 + sig.offset) % sig.cycle;
      return t < sig.green
        ? { state: "green", left: Math.round(sig.green - t) }
        : { state: "red", left: Math.round(sig.cycle - t) };
    }

    // Прогноз после меры (детерминированно, чтобы цифры не прыгали между расчётами)
    function afterMeasure(v, scenario) {
      const before = v.delay_pred_sec;
      if (before <= 30) return before; // идущим по графику мера не нужна
      let k = MEASURE_EFFECT[scenario] || 0;
      if (REC_FOR_REASON[v._reason] === scenario) k += 0.45;        // ровно то, что советует модель
      else if (MEASURE_FITS[scenario] && MEASURE_FITS[scenario].includes(v._reason)) k += 0.2; // тоже бьёт в причину
      const hash = [...v.vehicle_id].reduce((a, c) => a + c.charCodeAt(0), 0) % 10; // небольшой разброс по ТС
      k = Math.min(0.9, k * (0.9 + hash / 50));
      return Math.round(before * (1 - k));
    }

    // Резервное ТС выходит с конечной и идёт по графику
    let reserveSeq = 1;
    function addReserve(route_id) {
      const r = routeById[route_id];
      const v = {
        vehicle_id: `Р${reserveSeq++}-${route_id}`, route_id, _reserve: true,
        _pos: 0, _d: 0, _trouble: 0, _reason: "accumulated_delay",
        _feats: makeFeatures("accumulated_delay"), _conf: 0.8, speed: 22, _speedAvg: 20,
      };
      newTrip(v, -20);
      vehicles.push(v);
      byId.set(v.vehicle_id, v);
      step(0);
      return v;
    }

    let offline = false;
    let handlers = null;
    let last = Date.now();
    function tick(h) {
      const real = Date.now();
      let left = Math.min(5, (real - last) / 1000) * speedFactor; // сколько секунд симуляции прошло
      last = real;
      while (left > 0) { const d = Math.min(2, left); step(d); left -= d; } // мелкими шагами, чтобы не «перепрыгивать» остановки
      // «Обрыв связи»: симуляция идёт дальше, но данные на дашборд не приходят
      if (offline) return;
      h.onVehicles && h.onVehicles(vehicles.map(pub));

      // Время инцидента наступило — сверяем прогноз с фактом
      for (const a of [...alerts.values()]) {
        if (new Date(a.eta_incident).getTime() > simNow) continue;
        const v = byId.get(a.vehicle_id);
        alerts.delete(a.alert_id);
        if (!v) continue;
        // Факт: если рейс тот же — из текущего расписания, если уже закончился — из прошлого рейса
        let factDelay = null;
        if (v._tripId === a._trip) {
          const st = schedule(v).stops.find((s) => s.stop_id === a.target_stop_id);
          if (st) factDelay = st.delay_sec;
        } else if (v._lastTrip && v._lastTrip.id === a._trip && v._lastTrip.facts[a.target_stop_id]) {
          factDelay = Math.round((v._lastTrip.facts[a.target_stop_id] - a._plan) / 1000);
        }
        if (factDelay == null) continue;
        const stop = { delay_sec: factDelay };
        h.onAlertVerified && h.onAlertVerified({
          type: "alert.verified", alert_id: a.alert_id, vehicle_id: a.vehicle_id, route_id: a.route_id,
          target_stop_id: a.target_stop_id, target_stop_name: a.target_stop_name,
          delay_pred_sec: a.delay_pred_sec, delay_fact_sec: stop.delay_sec,
          verified_at: new Date(simNow).toISOString(),
        });
      }

      for (const v of vehicles) {
        const has = [...alerts.values()].find((a) => a.vehicle_id === v.vehicle_id);
        if (!has && v.risk_score >= App.config.RISK_RED) {
          const a = makeAlert(v);
          // по одной остановке рейса — один алерт
          v._alerted = v._alerted && v._alerted.trip === v._tripId ? v._alerted : { trip: v._tripId, stops: new Set() };
          if (v._alerted.stops.has(a.target_stop_id)) { alertSeq--; continue; }
          // риск по прогнозу именно для целевой остановки; если он не «красный» — алерт не нужен
          a.risk_score = +riskFromDelay(a.delay_pred_sec).toFixed(3);
          if (a.risk_score < App.config.RISK_RED) { alertSeq--; continue; }
          v._alerted.stops.add(a.target_stop_id);
          alerts.set(a.alert_id, a);
          h.onAlertNew && h.onAlertNew(a);
        }
        // Алерт не снимаем раньше времени: в момент инцидента сверим прогноз с фактом (см. выше)
      }
    }

    return {
      name: "mock",
      async getRoutes() {
        return routes.map((r) => ({
          route_id: r.route_id, name: r.name, transport_type: r.transport_type,
          geometry: r.dirs[0].line,                               // линия «туда» (для совместимости)
          stops: r.dirs[0].stops.map(({ pos_m, ...s }) => s),
          // оба направления: каждое по своей стороне дороги
          directions: r.dirs.map((d, i) => ({
            direction_id: i, name: d.name, geometry: d.line,
            stops: d.stops.map(({ pos_m, ...s }) => s),
          })),
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
        // Формат как у ML-сервиса (GET /metrics/model); MAE — реальный с labels_test (statistics/tables/model_metrics.csv)
        return { mae_test_s: 43.7, latency_ms_p50: 18, model_version: "catboost-ensemble-v1 (mock)" };
      },
      async getWorstStops(limit = 10) {
        // Топ-10 по прогнозируемой задержке среди всех предстоящих остановок.
        const agg = new Map();
        for (const v of vehicles) {
          if (v._reserve) continue;
          const sch = schedule(v);
          for (const s of sch.stops) {
            if (s.status === "passed" || !Number.isFinite(s.delay_sec)) continue;
            const key = `${v.route_id}|${sch.direction_id ?? 0}|${s.stop_id}`;
            const cell = agg.get(key) || {
              route_id: v.route_id, direction_id: sch.direction_id ?? 0,
              stop_id: s.stop_id, name: s.name, lat: s.lat, lon: s.lon,
              vehicles: 0, sum: 0, max: 0,
            };
            cell.vehicles += 1; cell.sum += s.delay_sec;
            if (s.delay_sec > cell.max) cell.max = s.delay_sec;
            agg.set(key, cell);
          }
        }
        return [...agg.values()]
          .map((c) => ({
            route_id: c.route_id, direction_id: c.direction_id,
            stop_id: c.stop_id, name: c.name, lat: c.lat, lon: c.lon,
            avg_delay_sec: Math.round(c.sum / c.vehicles),
            max_delay_sec: Math.round(c.max),
            vehicles: c.vehicles,
          }))
          .filter((r) => r.avg_delay_sec >= 30)
          .sort((a, b) => b.avg_delay_sec - a.avg_delay_sec)
          .slice(0, limit);
      },
      // What-if: как изменится прогноз у ТС маршрута, если применить меру
      async whatif({ scenario, route_id, at_stop_id }) {
        await new Promise((res) => setTimeout(res, 300 + Math.random() * 400)); // «модель считает»
        const list = vehicles.filter((v) => v.route_id === route_id && !v._reserve).map((v) => {
          const before = v.delay_pred_sec;
          return {
            vehicle_id: v.vehicle_id,
            delay_before_sec: before, delay_after_sec: afterMeasure(v, scenario),
            risk_before: v.risk_score, risk_after: +riskFromDelay(afterMeasure(v, scenario)).toFixed(3),
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
      // Применить меру в симуляции (только у мока; в живом режиме это решает диспетчер по-настоящему)
      async applyMeasure({ scenario, route_id }) {
        for (const v of vehicles.filter((x) => x.route_id === route_id)) {
          const gain = v.delay_pred_sec - afterMeasure(v, scenario);
          if (gain > 0) v._recover = (v._recover || 0) + Math.max(0, v.delay_now_sec) * (gain / Math.max(v.delay_pred_sec, 1));
          if (MEASURE_FITS[scenario].includes(v._reason)) v._trouble = 0; // причина устранена
        }
        if (scenario === "add_reserve") addReserve(route_id);
        if (scenario === "signal_priority") routeById[route_id]._priorityUntil = simNow + 20 * 60000; // зелёная волна на 20 мин
        return { ok: true };
      },
      // Светофоры маршрута с текущей фазой (только у мока: фазы — симуляция)
      getSignals(route_id) {
        const r = routeById[route_id];
        if (!r) return [];
        const seen = new Set(), out = [];
        for (const d of r.dirs) for (const sg of d.signals) {
          if (seen.has(sg.id)) continue;
          seen.add(sg.id);
          out.push({ signal_id: sg.id, lat: sg.lat, lon: sg.lon, ...signalState(r, sg) });
        }
        return out;
      },
      // Скорость симуляции (только у мока)
      setSpeed(x) { speedFactor = x; },
      getSpeed() { return speedFactor; },
      // Имитация обрыва связи с сервером (для демо критерия «надёжность»)
      setOffline(on) {
        offline = on;
        if (handlers) handlers.onStatus && handlers.onStatus(on ? "degraded" : "mock");
      },
      isOffline() { return offline; },
      start(h) {
        handlers = h;
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
