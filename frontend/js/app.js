// Главный файл: хранит состояние и связывает источник данных, карту и панель.
window.App = window.App || {};

(function (App) {
  const cfg = App.config;
  const source = App.createSource(cfg);
  const $ = (id) => document.getElementById(id);

  // ---------- Состояние ----------
  const state = {
    routes: new Map(),   // route_id -> route
    vehicles: new Map(), // vehicle_id -> vehicle
    alerts: new Map(),   // alert_id -> alert
    acked: new Set(),    // алерты, которые диспетчер отметил «Принято»
    whatif: new Map(),   // vehicle_id -> последний результат What-if
    selectedId: null,    // выбранное ТС
    schedule: null,      // расписание выбранного ТС
    status: "connecting",
    lastUpdate: null,
    riskMode: false,
  };

  // ---------- Производные данные ----------
  const activeAlerts = () =>
    [...state.alerts.values()].filter((a) => !state.acked.has(a.alert_id)).sort((a, b) => b.risk_score - a.risk_score);

  const alertFor = (vehicleId) => activeAlerts().find((a) => a.vehicle_id === vehicleId);

  function routeStats() {
    const stats = {};
    for (const r of state.routes.values()) stats[r.route_id] = { route_id: r.route_id, name: r.name, total: 0, red: 0, yellow: 0 };
    for (const v of state.vehicles.values()) {
      const s = stats[v.route_id];
      if (!s) continue;
      s.total++;
      const l = App.riskLevel(v.risk_score);
      if (l === "red") s.red++;
      if (l === "yellow") s.yellow++;
    }
    const rank = { red: 0, yellow: 1, green: 2 };
    return Object.values(stats)
      .map((s) => ({ ...s, level: s.red ? "red" : s.yellow ? "yellow" : "green" }))
      .sort((a, b) => rank[a.level] - rank[b.level] || b.red - a.red || b.yellow - a.yellow);
  }

  // ---------- Отрисовка ----------
  function renderKpis() {
    const c = { red: 0, yellow: 0, green: 0 };
    for (const v of state.vehicles.values()) c[App.riskLevel(v.risk_score)]++;
    $("kpi-total").textContent = state.vehicles.size;
    $("kpi-red").textContent = c.red;
    $("kpi-yellow").textContent = c.yellow;
    $("kpi-green").textContent = c.green;
  }

  function renderStatus() {
    $("status").className = "status status--" + state.status;
    $("status-text").textContent = {
      connecting: "подключение…",
      live: "онлайн",
      mock: "демо-данные",
      degraded: "нет связи · данные на " + (state.lastUpdate ? App.fmtTime(state.lastUpdate, true) : "—"),
    }[state.status];
  }

  function renderList() {
    App.sidebar.renderAlerts(activeAlerts(), { onSelect: selectVehicle });
  }

  function renderRoutes() {
    const stats = routeStats();
    App.sidebar.renderRoutes(stats, {
      onSelect: (id) => {
        // клик по маршруту: выбрать самое проблемное ТС на нём
        const vs = [...state.vehicles.values()].filter((v) => v.route_id === id).sort((a, b) => b.risk_score - a.risk_score);
        if (vs[0]) selectVehicle(vs[0].vehicle_id);
      },
    });
    App.map.setRouteLevels(Object.fromEntries(stats.map((s) => [s.route_id, s.level])));
  }

  function renderVehicle() {
    if (!state.selectedId) return;
    const v = state.vehicles.get(state.selectedId);
    App.sidebar.updateVehicle({
      vehicle: v,
      route: state.routes.get(v.route_id),
      alert: alertFor(v.vehicle_id),
      schedule: state.schedule,
      whatif: state.whatif.get(v.vehicle_id),
    });
  }

  // ---------- Выбор ТС ----------
  async function selectVehicle(id) {
    const v = state.vehicles.get(id);
    if (!v) return;
    state.selectedId = id;
    state.schedule = null;
    const route = state.routes.get(v.route_id);

    App.sidebar.openVehicle({
      onBack: clearSelection,
      onAck: () => {
        const a = alertFor(id);
        if (a) state.acked.add(a.alert_id);
        renderVehicle();
      },
      onWhatif: async (scenario) => {
        const a = alertFor(id);
        const res = await source.whatif({ scenario, route_id: v.route_id, at_stop_id: a ? a.target_stop_id : null });
        state.whatif.set(id, res);
        return res;
      },
    });
    renderVehicle();
    App.map.focusRoute(route, v);
    await refreshSchedule();
  }

  async function refreshSchedule() {
    const id = state.selectedId;
    if (!id) return;
    try {
      const sch = await source.getSchedule(id);
      if (state.selectedId !== id) return; // пока грузили — выбрали другое ТС
      const first = !state.schedule;
      state.schedule = sch;
      const v = state.vehicles.get(id);
      if (first) App.map.select(v, state.routes.get(v.route_id), sch);
      else App.map.updateSelection(v, state.routes.get(v.route_id), sch);
      renderVehicle();
    } catch (e) {
      console.warn("Расписание не загрузилось:", e);
    }
  }

  function clearSelection() {
    state.selectedId = null;
    state.schedule = null;
    App.sidebar.closeVehicle();
    App.map.clearSelection();
    renderList();
  }

  // ---------- События из источника данных ----------
  const handlers = {
    onStatus(s) {
      state.status = s;
      renderStatus();
    },
    onVehicles(list) {
      for (const v of list) state.vehicles.set(v.vehicle_id, { ...state.vehicles.get(v.vehicle_id), ...v });
      state.lastUpdate = new Date();
      App.map.updateVehicles(list);
      renderKpis();
      renderVehicle();
    },
    onAlertNew(a) {
      a._fresh = true;
      state.alerts.set(a.alert_id, a);
      setTimeout(() => (a._fresh = false), 4000);
      renderList();
    },
    onAlertResolved(id) {
      state.alerts.delete(id);
      renderList();
    },
  };

  // ---------- Запуск ----------
  async function init() {
    renderStatus();
    const tickClock = () => ($("clock").textContent = App.fmtTime(new Date(App.now()), true));
    tickClock();
    setInterval(tickClock, 1000);

    // Карту не ждём: панель и данные показываем сразу, маршруты дорисуются, когда подложка загрузится
    App.map.init(cfg, { onVehicle: selectVehicle, onEmptyClick: () => state.selectedId && clearSelection() })
      .then(() => { renderRoutes(); if (state.selectedId) refreshSchedule(); });

    // Скорость симуляции — только для демо-данных
    const speedBox = $("speed");
    if (source.setSpeed) {
      speedBox.hidden = false;
      const mark = () => speedBox.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.x === source.getSpeed())));
      speedBox.querySelectorAll("button").forEach((b) => (b.onclick = () => { source.setSpeed(+b.dataset.x); mark(); }));
      mark();
    }

    // Переключатель «риск на маршрутах»
    $("risk-toggle").onchange = (e) => App.map.setRiskMode(e.target.checked);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && state.selectedId) clearSelection(); });

    try {
      const [routes, vehicles, alerts] = await Promise.all([source.getRoutes(), source.getVehicles(), source.getAlerts()]);
      routes.forEach((r) => state.routes.set(r.route_id, r));
      App.map.drawRoutes(routes);
      handlers.onVehicles(vehicles);
      alerts.forEach((a) => state.alerts.set(a.alert_id, a));
    } catch (e) {
      console.error("Не удалось загрузить начальные данные:", e);
      state.status = "degraded";
      renderStatus();
    }

    source.getMetrics()
      .then((m) => ($("model-info").innerHTML = `MAE <b>${Math.round(m.mae_sec)} с</b> · p95 <b>${m.p95_latency_ms} мс</b>`))
      .catch(() => {});

    renderList();
    renderRoutes();
    source.start(handlers);

    setInterval(renderRoutes, 2000);            // светофор маршрутов
    setInterval(refreshSchedule, 2000);         // расписание выбранного ТС
    setInterval(() => !state.selectedId && renderList(), 15000); // «через N мин» в списке
    setInterval(() => state.status === "degraded" && renderStatus(), 5000);
  }

  App.state = state; // для отладки в консоли браузера
  document.addEventListener("DOMContentLoaded", init);
})(window.App);
