// Левая панель диспетчера: список алертов и маршрутов ИЛИ карточка выбранного ТС.
window.App = window.App || {};

(function (App) {
  const $ = (id) => document.getElementById(id);
  const t = (dict, code) => App.esc(App.labels.t(dict, code));

  App.sidebar = {
    // ================= Список =================
    renderAlerts(alerts, { onSelect }) {
      const ul = $("alerts");
      $("alerts-count").textContent = alerts.length;
      $("alerts-count").classList.toggle("badge--hot", alerts.length > 0);

      if (!alerts.length) {
        ul.innerHTML = `<li class="empty">Всё по графику. Опозданий в ближайшие 15 минут не ожидается.</li>`;
        return;
      }
      ul.innerHTML = alerts.map((a) => {
        const level = App.riskLevel(a.risk_score);
        return `
        <li>
          <button class="alert alert--${level} ${a._fresh ? "is-fresh" : ""}" data-vid="${App.esc(a.vehicle_id)}">
            <span class="alert__delay">${App.fmtDelayShort(a.delay_pred_sec)}</span>
            <span class="alert__body">
              <span class="alert__top"><span class="route-chip">${App.esc(a.route_id)}</span> ТС ${App.esc(a.vehicle_id)}</span>
              <span class="alert__where">к «${App.esc(a.target_stop_name || a.target_stop_id)}» · ${App.fmtIn(a.eta_incident)}</span>
              <span class="alert__reason">${t("reasons", a.reason_pattern)}</span>
            </span>
            <span class="alert__risk">${App.pct(a.risk_score)}</span>
          </button>
        </li>`;
      }).join("");
      ul.querySelectorAll("[data-vid]").forEach((b) => (b.onclick = () => onSelect(b.dataset.vid)));
    },

    renderRoutes(stats, { onSelect }) {
      $("routes").innerHTML = stats.map((r) => `
        <li>
          <button class="route" data-id="${App.esc(r.route_id)}">
            <span class="dot dot--${r.level}"></span>
            <span class="route-chip">${App.esc(r.route_id)}</span>
            <span class="route__name">${App.esc(r.name)}</span>
            <span class="route__stat">${r.red ? `<b class="t-red">${r.red}</b>` : ""}${r.yellow ? `<b class="t-yellow">${r.yellow}</b>` : ""}<span>${r.total} ТС</span></span>
          </button>
        </li>`).join("");
      $("routes").querySelectorAll("[data-id]").forEach((b) => (b.onclick = () => onSelect(b.dataset.id)));
    },

    // ================= Карточка ТС =================
    openVehicle(ctx) {
      $("view-list").hidden = true;
      $("view-vehicle").hidden = false;
      $("view-vehicle").innerHTML = `
        <div class="vh__nav">
          <button class="link" id="vh-back">← Все ТС под риском</button>
        </div>
        <section id="vh-hero"></section>
        <section id="vh-why" class="block"></section>
        <section id="vh-act"></section>
        <section id="vh-sched" class="block"></section>`;
      $("vh-back").onclick = ctx.onBack;
      this._ctx = ctx;
      this._actKey = null;
      this._whyKey = null;
    },

    closeVehicle() {
      $("view-vehicle").hidden = true;
      $("view-vehicle").innerHTML = "";
      $("view-list").hidden = false;
      this._ctx = null;
    },

    // Вызывается каждую секунду: обновить цифры
    updateVehicle({ vehicle: v, route, alert, schedule, whatif }) {
      if (!this._ctx || !v) return;
      const level = App.riskLevel(v.risk_score);
      const target = schedule && schedule.stops.find((s) => s.is_target);

      // --- Шапка ---
      const whereName = alert ? alert.target_stop_name || alert.target_stop_id : target && target.name;
      const whereTime = alert ? alert.eta_incident : target && (target.time_pred || target.time_plan);
      $("vh-hero").innerHTML = `
        <div class="hero hero--${level}">
          <div class="hero__top">
            <span class="route-chip route-chip--lg">${App.esc(v.route_id)}</span>
            <div class="hero__title">
              <b>ТС ${App.esc(v.vehicle_id)}</b>
              <span class="muted">${App.esc(schedule ? schedule.direction : route ? route.name : "")}</span>
            </div>
            <span class="level level--${level}">${App.levelName[level]}</span>
          </div>
          <div class="hero__label">Прогноз через 10–15 минут</div>
          <div class="hero__big">${App.fmtDelay(v.delay_pred_sec)}</div>
          ${whereName ? `<div class="hero__where">к остановке «${App.esc(whereName)}» · ${App.fmtTime(whereTime)} (${App.fmtIn(whereTime)})</div>` : ""}
          <div class="hero__grid">
            <div><span class="muted">Сейчас</span><b class="t-${App.delayLevel(v.delay_now_sec)}">${App.fmtDelayShort(v.delay_now_sec)}</b></div>
            <div><span class="muted">Скорость</span><b>${v.speed ?? "—"} км/ч</b></div>
            <div><span class="muted">Риск</span><b>${App.pct(v.risk_score)}</b></div>
            <div><span class="muted">Уверенность</span><b>${v.confidence != null ? App.pct(v.confidence) : alert && alert.confidence != null ? App.pct(alert.confidence) : "—"}</b></div>
          </div>
        </div>`;

      // --- Почему (перерисовываем только при смене причины) ---
      const reason = (alert && alert.reason_pattern) || v.reason_pattern;
      const feats = (alert && alert.top_features) || v.top_features || [];
      const whyKey = level === "green" ? "green" : reason + feats.map((f) => f.name).join();
      if (whyKey !== this._whyKey) {
        this._whyKey = whyKey;
        $("vh-why").innerHTML = level === "green"
          ? `<h4>Почему такой прогноз</h4><p class="why__ok">Идёт по графику. Модель не видит признаков опоздания в ближайшие 15 минут.</p>`
          : `<h4>Почему опаздывает</h4>
             <p class="why__title">${t("reasons", reason)}</p>
             <p class="why__text">${t("explanations", reason)}</p>
             ${feats.length ? `<div class="feats"><div class="muted small">Что сильнее всего повлияло на прогноз</div>${featBars(feats)}</div>` : ""}`;
      }

      // --- Что делать (перерисовываем только при смене уровня/алерта, чтобы не сбить выбор в What-if) ---
      const actKey = level === "green" ? "green" : `${(alert && alert.recommendation) || v.recommendation}|${alert ? alert.alert_id : ""}`;
      if (actKey !== this._actKey) {
        this._actKey = actKey;
        $("vh-act").innerHTML = level === "green" ? "" : actHtml((alert && alert.recommendation) || v.recommendation, alert, whatif);
        bindAct(this._ctx, v);
      }

      // --- Расписание ---
      $("vh-sched").innerHTML = schedule ? scheduleHtml(schedule) : `<h4>Расписание</h4><p class="muted">Загружаем…</p>`;
    },
  };

  function featBars(feats) {
    const list = feats.slice().sort((a, b) => b.contribution - a.contribution);
    const max = Math.max(...list.map((f) => f.contribution), 0.01);
    return list.map((f) => `
      <div class="feat">
        <span>${t("features", f.name)}</span>
        <span class="feat__bar"><i style="width:${(f.contribution / max) * 100}%"></i></span>
        <span class="feat__val">${Math.round(f.contribution * 100)}%</span>
      </div>`).join("");
  }

  function actHtml(rec, alert, whatif) {
    return `
      <div class="block block--rec">
        <h4>Рекомендация</h4>
        <p class="rec">${t("recommendations", rec)}</p>
        <div class="whatif-box">
          <div class="muted small">Проверить заранее: что будет, если…</div>
          <div class="scenarios">
            ${Object.entries(App.labels.scenarios).map(([k, v], i) => `
              <label class="scenario"><input type="radio" name="scenario" id="sc-${k}" value="${k}" ${i === 0 ? "checked" : ""}> ${v}</label>`).join("")}
          </div>
          <div class="row">
            <button class="btn" id="whatif-btn">Рассчитать эффект</button>
            ${alert ? `<button class="btn btn--ghost" id="ack-btn">Принято</button>` : ""}
          </div>
          <div id="whatif-result">${whatif ? whatifHtml(whatif) : ""}</div>
        </div>
      </div>`;
  }

  function bindAct(ctx, v) {
    const btn = $("whatif-btn");
    if (btn) btn.onclick = async () => {
      const scenario = document.querySelector('input[name="scenario"]:checked').value;
      btn.disabled = true;
      btn.textContent = "Считаем…";
      try {
        $("whatif-result").innerHTML = whatifHtml(await ctx.onWhatif(scenario));
      } catch (e) {
        $("whatif-result").innerHTML = `<p class="error">Не удалось рассчитать: ${App.esc(e.message)}. Проверьте связь с сервером и попробуйте ещё раз.</p>`;
      } finally {
        btn.disabled = false;
        btn.textContent = "Рассчитать эффект";
      }
    };
    const ack = $("ack-btn");
    if (ack) ack.onclick = ctx.onAck;
  }

  function whatifHtml(w) {
    const s = w.summary || {};
    const max = Math.max(60, ...(w.vehicles || []).map((v) => Math.abs(v.delay_before_sec)));
    const bar = (sec) => `${Math.max(2, (sec / max) * 100)}%`;
    return `
      <div class="whatif">
        <div class="whatif__summary">
          <div><span class="muted small">Среднее опоздание на маршруте</span>
            <b>${App.fmtDelayShort(s.avg_delay_before_sec)} → <span class="t-green">${App.fmtDelayShort(s.avg_delay_after_sec)}</span></b></div>
          <div><span class="muted small">Опаздывающих ТС</span>
            <b>${s.red_before ?? "—"} → <span class="t-green">${s.red_after ?? "—"}</span></b></div>
        </div>
        ${(w.vehicles || []).map((v) => `
          <div class="wrow">
            <span>ТС ${App.esc(v.vehicle_id)}</span>
            <span class="wrow__bars"><i class="b-before" style="width:${bar(v.delay_before_sec)}"></i><i class="b-after" style="width:${bar(v.delay_after_sec)}"></i></span>
            <span class="wrow__val">${App.fmtDelayShort(v.delay_before_sec)} → ${App.fmtDelayShort(v.delay_after_sec)}</span>
          </div>`).join("")}
        <div class="legend-inline muted small"><i class="b-before"></i>сейчас <i class="b-after"></i>после меры</div>
      </div>`;
  }

  function scheduleHtml(sch) {
    const rows = sch.stops.map((s) => {
      const lvl = App.delayLevel(s.delay_sec);
      const real = s.status === "passed" ? s.time_fact : s.time_pred;
      const tag = s.is_target ? `<span class="st__tag">цель прогноза</span>` : s.status === "next" ? `<span class="st__tag st__tag--next">следующая</span>` : "";
      return `
        <li class="st st--${s.status} ${s.is_target ? "st--target" : ""}">
          <span class="st__rail"><i class="st__dot dot--${s.status === "passed" ? "passed" : lvl}"></i></span>
          <span class="st__name">${App.esc(s.name)}${tag}</span>
          <span class="st__time">${App.fmtTime(s.time_plan)}</span>
          <span class="st__time ${s.status === "passed" ? "" : "st__time--pred"}">${real ? App.fmtTime(real) : "—"}</span>
          <span class="st__dev dev--${s.status === "passed" ? "passed" : lvl}">${App.fmtDelayShort(s.delay_sec)}</span>
        </li>`;
    }).join("");
    return `
      <h4>Расписание рейса</h4>
      <div class="sched__head"><span></span><span>Остановка</span><span>План</span><span>Факт/прогноз</span><span>Откл.</span></div>
      <ol class="sched">${rows}</ol>
      <p class="muted small">Прогноз времени для следующих остановок — от ML-модели. Прошедшие — фактическое время по телеметрии.</p>`;
  }
})(window.App);
