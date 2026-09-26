// Окно What-if: сравнение всех мер для маршрута и кнопка «Применить».
// Открывается из карточки автобуса кнопкой «Что если…».
window.App = window.App || {};

(function (App) {
  const $ = (id) => document.getElementById(id);
  let ctx = null;       // {source, vehicle, route, alert, rec, onApplied}
  let results = [];     // [{scenario, res}]
  let chosen = null;    // выбранная мера
  let lastFocus = null;

  App.whatif = {
    isOpen: () => !$("whatif-modal").hidden,

    async open(c) {
      ctx = c;
      results = [];
      chosen = null;
      lastFocus = document.activeElement;
      const modal = $("whatif-modal");
      modal.hidden = false;
      $("wi-title").textContent = `Сравнение мер · маршрут ${c.vehicle.route_id}`;
      $("wi-sub").textContent = c.route ? c.route.name : "";
      $("wi-body").innerHTML = `<div class="wi-loading">Модель пересчитывает прогноз для каждой меры…</div>`;
      $("wi-apply").hidden = true;
      $("wi-close").focus();

      // Считаем все меры параллельно
      const codes = Object.keys(App.labels.scenarios);
      try {
        const all = await Promise.all(codes.map((scenario) =>
          c.source.whatif({ scenario, route_id: c.vehicle.route_id, at_stop_id: c.alert ? c.alert.target_stop_id : null })
            .then((res) => ({ scenario, res }))));
        if (ctx !== c) return; // окно успели закрыть
        results = all;
        const best = bestOf(all);
        chosen = App.labels.scenarios[c.rec] ? c.rec : best;
        render();
      } catch (e) {
        $("wi-body").innerHTML = `<p class="error">Не удалось рассчитать: ${App.esc(e.message)}. Проверьте связь с сервером и откройте окно ещё раз.</p>`;
      }
    },

    close() {
      $("whatif-modal").hidden = true;
      ctx = null;
      if (lastFocus && lastFocus.focus) lastFocus.focus();
    },
  };

  // Лучшая мера — та, после которой меньше всего опаздывающих, при равенстве — меньше среднее опоздание
  function bestOf(all) {
    return all.slice().sort((a, b) =>
      (a.res.summary.red_after - b.res.summary.red_after) ||
      (a.res.summary.avg_delay_after_sec - b.res.summary.avg_delay_after_sec))[0].scenario;
  }

  function render() {
    const best = bestOf(results);
    const s0 = results[0].res.summary;
    const total = results[0].res.vehicles.length;
    const maxGain = Math.max(1, ...results.map((r) => r.res.summary.avg_delay_before_sec - r.res.summary.avg_delay_after_sec));

    const rows = results
      .slice()
      .sort((a, b) => (a.scenario === best ? -1 : b.scenario === best ? 1 : 0) || a.res.summary.avg_delay_after_sec - b.res.summary.avg_delay_after_sec)
      .map(({ scenario, res }) => {
        const s = res.summary;
        const gain = s.avg_delay_before_sec - s.avg_delay_after_sec;
        const tags = scenario === best && scenario === ctx.rec
          ? `<span class="tag tag--best">лучший · советует модель</span>`
          : (scenario === best ? `<span class="tag tag--best">лучший</span>` : "") +
            (scenario === ctx.rec ? `<span class="tag">советует модель</span>` : "");
        return `
        <button class="wi-opt ${scenario === chosen ? "is-chosen" : ""}" data-sc="${scenario}" role="radio" aria-checked="${scenario === chosen}">
          <span class="wi-opt__name">${App.esc(App.labels.scenarios[scenario])}${tags}</span>
          <span class="wi-opt__gain ${gain > 0 ? "t-green" : "t-red"}">${gain > 0 ? "−" : "+"}${App.fmtDelayShort(Math.abs(gain)).replace(/^[+−]/, "")}</span>
          <span class="wi-opt__bar" title="Чем длиннее, тем сильнее эффект"><i style="width:${Math.max(2, (Math.max(0, gain) / maxGain) * 100)}%"></i></span>
          <span class="wi-opt__red">опаздывают <b>${s.red_before} → ${s.red_after}</b></span>
        </button>`;
      }).join("");

    const cur = results.find((r) => r.scenario === chosen).res;
    $("wi-body").innerHTML = `
      <div class="wi-now">
        <div><span class="muted small">Сейчас на маршруте</span><b>${App.fmtDelayShort(s0.avg_delay_before_sec)}</b><span class="muted small">среднее опоздание</span></div>
        <div><span class="muted small">Опаздывают</span><b class="${s0.red_before ? "t-red" : ""}">${s0.red_before} из ${total}</b><span class="muted small">автобусов</span></div>
      </div>
      <h4>Насколько каждая мера уменьшит среднее опоздание</h4>
      <div class="wi-opts" role="radiogroup" aria-label="Меры">${rows}</div>
      <h4>Выбрано «${App.esc(App.labels.scenarios[chosen])}» — по автобусам</h4>
      ${vehiclesHtml(cur)}`;

    $("wi-body").querySelectorAll("[data-sc]").forEach((b) => (b.onclick = () => { chosen = b.dataset.sc; render(); }));

    const apply = $("wi-apply");
    if (ctx.source.applyMeasure) {
      apply.hidden = false;
      apply.disabled = false;
      apply.textContent = "Применить эту меру";
      apply.onclick = async () => {
        apply.disabled = true;
        await ctx.onApply(chosen);
        App.whatif.close();
      };
    }
  }

  function vehiclesHtml(w) {
    const max = Math.max(60, ...w.vehicles.map((v) => Math.abs(v.delay_before_sec)));
    const bar = (sec) => `${Math.max(2, (Math.max(0, sec) / max) * 100)}%`;
    return `
      <div class="wi-veh">
        ${w.vehicles.map((v) => `
          <div class="wrow ${v.vehicle_id === ctx.vehicle.vehicle_id ? "wrow--me" : ""}">
            <span>ТС ${App.esc(v.vehicle_id)}</span>
            <span class="wrow__bars"><i class="b-before" style="width:${bar(v.delay_before_sec)}"></i><i class="b-after" style="width:${bar(v.delay_after_sec)}"></i></span>
            <span class="wrow__val">${App.fmtDelayShort(v.delay_before_sec)} → <b class="t-${App.delayLevel(v.delay_after_sec)}">${App.fmtDelayShort(v.delay_after_sec)}</b></span>
          </div>`).join("")}
        <div class="legend-inline muted small"><i class="b-before"></i>сейчас <i class="b-after"></i>после меры</div>
      </div>`;
  }

  // Закрытие: крестик, кнопка «Закрыть», клик по фону, Esc
  document.addEventListener("DOMContentLoaded", () => {
    $("wi-close").onclick = App.whatif.close;
    $("wi-cancel").onclick = App.whatif.close;
  });
})(window.App);
