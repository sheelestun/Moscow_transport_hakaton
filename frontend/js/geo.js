// Геометрия на линии маршрута: расстояния, проекция точки на линию, вырезание участка.
// Координаты везде в формате [lat, lon].
window.App = window.App || {};

App.geo = (function () {
  const R = 6371000, toRad = Math.PI / 180;

  // Расстояние в метрах между двумя точками
  function dist(a, b) {
    const dLat = (b[0] - a[0]) * toRad, dLon = (b[1] - a[1]) * toRad;
    const h = Math.sin(dLat / 2) ** 2 + Math.cos(a[0] * toRad) * Math.cos(b[0] * toRad) * Math.sin(dLon / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(h));
  }

  // Накопленная длина линии в каждой вершине: [0, 120, 340, ...]
  function cumulative(line) {
    const c = [0];
    for (let i = 1; i < line.length; i++) c.push(c[i - 1] + dist(line[i - 1], line[i]));
    return c;
  }

  // Точка на линии, пройдя m метров от начала
  function pointAt(line, cum, m) {
    if (m <= 0) return line[0];
    for (let i = 1; i < line.length; i++) {
      if (m <= cum[i]) {
        const k = (m - cum[i - 1]) / (cum[i] - cum[i - 1] || 1);
        return [line[i - 1][0] + (line[i][0] - line[i - 1][0]) * k, line[i - 1][1] + (line[i][1] - line[i - 1][1]) * k];
      }
    }
    return line[line.length - 1];
  }

  // Ближайшая точка линии к p: {pos_m, point, off_m}
  function project(line, cum, p) {
    const kx = Math.cos(p[0] * toRad); // поправка на широту, чтобы работать в «плоских» координатах
    let best = { pos_m: 0, point: line[0], off_m: Infinity };
    for (let i = 1; i < line.length; i++) {
      const a = line[i - 1], b = line[i];
      const ax = a[1] * kx, ay = a[0], bx = b[1] * kx, by = b[0], px = p[1] * kx, py = p[0];
      const dx = bx - ax, dy = by - ay;
      const L = dx * dx + dy * dy || 1e-18;
      const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / L));
      const q = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
      const d = dist(p, q);
      if (d < best.off_m) best = { pos_m: cum[i - 1] + (cum[i] - cum[i - 1]) * t, point: q, off_m: d };
    }
    return best;
  }

  // Участок линии между from_m и to_m (порядок не важен)
  function slice(line, cum, from_m, to_m) {
    const a = Math.min(from_m, to_m), b = Math.max(from_m, to_m);
    const out = [pointAt(line, cum, a)];
    for (let i = 0; i < line.length; i++) if (cum[i] > a && cum[i] < b) out.push(line[i]);
    out.push(pointAt(line, cum, b));
    return out;
  }

  return { dist, cumulative, pointAt, project, slice };
})();
