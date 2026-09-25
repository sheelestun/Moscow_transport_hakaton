// Настройки дашборда.
// Любую настройку можно переопределить через адресную строку, например:
//   index.html?mode=live&api=http://localhost:8000&ws=ws://localhost:8000/ws
window.App = window.App || {};

// Текущее время. Мок подменяет его своими «ускоренными» часами.
App.now = () => Date.now();

App.config = (function () {
  const p = new URLSearchParams(location.search);
  return {
    // "mock" — фейковые данные прямо в браузере (бэкенд не нужен)
    // "live" — настоящий бэкенд Даниила (REST + WebSocket)
    MODE: p.get("mode") || "mock",
    API_BASE: p.get("api") || "http://localhost:8000",
    WS_URL: p.get("ws") || "ws://localhost:8000/ws",

    // Пороги светофора по risk_score (0..1), который приходит от ML
    RISK_RED: 0.7,
    RISK_YELLOW: 0.35,

    MAP_CENTER: [55.73, 37.57],
    MAP_ZOOM: 12,
    TIMEZONE: "Europe/Moscow",
  };
})();
