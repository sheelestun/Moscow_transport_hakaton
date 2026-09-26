#!/bin/sh
# Скрипт, который nginx-образ запускает сам через /docker-entrypoint.d/.
# Подменяет плейсхолдеры (__MODE__, __API_BASE__, __WS_URL__) в config.js
# на значения переменных окружения. Отсутствующая переменная → пустое поле,
# и config.js применит fallback-константу.
set -eu

SRC="/usr/share/nginx/html/js/config.js"

MODE="${MODE:-live}"
API_BASE="${API_BASE:-http://localhost:8000}"
WS_URL="${WS_URL:-ws://localhost:8000/ws}"

# sed -i прямо на месте — простая, атомарная замена плейсхолдеров.
sed -i \
  -e "s|__MODE__|${MODE}|g" \
  -e "s|__API_BASE__|${API_BASE}|g" \
  -e "s|__WS_URL__|${WS_URL}|g" \
  "$SRC"

echo "[frontend] config.js patched: MODE=${MODE} API_BASE=${API_BASE} WS_URL=${WS_URL}"
