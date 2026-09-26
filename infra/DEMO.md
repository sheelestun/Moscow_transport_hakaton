# E2E-демо: NDTP-эмулятор → backend → ML → фронт

Полный пайплайн одной командой + сценарий показа для жюри.

## Требования

- Docker + Docker Compose v2
- ~4 ГБ RAM свободных
- `dataset/` в корне репо (см. `ml/README.md`, ссылка на Яндекс.Диск)
- OCI-архив эмулятора: `dataset/ndtp-telemetry-emulator.tar` — организаторы прислали в чате

## Быстрый старт (60 секунд)

```bash
# 1. Один раз — подтягиваем эмулятор из OCI-архива
docker load -i dataset/ndtp-telemetry-emulator.tar

# 2. Поднимаем весь стек
docker compose up -d --build

# 3. Ждём health (обычно 20-30 сек)
docker compose ps
```

Все зелёные — открываем фронт:

- **Фронт:** http://localhost:3000
- **Backend API:** http://localhost:8000/docs (Swagger)
- **ML API:** http://localhost:8001/docs (Swagger)
- **NDTP эмулятор:** http://localhost:18080/api/cells

## Что смотрим на экране жюри

Порядок нажатий:

| Шаг | Что делаем | Что покажет |
|---|---|---|
| 1 | Открыть http://localhost:3000 | 24 ТС на 6 маршрутах, часть красных (`risk ≥ 0.7`) |
| 2 | Кликнуть по ТС | Карточка: `delay_pred_sec`, `reason_pattern`, `top_features`, `confidence` |
| 3 | Открыть боковую панель «Алерты» | Список активных инцидентов, у каждого — `recommendation` |
| 4 | В карточке ТС нажать «Расписание» | Прошедшие остановки с `delay_sec`, ближайшая с `is_target=true` в окне (T+10, T+15] |
| 5 | Открыть «What-if», выбрать `detour` для маршрута | POST `/whatif` — сравнение `red_before/red_after`, список ТС |
| 6 | Нажать «Применить» | `apply=true` → эффект вносится в симуляцию, `delay_pred` у ТС падает на следующих тиках |
| 7 | Показать `GET /metrics/model` в Swagger backend | `mae_test_s ≈ 43.7`, `score_estimate = 1.0`, `latency_ms_p50 ≈ 1.6ms` |
| 8 | Обрыв связи: остановить backend `docker compose stop backend`, показать `status: degraded` во фронте, потом снова `up` | Тест критерия «надёжность» из ТЗ |

## Проверка вручную (без фронта)

```bash
# Активные ТС и алерты
curl -s http://localhost:8000/vehicles | jq 'length'
curl -s http://localhost:8000/alerts?active=true | jq

# Расписание одного ТС
VID=$(curl -s http://localhost:8000/vehicles | jq -r '.[0].vehicle_id')
curl -s http://localhost:8000/vehicles/$VID/schedule | jq

# What-if для маршрута М1
curl -s -X POST http://localhost:8000/whatif \
  -H "Content-Type: application/json" \
  -d '{"scenario":"detour","route_id":"М1"}' | jq '.summary'

# WebSocket (websocat или wscat)
websocat ws://localhost:8000/ws
```

## Реальный NDTP-поток через эмулятор

Эмулятор шлёт TCP-поток в backend, у нас MVP-backend его не парсит (это работа
Даниила Германа). Пока пайплайн NDTP → backend не готов, используем два обходных пути:

- **Замена телеметрии из dataset:** `python backend/src/csv_replayer.py --dataset ./dataset --split validate --ml-url http://localhost:8001` — 151 точка validate → POST /predict.
- **Эмулятор как источник трафика в ML напрямую** (для проверки live-режима):
  `python infra/emulator_replay.py --dataset ./dataset` — реальные треки 13 ТС из
  `validate/traffic.csv` льются через эмулятор на условный listener; ML отвечает
  `data_status: live` (см. `ml/EMULATOR.md`).

## Остановка

```bash
docker compose down          # остановить контейнеры, volume оставить
docker compose down -v       # с volume (postgres-data)
```

## Как оно устроено

```
NDTP-эмулятор (Java Spring Boot)
   │ TCP:5555 (пока не разбираем в MVP)
   ▼
Backend (FastAPI :8000)
   • симуляция 24 ТС на 6 маршрутах (тот же shape, что у mock.js)
   • REST /routes /vehicles /alerts /schedule /whatif /metrics/model
   • WebSocket /ws → фронт (vehicle.update, alert.new, alert.verified)
   │ HTTP /whatif/predict
   ▼
ML (FastAPI :8001)
   • CatBoost-ансамбль, /predict + /whatif/predict
   • /metrics/model — MAE, score, latency
   │
Frontend (nginx :3000)
   • MapLibre + сайдбар + What-if
```

MVP-заглушка backend **не парсит NDTP** — держит симуляцию в памяти и зовёт ML
через service-name `ml`. Как только Даниил Герман закончит парсер и постгресный
слой, replace: движение ТС читается из потока, а не из симулятора.
