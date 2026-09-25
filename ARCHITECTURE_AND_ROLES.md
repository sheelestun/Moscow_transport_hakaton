# Архитектура и роли

Проект: **Предиктор изменений в графике движения городского транспорта** (хакатон Московского транспорта).

Задача — за 10–15 минут до фактической задержки на остановке предсказать её величину в секундах и предупредить диспетчера с рекомендацией.

**Дедлайн: 27 сентября 2026, 23:59 МСК.** Времени мало — план и приоритеты в разделах 7 и 12.

---

## 1. Что именно предсказываем

Прогнозная точка — пара `(tr_id, T)`. По телеметрии ТС до момента `T` нужно предсказать **фактическую задержку в секундах** (`target_delay_s`) на первой остановке, чьё плановое время попадает в окно `(T+10 мин, T+15 мин]`. Целевая остановка (`target_stop_id`) и её плановое время (`target_time_begin`) даны в точке.

**Знак важен:** `+` = опоздание, `−` = опережение.

**Анти-утечка:** при прогнозе для `T` использовать только `event_time ≤ T` и подсказку `cur_dev_s` (задержка на последней уже пройденной остановке).

**Метрика (Data Science):**

```
MAE      = mean(|факт − прогноз|)                                  # секунды
mae_zero = mean(|факт|)                                             # прогноз = 0
score    = max(0, min(1, (mae_zero − MAE) / (mae_zero − MAE_TARGET)))   ∈ [0,1]
```

Скор → баллы жюри: **≥0.70 → 6**, 0.60–0.69 → 5, 0.50–0.59 → 4, **0.40–0.49 → 3 (baseline)**, 0.25–0.39 → 2, 0.10–0.24 → 1, <0.10 → 0.

Baseline `prediction = cur_dev_s` уже даёт score ≈ 0.40 и лежит в `sample_submission.csv`. Наш пол — 0.40, цель — ≥0.70.

---

## 2. Что мы строим

Три независимых сервиса + вспомогалки, всё в `docker-compose`:

- **ML-модуль** — офлайн обучение (CatBoost + PyTorch/Transformer), онлайн инференс (FastAPI + ONNX). Владельцы: Степан, Фёдор.
- **Backend** — приём NDTP-потока из эмулятора (TCP-сервер), склейка с расписанием, feature store, оркестрация, REST + WebSocket API. Владелец: Даниил Герман.
- **Frontend (BI-дашборд)** — карта Москвы, светофор рисков, карточка инцидента, лента алертов, панель What-if. Владелец: Вероника.
- **Data-компонент** — EDA, качество данных, схема Postgres, витрины метрик. Владелец: Даниил Шелестов.

Плюс общие: Redis (шина + онлайн-фичи), Postgres (история и алерты), Nginx (реверс-прокси для демо), Swagger UI (из FastAPI бесплатно), Sphinx (документация по коду).

---

## 3. Ограничения (must / must not)

**Обязательно:**
- Стек **строго**: Python 3.12+, PyTorch, CatBoost, Docker.
- Горизонт прогноза строго **10–15 минут** — жюри проверяет.
- ML и Backend — **разные сервисы**, не монолит.
- **Документация:** PyDoc/Sphinx по коду + OpenAPI/Swagger по API.
- Дашборд читается за ~5 секунд без инструкции.
- Внешние API (Яндекс.Пробки, погода) разрешены, интернет на демо есть.

**Не подходит:**
- LightGBM / XGBoost / TensorFlow / JAX — вне стека.
- Kafka — избыточно, берём Redis Streams.
- Свой парсер сырого NDTP «с нуля» — используем спеку и эмулятор из `docs/`.

**Инфраструктура:** серверы **не дают**. Хостим сами (VPS команды + локальный ноут-fallback).

**Лимит сабмитов:** 36 попыток/день всего, 24 успешных/день. Экспериментируем на локальном `labels_test.csv`, сабмитим только осмысленные версии.

---

## 4. Данные (что реально даётся)

Датасет находится в `/home/stepan/Загрузки/dataset/` у Степана и на Яндекс.Диске.

| Файл | Строк | Что внутри |
|---|---|---|
| `train/traffic.csv` | ~500k | Телеметрия для трейна: `packet_id, tr_id, unit_id, event_time, device_event_id, location_valid, gps_time, lon, lat, alt, speed, heading, receive_time, is_hist_data`. Периодичность ~12–15 сек, много строк с `location_valid=False` (без координат). |
| `train/schedule.csv` | 16 675 | Расписание с фактом: `tt_action_item_id (=target_stop_id), time_begin (план), time_fact_begin (факт), order_date, manual_fill, tr_id, geom (POINT WKT), building_address`. |
| `labels/labels_train.csv` | 4 434 | Прогнозные точки трейна с таргетом: `sample_id, tr_id, T, target_stop_id, target_time_begin, cur_dev_s, target_delay_s, target_class`. 39 уникальных `tr_id`. |
| `test/traffic.csv` + `test/schedule.csv` + `labels/labels_test.csv` | 353 точки | То же для локальной валидации. |
| `validate/traffic.csv` | как train | Телеметрия validate. |
| `validate/schedule_plan.csv` | 5 559 | План без факта (это скрыто, оценивают на этом). |
| `validate/points.csv` | **151** | Прогнозные точки validate — по ним считаем сабмит. |
| `sample_submission.csv` | 151 | Готовый бейзлайн (`prediction = cur_dev_s`), формат `sample_id;prediction`. |
| `docs/Emulator-and-Telematic-Packets-Specification.md` | — | Полная спека NDTP-протокола и эмулятора. |
| `ndtp-telemetry-emulator.tar` | 128 MB | Docker-образ эмулятора для real-time контура. |

**Распределение таргета на train:** median 24 с, mean 49 с, `|target|` в среднем 97 с, диапазон [−372, +672]. Классы: 62% ontime, 15% early, 23% late.

**Маппинг `traffic.csv` ↔ NDTP-ячейка `G6CellNav00`** (нужен бэку для парсера потока эмулятора):

| CSV | NDTP | Преобразование |
|---|---|---|
| `event_time` / `gps_time` | `timestamp` | Unix-секунды → datetime |
| `lon` | `longitude` | `/1e7`, знак из `extraDopBit6` (E/W) |
| `lat` | `latitude` | `/1e7`, знак из `extraDopBit5` (N/S) |
| `alt` | `altitude` | метры |
| `speed` | `speedAvg` | км/ч |
| `heading` | `course` | градусы |
| `location_valid` | `extraDopBit7` | флаг валидности |
| `unit_id` | `peerAddress` (NPL) | ID терминала |

**Формат сабмита** (`;` разделитель, UTF-8, все `sample_id` из `validate/points.csv`, без дублей):

```
sample_id;prediction
131672_1767670500;120.0
122048_1767732000;45.0
130072_1767732000;-30.0
```

---

## 5. Команда и зоны ответственности

| Кто | Роль | Владеет |
|---|---|---|
| **Степан** | ML | Sequence-модель (PyTorch/Transformer/GRU), ансамбль, инференс-сервис, экспорт ONNX |
| **Фёдор** | ML | Табличный CatBoost, фиче-инжиниринг, сабмиты, оффлайн-валидация на `labels_test.csv` |
| **Даниил Герман** | Backend | NDTP TCP-сервер, парсер, ingestor→шина, feature store, REST+WS API, алерты, `docker-compose`, деплой на VPS |
| **Вероника** | Frontend / BI | Карта, светофор рисков, лента и карточки алертов, панель What-if, WebSocket-подписка |
| **Даниил Шелестов** | Data Analyst / DBA | EDA, качество данных, схема Postgres, витрины метрик, отчёт по производительности, помощь ML с фичами и валидацией |

**Что согласовать в первый час:**
1. **ML ↔ Backend:** контракт `/predict` (раздел 6.1) и правила версионирования модели.
2. **Backend ↔ Frontend:** список REST-эндпоинтов и схема WebSocket-событий (раздел 6.2).
3. **Data ↔ ML:** определение фичей `cur_dev_s`, `current_delay_sec`, окно последних N пингов.
4. **Все:** общий словарь ключей (`tr_id`, `target_stop_id`, `T` в UTC), timezone (используем UTC внутри, Europe/Moscow на UI).

Владелец задачи на стыке — тот, у кого хранится итоговый артефакт.

---

## 6. Архитектура

```
┌────────────────────────┐
│  NDTP-эмулятор (Docker)│  ndtp-telemetry-emulator:1.0
│  REST :18080 → config  │  configure via POST /api/config
└──────────┬─────────────┘
           │ TCP: бинарные NDTP-пакеты (G6CellNav00 и др.)
           ▼
┌────────────────────────┐
│  NDTP TCP-сервер       │  backend/src/ingestor/
│  handshake + parse     │  парсит по спеке (NPL+NPH+ячейки)
└──────────┬─────────────┘
           │ нормализованные события (та же схема, что traffic.csv)
           ▼
┌────────────────────────┐       ┌──────────────────────┐
│  Redis Streams (шина)  │◀─────▶│  Schedule loader     │
│  telemetry.raw         │       │  (CSV → Redis/Postgr)│
└──────────┬─────────────┘       └──────────────────────┘
           ▼
┌────────────────────────┐
│  Feature builder       │  online-фичи из окна пингов + расписания:
│  + Map Matching        │  current_delay, speed_avg_5min,
│                        │  dwell_time, headway, segment_id, ...
└──────────┬─────────────┘
           ▼
┌────────────────────────┐       ┌──────────────────────┐
│  Feature Store (Redis) │──────▶│  ML Inference Service│
│  online-фичи по ТС     │       │  FastAPI :8001       │
└──────────┬─────────────┘       │  CatBoost + PyTorch  │
           │                     │  ONNX Runtime        │
           │                     └──────────┬───────────┘
           │                                │ delay_pred_sec, risk
           │                                ▼
           │       ┌─────────────────────────────────────┐
           └──────▶│  Backend API :8000 (FastAPI)        │
                   │  правила алертов (порог+дедуп),     │
                   │  What-if оркестрация,               │
                   │  REST + WebSocket + Swagger UI      │
                   └──────────┬──────────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐   ┌───────────────────┐
                   │  Frontend :3000      │   │  Postgres         │
                   │  карта, алерты,      │   │  история, алерты, │
                   │  What-if             │   │  метрики модели   │
                   └──────────────────────┘   └───────────────────┘
```

**Два режима инференса:**
1. **Батч (главное для очков):** скрипт `ml/src/predict_submission.py` читает `validate/traffic.csv` + `validate/schedule_plan.csv` + `validate/points.csv`, строит фичи, зовёт модель, пишет `submission.csv`. Это то, чем меряют MAE.
2. **Онлайн (для системы и демо):** эмулятор → NDTP-сервер → шина → фичи → инференс → алерты → дашборд. Модель та же самая (единый пайплайн фичей).

**Почему такие решения:**
- **Redis Streams, не Kafka** — за 2 суток Kafka не окупится.
- **Feature Store на Redis** — фичи должны отдаваться за миллисекунды, и одна и та же логика фичей должна работать и в батче, и онлайн (общий модуль `ml/src/features/`).
- **Postgres, не ClickHouse** — экономим стек, история небольшая.
- **Один VPS + docker-compose** — на демо один хост, все сервисы рядом.
- **ML отдельным сервисом** — прямое требование оргов + позволит перекатывать модель без деплоя бэка.

---

## 7. Контракты между сервисами

Заморозить в первый час. После — правки только через явное согласование.

### 7.1 Backend → ML: `POST /predict`

```json
// request
{
  "sample_id": "131672_1767670500",
  "vehicle_id": "131672",
  "T": "2026-01-06T03:35:00Z",
  "target_stop_id": "53700172828",
  "target_time_begin": "2026-01-06T03:50:00Z",
  "features": {
    "cur_dev_s": 274.0,
    "current_delay_sec": 260,
    "speed_avg_5min": 18.3,
    "speed_avg_15min": 15.1,
    "dwell_last_stop_sec": 25,
    "headway_to_next_sec": 380,
    "headway_to_prev_sec": 210,
    "distance_to_target_m": 4200,
    "planned_time_to_target_sec": 900,
    "hour_of_day": 3,
    "day_of_week": 1,
    "weather_code": 2,
    "traffic_score": 0.7,
    "history": [
      {"t": -60, "lat": 55.75, "lon": 37.61, "speed": 15},
      {"t": -45, "lat": 55.751, "lon": 37.611, "speed": 20}
    ]
  }
}

// response
{
  "sample_id": "131672_1767670500",
  "delay_pred_sec": 187.0,     // главный таргет, MAE считается по нему
  "risk_score": 0.83,          // sigmoid((delay_pred - 120) / 60), для UI
  "confidence": 0.71,          // 1 - нормированная дисперсия ансамбля
  "top_features": [
    {"name": "cur_dev_s", "contribution": 0.42},
    {"name": "traffic_score", "contribution": 0.18}
  ],
  "model_version": "ens-0.3.1"
}
```

Двойной выход (`delay_pred_sec` + `risk_score`) закрывает расхождение между слайдом «вычисляет вероятность» и метрикой MAE: одна регрессионная модель на секунды, `risk_score` — детерминированная функция от предсказания.

Батч-вариант: `POST /predict/batch` принимает массив таких же запросов и возвращает массив ответов. Используется скриптом `predict_submission.py`.

### 7.2 Backend → Frontend

REST (все с Swagger в `/docs`):
- `GET /routes` — список маршрутов и геометрия для карты.
- `GET /vehicles?route_id=…` — текущие ТС.
- `GET /alerts?active=true` — активные алерты.
- `POST /whatif` — сценарий: `{scenario: "add_reserve", route_id, at_stop_id}` → пересчитанные прогнозы.
- `GET /metrics/model` — свежий MAE / p95-latency (для витрины на демо).

WebSocket `/ws`:
```json
// событие "alert.new"
{
  "type": "alert.new",
  "alert_id": "a-91021",
  "vehicle_id": "131672",
  "route_id": "A-905",
  "target_stop_id": "53700172828",
  "delay_pred_sec": 187,
  "risk_score": 0.83,
  "eta_incident": "2026-01-06T03:47:00Z",
  "reason_pattern": "traffic_jam_ahead",
  "recommendation": "release_reserve"
}
```

Также события `vehicle.update`, `whatif.result`.

### 7.3 События в шине Redis Streams

- `telemetry.raw` — сырые события от NDTP-сервера.
- `telemetry.matched` — после Map Matching, с `segment_id` и текущим отклонением.
- `predictions.new` — прогнозы, публикуются ML или бэком после запроса.
- `alerts.new` — после порогов и дедупа.

Frontend не подписывается на Redis, только на WebSocket бэка.

---

## 8. Структура репозитория

```
Moscow_transport_hakaton/
├── docker-compose.yml
├── .env.example
├── README.md                        (жюри: как запустить)
├── ARCHITECTURE_AND_ROLES.md        (этот файл — для команды)
│
├── ml/                              владельцы: Степан, Фёдор
│   ├── notebooks/                   EDA, эксперименты
│   ├── src/
│   │   ├── features/                общий модуль фичей (батч + онлайн)
│   │   │   ├── from_csv.py          для трейна и submission
│   │   │   └── from_stream.py       для онлайн (тот же контракт фичей)
│   │   ├── models/
│   │   │   ├── catboost_tab.py
│   │   │   ├── torch_seq.py
│   │   │   └── ensemble.py
│   │   ├── train.py                 обучение по train + labels_train
│   │   ├── eval.py                  валидация по labels_test
│   │   ├── predict_submission.py    построение submission.csv
│   │   ├── export_onnx.py           экспорт + квантизация
│   │   └── inference_service.py     FastAPI, POST /predict, /predict/batch
│   ├── artifacts/                   веса, ONNX (в .gitignore)
│   ├── Dockerfile
│   └── requirements.txt
│
├── backend/                         владелец: Даниил Герман
│   ├── src/
│   │   ├── ingestor/                NDTP TCP-сервер + парсер NPL/NPH/ячеек
│   │   ├── schedule/                загрузка CSV, поиск target_stop_id
│   │   ├── mapmatch/                снап GPS на маршрут / segment_id
│   │   ├── feature_store/           Redis-обёртка
│   │   ├── api/                     FastAPI (REST + WS + Swagger)
│   │   ├── alerts/                  пороги, дедуп, история
│   │   └── whatif/                  оркестрация сценариев
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/                        владелец: Вероника
│   ├── src/                         карта (Leaflet/MapLibre + OSM), лента, What-if
│   ├── Dockerfile
│   └── package.json
│
├── data/                            владелец: Даниил Шелестов
│   ├── eda/                         ноутбуки по EDA
│   ├── sql/                         схемы Postgres, миграции
│   ├── quality_reports/             отчёты по качеству
│   └── metrics/                     скрипты подсчёта витрин
│
├── docs/                            (сгенерированное: Sphinx build + Swagger export)
│   └── sphinx/                      конфиг Sphinx для PyDoc
│
└── infra/
    ├── nginx.conf                   реверс-прокси для демо
    └── deploy.md                    инструкция на VPS
```

Крупные артефакты (веса, ONNX, датасет) — **не в git**, синхронизируем через диск / scp на VPS.

---

## 9. План работ (2.5 суток, до 27 сент 23:59 МСК)

**Фаза 0 — Kickoff, T+0…3ч (25.09 вечер):**
- Все читают ТЗ и `dataset/README.md` + спеку эмулятора.
- Замораживаем контракты из раздела 7.
- Даниил Ш. — быстрый EDA `labels_train.csv` + `traffic.csv` (распределения, gap'ы, качество склейки).
- Фёдор — минимальный baseline: `predict = cur_dev_s`, генерация `submission.csv`, первый сабмит (проверить формат и score, ожидание ≈0.40 = 3 балла в кармане).
- Степан — каркас `ml/src/features/from_csv.py` + `train.py` с CatBoost на 5 базовых фичах.
- Даниил Г. — каркас `backend/src/api/` (FastAPI, Swagger, заглушка `/predict`).
- Вероника — каркас фронта: карта + пустая лента алертов + WebSocket-подключение.

**Фаза 1 — MVP end-to-end, T+3…12ч (25.09 ночь → 26.09 утро):**
- ML: CatBoost с реальными фичами (окно телеметрии, headway, планируемое время до целевой, dwell, час/день недели). Второй сабмит.
- Backend: NDTP TCP-сервер парсит handshake+realtime, публикует в `telemetry.raw`, feature builder + `/predict` работает end-to-end на моковом потоке. Расписание загружается из CSV в Postgres.
- Frontend: карта тянет `GET /vehicles`, лента слушает WebSocket, отображает alert.new.
- Data: схема Postgres развёрнута, витрина `GET /metrics/model` начинает наполняться.
- **Цель фазы: сквозной поток данных работает, submission.csv улучшил baseline.**

**Фаза 2 — Итерация качества, T+12…36ч (26.09):**
- Степан — sequence-модель (PyTorch/Transformer или GRU) на окне телеметрии.
- Фёдор — дожать CatBoost, добавить внешние фичи (погода, пробки через API если успеем).
- Ансамбль (weighted blend по MAE на labels_test).
- Даниил Г. — логика алертов (пороги, дедуп по (tr_id, target_stop_id)), What-if эндпоинт.
- Вероника — светофор рисков (heatmap по сегментам), карточка инцидента, панель What-if.
- Данил Ш. — метрики MAE в разрезе маршрут/час, отчёт по качеству.
- Сабмиты по мере улучшения (не тратим лимит впустую).

**Фаза 3 — Hardening, T+36…52ч (26.09 ночь → 27.09 день):**
- ONNX-экспорт + INT8 квантизация, замер latency до/после.
- Реконнект в NDTP-сервере, деградация при потере эмулятора.
- Стабилизация UI на реальных данных.
- Настоящий Map Matching (Valhalla/OSRM) — **только если модель достигла 0.55+ и есть свободные руки**. Иначе оставляем «снап на ближайшую остановку из расписания».
- Sphinx-документация: генерируем из docstrings.
- README для жюри с инструкцией запуска.

**Фаза 4 — Демо и сдача, T+52…58ч (27.09 до 23:59):**
- Финальный submission.csv.
- Скрипт демо: `docker compose up` → эмулятор → показать алерты за 10–15 мин → What-if.
- Резервный VPS + локальный ноут с ngrok как fallback.
- Заполнить форму на платформе: ссылки на репо, документацию, описание производительности, список бонусов.

---

## 10. Бонусные фичи (по стоимости/выгоде)

1. **Ансамбль CatBoost + PyTorch/Transformer** — прямо в ТЗ, у нас 2 ML, делаем обязательно.
2. **ONNX + INT8 квантизация** — быстро и даёт материал для критерия «Производительность» (замерим latency до/после).
3. **What-if** — минимум сценарий «выпуск резерва»; больше — если время.
4. **Map Matching через OSRM/Valhalla** — только если основной пайплайн дожат до ≥0.55.
5. **TensorRT** — не трогаем, если демо на CPU или без GPU-VPS.

---

## 11. Метрики и что показываем на демо

**Модельные:**
- Score из формулы выше (главный, определяет 0-6 баллов).
- MAE (сек) в разрезе: маршрут / час дня / severity.
- Дельта относительно baseline `cur_dev_s`.

**Системные:**
- Latency инференса p50 / p95 (до и после ONNX).
- End-to-end latency: пинг эмулятора → алерт в UI.
- Пропускная способность ingestor (пингов/сек).
- Восстановление после reconnect эмулятора (сколько секунд простоя).

**UX-демо-сценарий (2 минуты):**
1. `docker compose up` — все сервисы поднимаются.
2. Запускаем эмулятор с реалистичным `intervalMs`.
3. Диспетчер видит карту с ТС и светофором рисков.
4. Через ~10 сек всплывает алерт: «ТС 131672 опоздает на остановку X через 12 мин на 3 мин».
5. Клик по алерту → карточка с топ-фичами и рекомендацией.
6. Нажимаем «What-if: выпуск резерва» → пересчитанные риски.
7. Кладём эмулятор → сервис не падает, статус переходит в «degraded», UI показывает последнее состояние.

---

## 12. Definition of Done по ролям

**ML (Степан + Фёдор):**
- `submission.csv` с score ≥ 0.55 (минимум) / ≥ 0.70 (цель).
- `/predict` работает в отдельном контейнере, p95 latency < 200 мс.
- Ансамбль обучен, чекпоинты и ONNX в `ml/artifacts/`.
- Latency замерена до/после ONNX и вписана в отчёт.
- Модуль фичей `ml/src/features/` идентичен для батча и онлайна.

**Backend (Даниил Г.):**
- `docker compose up` поднимает весь стек одной командой.
- NDTP TCP-сервер принимает handshake и realtime, парсит `G6CellNav00`, публикует в шину.
- Все API из раздела 7 отвечают, Swagger UI открывается по `/docs`.
- Алерты не дублируются (дедуп по (tr_id, target_stop_id) с TTL).
- Реконнект отрабатывает без падения сервиса.

**Frontend (Вероника):**
- Карта Москвы с маршрутами и живыми ТС, светофор риска.
- Лента алертов обновляется по WebSocket в реальном времени.
- Клик по алерту показывает карточку с top_features и рекомендацией.
- Панель What-if вызывает бэкенд и рендерит дельту.
- Понятно за 5 секунд без инструкции (проверить на постороннем).

**Data (Даниил Ш.):**
- Отчёт по качеству данных сдан к концу Фазы 1.
- Схема Postgres развёрнута, миграции в `data/sql/`.
- Витрина `GET /metrics/model` наполняется.
- Метрики для демо-слайда посчитаны.

---

## 13. Риски и что с ними делать

| Риск | Митигация |
|---|---|
| Не разберёмся в NDTP-парсинге вовремя | Даниил Г. сразу читает `docs/Emulator-and-Telematic-Packets-Specification.md`, пишет юнит-тесты на handshake и один `G6CellNav00` пакет. Если совсем плохо — фолбэк: онлайн-контур работает от CSV-репроигрывателя (свой скрипт, читает `validate/traffic.csv` и льёт в шину по расписанию `event_time`). |
| ML не достигает >0.40 | Baseline `cur_dev_s` уже даёт 0.40 = 3 балла. CatBoost с 5 фичами почти всегда даёт 0.50+. |
| VPS падает на демо | Локальный ноут с ngrok / tailscale как второй хост. Все сервисы поднимаются одной командой. |
| Frontend не успевает | Минимум: карта + лента алертов. What-if — резать первым. |
| Sphinx-доки забыли | В Фазе 3 — 1 час на генерацию из docstrings. OpenAPI/Swagger уже есть бесплатно из FastAPI. |
| Лимит сабмитов исчерпан | Строго не спамим: сабмитим только когда локальный MAE на `labels_test.csv` улучшился на ≥3 сек. |
| Взаимозаменяемость | Backend ↔ Data Analyst и ML1 ↔ ML2 частично пересекаются, задачи разбиваем на маленькие PR. |

---

## 14. Что уже точно известно (не задавать оргам)

- Формат данных, схема CSV, маппинг NDTP — в `docs/Emulator-and-Telematic-Packets-Specification.md` и `dataset/README.md`.
- Метрика — MAE, скор считается по формуле выше, `MAE_TARGET` известен платформе.
- Baseline score = 0.40.
- Эмулятор — Docker `ndtp-telemetry-emulator:1.0`, REST `:18080`, TCP шлёт на `targetHost:targetPort`.
- Дедлайн: 27 сент 23:59 МСК. Лимит сабмитов: 36/24 в день.
