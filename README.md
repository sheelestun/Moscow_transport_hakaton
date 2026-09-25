# Moscow Transport Hackathon — Предиктор задержек

ИИ-система раннего прогнозирования отклонений от графика наземного городского транспорта Москвы за **10–15 минут** до фактической задержки. Хакатон Московского транспорта, дедлайн — **27 сентября 2026, 23:59 МСК**.

## Что делает

Непрерывно анализирует потоковую телеметрию NDTP, накладывает на расписание и предсказывает **фактическую задержку в секундах** на первой остановке в окне T+10..15 минут. Выводит алерты на дашборд диспетчера с топ-фичами, предполагаемой причиной и рекомендацией (выпуск резерва, объезд, коррекция интервалов).

Метрика: **MAE**, скор в [0, 1] по формуле `max(0, min(1, (mae_zero − MAE) / (mae_zero − MAE_TARGET)))`. Целевой скор: **≥ 0.70** (максимум 6 баллов).

## Архитектура

Три независимых сервиса в `docker-compose`:

- **ML-модуль** — CatBoost (табличка) + PyTorch/Transformer (последовательности), ансамбль, экспорт в ONNX. FastAPI-сервис с `/predict` и `/predict/batch`.
- **Backend** — NDTP TCP-сервер, парсер, feature store на Redis, оркестрация, REST + WebSocket API, Swagger UI.
- **Frontend (BI-дашборд)** — карта Москвы, светофор рисков по маршрутам, карточка инцидента, лента алертов, панель What-if.

Плюс: Redis Streams (шина + онлайн-фичи), Postgres (история и алерты), Nginx (реверс-прокси на демо).

Подробно — см. [`ARCHITECTURE_AND_ROLES.md`](./ARCHITECTURE_AND_ROLES.md).

## Стек (строго)

Python 3.12+, PyTorch, CatBoost, Docker. Никаких LightGBM/XGBoost/TF/JAX. Фронтенд-стек не оговорён.

## Команда

| Роль | Кто |
|---|---|
| ML | Степан, Фёдор |
| Backend | Даниил Герман |
| Frontend / BI | Вероника |
| Data Analyst / DBA | Даниил Шелестов |

## Данные

Датасет — по ссылке из ТЗ на Яндекс.Диске (`https://disk.yandex.ru/d/CA6tsj4aJJ4Aaw`). Не коммитим в git. Локально ожидается в `./dataset/` или указывается через переменную окружения `DATASET_DIR`.

- `train/`, `test/` — телеметрия + расписание с фактом.
- `labels/labels_train.csv`, `labels_test.csv` — прогнозные точки с таргетом.
- `validate/` — телеметрия + плановое расписание + `points.csv` (без факта).
- `sample_submission.csv` — бейзлайн `prediction = cur_dev_s` (score ≈ 0.40).
- `ndtp-telemetry-emulator.tar` — Docker-образ эмулятора для real-time контура.
- `docs/Emulator-and-Telematic-Packets-Specification.md` — спека NDTP-протокола.

## Как запустить (после MVP)

```bash
# Поднять весь стек
docker compose up -d

# Загрузить эмулятор NDTP
docker load -i ./dataset/ndtp-telemetry-emulator.tar
docker run --rm -p 18080:18080 --add-host=host.docker.internal:host-gateway \
  --name ndtp-emu ndtp-telemetry-emulator:1.0

# Настроить эмулятор (пример)
curl -X POST http://localhost:18080/api/config \
  -H 'Content-Type: application/json' \
  -d @infra/emulator-config.json
```

- Дашборд: `http://localhost:3000`
- Swagger API: `http://localhost:8000/docs`
- Метрики модели: `http://localhost:8000/metrics/model`

## Как построить submission.csv

```bash
python ml/src/predict_submission.py \
  --dataset ./dataset \
  --model ml/artifacts/ensemble.pt \
  --out submission.csv
```

## Документация

- Код: PyDoc/Sphinx — `docs/sphinx/_build/html/index.html` после `make html`.
- API: OpenAPI/Swagger — `http://localhost:8000/docs`.
- Команде: [`ARCHITECTURE_AND_ROLES.md`](./ARCHITECTURE_AND_ROLES.md) — архитектура, роли, контракты, план работ.
