# Moscow Transport Hackathon — Предиктор задержек

ИИ-система раннего прогнозирования отклонений наземного транспорта Москвы от графика за **10–15 минут** до факта. Хакатон Московского транспорта, дедлайн — **27 сентября 2026, 23:59 МСК**.

## Актуальное состояние

| Компонент | Статус | Комментарий |
|---|---|---|
| ML-модель | ✅ готова | CatBoost-ансамбль, локальный **score ≈ 1.0** (платформенный тоже 1.0) |
| Инференс-сервис | ✅ | FastAPI `:8001`, `/predict`, `/predict/batch`, `/metrics/model`, `/whatif/predict`, `/reload`. `p50 = 1.6 мс` |
| ONNX-экспорт | ✅ | fp32 6× быстрее CatBoost native |
| Backend-шлюз | 🟡 MVP-заглушка | FastAPI `:8000`, WebSocket, симулирует движение ТС + зовёт ML |
| NDTP-парсер | ⏳ Даниил Герман | пока замещается симулятором в backend / `csv_replayer.py` фолбэком |
| Frontend | ✅ | MapLibre, карточка ТС, алерты, What-if, обрыв связи, шкала 15 мин |
| Docker Compose | ✅ | `docker compose up -d --build` — весь стек |
| Отчёт по данным | ✅ | `statistics/REPORT.md`, 8 графиков, все метрики |
| Sphinx-документация | ✅ | `docs/sphinx/` — `make html` |

## Что делает система

1. **NDTP-эмулятор** льёт бинарный TCP-поток телеметрии (13 реальных ТС из validate/traffic.csv).
2. **Backend** держит стейт по ТС, зовёт ML для прогноза `delay_pred_sec`, `reason_pattern`, `top_features`, `confidence`.
3. **ML-сервис** на CatBoost-ансамбле: `MAE ~ 40 с` на реальных точках, `p95 latency < 60 мс` в контейнере.
4. **Frontend** пушит `vehicle.update` по WebSocket, рисует карту + алерты + What-if-сценарии.

Метрика хакатона: `MAE`, скор `max(0, min(1, (mae_zero − MAE) / (mae_zero − MAE_TARGET)))` — у нас упирается в потолок **1.0**.

## Как запустить всё

Требования: Docker + Docker Compose v2, ~4 ГБ RAM. Датасет — на Яндекс.Диске (`https://disk.yandex.ru/d/CA6tsj4aJJ4Aaw`), кладём в `./dataset/`.

```bash
docker load -i dataset/ndtp-telemetry-emulator.tar   # эмулятор из OCI-архива
docker compose up -d --build                          # весь стек
docker compose ps                                     # ждём healthy
```

- **Дашборд**: http://localhost:3000
- **Backend Swagger**: http://localhost:8000/docs
- **ML Swagger**: http://localhost:8001/docs

Полная инструкция (что клонить, куда положить датасет, как поднять только backend или только ML для NDTP-парсера) — [`LOCAL_SETUP.md`](./LOCAL_SETUP.md). Сценарий демо для жюри — [`infra/DEMO.md`](./infra/DEMO.md).

## Тесты

```bash
pip install -r backend/requirements.txt -r ml/requirements.txt pytest httpx
pytest      # backend/tests + ml/tests + statistics/tests, ~2 сек
```

CI гоняет то же самое на каждом пуше (`.github/workflows/tests.yml`).

## Ключевые документы

- [`LOCAL_SETUP.md`](./LOCAL_SETUP.md) — как поднять весь стек локально
- [`ARCHITECTURE_AND_ROLES.md`](./ARCHITECTURE_AND_ROLES.md) — архитектура, роли, контракты
- [`ml/README.md`](./ml/README.md) — ML-трек: как тренировать, инференс-сервис, ONNX
- [`ml/PERFORMANCE.md`](./ml/PERFORMANCE.md) — цифры: MAE, latency, размеры, деградация
- [`ml/EMULATOR.md`](./ml/EMULATOR.md) — NDTP-эмулятор end-to-end
- [`infra/DEMO.md`](./infra/DEMO.md) — сценарий показа
- [`statistics/REPORT.md`](./statistics/REPORT.md) — анализ датасета от Шелестова
- [`docs/sphinx/`](./docs/sphinx/) — Sphinx-документация модулей (`make html`)

## Как построить `submission.csv`

```bash
# обучить ансамбль на train+test и получить сабмит на validate
python ml/src/train_catboost.py --dataset ./dataset --fit --out submission.csv

# верифицировать: онлайн-инференс == батч
python ml/src/verify_streaming.py --dataset ./dataset --submission submission.csv
```

## Команда

| Роль | Кто | Основные ветки |
|---|---|---|
| ML | Степан, Фёдор | `main`, `ml/catboost-tabular` |
| Backend | Даниил Герман | (в работе) |
| Frontend | Вероника | `frontend` |
| Data / DBA | Даниил Шелестов | `feature/DBA` |

## Стек

Python 3.12+, CatBoost, FastAPI, Docker Compose, Redis, Postgres, nginx, MapLibre. Без LightGBM/XGBoost/TF/JAX.
