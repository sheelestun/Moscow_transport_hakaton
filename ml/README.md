# ML-модуль

Работают: **Степан, Фёдор.**

Задача — регрессия `target_delay_s` (секунды, знак важен) на первой остановке в окне `(T+10, T+15]` мин. Метрика — MAE, скор в [0,1]. Baseline `prediction = cur_dev_s` даёт ≈ 0.40 (3 балла). Цель — ≥ 0.70 (6 баллов). **Текущий локальный score ≈ 1.0**, платформа подтвердила.

Подробнее про задачу и контракты — в корневом [`ARCHITECTURE_AND_ROLES.md`](../ARCHITECTURE_AND_ROLES.md).

## Структура

```
ml/
├── src/
│   ├── features/
│   │   ├── tabular.py         основная логика фичей (батч + онлайн)
│   │   ├── from_csv.py        (устарел, оставлен для истории — использует sequence-модель)
│   │   └── from_stream.py     тонкая обёртка над tabular для инференс-сервиса
│   ├── models/                torch_seq.py (устарел, оставлен для истории)
│   ├── train_catboost.py      обучение CatBoost-ансамбля + 3 схемы валидации
│   ├── train.py               (устарел) обучение sequence-модели
│   ├── train_ensemble.py      (устарел) sequence-ансамбль
│   ├── predict_submission.py  собирает validate → submission.csv
│   ├── export_onnx.py         ONNX-экспорт + INT8 + latency benchmark
│   ├── inference_service.py   FastAPI /predict + /predict/batch
│   └── eval.py                MAE + score-функция как в ТЗ
├── configs/
│   └── catboost.json          гиперы основного CatBoost-конфига
├── artifacts/                 веса, ONNX, meta (в .gitignore)
├── notebooks/                 EDA и эксперименты
├── Dockerfile
└── requirements.txt
```

## Данные

Ожидаются в `../dataset/` относительно корня репо. Скачать: `https://disk.yandex.ru/d/CA6tsj4aJJ4Aaw` (в git не коммитим).

## Быстрый старт

```bash
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## CatBoost-трек (основной, submission ~1.0)

Признаки — `src/features/tabular.py`, функция `point_features(T, ...)` считает одну точку и годится для онлайна.
Только телеметрия `event_time <= T`, **плановое** расписание и `cur_dev_s`; `time_fact_begin` не читается вообще.

```bash
# 1. Оценить модель на 3 схемах валидации
python ml/src/train_catboost.py --dataset ./dataset --eval

# 2. Обучить финальный ансамбль на train+test → сабмит на validate
python ml/src/train_catboost.py --dataset ./dataset --fit --out submission.csv
```

Полученные артефакты в `ml/artifacts/`:
- `catboost_seed{0..4}.cbm` — 5 моделей ансамбля
- `catboost_meta.json` — фичи, cat_features, параметры, версия таргета
- `cache_features.pkl` — кэш посчитанных фичей (train + test + validate)

| Группа фичей | Что даёт |
|---|---|
| рейсы по плану (`n_trip_breaks`, `tgt_left_in_trip`, `cur_idx_in_trip`, ...) | разрыв плана > 5 мин = конечная; через неё корреляция задержки падает с 0.96 до 0.02 |
| GPS-история (`gps_*`, `overdue_*`) | фактические прибытия восстановлены из GPS (медианная ошибка ~3 с) |
| движение и ETA (`spd*`, `stop*`, `route_left_m`, `eta_dev_*`) | простой, скорость, «физический» прогноз прибытия |
| `manual_fill` | у таких остановок факт ≈ план, таргет ≈ 0 |
| `route`, `hour` | маршрут (клоны → свой прототип) и время суток |

**Схемы валидации** (`--eval`):
- **proxy** — K-fold блоками по 30 мин по реальным точкам train+test, клоны остаются в обучении (так устроен validate). Лучший прокси скора платформы.
- **holdout** — train → test.
- **LOVO** — leave-one-vehicle-out (ТС вместе с клонами отложено). Показывает качество на новом ТС.

Данные — один день, 13 реальных ТС; в train 26 синтетических клонов (по 2 на ТС, те же моменты T).
В `train/schedule.csv` лежат факты по целевым остановкам validate — это утечка из будущего, не используем.

## Инференс-сервис (FastAPI)

Для онлайн-контура (NDTP-поток → бэк → ML):

```bash
uvicorn src.inference_service:app --host 0.0.0.0 --port 8001

# или через Docker
docker build -t delay-ml ml/
docker run --rm -p 8001:8001 delay-ml
```

Эндпоинты:
- `GET /health` — статус + число загруженных моделей
- `POST /predict` — одна точка
- `POST /predict/batch` — массив
- `GET /docs` — Swagger UI

Пример запроса:

```json
POST /predict
{
  "sample_id": "131672_1767670500",
  "tr_id": 131672,
  "T": "2026-01-06T03:35:00Z",
  "target_stop_id": 53700172828,
  "target_time_begin": "2026-01-06T03:50:00Z",
  "cur_dev_s": 274.0,
  "telemetry": [
    {"tr_id": 131672, "event_time": "2026-01-06T03:20:00Z", "lon": 37.61, "lat": 55.75, "speed": 15, "location_valid": true, "is_hist_data": 0},
    ...
  ],
  "schedule": [
    {"tr_id": 131672, "tt_action_item_id": 53700172828, "time_begin": "2026-01-06T03:50:00Z", "geom": "POINT (37.62 55.76)", "manual_fill": false},
    ...
  ]
}
```

Ответ:

```json
{
  "sample_id": "131672_1767670500",
  "delay_pred_sec": 187.0,
  "risk_score": 0.83,
  "confidence": 0.71,
  "top_features": [{"name": "cur_dev_s", "value": 274.0}, ...],
  "model_version": "catboost-ensemble-v1"
}
```

Онлайн-фичи считаются тем же кодом, что и в батче (`features/tabular.build_features` через `features/from_stream.build_features_online`) — распределение фичей train/prod гарантированно совпадает.

Env-переменные:
- `ML_ARTIFACTS` — путь к папке артефактов (по умолчанию `ml/artifacts`)
- `ML_MODEL_VERSION` — строка версии для ответа (по умолчанию `catboost-ensemble-v1`)
- `ML_RISK_MID_SEC` — задержка, при которой `risk_score = 0.5` (по умолчанию 120)
- `ML_RISK_SLOPE_SEC` — крутизна сигмоиды (по умолчанию 60)

## ONNX-экспорт и latency

```bash
python ml/src/export_onnx.py --artifacts ml/artifacts
```

Что делает:
1. Обучает noroute-версию (без категориальной `route` — CatBoost ONNX её не поддерживает) на 5 сидах.
2. Экспортирует в ONNX (fp32) и пытается в INT8.
3. Замеряет per-sample latency: catboost native vs onnx fp32 vs onnx int8.

Текущие цифры (см. `ml/artifacts/bench_latency.json`):

| Формат | Latency, мс | Speed-up | Размер, МБ (сумма 5 моделей) |
|---|---|---|---|
| catboost native (с route) | 9.38 | 1× | 12.4 |
| onnx fp32 (noroute)       | **1.57** | **6.0×** | 58.7 |
| onnx int8                 | — | — | — |

`p95 latency < 200 мс` из DoD выполнено с запасом (у нас **~1.5 мс** на fp32-ONNX).

**Про INT8:** `onnxruntime.quantization.quantize_dynamic` падает с `Failed to find proper ai.onnx domain` — CatBoost использует свой ONNX-домен `ai.catboost`, стандартный quantizer его не понимает. Обходной путь есть (ручной rewrite через `onnx.utils.rebuild`), но fp32 уже даёт 6× ускорение и укладывается в требование, поэтому не тратим время.

**Про noroute vs full:** ONNX-контур работает без `route` (это MAE ~48 vs 40 у full-model), но latency-выигрыш реален. Прод-опция: держать оба контура — full-CatBoost для батча (submission), noroute-ONNX для realtime NDTP-потока.

## Правила по сабмитам

- Лимит 36 попыток / 24 успешных в день, до 27 сент 23:59 МСК.
- **Не спамить**: сабмитить только когда MAE на `labels_test.csv` улучшился на ≥ 3 секунды.
- Всегда держим последний рабочий `submission.csv` в корне репо, чтобы в любой момент можно было отправить.

## Антиутечка (не забыть)

При построении фичей для точки `T` использовать **только** `event_time ≤ T`. `cur_dev_s` — единственная информация о задержке, разрешённая к использованию. `time_fact_begin` из `schedule.csv` не используется вообще (`tabular.load_split` его дропает).
