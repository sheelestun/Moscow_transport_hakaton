# ML-модуль

Работают: **Степан, Фёдор.**

Задача — регрессия `target_delay_s` (секунды, знак важен) на первой остановке в окне `(T+10, T+15]` мин. Метрика — MAE, скор в [0,1]. Baseline `prediction = cur_dev_s` даёт ≈ 0.40 (3 балла). Цель — ≥ 0.70 (6 баллов).

Подробнее про задачу и контракты — в корневом [`ARCHITECTURE_AND_ROLES.md`](../ARCHITECTURE_AND_ROLES.md).

## Структура

```
ml/
├── src/
│   ├── features/          общий модуль фичей (батч + онлайн)
│   ├── models/            catboost_tab.py, torch_seq.py, ensemble.py
│   ├── eval.py            MAE + score-функция как в ТЗ
│   ├── predict_submission.py   собирает validate → submission.csv
│   ├── train.py           обучение (TODO)
│   ├── export_onnx.py     квантизация + экспорт (TODO)
│   └── inference_service.py    FastAPI /predict (TODO)
├── notebooks/             EDA и эксперименты
├── artifacts/             веса, ONNX (в .gitignore)
├── Dockerfile
└── requirements.txt
```

## Данные

Ожидаются в `../dataset/` относительно корня репо или в пути из `DATASET_DIR`.

Скачать: `https://disk.yandex.ru/d/CA6tsj4aJJ4Aaw` (в git не коммитим).

## Быстрый старт

```bash
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# бейзлайн cur_dev_s → submission.csv (сразу ≈0.40)
python src/predict_submission.py \
  --dataset ../dataset \
  --out ../submission_baseline.csv \
  --baseline

# оценка на локальном тесте (после того как появится модель)
python src/eval.py \
  --labels ../dataset/labels/labels_test.csv \
  --pred my_test_predictions.csv
```

## Разделение работ (черновик, скорректируем по ходу)

**Фёдор — табличный трек:**
- `src/features/from_csv.py` — фичи из `traffic.csv` + `schedule.csv` до момента `T` (окно последних N пингов, средняя скорость, dwell, headway, планируемое время до целевой, час/день, погода/пробки если успеем).
- `src/models/catboost_tab.py` — обучение CatBoost на `labels_train` + `traffic_train`, валидация на `labels_test`.
- Первый MVP-сабмит с рабочей CatBoost-моделью (сверху бейзлайна).

**Степан — sequence-трек:**
- `src/models/torch_seq.py` — Transformer/GRU на окне телеметрии (последние ~30 пингов = ~6 минут).
- Единый интерфейс `predict(features_batch)` совместимый с CatBoost.
- `src/models/ensemble.py` — weighted blend по MAE на `labels_test`.
- `src/export_onnx.py` — экспорт ансамбля + INT8 квантизация, замер latency до/после.

**Совместно:**
- `src/features/` — общий модуль фичей: **один и тот же код** работает и в батче (для сабмита), и онлайн (для инференс-сервиса). Договориться о схеме.
- `src/inference_service.py` — FastAPI с `POST /predict` и `POST /predict/batch` по контракту из `ARCHITECTURE_AND_ROLES.md` §7.1.

## Правила по сабмитам

- Лимит 36 попыток / 24 успешных в день, до 27 сент 23:59 МСК.
- **Не спамить**: сабмитить только когда MAE на `labels_test.csv` улучшился на ≥ 3 секунды.
- Всегда держим последний рабочий `submission.csv` в корне репо, чтобы в любой момент можно было отправить.
- В зачёт идёт лучший результат — не бойтесь сабмитить сильные версии.

## Антиутечка (не забыть)

При построении фичей для точки `T` использовать **только** `event_time ≤ T`. `cur_dev_s` — единственная информация о задержке, разрешённая к использованию. `time_fact_begin` из `schedule.csv` можно использовать только для точек, где остановка уже пройдена (обычно достаточно `cur_dev_s`).
