# Датасет

Скачивается с Яндекс.Диска: **https://disk.yandex.ru/d/CA6tsj4aJJ4Aaw**

Содержимое этой папки в git **не коммитим** (см. корневой `.gitignore`). Здесь должен получиться такой набор:

```
dataset/
├── README.md                       (этот файл, единственный под git)
├── train/
│   ├── traffic.csv                 телеметрия
│   └── schedule.csv                расписание с фактом (time_fact_begin)
├── test/
│   ├── traffic.csv
│   └── schedule.csv
├── labels/
│   ├── labels_train.csv            прогнозные точки с target_delay_s
│   └── labels_test.csv
├── validate/
│   ├── traffic.csv
│   ├── schedule_plan.csv           только план, без факта
│   └── points.csv                  151 точка для сабмита
├── sample_submission.csv           бейзлайн cur_dev_s, score ≈ 0.40
├── ndtp-telemetry-emulator.tar     Docker-образ эмулятора (~128 MB)
└── docs/
    └── Emulator-and-Telematic-Packets-Specification.md
```

## Как разложить

Скачайте архив с Яндекс.Диска и распакуйте прямо в эту папку так, чтобы структура выше сложилась. Проверить:

```bash
ls dataset/validate/points.csv                  # должен существовать
wc -l dataset/labels/labels_train.csv           # 4435 (с заголовком)
wc -l dataset/validate/points.csv               # 152 (с заголовком)
```

## Как использовать

Скрипты по умолчанию ищут датасет в этой папке:

```bash
python ml/src/predict_submission.py --dataset ./dataset --out submission.csv --baseline
python ml/src/eval.py --labels ./dataset/labels/labels_test.csv --pred my_pred.csv
```

Если хочется хранить датасет в другом месте — можно указывать флагом `--dataset /путь/к/датасету` или через переменную `DATASET_DIR` (когда добавим поддержку).

## Эмулятор NDTP

Загрузка и запуск (нужны только для real-time контура, для обучения не требуется):

```bash
docker load -i dataset/ndtp-telemetry-emulator.tar
docker run --rm -p 18080:18080 --add-host=host.docker.internal:host-gateway \
  --name ndtp-emu ndtp-telemetry-emulator:1.0
```

Спека протокола и REST API: `dataset/docs/Emulator-and-Telematic-Packets-Specification.md`.
