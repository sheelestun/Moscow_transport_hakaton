# Локальный запуск — Moscow Transport Hackathon

Полный стек (ML + backend + frontend + NDTP-эмулятор + redis + postgres) поднимается одной командой в Docker Compose. Отдельно можно гонять только backend с симулятором — так работает Даниил Герман над NDTP-парсером, не поднимая ML.

## 0. Что нужно на машине

- Docker и Docker Compose v2 (`docker compose version` должен работать)
- Python 3.12+ и `pip` (для тестов и локального dev-loop без docker)
- Node.js 20+ (только если хочется править фронт без docker)
- ~4 ГБ RAM, ~2 ГБ диска

## 1. Клонировать репу

```bash
git clone git@github.com:sheelestun/Moscow_transport_hakaton.git
cd Moscow_transport_hakaton
git checkout integration/backend-e2e-demo   # ветка интеграции, здесь свежее чем в main
```

## 2. Забрать датасет

Датасет лежит на Яндекс.Диске (~200 МБ, в git не пушим): <https://disk.yandex.ru/d/CA6tsj4aJJ4Aaw>

Скачать и распаковать в `./dataset/` — должна получиться такая структура:

```
dataset/
├── docs/                            # PDF-описание NDTP-протокола
├── labels/                          # labels_{train,test,validate}.csv
├── train/  test/  validate/         # traffic.csv, schedule.csv
├── sample_submission.csv
├── ndtp-telemetry-emulator.tar      # OCI-образ Java-эмулятора (129 МБ)
└── README.md
```

Быстрый чек, что подмонтировали правильно:

```bash
ls dataset/validate/     # должен показать traffic.csv, schedule.csv
du -sh dataset/          # ~200 МБ
```

## 3. Один раз — загрузить образ эмулятора

Эмулятор поставляется OCI-архивом, в Docker Hub его нет:

```bash
docker load -i dataset/ndtp-telemetry-emulator.tar
docker image ls | grep ndtp-telemetry-emulator   # должен появиться :1.0
```

## 4. Поднять весь стек

```bash
docker compose up -d --build
docker compose ps                         # ждём пока все healthy (10-20 сек)
```

- **Дашборд**: <http://localhost:3000>
- **Backend Swagger**: <http://localhost:8000/docs>
- **ML Swagger**: <http://localhost:8001/docs>
- **Эмулятор API**: <http://localhost:18080> (POST `/api/config` — конфиг из `infra/emulator-config.json`)

Логи одного сервиса: `docker compose logs -f backend`. Остановить: `docker compose down` (persistent postgres не удаляется).

## 5. Проверить, что работает

Открыть <http://localhost:3000>, на карте должны быть:
- 24 автобуса на 6 маршрутах (м1/м2/м5/196/Б/…)
- в правой панели «Требуют внимания» — 3-5 алертов
- виджеты: «Опоздания по часам», «Проблемные остановки», «Слипание паровозиком», «Точность по горизонту»

Полный сценарий демо (8 шагов) — [`infra/DEMO.md`](./infra/DEMO.md).

## 6. Автотесты (без docker)

Быстрые unit/smoke-тесты бэкенда и ML — гоняются в CI и локально:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt -r ml/requirements.txt pytest httpx
pytest                # backend/tests + ml/tests + statistics/tests
```

Ждём `33 passed`. Если что-то красное — до пуша в main не мёржим.

## 7. Только backend (для Даниила Германа, NDTP-парсер)

Бэкенд сам содержит симулятор ТС и вызывает ML только на What-if. Для разработки парсера можно поднять backend отдельно (без ML), чтобы не грузить CatBoost:

```bash
cd backend
pip install -r requirements.txt
ML_URL=http://127.0.0.1:1 uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

`ML_URL` бьётся в пустоту — бэкенд использует локальный фолбэк (`Simulator.measure_effect`). Все эндпоинты (`/vehicles`, `/routes`, `/whatif`, `/ws`) работают, дашборд можно открыть на <http://localhost:3000> (или собрать фронт напрямую: `cd frontend && python -m http.server 3000`).

Контракт для NDTP-парсера — заменить симулятор в `backend/src/simulator.py` на потребителя из `ndtp-emu` TCP-потока. Формат `pub_vehicle()` — единственное, что нужно сохранить как есть (см. `frontend/js/mock.js::pub` — source of truth для фронта).

## 8. Только ML (переобучение)

```bash
cd ml
pip install -r requirements.txt
# оценка (~5 мин, пишет ml/artifacts/catboost_metrics.json)
python src/train_catboost.py --dataset ../dataset --eval
# финальная модель на train+test + сабмит на validate (~3 мин)
python src/train_catboost.py --dataset ../dataset --fit --out submission.csv
```

Артефакты (`catboost_seed*.cbm`, `catboost_meta.json`) читаются live-контейнером ML при старте — если поменяли, дёрни `curl -X POST http://localhost:8001/reload`.

## 9. Порты и что на каких сидит

| Сервис | Хост-порт | Внутри compose | Наружу через |
|---|---|---|---|
| frontend | 3000 | nginx :80 | Браузер |
| backend | 8000 | uvicorn :8000 | Браузер / фронт |
| ml | 8001 | uvicorn :8001 | Backend, curl |
| ndtp-emu | 18080 (HTTP), TCP на targetHost | :18080 | Backend по TCP |
| postgres | 5432 | :5432 | psql |
| redis | — | :6379 | только внутри сети |

## 10. Типовые проблемы

- **`ndtp-telemetry-emulator:1.0 not found`** → забыт `docker load -i` из шага 3
- **`ML healthcheck failed`** → в `ml/artifacts/` нет `catboost_seed*.cbm`; обучить моделью (шаг 8) или скачать артефакты из ветки `ml/catboost-tabular`
- **Дашборд пустой, `status: degraded`** → backend не поднялся; `docker compose logs backend`
- **`No module named 'catboost'` в pytest** → `pip install -r ml/requirements.txt` (для юнит-тестов ML нужен catboost)
- **`Address already in use :8000`** → у тебя уже что-то на 8000; `docker compose down` или поменять порт в `docker-compose.yml`

## Что где искать

- `ARCHITECTURE_AND_ROLES.md` — как компоненты соединены, кто что делает
- `ml/README.md` — детали ML-трека (какие фичи, чем валидируется, ONNX)
- `infra/DEMO.md` — сценарий показа жюри
- `statistics/REPORT.md` — анализ датасета Шелестова с графиками
