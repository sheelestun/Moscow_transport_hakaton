# ML-модуль: FastAPI-сервис прогноза задержек (CatBoost-ансамбль).
# Лёгкий образ: только зависимости инференса (без PyTorch/ONNX — они нужны лишь для обучения и экспериментов).
FROM python:3.12-slim

WORKDIR /app

# curl — для HEALTHCHECK (его же использует docker-compose)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt     && pip install --no-cache-dir --no-deps "catboost>=1.2"

COPY src/ ./src/
# Модели: train_catboost.py --fit кладёт их в ml/artifacts. В docker-compose папка ещё и монтируется,
# так что переобученные модели подхватываются без пересборки (POST /reload).
COPY artifacts/ ./artifacts/

ENV ML_ARTIFACTS=/app/artifacts \
    PYTHONUNBUFFERED=1

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["uvicorn", "src.inference_service:app", "--host", "0.0.0.0", "--port", "8001"]
