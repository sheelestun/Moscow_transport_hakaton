Backend-шлюз
=============

Диспетчерский шлюз между NDTP-эмулятором/фронтом и ML-сервисом. FastAPI + WebSocket,
симуляция движения ТС по маршрутам (то же, что ``frontend/js/mock.js``), проксирование
``/whatif`` в ML.

Контракт вход/выход зафиксирован во ``frontend/js/api.js`` — live-режим отдаёт ровно
то же, что мок, чтобы фронт-код не различал источник.

Модули
------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   main
   simulator
   routes_data
   csv_replayer

Endpoints
---------

``GET /health``
    Статус: число ТС, алертов, клиентов WS, доступность ML.

``GET /routes``
    6 маршрутов с polyline-геометрией (те же ``MOCK_ROUTES``, что во фронт-моке).

``GET /vehicles``
    Все ТС в текущий момент: координаты, ``delay_now_sec``, ``delay_pred_sec``,
    ``risk_score``, у не-зелёных дополнительно ``reason_pattern``, ``top_features``.

``GET /vehicles/{id}/schedule``
    Полное расписание рейса с ``passed / next / upcoming`` и флагом ``is_target``.

``GET /alerts?active=true``
    Активные алерты, каждый содержит ``recommendation`` и ``eta_incident``.

``GET /metrics/model``
    Прокси на ML ``/metrics/model``; при недоступности ML — последние закешированные
    метрики или заглушка.

``POST /whatif``
    Сценарий по маршруту. Локальная эвристика ``MEASURE_EFFECT`` +
    точечный вызов ML ``/whatif/predict`` для каждого ТС.

``POST /apply``
    Применить меру в симуляции (пока demo-режим).

``WS /ws``
    Пуш ``vehicle.update`` каждый тик; ``alert.new`` / ``alert.verified`` /
    ``alert.resolved`` по факту.
