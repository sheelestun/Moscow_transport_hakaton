"""Юнит-тесты утилит `train_catboost` без загрузки данных и моделей.

Каждый — быстрый (<0.1 с). Ловят регрессии контракта метрик, которые фронт и
подсчёт скора хакатона зависят.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from train_catboost import LEAD_BINS, mae, mae_by_lead, real_weight, score


def test_mae_zero_when_pred_equals_y() -> None:
    assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_mae_matches_np() -> None:
    y = np.array([100, -30, 45, 0], dtype=float)
    p = np.array([80, -20, 90, 5], dtype=float)
    assert mae(y, p) == pytest.approx(float(np.mean(np.abs(y - p))))


def test_score_bounded() -> None:
    """score = 1.0 если модель идеальна, 0.0 если хуже baseline."""
    y = np.array([100.0, 200.0, 50.0])
    assert score(y, y, mae_target=10.0) == pytest.approx(1.0)
    assert score(y, np.zeros_like(y), mae_target=10.0) == pytest.approx(0.0)


def test_score_clipped_to_unit_interval() -> None:
    """Даже при отрицательной разнице скор не уходит ниже 0."""
    y = np.array([50.0, 50.0])
    huge_error = np.array([500.0, 500.0])
    assert score(y, huge_error, mae_target=10.0) == 0.0


def test_real_weight_synthetic_less() -> None:
    meta = pd.DataFrame({"is_real": [True, False, True, False]})
    w = real_weight(meta, w_syn=0.5)
    assert list(w) == [1.0, 0.5, 1.0, 0.5]


def test_mae_by_lead_bins() -> None:
    """5 бинов, точка в каждом → 5 групп. MAE считается ровно на своих точках."""
    lead = pd.Series([60, 240, 400, 700, 1200])   # 0-3м, 3-5м, 5-10м, 10-15м, 15+
    y = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0])
    p = pd.Series([90.0, 80.0, 60.0, 30.0, 10.0])  # ошибки 10, 20, 40, 70, 90
    out = mae_by_lead(y, p, lead)
    assert len(out) == 5
    by_bin = {r["bin"]: r["mae_s"] for r in out}
    assert by_bin["0-3м"] == pytest.approx(10.0)
    assert by_bin["3-5м"] == pytest.approx(20.0)
    assert by_bin["5-10м"] == pytest.approx(40.0)
    assert by_bin["10-15м"] == pytest.approx(70.0)
    assert by_bin["15м+"] == pytest.approx(90.0)


def test_mae_by_lead_skips_empty_bins() -> None:
    """Пустой бин не попадает в отчёт (иначе UI рисует пустые полосы)."""
    lead = pd.Series([60, 60, 60])
    y = pd.Series([10.0, 20.0, 30.0])
    p = pd.Series([10.0, 20.0, 30.0])
    out = mae_by_lead(y, p, lead)
    assert len(out) == 1
    assert out[0]["bin"] == "0-3м"
    assert out[0]["points"] == 3


def test_lead_bins_cover_zero_to_infinity() -> None:
    """Не должно быть дыр между бинами — иначе часть точек молча теряется."""
    prev_hi = 0.0
    for lo, hi, _ in LEAD_BINS:
        assert lo == prev_hi
        prev_hi = hi
    assert prev_hi == float("inf")
