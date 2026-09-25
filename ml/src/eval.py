"""Метрика хакатона: MAE + скор в [0, 1].

Формула из dataset/README.md:

    MAE      = mean(|факт - прогноз|)
    mae_zero = mean(|факт|)
    score    = max(0, min(1, (mae_zero - MAE) / (mae_zero - MAE_TARGET)))

MAE_TARGET скрыт платформой. Для локальной оценки его можно приблизить,
исходя из того, что baseline `prediction = cur_dev_s` даёт score ≈ 0.40.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def compute_score(pred: np.ndarray, true: np.ndarray, mae_target: float) -> float:
    mae = compute_mae(pred, true)
    mae_zero = float(np.mean(np.abs(np.asarray(true))))
    if mae_zero <= mae_target:
        return 1.0 if mae <= mae_target else 0.0
    return max(0.0, min(1.0, (mae_zero - mae) / (mae_zero - mae_target)))


def infer_mae_target(baseline_mae: float, baseline_score: float, mae_zero: float) -> float:
    """Восстановить MAE_TARGET по известной точке (обычно baseline cur_dev_s ≈ 0.40)."""
    if not 0.0 < baseline_score < 1.0:
        raise ValueError("baseline_score должен быть в (0, 1) для восстановления MAE_TARGET")
    return mae_zero - (mae_zero - baseline_mae) / baseline_score


def _load_labels_and_pred(labels_path: Path, pred_path: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    if "target_delay_s" not in labels.columns:
        raise ValueError(f"{labels_path} должен содержать колонку target_delay_s")

    pred_sep = ";" if pred_path.suffix == ".csv" and _looks_like_submission(pred_path) else ","
    pred = pd.read_csv(pred_path, sep=pred_sep)
    if "sample_id" not in pred.columns or "prediction" not in pred.columns:
        raise ValueError(f"{pred_path} должен содержать колонки sample_id, prediction")

    merged = labels.merge(pred[["sample_id", "prediction"]], on="sample_id", how="inner")
    missing = len(labels) - len(merged)
    if missing:
        print(f"[warn] {missing} sample_id из labels отсутствуют в предсказаниях")
    return merged


def _looks_like_submission(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as f:
        head = f.readline()
    return ";" in head and "," not in head.split(";", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Оценка MAE и скор-функции хакатона")
    ap.add_argument("--labels", required=True, type=Path, help="CSV с target_delay_s (labels_test.csv)")
    ap.add_argument("--pred", required=True, type=Path, help="CSV с колонками sample_id, prediction")
    ap.add_argument(
        "--mae-target",
        type=float,
        default=None,
        help="Целевой MAE для перевода в скор [0,1]. Если не задан — используется приближение по baseline.",
    )
    args = ap.parse_args()

    df = _load_labels_and_pred(args.labels, args.pred)
    mae = compute_mae(df["prediction"].values, df["target_delay_s"].values)
    mae_zero = float(np.mean(np.abs(df["target_delay_s"].values)))

    mae_target = args.mae_target
    if mae_target is None:
        mae_baseline = compute_mae(df["cur_dev_s"].values, df["target_delay_s"].values)
        mae_target = infer_mae_target(mae_baseline, baseline_score=0.40, mae_zero=mae_zero)
        print(f"[info] MAE_TARGET восстановлен по baseline: {mae_target:.2f} c")

    score = compute_score(df["prediction"].values, df["target_delay_s"].values, mae_target)
    print(f"MAE       = {mae:.2f} c")
    print(f"mae_zero  = {mae_zero:.2f} c")
    print(f"MAE_TARGET= {mae_target:.2f} c (для перевода в скор)")
    print(f"score     = {score:.4f}")


if __name__ == "__main__":
    main()
