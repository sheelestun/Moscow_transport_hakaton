"""Собирает submission.csv по validate/points.csv.

По умолчанию — бейзлайн prediction = cur_dev_s (даёт score ≈ 0.40).
Когда появится обученная модель — заменить `predict_fn` в main().
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def baseline_cur_dev(points: pd.DataFrame) -> np.ndarray:
    """Прогноз = задержка на последней уже пройденной остановке."""
    return points["cur_dev_s"].fillna(0.0).values


def build_submission(points: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    if len(points) != len(predictions):
        raise ValueError(f"размеры не совпадают: {len(points)} points vs {len(predictions)} predictions")
    out = pd.DataFrame({"sample_id": points["sample_id"].values, "prediction": predictions})
    if out["sample_id"].duplicated().any():
        raise ValueError("дублирующиеся sample_id в сабмите")
    return out


def write_submission(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep=";", index=False, encoding="utf-8", lineterminator="\r\n")
    print(f"submission записан в {path} ({len(df)} строк)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Сборка submission.csv для validate")
    ap.add_argument("--dataset", required=True, type=Path, help="Корень датасета (там есть validate/)")
    ap.add_argument("--out", required=True, type=Path, help="Куда писать submission.csv")
    ap.add_argument("--baseline", action="store_true", help="Использовать бейзлайн cur_dev_s")
    args = ap.parse_args()

    points_path = args.dataset / "validate" / "points.csv"
    points = pd.read_csv(points_path)
    print(f"загружено точек: {len(points)} из {points_path}")

    if args.baseline:
        predictions = baseline_cur_dev(points)
    else:
        raise NotImplementedError(
            "Добавьте вызов обученной модели вместо baseline. Пока используйте --baseline."
        )

    sub = build_submission(points, predictions)
    write_submission(sub, args.out)


if __name__ == "__main__":
    main()
