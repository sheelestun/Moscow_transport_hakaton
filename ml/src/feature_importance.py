"""CatBoost feature-importance топ-N в формате Шелестова.

Усредняем `PredictionValuesChange` (native) по всем 5 сидам ансамбля, добавляем
русское описание из `feature_labels_ml.json`. Пишем `statistics/tables/feature_importance.csv`::

    rank,feature,importance,label

Запуск::

    python ml/src/feature_importance.py \\
        --artifacts ml/artifacts \\
        --out statistics/tables/feature_importance.csv \\
        --top 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    ap.add_argument("--out", type=Path, default=Path("./statistics/tables/feature_importance.csv"))
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    meta = json.loads((args.artifacts / "catboost_meta.json").read_text())
    labels_path = Path(__file__).resolve().parents[1] / "configs" / "feature_labels_ml.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}

    feats = meta["features"]
    imps = []
    for i in range(meta["n_models"]):
        m = CatBoostRegressor()
        m.load_model(str(args.artifacts / f"catboost_seed{i}.cbm"))
        imps.append(m.get_feature_importance())
    avg = np.mean(imps, axis=0)

    df = (pd.DataFrame({"feature": feats, "importance": avg})
          .sort_values("importance", ascending=False)
          .reset_index(drop=True))
    df.insert(0, "rank", df.index + 1)
    df["label"] = df["feature"].map(lambda f: labels.get(f, f))
    df["importance"] = df["importance"].round(3)
    top = df.head(args.top)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(args.out, index=False)
    print(f"[save] {args.out} — топ {len(top)} из {len(df)} фичей")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
