"""Собирает submission.csv по validate/points.csv.

Режимы:
  --baseline               prediction = cur_dev_s (score ≈ 0.40)
  --model <path.pt>        обученная sequence-модель из train.py (residual + cur_dev_s)
  --catboost [dir]         ансамбль CatBoost из train_catboost.py (residual + cur_dev_s)

Формат сабмита: sample_id;prediction, разделитель ';', CRLF (совпадает с sample_submission.csv).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def baseline_cur_dev(points: pd.DataFrame) -> np.ndarray:
    """Прогноз = задержка на последней уже пройденной остановке."""
    return points["cur_dev_s"].fillna(0.0).values


def model_predict(dataset_dir: Path, model_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Загружает чекпоинт, строит фичи на validate, возвращает (points, абсолютные предсказания)."""
    import torch

    from features.from_csv import build_features, load_split
    from models.torch_seq import DelaySeqDataset, SeqDelayModel, predict_residual

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    points, traffic, schedule = load_split(dataset_dir, "validate")
    history = ckpt.get("history")
    if history is None:
        print("[warn] в чекпоинте нет history — фичи tr_hist_* заполнятся глобальным средним")
    static_df, seq_array = build_features(
        points, traffic, schedule, seq_len=ckpt["seq_len"], history=history
    )
    static_np = static_df.to_numpy(dtype=np.float32)

    ds = DelaySeqDataset(
        seq_array,
        static_np,
        target=None,
        static_mean=ckpt["static_mean"],
        static_std=ckpt["static_std"],
    )
    model = SeqDelayModel(
        seq_feat_dim=ckpt["seq_feat_dim"],
        static_feat_dim=ckpt["static_feat_dim"],
        hidden=ckpt["hidden"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["state_dict"])

    residuals = predict_residual(model, ds)
    cur_dev = points["cur_dev_s"].fillna(0.0).to_numpy(dtype=np.float32)
    predictions = residuals + cur_dev
    return points, predictions


def catboost_predict(dataset_dir: Path, art_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Ансамбль CatBoost из train_catboost.py (artifacts/catboost_seed*.cbm + catboost_meta.json)."""
    import json

    from catboost import CatBoostRegressor, Pool

    from features.tabular import build_features, clone_sources, load_split

    meta = json.loads((art_dir / "catboost_meta.json").read_text(encoding="utf-8"))
    feats, cats = meta["features"], meta["cat_features"]
    _, _, sched_train = load_split(dataset_dir, "train")
    points, traffic, schedule = load_split(dataset_dir, "validate")
    X = build_features(points, traffic, schedule, route_of=clone_sources(sched_train))
    pool = Pool(X[feats], cat_features=cats)
    models = [CatBoostRegressor().load_model(str(art_dir / f"catboost_seed{i}.cbm")) for i in range(meta["n_models"])]
    resid = np.mean([m.predict(pool) for m in models], axis=0)
    return points, resid + points["cur_dev_s"].fillna(0.0).to_numpy()


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
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline", action="store_true", help="Использовать бейзлайн cur_dev_s")
    mode.add_argument("--model", type=Path, help="Путь к чекпоинту .pt из train.py")
    mode.add_argument("--catboost", type=Path, nargs="?", const=Path(__file__).resolve().parents[1] / "artifacts",
                      help="Папка с моделями CatBoost из train_catboost.py (по умолчанию ml/artifacts)")
    args = ap.parse_args()

    if args.catboost:
        print(f"[catboost] {args.catboost}")
        points, predictions = catboost_predict(args.dataset, args.catboost)
    elif args.baseline:
        points_path = args.dataset / "validate" / "points.csv"
        points = pd.read_csv(points_path)
        print(f"загружено точек: {len(points)} из {points_path}")
        predictions = baseline_cur_dev(points)
    else:
        print(f"[model] {args.model}")
        points, predictions = model_predict(args.dataset, args.model)
        print(f"загружено точек: {len(points)}")

    sub = build_submission(points, predictions)
    write_submission(sub, args.out)


if __name__ == "__main__":
    main()
