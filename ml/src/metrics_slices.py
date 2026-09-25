"""MAE нашего CatBoost-ансамбля в срезах cohort/class/hour, в формате Шелестова.

**Важно про честность:** финальные `catboost_seed*.cbm` обучены на `train+test` (для
сабмита на validate), поэтому прямая оценка на train/test даёт leakage-число. Здесь
используем **OOF-предикты proxy K-fold** (5 фолдов блоками по 30 мин + tr_id, как
устроен validate) — они получены моделями, которые ТЕСТОВЫЕ точки не видели. Это
матчит «честный» proxy_mae из `train_catboost.py --eval`.

Формат выхода `statistics/tables/model_metrics.csv`::

    split,dimension,group,points,mae_model_s

Тот же ключ (split,dimension,group) что и в `baseline_metrics.csv` — Шелестов
делает join и получает MAE модели рядом с baseline.

Запуск::

    python ml/src/metrics_slices.py \\
        --labels ./dataset/labels \\
        --artifacts ml/artifacts \\
        --out statistics/tables/model_metrics.csv
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_catboost import PARAMS, fit, predict, pooled, real_weight  # noqa: E402
from features.tabular import FEATURES  # noqa: E402


def oof_predictions(data: dict, feats: list[str], params: dict, k: int = 5,
                    block_min: int = 30, seeds=(0,), w_syn: float = 1.0) -> pd.Series:
    """OOF предикт proxy K-fold: клоны в обучении, реальные точки — в отложенных фолдах."""
    X, M = pooled(data)
    real = M.index[M["is_real"]]
    blocks = (M.loc[real, "tr_id"].astype(str)
              + "_" + (M.loc[real, "T"].astype("int64") // (block_min * 60 * 10**9)).astype(str))
    ub = blocks.unique()
    rng = np.random.RandomState(0)
    fold_of = dict(zip(ub, rng.randint(0, k, len(ub))))
    folds = blocks.map(fold_of)
    pred = pd.Series(np.nan, index=real, dtype=float)
    for f in range(k):
        te = folds.index[folds == f]
        tr = M.index.difference(te)
        y = (M.loc[tr, "y"] - M.loc[tr, "cur_dev_s"]).to_numpy()
        ms = [fit(X.loc[tr], y, feats, params, s, real_weight(M.loc[tr], w_syn)) for s in seeds]
        pred.loc[te] = predict(ms, X.loc[te], feats, M.loc[te, "cur_dev_s"].to_numpy())
        print(f"  fold {f+1}/{k}: |te|={len(te)}")
    return pred


def slice_metrics(split: str, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    err = (meta["y_pred"] - meta["y"]).abs()

    def add(dim: str, group: str, mask: pd.Series) -> None:
        n = int(mask.sum())
        if n == 0:
            return
        rows.append({"split": split, "dimension": dim, "group": str(group),
                     "points": n, "mae_model_s": float(err[mask].mean())})

    add("cohort", "all", pd.Series(True, index=meta.index))
    for value in ("real", "synthetic_candidate"):
        add("cohort", value, meta["cohort"] == value)
    for value in ("early", "ontime", "late"):
        add("class", value, meta["target_class"] == value)
    for hour in sorted(meta["hour"].unique()):
        add("hour", str(hour), meta["hour"] == hour)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("./dataset/labels"))
    ap.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    ap.add_argument("--out", type=Path, default=Path("./statistics/tables/model_metrics.csv"))
    args = ap.parse_args()

    print(f"[load] cache {args.artifacts / 'cache_features.pkl'}")
    data = pickle.loads((args.artifacts / "cache_features.pkl").read_bytes())

    print(f"[oof] proxy K-fold, 5 фолдов, 1 сид (для честной оценки без утечки)")
    oof = oof_predictions(data, FEATURES, PARAMS, k=5, seeds=(0,))

    labels = {s: pd.read_csv(args.labels / f"labels_{s}.csv",
                             usecols=["sample_id", "target_class"], dtype={"sample_id": "string"})
              for s in ("train", "test")}

    _, M_all = pooled(data)
    M_all = M_all.copy()
    M_all["y_pred"] = oof
    M_all["hour"] = pd.to_datetime(M_all["T"]).dt.hour
    M_all["cohort"] = np.where(M_all["is_real"], "real", "synthetic_candidate")

    # цепляем target_class по source (в M cache есть source = train/test/... через split)
    n_train = len(data["train"][1])
    M_all = M_all.reset_index(drop=True)
    tc = pd.Series(index=M_all.index, dtype="object")
    tc.iloc[:n_train] = labels["train"]["target_class"].values
    tc.iloc[n_train:] = labels["test"]["target_class"].values
    M_all["target_class"] = tc
    M_all["split"] = ["train"] * n_train + ["test"] * (len(M_all) - n_train)

    all_rows = []
    for split in ("train", "test"):
        m = M_all[(M_all["split"] == split) & M_all["y_pred"].notna()].copy()
        if m.empty:
            continue
        mae_split = float((m["y_pred"] - m["y"]).abs().mean())
        base_split = float(m["y"].abs().mean())
        print(f"[{split}] OOF n={len(m)} MAE(model)={mae_split:.2f} baseline MAE(zero)={base_split:.2f}")
        all_rows.append(slice_metrics(split, m))

    out = pd.concat(all_rows, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[save] {args.out} — {len(out)} строк")
    print(out.groupby(["split", "dimension"])["mae_model_s"].mean().round(2))


if __name__ == "__main__":
    main()
