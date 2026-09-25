"""CatBoost-трек: признаки -> оценка по трём схемам -> финальная модель -> submission.csv.

Схемы валидации (данные — один день, 13 реальных ТС + 26 синтетических клонов в train):

* ``proxy`` — **как validate**: K-fold по реальным точкам train+test блоками по времени; клоны и остальные
  точки остаются в обучении. Validate устроен так же (соседние моменты T тех же ТС), поэтому это лучший
  прокси скора платформы. Дополнительно печатается классический holdout train -> test.
* ``lovo`` — **честная**: откладываем реальное ТС вместе с клонами. Качество на новом ТС / другом дне.

Запуск::

    python ml/src/train_catboost.py --dataset ./dataset --eval          # оценка
    python ml/src/train_catboost.py --dataset ./dataset --fit --out submission.csv
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features.tabular import CAT_FEATURES, FEATURES, build_features, clone_sources, load_split  # noqa: E402

ART = Path(__file__).resolve().parents[1] / "artifacts"

PARAMS = dict(iterations=1500, learning_rate=0.03, depth=6, loss_function="RMSE", l2_leaf_reg=3.0,
              verbose=0, thread_count=-1, allow_writing_files=False)


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def score(y, p, mae_target: float) -> float:
    mz = mae(y, 0)
    return float(np.clip((mz - mae(y, p)) / (mz - mae_target), 0, 1))


# ----------------------------------------------------------------------------- данные


def load_all(dataset: Path, cache: bool = True) -> dict:
    """Признаки для train/test/validate (+ кеш в artifacts/cache)."""
    ART.mkdir(parents=True, exist_ok=True)
    cpath = ART / "cache_features.pkl"
    if cache and cpath.exists() and cpath.stat().st_mtime > Path(__file__).with_name("features").joinpath("tabular.py").stat().st_mtime:
        return pickle.loads(cpath.read_bytes())
    out = {}
    _, _, sched_train = load_split(dataset, "train")
    src = clone_sources(sched_train)
    for split in ("train", "test", "validate"):
        t0 = time.time()
        pts, tr, sch = load_split(dataset, split)
        X = build_features(pts, tr, sch, route_of=src)
        meta = pts.set_index("sample_id")[["tr_id", "T", "cur_dev_s"]].copy()
        meta["source"] = meta["tr_id"].map(lambda t: src.get(int(t), int(t)))
        meta["is_real"] = meta["tr_id"] < 9_000_000
        if "target_delay_s" in pts:
            meta["y"] = pts.set_index("sample_id")["target_delay_s"]
        out[split] = (X.loc[meta.index], meta)
        print(f"[features] {split}: {X.shape} за {time.time() - t0:.0f} c")
    cpath.write_bytes(pickle.dumps(out))
    return out


# ----------------------------------------------------------------------------- модель


def fit(X: pd.DataFrame, y_resid: np.ndarray, feats: list[str], params: dict, seed: int = 0,
        weight: np.ndarray | None = None) -> CatBoostRegressor:
    cats = [c for c in CAT_FEATURES if c in feats]
    m = CatBoostRegressor(**{**params, "random_seed": seed})
    m.fit(Pool(X[feats], y_resid, cat_features=cats, weight=weight))
    return m


def predict(models: list[CatBoostRegressor], X: pd.DataFrame, feats: list[str], cur_dev: np.ndarray) -> np.ndarray:
    cats = [c for c in CAT_FEATURES if c in feats]
    r = np.mean([m.predict(Pool(X[feats], cat_features=cats)) for m in models], axis=0)
    return r + cur_dev


def real_weight(meta: pd.DataFrame, w_syn: float) -> np.ndarray:
    return np.where(meta["is_real"], 1.0, w_syn)


# ----------------------------------------------------------------------------- схемы оценки


def pooled(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = pd.concat([data["train"][0], data["test"][0]])
    M = pd.concat([data["train"][1], data["test"][1]])
    return X, M


def eval_proxy(data: dict, feats: list[str], params: dict, k: int = 5, block_min: int = 30,
               w_syn: float = 1.0, seeds=(0,)) -> dict:
    """K-fold по реальным точкам блоками по времени (как validate: соседние T, клоны остаются в обучении)."""
    X, M = pooled(data)
    real = M.index[M["is_real"]]
    blocks = (M.loc[real, "tr_id"].astype(str) + "_" + (M.loc[real, "T"].astype("int64") // (block_min * 60 * 10**9)).astype(str))
    ub = blocks.unique()
    rng = np.random.RandomState(0)
    fold_of = dict(zip(ub, rng.randint(0, k, len(ub))))
    folds = blocks.map(fold_of)
    pred = pd.Series(np.nan, index=real)
    for f in range(k):
        te = folds.index[folds == f]
        tr = M.index.difference(te)
        y = (M.loc[tr, "y"] - M.loc[tr, "cur_dev_s"]).to_numpy()
        ms = [fit(X.loc[tr], y, feats, params, s, real_weight(M.loc[tr], w_syn)) for s in seeds]
        pred.loc[te] = predict(ms, X.loc[te], feats, M.loc[te, "cur_dev_s"].to_numpy())
    y = M.loc[real, "y"]
    return {"proxy_mae": mae(y, pred), "proxy_base": mae(y, M.loc[real, "cur_dev_s"]), "pred": pred}


def eval_holdout(data: dict, feats: list[str], params: dict, w_syn: float = 1.0, seeds=(0,)) -> dict:
    (Xtr, Mtr), (Xte, Mte) = data["train"], data["test"]
    y = (Mtr["y"] - Mtr["cur_dev_s"]).to_numpy()
    ms = [fit(Xtr, y, feats, params, s, real_weight(Mtr, w_syn)) for s in seeds]
    p = predict(ms, Xte, feats, Mte["cur_dev_s"].to_numpy())
    return {"test_mae": mae(Mte["y"], p), "test_base": mae(Mte["y"], Mte["cur_dev_s"]), "pred": p, "models": ms}


def eval_lovo(data: dict, feats: list[str], params: dict, w_syn: float = 1.0) -> dict:
    X, M = pooled(data)
    errs, base = [], []
    for v in sorted(M.loc[M["is_real"], "source"].unique()):
        te = M.index[(M["source"] == v) & M["is_real"]]
        tr = M.index[M["source"] != v]
        y = (M.loc[tr, "y"] - M.loc[tr, "cur_dev_s"]).to_numpy()
        m = fit(X.loc[tr], y, feats, {**params, "iterations": min(params["iterations"], 800)}, 0, real_weight(M.loc[tr], w_syn))
        p = predict([m], X.loc[te], feats, M.loc[te, "cur_dev_s"].to_numpy())
        errs += list(np.abs(M.loc[te, "y"] - p))
        base += list(np.abs(M.loc[te, "y"] - M.loc[te, "cur_dev_s"]))
    return {"lovo_mae": float(np.mean(errs)), "lovo_base": float(np.mean(base))}


def mae_target_estimate(data: dict) -> float:
    """MAE_TARGET по условию «бейзлайн cur_dev_s даёт score 0.40» (оценка на test)."""
    _, Mte = data["test"]
    mz, mb = mae(Mte["y"], 0), mae(Mte["y"], Mte["cur_dev_s"])
    return mz - (mz - mb) / 0.40


# ----------------------------------------------------------------------------- финал


def fit_final(data: dict, feats: list[str], params: dict, w_syn: float, seeds) -> list[CatBoostRegressor]:
    X, M = pooled(data)
    y = (M["y"] - M["cur_dev_s"]).to_numpy()
    return [fit(X, y, feats, params, s, real_weight(M, w_syn)) for s in seeds]


def write_submission(dataset: Path, ids: pd.Index, pred: np.ndarray, out: Path) -> pd.DataFrame:
    sample = pd.read_csv(dataset / "sample_submission.csv", sep=";")
    sub = sample[["sample_id"]].merge(pd.DataFrame({"sample_id": ids, "prediction": np.round(pred, 1)}),
                                      on="sample_id", how="left")
    assert len(sub) == len(sample) and sub["sample_id"].is_unique and sub["prediction"].notna().all()
    sub.to_csv(out, sep=";", index=False, encoding="utf-8", lineterminator="\r\n")
    return sub


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("./dataset"))
    ap.add_argument("--eval", action="store_true", help="оценить по трём схемам")
    ap.add_argument("--fit", action="store_true", help="обучить на train+test и записать сабмит")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "catboost.json")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    data = load_all(args.dataset, cache=not args.no_cache)
    cfg = json.loads(args.config.read_text()) if args.config.exists() else {}
    feats = cfg.get("features", FEATURES)
    params = {**PARAMS, **cfg.get("params", {})}
    w_syn = cfg.get("w_syn", 1.0)
    seeds = list(range(cfg.get("n_seeds", 5)))
    mt = mae_target_estimate(data)

    if args.eval:
        ho = eval_holdout(data, feats, params, w_syn)
        px = eval_proxy(data, feats, params, w_syn=w_syn)
        lv = eval_lovo(data, feats, params, w_syn)
        print(f"MAE_TARGET ≈ {mt:.1f}")
        print(f"holdout train->test: MAE {ho['test_mae']:.2f} (бейзлайн {ho['test_base']:.2f})  score≈{score(data['test'][1]['y'], ho['pred'], mt):.3f}")
        print(f"proxy K-fold (как validate): MAE {px['proxy_mae']:.2f} (бейзлайн {px['proxy_base']:.2f})")
        print(f"LOVO (честная): MAE {lv['lovo_mae']:.2f} (бейзлайн {lv['lovo_base']:.2f})")

    if args.fit:
        models = fit_final(data, feats, params, w_syn, seeds)
        Xv, Mv = data["validate"]
        pred = predict(models, Xv, feats, Mv["cur_dev_s"].to_numpy())
        sub = write_submission(args.dataset, Xv.index, pred, args.out)
        ART.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(models):
            m.save_model(str(ART / f"catboost_seed{i}.cbm"))
        (ART / "catboost_meta.json").write_text(json.dumps(
            {"features": feats, "cat_features": [c for c in CAT_FEATURES if c in feats], "params": params,
             "w_syn": w_syn, "n_models": len(models), "target": "target_delay_s - cur_dev_s"}, ensure_ascii=False, indent=2))
        print(f"[submission] {args.out}: {len(sub)} строк; прогноз {sub['prediction'].describe().round(1).to_dict()}")


if __name__ == "__main__":
    main()
