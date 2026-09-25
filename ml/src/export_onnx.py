"""Экспорт CatBoost-ансамбля в ONNX + INT8 квантизация + замер latency.

Технический нюанс: CatBoost ONNX-экспорт **не поддерживает категориальные фичи**
(`route` у нас категориальная). Поэтому онлайн-контур на ONNX использует noroute-версию
модели (без фичи `route`) — на нашем сплите она MAE ~48 (vs 40 у полной), но зато экспортируется
в ONNX + квантуется в INT8. Основной submission-путь остаётся на native CatBoost.

Что делает скрипт:
  1) Если нет `catboost_noroute_seed*.cbm` — обучает их из кэша фичей (те же гиперы, без route).
  2) Экспортирует каждый seed в ONNX (fp32) и INT8.
  3) Замеряет per-sample latency: catboost native vs onnx fp32 vs onnx int8.
  4) Пишет `bench_latency.json` в артефакты.

Запуск::

    python ml/src/export_onnx.py --artifacts ml/artifacts
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features.tabular import FEATURES  # noqa: E402
from train_catboost import PARAMS, fit, pooled, real_weight  # noqa: E402


NOROUTE_PARAMS = dict(PARAMS, iterations=600, learning_rate=0.07, depth=8, l2_leaf_reg=3.0)
NOROUTE_FEATURES = [f for f in FEATURES if f != "route"]
NOROUTE_SEEDS = 5


def ensure_noroute_models(art: Path) -> list[Path]:
    """Гарантирует существование noroute-моделей. Обучает их из cache_features.pkl если нужно."""
    paths = [art / f"catboost_noroute_seed{i}.cbm" for i in range(NOROUTE_SEEDS)]
    if all(p.exists() for p in paths):
        print(f"[noroute] найдены готовые чекпоинты: {len(paths)}")
        return paths

    cache_path = art / "cache_features.pkl"
    if not cache_path.exists():
        raise RuntimeError(f"нет {cache_path} — сначала запусти train_catboost.py --fit")
    data = pickle.loads(cache_path.read_bytes())
    X, M = pooled(data)
    y = (M["y"] - M["cur_dev_s"]).to_numpy()
    w = real_weight(M, w_syn=1.0)
    print(f"[noroute] тренирую {NOROUTE_SEEDS} сидов, {len(NOROUTE_FEATURES)} фичей, {len(X)} строк")
    for seed, path in enumerate(paths):
        if path.exists():
            continue
        m = fit(X, y, NOROUTE_FEATURES, NOROUTE_PARAMS, seed=seed, weight=w)
        m.save_model(str(path))
    return paths


def export_onnx(cbm_paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Каждый CBM → ONNX (fp32) + ONNX (INT8). Возвращает (fp32_paths, int8_paths)."""
    onnx_paths, int8_paths = [], []
    for cbm in cbm_paths:
        onnx = cbm.with_suffix(".onnx")
        int8 = cbm.parent / (cbm.stem + "_int8.onnx")
        m = CatBoostRegressor()
        m.load_model(str(cbm))
        m.save_model(str(onnx), format="onnx",
                     export_parameters={"onnx_domain": "ai.catboost",
                                        "onnx_model_version": 1,
                                        "onnx_doc_string": f"Delay predictor {cbm.stem}",
                                        "onnx_graph_name": cbm.stem})
        onnx_paths.append(onnx)
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(str(onnx), str(int8), weight_type=QuantType.QInt8)
            int8_paths.append(int8)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] INT8 квантизация {cbm.name} упала: {e}")
    return onnx_paths, int8_paths


def _sizes_kb(paths: list[Path]) -> list[float]:
    return [p.stat().st_size / 1024 for p in paths]


def bench(art: Path, n_calls: int = 200) -> dict:
    import onnxruntime as ort
    data = pickle.loads((art / "cache_features.pkl").read_bytes())
    X_full = data["test"][0][NOROUTE_FEATURES].head(n_calls).copy()
    X_num = X_full.to_numpy(dtype=np.float32)

    cbm_paths = [art / f"catboost_noroute_seed{i}.cbm" for i in range(NOROUTE_SEEDS)]
    fp32_paths = [p.with_suffix(".onnx") for p in cbm_paths]
    int8_paths = [p.parent / (p.stem + "_int8.onnx") for p in cbm_paths]

    cb_models = [CatBoostRegressor() for _ in cbm_paths]
    for m, p in zip(cb_models, cbm_paths):
        m.load_model(str(p))
    fp32_sessions = [ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]) for p in fp32_paths]
    int8_sessions = ([ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]) for p in int8_paths]
                     if all(p.exists() for p in int8_paths) else None)

    def bench_cb() -> float:
        # предикт по одному ряду, чтобы честно померить latency, как в проде
        t0 = time.perf_counter()
        for row in range(len(X_full)):
            r = X_full.iloc[row:row + 1]
            _ = float(np.mean([m.predict(r)[0] for m in cb_models]))
        return (time.perf_counter() - t0) / len(X_full) * 1000

    def bench_onnx(sessions) -> float:
        inp_name = sessions[0].get_inputs()[0].name
        t0 = time.perf_counter()
        for row in range(len(X_full)):
            feed = {inp_name: X_num[row:row + 1]}
            _ = float(np.mean([s.run(None, feed)[0] for s in sessions]))
        return (time.perf_counter() - t0) / len(X_full) * 1000

    print(f"[bench] catboost native ({len(X_full)} запросов)...")
    cb_ms = bench_cb()
    print(f"[bench] onnx fp32...")
    onnx_ms = bench_onnx(fp32_sessions)
    int8_ms = None
    if int8_sessions:
        print(f"[bench] onnx int8...")
        int8_ms = bench_onnx(int8_sessions)

    return {
        "n_calls": len(X_full),
        "n_models": NOROUTE_SEEDS,
        "features": NOROUTE_FEATURES,
        "sizes_kb": {"catboost": _sizes_kb(cbm_paths), "onnx_fp32": _sizes_kb(fp32_paths),
                     "onnx_int8": _sizes_kb(int8_paths) if int8_sessions else None},
        "latency_ms_per_sample": {"catboost": cb_ms, "onnx_fp32": onnx_ms, "onnx_int8": int8_ms},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    ap.add_argument("--bench-calls", type=int, default=200)
    args = ap.parse_args()

    print("[1/3] noroute-модели")
    cbm_paths = ensure_noroute_models(args.artifacts)

    print("[2/3] ONNX + INT8 экспорт")
    fp32, int8 = export_onnx(cbm_paths)
    print(f"[export] fp32: {len(fp32)}, int8: {len(int8)}")

    print("[3/3] latency benchmark")
    result = bench(args.artifacts, args.bench_calls)
    sizes = result["sizes_kb"]
    lat = result["latency_ms_per_sample"]
    print(f"\nsize (KB / файл, сумма по {result['n_models']} моделям):")
    for k, v in sizes.items():
        if v:
            print(f"  {k:12s}: avg {sum(v)/len(v):6.1f}, total {sum(v):7.1f}")
    print(f"\nlatency per-sample (мс):")
    print(f"  catboost   : {lat['catboost']:.2f}")
    print(f"  onnx_fp32  : {lat['onnx_fp32']:.2f}  (× {lat['catboost']/lat['onnx_fp32']:.2f} от catboost)")
    if lat["onnx_int8"]:
        print(f"  onnx_int8  : {lat['onnx_int8']:.2f}  (× {lat['catboost']/lat['onnx_int8']:.2f} от catboost)")

    (args.artifacts / "bench_latency.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[save] {args.artifacts / 'bench_latency.json'}")


if __name__ == "__main__":
    main()
