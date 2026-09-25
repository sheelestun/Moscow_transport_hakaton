"""Обучение ансамбля GRU (N моделей с разными seed'ами).

Все N моделей делят:
  - тот же train/test-сплит
  - те же static_mean/std (нормализация от первой модели)
  - ту же history из train

Разные seed → разная инициализация весов и порядок батчей → разные ошибки на test,
среднее по ним обычно даёт +1–3 сек MAE.

Формат чекпоинта совместим с обычным seq_model.pt, только:
  - "state_dict" отсутствует
  - "state_dicts" — список из N словарей

predict_submission.py умеет читать оба формата.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features.from_csv import (  # noqa: E402
    SEQ_FEATURES,
    SEQ_LEN,
    STATIC_FEATURES,
    build_features,
    build_target_residual,
    compute_history_stats,
    load_split,
)
from models.torch_seq import (  # noqa: E402
    DelaySeqDataset,
    SeqDelayModel,
    TrainConfig,
    predict_residual,
    train_model,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Тренировка ансамбля sequence-моделей")
    ap.add_argument("--dataset", type=Path, default=Path("./dataset"))
    ap.add_argument("--out", type=Path, default=Path("./ml/artifacts/seq_ensemble.pt"))
    ap.add_argument("--n-seeds", type=int, default=5, help="Сколько моделей в ансамбле")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    print("[load] train + test")
    tr_points, tr_traffic, tr_schedule = load_split(args.dataset, "train")
    te_points, te_traffic, te_schedule = load_split(args.dataset, "test")

    history = compute_history_stats(tr_points)
    print(f"[hist] tr_ids={len(history['tr_delay_mean'])}, "
          f"global_delay_mean={history['global_delay_mean']:.1f}")

    print("[features] train + test")
    tr_static, tr_seq = build_features(tr_points, tr_traffic, tr_schedule, history=history)
    tr_y = build_target_residual(tr_points)
    te_static, te_seq = build_features(te_points, te_traffic, te_schedule, history=history)
    te_y = build_target_residual(te_points)

    tr_static_np = tr_static.to_numpy(dtype=np.float32)
    te_static_np = te_static.to_numpy(dtype=np.float32)
    cur_dev = te_static["cur_dev_s"].to_numpy(dtype=np.float32)
    y_true = te_points["target_delay_s"].to_numpy(dtype=np.float32)
    mae_baseline = float(np.mean(np.abs(cur_dev - y_true)))
    print(f"[baseline] test MAE (cur_dev_s) = {mae_baseline:.2f} c")

    train_ds = DelaySeqDataset(tr_seq, tr_static_np, tr_y)
    test_ds = DelaySeqDataset(
        te_seq, te_static_np, te_y,
        static_mean=train_ds.static_mean, static_std=train_ds.static_std,
    )

    seq_dim = tr_seq.shape[2]
    static_dim = tr_static_np.shape[1]

    state_dicts = []
    per_model_mae = []
    ensemble_sum = np.zeros(len(te_y), dtype=np.float64)

    for seed in range(args.n_seeds):
        print(f"\n===== [seed {seed}] =====")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = SeqDelayModel(
            seq_feat_dim=seq_dim, static_feat_dim=static_dim,
            hidden=args.hidden, num_layers=args.num_layers, dropout=args.dropout,
        )
        cfg = TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, patience=args.patience,
        )
        model, best_val = train_model(model, train_ds, test_ds, cfg)
        residuals = predict_residual(model, test_ds)
        pred_abs = residuals + cur_dev
        mae_seed = float(np.mean(np.abs(pred_abs - y_true)))
        print(f"[seed {seed}] test MAE abs = {mae_seed:.2f}  (residual best = {best_val:.2f})")

        state_dicts.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        per_model_mae.append(mae_seed)
        ensemble_sum += residuals

    ensemble_residual = ensemble_sum / args.n_seeds
    ensemble_pred = ensemble_residual + cur_dev
    mae_ensemble = float(np.mean(np.abs(ensemble_pred - y_true)))
    mae_zero = float(np.mean(np.abs(y_true)))

    print("\n===== [ensemble summary] =====")
    print(f"per-seed MAE: {[round(m, 2) for m in per_model_mae]}")
    print(f"mean seed MAE = {np.mean(per_model_mae):.2f}")
    print(f"ensemble MAE  = {mae_ensemble:.2f}")
    print(f"baseline MAE  = {mae_baseline:.2f}")
    print(f"mae_zero      = {mae_zero:.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dicts": state_dicts,
            "n_seeds": args.n_seeds,
            "static_mean": np.asarray(train_ds.static_mean, dtype=np.float32),
            "static_std": np.asarray(train_ds.static_std, dtype=np.float32),
            "static_features": STATIC_FEATURES,
            "seq_features": SEQ_FEATURES,
            "seq_len": SEQ_LEN,
            "seq_feat_dim": seq_dim,
            "static_feat_dim": static_dim,
            "hidden": args.hidden,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "history": history,
            "test_mae_ensemble": mae_ensemble,
            "test_mae_per_seed": per_model_mae,
            "test_mae_baseline": mae_baseline,
        },
        args.out,
    )
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
