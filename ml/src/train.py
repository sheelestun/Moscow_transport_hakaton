"""Обучение sequence-модели задержек.

Пайплайн:
  train  → build_features → DelaySeqDataset (учим residual = target − cur_dev)
  test   → build_features → DelaySeqDataset (валидация + early stopping)
  сохраняем: state_dict, static_mean/std, конфиг фичей и модели

На выходе:
  - artefact .pt в --out (по умолчанию ml/artifacts/seq_model.pt)
  - лог с train/val MAE по эпохам
  - финальный test MAE в абсолютных секундах (для сравнения с baseline cur_dev_s)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Скрипт запускается как `python ml/src/train.py`; добавим ml/src в sys.path,
# чтобы работали импорты features.* и models.*
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features.from_csv import (  # noqa: E402
    SEQ_FEATURES,
    SEQ_LEN,
    STATIC_FEATURES,
    build_features,
    build_target_residual,
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
    ap = argparse.ArgumentParser(description="Тренировка sequence-модели задержек")
    ap.add_argument("--dataset", type=Path, default=Path("./dataset"), help="Корень датасета")
    ap.add_argument("--out", type=Path, default=Path("./ml/artifacts/seq_model.pt"), help="Куда сохранить чекпоинт")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("[load] train")
    tr_points, tr_traffic, tr_schedule = load_split(args.dataset, "train")
    print(f"[load] train: points={len(tr_points)}, traffic={len(tr_traffic)}, schedule={len(tr_schedule)}")

    print("[load] test")
    te_points, te_traffic, te_schedule = load_split(args.dataset, "test")
    print(f"[load] test:  points={len(te_points)}, traffic={len(te_traffic)}, schedule={len(te_schedule)}")

    print("[features] train")
    tr_static, tr_seq = build_features(tr_points, tr_traffic, tr_schedule)
    tr_y = build_target_residual(tr_points)

    print("[features] test")
    te_static, te_seq = build_features(te_points, te_traffic, te_schedule)
    te_y = build_target_residual(te_points)

    if tr_y is None or te_y is None:
        raise RuntimeError("В train/test нет target_delay_s — не на чем учиться")

    tr_static_np = tr_static.to_numpy(dtype=np.float32)
    te_static_np = te_static.to_numpy(dtype=np.float32)

    train_ds = DelaySeqDataset(tr_seq, tr_static_np, tr_y)
    test_ds = DelaySeqDataset(
        te_seq,
        te_static_np,
        te_y,
        static_mean=train_ds.static_mean,
        static_std=train_ds.static_std,
    )

    seq_dim = tr_seq.shape[2]
    static_dim = tr_static_np.shape[1]
    model = SeqDelayModel(
        seq_feat_dim=seq_dim,
        static_feat_dim=static_dim,
        hidden=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] SeqDelayModel: seq_dim={seq_dim}, static_dim={static_dim}, params={n_params}")

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
    print(f"[train] device={cfg.device}, epochs={cfg.epochs}, batch={cfg.batch_size}, lr={cfg.lr}")
    model, best_val_residual = train_model(model, train_ds, test_ds, cfg)
    print(f"[done] best test MAE (residual) = {best_val_residual:.2f} c")

    residuals = predict_residual(model, test_ds)
    cur_dev = te_static["cur_dev_s"].to_numpy(dtype=np.float32)
    y_true = te_points["target_delay_s"].to_numpy(dtype=np.float32)
    pred_abs = residuals + cur_dev
    mae_model = float(np.mean(np.abs(pred_abs - y_true)))
    mae_baseline = float(np.mean(np.abs(cur_dev - y_true)))
    mae_zero = float(np.mean(np.abs(y_true)))
    print(f"[abs MAE] model={mae_model:.2f}  baseline(cur_dev)={mae_baseline:.2f}  mae_zero={mae_zero:.2f}")
    if mae_model < mae_baseline:
        gain = mae_baseline - mae_model
        print(f"[gain]    model лучше baseline на {gain:.2f} c")
    else:
        print("[warn]    модель не бьёт baseline — проверь фичи/эпохи/lr")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
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
            "test_mae_abs": mae_model,
            "test_mae_baseline": mae_baseline,
        },
        args.out,
    )
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
