"""Sequence-модель на PyTorch: GRU + статик-голова.

Учим предсказывать residual = target_delay_s − cur_dev_s (см. features/from_csv.py).
На инференсе абсолютный прогноз получаем как `model_out + cur_dev_s` снаружи.

Размерности:
- seq: (batch, SEQ_LEN, SEQ_FEAT_DIM) — GPS-пинги от features/from_csv.py
- static: (batch, STATIC_FEAT_DIM) — статик-вектор от features/from_csv.py

Loss = L1 (MAE в лоссе напрямую, чтобы модель училась под целевую метрику).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Значения по умолчанию согласованы с features/from_csv.py.
# При реальном использовании передавайте фактические размеры через SeqDelayModel(...) явно.
SEQ_LEN = 30
SEQ_FEAT_DIM = 5
STATIC_FEAT_DIM = 20


class DelaySeqDataset(Dataset):
    """Обёртка над np.ndarray. Все три массива должны быть выровнены по axis=0."""

    def __init__(
        self,
        seq: np.ndarray,
        static: np.ndarray,
        target: Optional[np.ndarray] = None,
        static_mean: Optional[np.ndarray] = None,
        static_std: Optional[np.ndarray] = None,
    ):
        assert seq.ndim == 3 and static.ndim == 2 and len(seq) == len(static)
        if target is not None:
            assert len(target) == len(seq)

        self.static_mean = static.mean(axis=0) if static_mean is None else static_mean
        self.static_std = np.where(static.std(axis=0) < 1e-6, 1.0, static.std(axis=0)) if static_std is None else static_std
        static_norm = (static - self.static_mean) / self.static_std

        self.seq = torch.from_numpy(seq).float()
        self.static = torch.from_numpy(static_norm).float()
        self.target = torch.from_numpy(target).float() if target is not None else None

    def __len__(self) -> int:
        return len(self.seq)

    def __getitem__(self, i: int):
        if self.target is None:
            return self.seq[i], self.static[i]
        return self.seq[i], self.static[i], self.target[i]


class SeqDelayModel(nn.Module):
    """GRU по seq + concat со static → MLP → скаляр (residual в секундах)."""

    def __init__(
        self,
        seq_feat_dim: int = SEQ_FEAT_DIM,
        static_feat_dim: int = STATIC_FEAT_DIM,
        hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=seq_feat_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + static_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(seq)          # h: (num_layers, B, hidden)
        h_last = h[-1]                # (B, hidden)
        x = torch.cat([h_last, static], dim=1)
        return self.head(x).squeeze(-1)   # (B,)


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(
    model: SeqDelayModel,
    train_ds: DelaySeqDataset,
    val_ds: DelaySeqDataset,
    cfg: TrainConfig = field(default_factory=TrainConfig),
) -> tuple[SeqDelayModel, float]:
    """Обучает модель, возвращает (модель с лучшими весами, лучший val MAE)."""
    cfg = cfg if isinstance(cfg, TrainConfig) else TrainConfig()
    model.to(cfg.device)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.L1Loss()

    best_val = float("inf")
    best_state = None
    bad = 0
    for epoch in range(cfg.epochs):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for seq, static, y in train_loader:
            seq, static, y = seq.to(cfg.device), static.to(cfg.device), y.to(cfg.device)
            pred = model(seq, static)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * len(y)
            train_n += len(y)
        train_mae = train_loss_sum / max(train_n, 1)

        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for seq, static, y in val_loader:
                seq, static, y = seq.to(cfg.device), static.to(cfg.device), y.to(cfg.device)
                pred = model(seq, static)
                val_sum += torch.nn.functional.l1_loss(pred, y, reduction="sum").item()
                val_n += len(y)
        val_mae = val_sum / max(val_n, 1)

        print(f"epoch {epoch:02d}  train MAE {train_mae:6.2f}   val MAE {val_mae:6.2f}")

        if val_mae < best_val - 1e-3:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"early stop at epoch {epoch} (best val MAE {best_val:.2f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def predict_residual(model: SeqDelayModel, dataset: DelaySeqDataset, batch_size: int = 256, device: Optional[str] = None) -> np.ndarray:
    """Возвращает residual (secondsdelta). Абсолютный ответ = residual + cur_dev_s (прибавьте снаружи)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outs = []
    with torch.no_grad():
        for batch in loader:
            seq, static = batch[0].to(device), batch[1].to(device)
            outs.append(model(seq, static).cpu().numpy())
    return np.concatenate(outs)


if __name__ == "__main__":
    torch.manual_seed(42)
    b, l, sf, af = 4, SEQ_LEN, SEQ_FEAT_DIM, STATIC_FEAT_DIM
    seq = torch.randn(b, l, sf)
    static = torch.randn(b, af)
    model = SeqDelayModel(sf, af)
    out = model(seq, static)
    assert out.shape == (b,), f"expected ({b},), got {out.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"forward pass ok, output {out.shape}, params={n_params}")
