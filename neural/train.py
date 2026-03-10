"""Training script for PatchCNN.

Usage:
    python neural/train.py [--epochs 30] [--batch 2048] [--workers 4] [--resume]

Checkpoints saved to neural/checkpoints/best.pt (best val BCE).
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MinesweeperDataset
from model import PatchCNN

DATA_DIR = Path(__file__).parent / "data"
CKPT_DIR = Path(__file__).parent / "checkpoints"


def bce_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy(pred, target)
    mse = nn.functional.mse_loss(pred, target)
    return bce + 0.1 * mse


def evaluate(model, loader, device):
    model.eval()
    total_bce = 0.0
    total_mae = 0.0
    n = 0
    with torch.no_grad():
        for patches, labels in loader:
            patches, labels = patches.to(device), labels.to(device)
            preds = model(patches)
            total_bce += nn.functional.binary_cross_entropy(preds, labels).item() * len(labels)
            total_mae += (preds - labels).abs().mean().item() * len(labels)
            n += len(labels)
    return total_bce / n, total_mae / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = MinesweeperDataset(DATA_DIR / "train.jsonl", augment=True)
    val_ds = MinesweeperDataset(DATA_DIR / "val.jsonl", augment=False)
    print(f"Train samples: {len(train_ds):,}  Val samples: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )

    model = PatchCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_bce = float("inf")

    if args.resume:
        ckpt_path = CKPT_DIR / "best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_val_bce = ckpt["val_bce"]
            print(f"Resumed from epoch {start_epoch}, best val BCE = {best_val_bce:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(patches)
            loss = bce_mse_loss(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)
        val_bce, val_mae = evaluate(model, val_loader, device)

        saved = ""
        if val_bce < best_val_bce:
            best_val_bce = val_bce
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "val_bce": val_bce,
                },
                CKPT_DIR / "best.pt",
            )
            saved = "  ← saved"

        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_bce={val_bce:.4f}  val_mae={val_mae:.4f}  "
            f"lr={lr:.2e}{saved}"
        )

    print(f"\nBest val BCE: {best_val_bce:.4f}")


if __name__ == "__main__":
    main()
