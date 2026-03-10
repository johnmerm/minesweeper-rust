"""PatchCNN: per-cell mine probability predictor.

Input:  (B, 8, 9, 9)  — 8-channel 9×9 patch centred on the target hidden cell
Output: (B,)           — P(mine) in [0, 1]

Architecture (~180k parameters):
  4× Conv2d(→32/64, 3×3, pad=1) + BN + ReLU
  AdaptiveAvgPool2d(3×3)  → flatten 288-d
  Append mines_remaining_ratio (channel 6 centre value)
  MLP: 289→128→64→1, Sigmoid output
"""

import torch
import torch.nn as nn


class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(8, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)),  # → (B, 32, 3, 3)
        )
        # 32*3*3 = 288 from conv, +1 for mines_ratio
        self.fc = nn.Sequential(
            nn.Linear(288 + 1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 8, 9, 9)
        # Extract global mines_ratio from channel 6 at centre (4,4)
        ratio = x[:, 6, 4, 4].unsqueeze(1)  # (B, 1)
        feat = self.conv(x)                  # (B, 32, 3, 3)
        feat = feat.flatten(1)               # (B, 288)
        feat = torch.cat([feat, ratio], dim=1)  # (B, 289)
        out = self.fc(feat).squeeze(1)       # (B,)
        return torch.sigmoid(out)


if __name__ == "__main__":
    model = PatchCNN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PatchCNN parameters: {n_params:,}")
    dummy = torch.zeros(4, 8, 9, 9)
    out = model(dummy)
    print(f"Output shape: {out.shape}, range: [{out.min():.4f}, {out.max():.4f}]")
