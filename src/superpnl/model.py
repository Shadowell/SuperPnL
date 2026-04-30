from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class TCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        kernel_size: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [TCNBlock(hidden_dim, kernel_size, dilation, dropout) for dilation in dilations]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        x = x.transpose(1, 2)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return x.transpose(1, 2)


class SuperPnLModel(nn.Module):
    def __init__(
        self,
        bar_dim: int,
        feature_dim: int,
        num_horizons: int,
        hidden_dim: int = 128,
        dropout: float = 0.05,
        use_features: bool = True,
    ) -> None:
        super().__init__()
        self.use_features = use_features and feature_dim > 0
        self.bar_encoder = TCNEncoder(bar_dim, hidden_dim, dropout=dropout)
        if self.use_features:
            self.feature_encoder = TCNEncoder(
                feature_dim,
                hidden_dim,
                dilations=(1, 2, 4, 8, 16, 32),
                dropout=dropout,
            )
            self.film = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 3),
            )
        else:
            self.feature_encoder = None
            self.film = None
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_horizons * 2),
        )
        self.num_horizons = num_horizons

    def forward(self, bar: torch.Tensor, features: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        bar_hidden = self.bar_encoder(bar)[:, -1, :]
        fused = bar_hidden
        if self.use_features:
            if features is None:
                raise ValueError("features required when use_features=True")
            feature_hidden = self.feature_encoder(features)[:, -1, :]
            gamma, beta, gate = self.film(feature_hidden).chunk(3, dim=-1)
            mod = bar_hidden * (1.0 + torch.tanh(gamma)) + beta
            fused = bar_hidden + torch.sigmoid(gate) * mod
        out = self.head(fused).view(-1, self.num_horizons, 2)
        pred_ret = out[:, :, 0]
        pos_logit = out[:, :, 1]
        return pred_ret, pos_logit
