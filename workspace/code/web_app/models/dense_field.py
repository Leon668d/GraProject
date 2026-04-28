"""
Dense-field nonrigid registration network.

This is an experimental U-Net style model that predicts a dense 2D displacement
field in pixel units for SAR/optical registration.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class DenseFieldRegistrationCNN(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        in_channels = 4
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

        self.dec4 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.dec3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec1 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.flow_head = nn.Conv2d(base_channels, 2, kernel_size=3, padding=1)

    @staticmethod
    def build_features(sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(sar - optical)
        product = sar * optical
        return torch.cat([sar, optical, diff, product], dim=1)

    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        x = self.build_features(sar, optical)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.flow_head(d1)


def flow_to_grid(flow: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = flow.shape
    device = flow.device
    dtype = flow.dtype

    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype),
        indexing="ij",
    )
    base_grid = torch.stack((xs, ys), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)

    flow_x = flow[:, 0] / max(width - 1, 1) * 2.0
    flow_y = flow[:, 1] / max(height - 1, 1) * 2.0
    normalized_flow = torch.stack((flow_x, flow_y), dim=-1)
    return base_grid + normalized_flow


def warp_tensor(image: torch.Tensor, flow: torch.Tensor, padding_mode: str = "border") -> torch.Tensor:
    grid = flow_to_grid(flow)
    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )


def smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1]).mean()
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
    return dx + dy


def local_ncc_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x_centered = x - x.mean(dim=(-2, -1), keepdim=True)
    y_centered = y - y.mean(dim=(-2, -1), keepdim=True)
    numerator = (x_centered * y_centered).mean(dim=(-2, -1))
    denominator = torch.sqrt(
        (x_centered.square().mean(dim=(-2, -1)) + eps)
        * (y_centered.square().mean(dim=(-2, -1)) + eps)
    )
    ncc = numerator / denominator
    return 1.0 - ncc.mean()
