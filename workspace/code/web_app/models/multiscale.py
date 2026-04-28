"""
Multi-scale registration models.

The original architecture predicts a global translation ``(dx, dy)`` from a
coarse-to-fine feature stack. This module now also exposes a homography
variant that reuses the same backbone while regressing 8 free homography
parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise + pointwise block with state-dict names aligned to the checkpoint."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.pw_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = F.relu(self.pw_bn(self.pw_conv(x)), inplace=True)
        return x


class MultiScaleBranch(nn.Module):
    """
    One branch of the coarse-to-fine pyramid.

    Each branch sees a different input scale and keeps module indices aligned
    with the checkpoint: block / pooling / block / block.
    """

    def __init__(self, branch_channels: tuple[int, int, int], input_scale: int) -> None:
        super().__init__()
        c1, c2, c3 = branch_channels
        self.input_scale = input_scale
        self.branch = nn.Sequential(
            DepthwiseSeparableBlock(1, c1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DepthwiseSeparableBlock(c1, c2),
            DepthwiseSeparableBlock(c2, c3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_scale > 1:
            height, width = x.shape[-2:]
            x = F.interpolate(
                x,
                size=(max(1, height // self.input_scale), max(1, width // self.input_scale)),
                mode="bilinear",
                align_corners=False,
            )
        return self.branch(x)


class MultiScaleFusion(nn.Module):
    """Project multi-scale features to a common width, then fuse them."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1x1 = nn.ModuleList(
            [
                nn.Conv2d(64, 256, kernel_size=1, bias=True),
                nn.Conv2d(32, 256, kernel_size=1, bias=True),
                nn.Conv2d(16, 256, kernel_size=1, bias=True),
            ]
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256 * 3, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        projected = [conv(feature) for conv, feature in zip(self.conv1x1, features)]
        reference_size = projected[0].shape[-2:]
        aligned = [
            feature
            if feature.shape[-2:] == reference_size
            else F.interpolate(feature, size=reference_size, mode="bilinear", align_corners=False)
            for feature in projected
        ]
        return self.fusion_conv(torch.cat(aligned, dim=1))


class MultiScaleRegistrationCNN(nn.Module):
    """
    Coarse-to-fine global registration model.

    The checkpoint is single-channel at the branch input, so the two modalities
    are first compressed into a robust mismatch response before entering the
    feature pyramid.
    """

    def __init__(self, output_dim: int = 2) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.branch1 = MultiScaleBranch((16, 32, 64), input_scale=1)
        self.branch2 = MultiScaleBranch((8, 16, 32), input_scale=2)
        self.branch3 = MultiScaleBranch((4, 8, 16), input_scale=4)
        self.fusion = MultiScaleFusion()
        self.match_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, output_dim),
        )

    @staticmethod
    def build_mismatch_map(sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(sar - optical)
        product = sar * optical
        return 0.7 * diff + 0.3 * (1.0 - product)

    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        x = self.build_mismatch_map(sar, optical)
        fused = self.fusion(
            [
                self.branch1(x),
                self.branch2(x),
                self.branch3(x),
            ]
        )
        return self.match_head(fused)


class MultiScaleHomographyCNN(MultiScaleRegistrationCNN):
    """Homography variant of the coarse-to-fine registration model."""

    def __init__(self) -> None:
        super().__init__(output_dim=8)
