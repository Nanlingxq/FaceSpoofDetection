"""Depth auxiliary head and full-image supervision losses.

The head is used only while training.  It predicts a 40x40 pseudo-depth map
from the shared feature map used by the existing Fourier reconstruction head.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DepthDecoderBlock(nn.Module):
    """Upsample by two and refine with two lightweight convolutions."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.block(x)


class DepthGenerator(nn.Module):
    """Decode the 10x10 shared feature map into a normalized depth map at the configured supervision size."""

    def __init__(self, in_channels: int = 128, output_size=(80, 80)) -> None:
        super().__init__()
        self.output_size = tuple(output_size)
        # The shared feature map is 10x10. Decode only to the requested
        # supervision resolution instead of always constructing an 80x80 map.
        feature_size = 10
        target_size = max(self.output_size)
        num_upsamples = max(0, int(math.ceil(math.log2(max(target_size, 1) / feature_size))))
        channel_schedule = [in_channels, 96, 64, 32]
        blocks = []
        for index in range(num_upsamples):
            in_ch = channel_schedule[min(index, len(channel_schedule) - 1)]
            out_ch = channel_schedule[min(index + 1, len(channel_schedule) - 1)]
            blocks.append(DepthDecoderBlock(in_ch, out_ch))
        self.decoder = nn.Sequential(*blocks)
        decoder_channels = channel_schedule[min(num_upsamples, len(channel_schedule) - 1)]
        self.output = nn.Conv2d(decoder_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.output(self.decoder(x))
        if tuple(x.shape[-2:]) != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(x)


def smooth_l1_depth_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean Smooth-L1 error over every pixel in the depth map."""
    return F.smooth_l1_loss(prediction, target, reduction="mean")


def _gradient(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return tensor[..., :, 1:] - tensor[..., :, :-1], tensor[..., 1:, :] - tensor[..., :-1, :]


def gradient_depth_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Match local horizontal and vertical depth gradients over the full map."""
    pred_x, pred_y = _gradient(prediction)
    target_x, target_y = _gradient(target)
    loss_x = F.smooth_l1_loss(pred_x, target_x, reduction="mean")
    loss_y = F.smooth_l1_loss(pred_y, target_y, reduction="mean")
    return 0.5 * (loss_x + loss_y)


def depth_auxiliary_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gradient_weight: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, pixel and gradient losses using all depth pixels."""
    pixel_loss = smooth_l1_depth_loss(prediction, target)
    gradient_loss = gradient_depth_loss(prediction, target)
    return pixel_loss + gradient_weight * gradient_loss, pixel_loss, gradient_loss
