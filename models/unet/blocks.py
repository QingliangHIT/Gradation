# -*- coding: utf-8 -*-
"""网络基础卷积块：UNet 系列共用的双卷积 / 残差卷积 / nnUNet 卷积块。"""
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ResConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class NNUNetConvBlock(nn.Module):
    """nnUNet v2 基础卷积块: Conv -> InstanceNorm -> LeakyReLU"""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        padding = [(k - 1) // 2 for k in ([kernel_size] * 2)]
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, eps=1e-5, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class NNUNetStackedBlocks(nn.Module):
    """nnUNet v2 堆叠卷积块（每级 num_blocks 个 ConvBlock）"""

    def __init__(self, num_blocks, in_ch, out_ch, stride=1):
        super().__init__()
        blocks = [NNUNetConvBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, num_blocks):
            blocks.append(NNUNetConvBlock(out_ch, out_ch, stride=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)
