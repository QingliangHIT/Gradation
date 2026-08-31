# -*- coding: utf-8 -*-
"""nnUNet 与 nnUNetv2（深度监督、步长卷积下采样）网络定义。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet.blocks import NNUNetStackedBlocks


class NNUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        features = [32, 64, 128, 256, 320, 320]
        num_blocks = [2, 2, 2, 2, 2, 2]

        self.enc1 = NNUNetStackedBlocks(num_blocks[0], in_channels, features[0])
        self.down1 = nn.Conv2d(features[0], features[0], 2, stride=2)

        self.enc2 = NNUNetStackedBlocks(num_blocks[1], features[0], features[1])
        self.down2 = nn.Conv2d(features[1], features[1], 2, stride=2)

        self.enc3 = NNUNetStackedBlocks(num_blocks[2], features[1], features[2])
        self.down3 = nn.Conv2d(features[2], features[2], 2, stride=2)

        self.enc4 = NNUNetStackedBlocks(num_blocks[3], features[2], features[3])
        self.down4 = nn.Conv2d(features[3], features[3], 2, stride=2)

        self.enc5 = NNUNetStackedBlocks(num_blocks[4], features[3], features[4])
        self.down5 = nn.Conv2d(features[4], features[4], 2, stride=2)

        self.bottleneck = NNUNetStackedBlocks(num_blocks[5], features[4], features[5])

        self.up5 = nn.ConvTranspose2d(features[5], features[4], 2, stride=2)
        self.dec5 = NNUNetStackedBlocks(num_blocks[4], features[4] * 2, features[4])

        self.up4 = nn.ConvTranspose2d(features[4], features[3], 2, stride=2)
        self.dec4 = NNUNetStackedBlocks(num_blocks[3], features[3] * 2, features[3])

        self.up3 = nn.ConvTranspose2d(features[3], features[2], 2, stride=2)
        self.dec3 = NNUNetStackedBlocks(num_blocks[2], features[2] * 2, features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)
        self.dec2 = NNUNetStackedBlocks(num_blocks[1], features[1] * 2, features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)
        self.dec1 = NNUNetStackedBlocks(num_blocks[0], features[0] * 2, features[0])

        self.out_conv = nn.Conv2d(features[0], num_classes, 1)

        if self.deep_supervision:
            self.ds5 = nn.Conv2d(features[4], num_classes, 1)
            self.ds4 = nn.Conv2d(features[3], num_classes, 1)
            self.ds3 = nn.Conv2d(features[2], num_classes, 1)
            self.ds2 = nn.Conv2d(features[1], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        e5 = self.enc5(self.down4(e4))

        b = self.bottleneck(self.down5(e5))

        d5 = self.dec5(torch.cat([self.up5(b), e5], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.out_conv(d1)

        if self.deep_supervision and self.training:
            ds5 = self.ds5(d5)
            ds4 = self.ds4(d4)
            ds3 = self.ds3(d3)
            ds2 = self.ds2(d2)
            ds5 = nn.functional.interpolate(ds5, size=out.shape[2:], mode='bilinear', align_corners=False)
            ds4 = nn.functional.interpolate(ds4, size=out.shape[2:], mode='bilinear', align_corners=False)
            ds3 = nn.functional.interpolate(ds3, size=out.shape[2:], mode='bilinear', align_corners=False)
            ds2 = nn.functional.interpolate(ds2, size=out.shape[2:], mode='bilinear', align_corners=False)
            return [out, ds2, ds3, ds4, ds5]

        return out


class NNUNetv2(nn.Module):
    """
    nnUNet v2 风格 2D 分割网络
    - InstanceNorm + LeakyReLU
    - 步长卷积下采样（非池化）
    - 转置卷积上采样
    - 训练时深度监督（多尺度输出），推理时单输出
    - 可配置级数、通道数、每级卷积块数
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=2,
        num_stages=5,
        features=(32, 64, 128, 256, 320, 320),
        blocks_per_stage=(2, 2, 2, 2, 2, 2),
        deep_supervision=True,
    ):
        super().__init__()
        assert num_stages >= 2, "num_stages must >= 2"
        assert len(features) == num_stages + 1
        assert len(blocks_per_stage) == num_stages + 1

        self.num_stages = num_stages
        self.deep_supervision = deep_supervision
        self.features = features

        # ── 编码器 ──
        self.encoder = nn.ModuleList()
        # 第一级无下采样 stride=1
        self.encoder.append(
            NNUNetStackedBlocks(blocks_per_stage[0], in_channels, features[0], stride=1)
        )
        # 后续级使用 stride=2 下采样
        for s in range(1, num_stages):
            self.encoder.append(
                NNUNetStackedBlocks(blocks_per_stage[s], features[s - 1], features[s], stride=2)
            )

        # ── Bottleneck（最深层）──
        self.bottleneck = NNUNetStackedBlocks(
            blocks_per_stage[num_stages], features[num_stages - 1], features[num_stages], stride=2
        )

        # ── 解码器 ──
        self.decoder = nn.ModuleList()
        self.seg_outputs = nn.ModuleList()
        for s in range(num_stages - 1, -1, -1):
            # 转置卷积上采样
            up = nn.ConvTranspose2d(
                features[s + 1], features[s], kernel_size=2, stride=2, bias=False
            )
            # skip connection concat 后卷积
            dec = NNUNetStackedBlocks(blocks_per_stage[s], features[s] * 2, features[s], stride=1)
            self.decoder.append(nn.ModuleDict({"up": up, "dec": dec}))
            # 每级分割输出头（用于深度监督）
            self.seg_outputs.append(nn.Conv2d(features[s], num_classes, 1, bias=False))

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 编码
        skips = []
        feat = x
        for enc in self.encoder:
            feat = enc(feat)
            skips.append(feat)

        # Bottleneck
        feat = self.bottleneck(feat)

        # 解码 + 深度监督
        ds_outputs = []
        for i, dec_block in enumerate(self.decoder):
            feat = dec_block["up"](feat)
            # 处理尺寸不匹配
            skip = skips[self.num_stages - 1 - i]
            if feat.shape[2:] != skip.shape[2:]:
                feat = F.interpolate(feat, size=skip.shape[2:], mode='bilinear', align_corners=False)
            feat = torch.cat([feat, skip], dim=1)
            feat = dec_block["dec"](feat)
            ds_outputs.append(self.seg_outputs[i](feat))

        # ds_outputs[0] 是最高分辨率（最终输出）
        # ds_outputs[-1] 是最低分辨率
        if self.deep_supervision and self.training:
            # 将所有辅助输出上采样到最终输出尺寸
            target_size = ds_outputs[0].shape[2:]
            result = [ds_outputs[0]]
            for ds in ds_outputs[1:]:
                result.append(
                    F.interpolate(ds, size=target_size, mode='bilinear', align_corners=False)
                )
            return result  # [full_res, ds1, ds2, ...]

        return ds_outputs[0]
