# -*- coding: utf-8 -*-
"""颗粒分割训练数据集：images/ 与 masks/ 成对目录。"""
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ParticleDataset(Dataset):
    """
    数据集目录结构:
        data_dir/
            images/   (原始图或预处理后的灰度图)
            masks/    (对应的二值标注, 颗粒=255, 背景=0)
    """

    def __init__(self, data_dir, img_size=256):
        self.data_dir = data_dir
        self.img_size = img_size

        img_dir = os.path.join(data_dir, "images")
        self.img_files = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith((".png", ".jpg", ".bmp"))
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = img_path.replace("images", "masks")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.int64)

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask)

        return img, mask
