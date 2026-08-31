# -*- coding: utf-8 -*-
"""UNet 系列模型预测与结果叠加（供 viewer 交互查看器与命令行使用）。"""
import cv2
import numpy as np
import torch

from common.image_io import imread
from models.unet import NNUNet, NNUNetv2, ResUNet, UNet
from models.unet.metrics import compute_metrics_numpy

# 兼容旧命名：中文路径安全读取
imread_unicode = imread


def load_model(model_path, model_name, device):
    models = {"unet": UNet, "resunet": ResUNet, "nnunet": NNUNet, "nnunetv2": NNUNetv2}
    model = models[model_name](in_channels=3, num_classes=2).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model: {model_name} from {model_path}")
    return model


def preprocess_image(img_path, img_size):
    img = imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return img_rgb, img_tensor


def predict(model, img_tensor, device):
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.to(device))
        return torch.argmax(output, dim=1).squeeze(0).cpu().numpy()


def make_overlay(img_rgb, pred):
    overlay = img_rgb.astype(np.float32) / 255.0
    overlay[:, :, 0] = np.where(pred == 1, 1.0, overlay[:, :, 0])
    overlay[:, :, 1] = np.where(pred == 1, 0.0, overlay[:, :, 1])
    overlay[:, :, 2] = np.where(pred == 1, 0.0, overlay[:, :, 2])
    return np.clip(overlay * 0.6 + img_rgb.astype(np.float32) / 255.0 * 0.4, 0, 1)
