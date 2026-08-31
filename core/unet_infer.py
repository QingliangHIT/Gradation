# -*- coding: utf-8 -*-
"""UNet 语义分割推理适配器（权重加载与单例缓存）。"""
import cv2
import numpy as np
import torch

from core import config

_MODELS = None  # 延迟构建，避免 import 时就加载 torch 模型类
_IMG_SIZE = 256
_DEFAULT_MODEL_NAME = "resunet"

_cache = {"sig": None, "model": None}


def _signature():
    """缓存签名：设备偏好或权重路径变化时自动重建模型。"""
    c = config.get_config()
    return (c["device"], c["unet_model_path"])


def _load_model():
    global _MODELS
    if _MODELS is None:
        from models.unet import UNet, ResUNet
        _MODELS = {"unet": UNet, "resunet": ResUNet}

    sig = _signature()
    if _cache["model"] is not None and _cache["sig"] == sig:
        return _cache["model"]

    device = config.resolve_device()
    c = config.get_config()
    name = _DEFAULT_MODEL_NAME
    model = _MODELS[name](in_channels=3, num_classes=2).to(device)
    path = c["unet_model_path"]
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    _cache["sig"] = sig
    _cache["model"] = model
    return model


def unet_predict(img, model_path=None, model_name=None):
    """
    输入: BGR 彩色图 (H, W, 3) uint8 或 灰度图 (H, W) uint8
    输出: 二值图 (H, W) uint8, 颗粒=255
    """
    model = _load_model()

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    resized = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE))
    tensor = torch.from_numpy(
        resized.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        out = model(tensor)
        pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()

    pred = (pred * 255).astype(np.uint8)
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    return pred
