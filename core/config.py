# -*- coding: utf-8 -*-
"""推理运行配置：计算设备与各模型权重路径（全局单例）。

各模型适配器（unet_infer / sam_infer / registry 的 YOLO 适配）读取本模块配置，
并通过签名比对在配置变更后自动重建/失效模型缓存。
"""
import torch

# 默认权重路径（可在“设置 → 系统与推理”中修改）
_UNET_MODEL_PATH = r"A:\05-Codes\Gradation\DATA\resunet_best.pth"
_SAM_CHECKPOINT = r"A:\05-Codes\Gradation\DATA\sam_vit_h_4b8939.pth"
_YOLO_WEIGHTS = r"A:\05-Codes\Gradation\DATA\yolov8n-seg.pt"

# 计算设备偏好: "auto" / "cpu" / "cuda"
_DEVICE_PREF = "auto"


def _resolve_device(pref):
    """根据偏好解析实际计算设备。"""
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_device():
    """按当前设备偏好解析实际计算设备（供模型适配器使用）。"""
    return _resolve_device(_DEVICE_PREF)


def configure(device=None, unet_model_path=None, sam_checkpoint=None,
              yolo_weights=None):
    """运行时配置计算设备与模型权重路径。

    device: 'auto' / 'cpu' / 'cuda'
    返回配置后的完整配置字典。
    """
    global _DEVICE_PREF, _UNET_MODEL_PATH, _SAM_CHECKPOINT, _YOLO_WEIGHTS
    if device in ("auto", "cpu", "cuda"):
        _DEVICE_PREF = device
    if unet_model_path and unet_model_path != _UNET_MODEL_PATH:
        _UNET_MODEL_PATH = unet_model_path
    if sam_checkpoint and sam_checkpoint != _SAM_CHECKPOINT:
        _SAM_CHECKPOINT = sam_checkpoint
    if yolo_weights is not None:
        _YOLO_WEIGHTS = yolo_weights
    return get_config()


def get_config():
    """返回当前推理配置。"""
    return {
        "device": _DEVICE_PREF,
        "unet_model_path": _UNET_MODEL_PATH,
        "sam_checkpoint": _SAM_CHECKPOINT,
        "yolo_weights": _YOLO_WEIGHTS,
    }
