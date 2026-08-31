# -*- coding: utf-8 -*-
"""
分割模型注册表 —— 模型模块化管理中心。

设计目标：
    1. 所有分割模型（传统分水岭 / UNet / SAM / YOLO / …）统一注册、统一调用；
    2. 新增模型只需实现 run(img, params, stage) -> (binary, markers)
       并调用 register(...) 注册一条 ModelSpec，界面下拉框自动出现该模型；
    3. 每个模型声明所属参数组（param_group），参数对话框据此显示对应参数面板。

约定：
    - 输入  img:    BGR 彩色图 (H, W, 3) uint8
    - 输出  binary: 二值图 (H, W) uint8（可为 None）
    - 输出  markers: 实例标签图 (H, W) int32，背景<=1，实例从 2 开始
"""
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from algorithms import segmentation
from algorithms.process import traditional_binary

# 参数组常量：对话框按该值显示对应参数面板
GROUP_WATERSHED = "watershed"
GROUP_SAM = "sam"
GROUP_YOLO = "yolo"


@dataclass
class ModelSpec:
    """分割模型描述。"""
    key: str                          # 唯一键，保存于参数 seg_type
    label: str                        # 界面显示名称
    param_group: str                  # 参数组：watershed / sam / yolo
    run: Callable                     # run(img, params, stage) -> (binary, markers)
    description: str = ""


_REGISTRY: "OrderedDict[str, ModelSpec]" = OrderedDict()


def register(spec: ModelSpec):
    """注册一个分割模型（同名键覆盖注册）。"""
    _REGISTRY[spec.key] = spec
    return spec


def list_models():
    """按注册顺序返回所有模型描述。"""
    return list(_REGISTRY.values())


def get_model(key: str) -> Optional[ModelSpec]:
    return _REGISTRY.get(key)


def run_model(key: str, img, params, stage=None):
    """按模型键执行分割。"""
    spec = _REGISTRY.get(key)
    if spec is None:
        raise ValueError(f"未知分割模型: {key}")
    stage = stage or (lambda s: None)
    return spec.run(img, params, stage)


# ============================
# 内置模型实现
# ============================
def _run_watershed(img, p, stage):
    """传统分水岭：传统方法专用二值化路径 + 分水岭实例分割。"""
    stage("图像二值化中...")
    binary = traditional_binary(img)
    stage("分水岭分割中...")
    markers = segmentation.segment_particles(
        binary,
        dist_thresh_ratio=p["dist_thresh_ratio"],
        kernel_size=p["kernel_size"],
        close_iterations=p["close_iterations"],
        open_iterations=p["open_iterations"],
    )
    return binary, markers


def _run_unet(img, p, stage):
    """UNet 语义分割 + 分水岭实例分割。"""
    stage("UNet 推理中...")
    binary = segmentation.unet_predict(img)
    stage("分水岭实例分割中...")
    markers = segmentation.segment_particles_unet(
        binary,
        dist_thresh_ratio=p["dist_thresh_ratio"],
        kernel_size=p["kernel_size"],
        close_iterations=p["close_iterations"],
        open_iterations=p["open_iterations"],
    )
    return binary, markers


def _run_sam(img, p, stage):
    """SAM 自动实例分割。"""
    stage("SAM 模型推理中...")
    binary, markers = segmentation.sam_segment(
        img,
        points_per_side=p["sam_points_per_side"],
        pred_iou_thresh=p["sam_pred_iou_thresh"],
        stability_score_thresh=p["sam_stability_score_thresh"],
        crop_n_layers=p["sam_crop_n_layers"],
        crop_n_points_downscale_factor=p["sam_crop_n_points_downscale_factor"],
        min_mask_region_area=p["sam_min_mask_region_area"],
    )
    return binary, markers


# ---- YOLO 实例分割适配器（ultralytics）----
_yolo_model = None
_yolo_loaded_path = None


def _run_yolo(img, p, stage):
    """YOLO-seg 实例分割：需要 ultralytics 与有效权重文件。"""
    global _yolo_model, _yolo_loaded_path
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("未安装 ultralytics，无法使用 YOLO 模型（pip install ultralytics）")

    weights = segmentation.get_config().get("yolo_weights", "")
    if not weights or not os.path.isfile(weights):
        raise RuntimeError("YOLO 权重未配置：请在 设置 → 系统与推理 中指定有效的 .pt 权重文件")

    if _yolo_model is None or weights != _yolo_loaded_path:
        stage("加载 YOLO 权重...")
        _yolo_model = YOLO(weights)
        _yolo_loaded_path = weights

    stage("YOLO 推理中...")
    device = segmentation.resolve_device()
    results = _yolo_model.predict(
        img,
        device=str(device),
        conf=float(p.get("yolo_conf", 0.25)),
        verbose=False,
    )

    h, w = img.shape[:2]
    binary = np.zeros((h, w), np.uint8)
    markers = np.zeros((h, w), np.int32)

    r = results[0]
    if getattr(r, "masks", None) is None:
        # 无检测结果：无二值掩膜也无实例，不展示对应图层（保留全零标记图以走完流程）
        return None, markers

    masks = r.masks.data.cpu().numpy()  # (N, Hm, Wm) float
    if len(masks) == 0:
        return None, markers
    next_id = 2  # 与分水岭约定一致：背景<=1，实例从 2 开始
    for m in masks:
        mm = (m > 0.5).astype(np.uint8)
        if mm.shape != (h, w):
            mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
        binary[mm > 0] = 255
        # 单个掩码可能包含多个连通域，逐一赋予独立实例标签
        num, labels = cv2.connectedComponents(mm)
        for lab in range(1, num):
            markers[labels == lab] = next_id
            next_id += 1
    return binary, markers


# ============================
# 注册内置模型
# ============================
register(ModelSpec(
    key="watershed", label="传统分水岭", param_group=GROUP_WATERSHED,
    run=_run_watershed, description="传统二值化路径 + 距离变换分水岭",
))
register(ModelSpec(
    key="unet_watershed", label="UNet + 分水岭", param_group=GROUP_WATERSHED,
    run=_run_unet, description="UNet 语义分割 + 分水岭实例化（默认）",
))
register(ModelSpec(
    key="sam", label="SAM 实例分割", param_group=GROUP_SAM,
    run=_run_sam, description="Segment Anything 自动实例分割",
))
register(ModelSpec(
    key="yolo", label="YOLO 实例分割", param_group=GROUP_YOLO,
    run=_run_yolo, description="ultralytics YOLO-seg，需配置权重",
))
