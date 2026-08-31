# -*- coding: utf-8 -*-
"""SAM（Segment Anything）自动实例分割适配器。

SAM 官方源码位于 third_party/segment_anything，本模块负责把包路径加入
sys.path 并提供掩膜生成与标签图转换。
"""
import os
import sys

import cv2
import numpy as np

from core import config

# third_party 目录（segment_anything 包所在位置）
_THIRD_PARTY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party")
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

_SAM_MODEL_TYPE = "vit_h"

_cache = {"sig": None, "model": None}


def _signature():
    """缓存签名：设备偏好或检查点变化时自动重建模型。"""
    c = config.get_config()
    return (c["device"], c["sam_checkpoint"])


def _load_sam_model():
    """加载 SAM 模型（带缓存）。"""
    if _cache["model"] is not None and _cache["sig"] == _signature():
        return _cache["model"]

    device = config.resolve_device()
    c = config.get_config()
    model = sam_model_registry[_SAM_MODEL_TYPE](checkpoint=c["sam_checkpoint"])
    model.to(device=device)
    model.eval()
    _cache["sig"] = _signature()
    _cache["model"] = model
    return model


def sam_segment(
    img,
    points_per_side=4,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
):
    """使用 SAM 进行实例分割。

    输入: BGR 彩色图 (H, W, 3) uint8
    返回: (binary, markers)。SAM 不产生独立的二值掩膜，binary 恒为 None，
          markers 为标签图 (H, W) int32，每个实例有唯一标签（背景 <= 1）。
    """
    sam = _load_sam_model()

    # BGR -> RGB
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 创建 mask 生成器
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
    )

    # 生成 masks
    masks = mask_generator.generate(img_rgb)

    # 将 masks 转换为 markers 标签图
    h, w = img_rgb.shape[:2]
    markers = np.zeros((h, w), dtype=np.int32)

    # 按面积从大到小排序，大物体优先分配标签
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    # 实例标签从 2 开始（与分水岭一致，1 为背景约定），避免首个颗粒被测量环节跳过
    for idx, mask_dict in enumerate(sorted_masks, start=2):
        seg = mask_dict['segmentation']
        if seg.dtype != bool:
            seg = seg.astype(bool)
        # 只标记尚未被分配的区域
        markers[seg & (markers == 0)] = idx

    return None, markers


def overlay_masks(image, masks, alpha=0.5, random_color=True, save_path=None):
    """将 SAM 输出的掩码列表叠加显示在原图上，返回叠加后的图片。

    Args:
        image: 原图 (H, W, 3) uint8 RGB格式
        masks: mask列表，每个元素含 'segmentation' 键
        alpha: 掩码透明度，0=全透明 1=不透明
        random_color: 是否用随机颜色
        save_path: 若指定则保存图片到该路径

    Returns:
        overlay: 叠加后的图片 (H, W, 3) uint8
    """
    overlay = image.copy()
    if len(masks) == 0:
        return overlay

    sorted_anns = sorted(masks, key=lambda x: x['area'], reverse=True)
    for ann in sorted_anns:
        m = ann['segmentation']
        if m.dtype != bool:
            m = m.astype(bool)
        if random_color:
            color = (np.random.random(3) * 255).astype(np.uint8)
        else:
            color = np.array([128, 128, 128], dtype=np.uint8)
        overlay[m] = (overlay[m] * (1 - alpha) + color * alpha).astype(np.uint8)

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"已保存: {save_path}")

    return overlay
