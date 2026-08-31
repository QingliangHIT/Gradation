# -*- coding: utf-8 -*-
"""通用图像 I/O 与几何处理工具（兼容中文路径）。"""
import os

import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    """读取图像（np.fromfile + imdecode，兼容中文路径）。失败返回 None。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, flags) if data.size else None


def imwrite(path, img, params=None):
    """保存图像（imencode + tofile，兼容中文路径）。返回是否成功。"""
    ext = os.path.splitext(path)[1] or ".png"
    try:
        ok, buf = cv2.imencode(ext, img, params or [])
        if ok:
            buf.tofile(path)
        return bool(ok)
    except Exception:
        return False


def resize_image(image, scale=None, target_size=None):
    """缩放图片。scale: 缩放比例; target_size: (w, h) 目标尺寸。二选一。"""
    h, w = image.shape[:2]
    if target_size is not None:
        new_w, new_h = target_size
    elif scale is not None:
        new_w, new_h = int(w * scale), int(h * scale)
    else:
        return image
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def split_image(image, rows, cols, overlap=0):
    """将图片切分为 rows x cols 块。overlap: 重叠像素数。返回 (块列表, 每块的(x0,y0,x1,y1))。"""
    h, w = image.shape[:2]
    block_h = (h + overlap * (rows - 1)) // rows
    block_w = (w + overlap * (cols - 1)) // cols
    patches, coords = [], []
    for r in range(rows):
        for c in range(cols):
            y0 = max(r * (block_h - overlap), 0)
            x0 = max(c * (block_w - overlap), 0)
            y1 = min(y0 + block_h, h)
            x1 = min(x0 + block_w, w)
            patches.append(image[y0:y1, x0:x1])
            coords.append((x0, y0, x1, y1))
    return patches, coords


def merge_patches(patches, coords, original_size):
    """将切分块拼回原图。patches: 块列表; coords: 对应的(x0,y0,x1,y1); original_size: (h, w)。"""
    h, w = original_size
    result = np.zeros((h, w, 3), dtype=np.uint8)
    for patch, (x0, y0, x1, y1) in zip(patches, coords):
        result[y0:y1, x0:x1] = patch
    return result
