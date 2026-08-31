# -*- coding: utf-8 -*-
"""分水岭实例分割：基于距离变换的前景提取 + 形态学清理。"""
import cv2
import numpy as np


def segment_particles(binary, dist_thresh_ratio=0.45, kernel_size=3,
                      close_iterations=0, open_iterations=2):
    """传统分水岭：输入二值图，返回实例标签图 markers。"""
    img_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=open_iterations)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    _, sure_fg = cv2.threshold(dist, dist_thresh_ratio * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img_color, markers)

    return markers


def segment_particles_unet(binary, dist_thresh_ratio=0.4, kernel_size=3,
                           close_iterations=2, open_iterations=2):
    """UNet 分割后处理：
    1. 形态学清理二值掩膜
    2. Watershed 实例分割

    输入: 二值图 (H, W) uint8
    返回: markers 标签图
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=open_iterations)

    img_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    sure_bg = cv2.dilate(binary, kernel, iterations=3)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, dist_thresh_ratio * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers[unknown == 255] = 0

    markers = cv2.watershed(img_color, markers)

    return markers
