# -*- coding: utf-8 -*-
"""图像预处理：通用预处理（所有模型共用）与传统方法专用二值化。"""
import cv2
import numpy as np


def preprocess(img):
    """通用预处理（所有模型共用的正常预处理）：灰度化 → CLAHE 对比度增强 → 高斯去噪。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    img_list = [denoised]
    name_list = ["去噪"]
    return img_list, name_list, "预处理：灰度化-对比度增强-高斯去噪"


def traditional_binary(img):
    """传统方法专用路径：在通用预处理基础上做 Otsu 二值化 + 形态学闭运算，
    得到颗粒前景二值图（供分水岭实例分割使用）。"""
    img_list, _, _ = preprocess(img)
    denoised = img_list[-1]
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = 255 - binary
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return binary
