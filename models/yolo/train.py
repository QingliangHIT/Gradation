# -*- coding: utf-8 -*-
"""YOLO 分割模型预测演示：加载权重、推理并用 matplotlib 对比显示。

用法:
    python -m models.yolo.train
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_imgs(*imgs, titles=None):
    """并排显示多张 BGR 图像（替代旧 common.show.show_ui 的本地实现）。"""
    n = len(imgs)
    if titles is None:
        titles = [f"Image {i + 1}" for i in range(n)]
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from ultralytics import YOLO

    # 加载模型
    model = YOLO(r"A:\05-Codes\Gradation\DATA\yolov8n-seg.pt")

    # 预测图片
    img = cv2.imread(r"A:\05-Codes\Gradation\DATA\demo\capture_20260830_174523.jpg")
    results = model(img)

    # results[0] 表示第一张图片（results 是列表），plot() 返回绘制好的 BGR 图像数组
    annotated_img = results[0].plot()
    show_imgs(img, annotated_img, titles=["原图", "YOLO 分割结果"])
