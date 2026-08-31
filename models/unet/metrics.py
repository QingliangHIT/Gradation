# -*- coding: utf-8 -*-
"""分割评价指标：IoU / Dice / Accuracy（tensor 版供训练，numpy 版供预测）。"""
import numpy as np
import torch


def compute_metrics_tensor(outputs, masks):
    """训练/验证用：输入网络输出（B,C,H,W tensor）与标签，返回 (iou, dice, acc)。"""
    preds = torch.argmax(outputs, dim=1)
    tp = ((preds == 1) & (masks == 1)).sum().item()
    fp = ((preds == 1) & (masks == 0)).sum().item()
    fn = ((preds == 0) & (masks == 1)).sum().item()
    tn = ((preds == 0) & (masks == 0)).sum().item()

    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    return iou, dice, acc


def compute_metrics_numpy(pred, mask):
    """预测对比用：无标注（mask 为 None）时返回 (None, None, None)。"""
    if mask is None:
        return None, None, None
    mask = (mask > 127).astype(np.int64)
    tp = ((pred == 1) & (mask == 1)).sum()
    fp = ((pred == 1) & (mask == 0)).sum()
    fn = ((pred == 0) & (mask == 1)).sum()
    tn = ((pred == 0) & (mask == 0)).sum()
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    return iou, dice, acc
