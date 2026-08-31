# -*- coding: utf-8 -*-
"""多图层切换浏览控件（普通滚轮切换图层，Ctrl+滚轮缩放）。"""
from PyQt5.QtCore import Qt, pyqtSignal

from app.ui.image_viewer import ImageViewer


class ImageViewerEx(ImageViewer):
    """扩展ImageViewer，集成多图切换功能。
    普通滚轮=切换图片，ctrl+滚轮=缩放。"""

    imageChanged = pyqtSignal(int, str)  # (当前索引, 当前名称)

    def __init__(self):
        super().__init__()
        self._img_list = []
        self._img_names = []
        self._img_idx = 0

    def setImages(self, imgs, names):
        """设置多图列表，自动显示第一张。"""
        self._img_list = imgs
        self._img_names = names
        self._img_idx = 0
        self._showCurrent()

    def clear(self):
        """清空多图列表和显示。"""
        self._img_list = []
        self._img_names = []
        self._img_idx = 0
        super().clear()

    def _showCurrent(self):
        if not self._img_list:
            return
        img = self._img_list[self._img_idx]
        name = self._img_names[self._img_idx] if self._img_idx < len(self._img_names) else ""
        self.showImage(img)
        self._emitLabelChanged()
        self.imageChanged.emit(self._img_idx, name)

    def switchImage(self, direction):
        if not self._img_list:
            return
        self._img_idx = (self._img_idx + direction) % len(self._img_list)
        self._showCurrent()

    def setCurrentIndex(self, idx):
        """直接跳转到指定图层。"""
        if not self._img_list:
            return
        self._img_idx = max(0, min(idx, len(self._img_list) - 1))
        self._showCurrent()

    def _emitLabelChanged(self):
        name = self._img_names[self._img_idx] if self._img_idx < len(self._img_names) else ""
        total = len(self._img_list)
        zoom = int(self.scale_factor * 100)
        self.labelTextChanged.emit(f"{name} [{self._img_idx + 1}/{total}]\t🔍 {zoom}%")
        self.zoomChanged.emit(self.scale_factor)

    @property
    def labelText(self):
        name = self._img_names[self._img_idx] if self._img_idx < len(self._img_names) else ""
        total = len(self._img_list)
        zoom = int(self.scale_factor * 100)
        if total > 0:
            return f"{name} [{self._img_idx + 1}/{total}]\t🔍 {zoom}%"
        else:
            return f"🔍 {zoom}%"

    def wheelEvent(self, event):
        if event.modifiers() == Qt.ControlModifier:
            self._zoom_at(event)
            self._emitLabelChanged()
        else:
            # 普通滚轮: 切换图片（向上滚切换到下一张）
            if event.angleDelta().y() > 0:
                self.switchImage(-1)
            else:
                self.switchImage(1)
