# -*- coding: utf-8 -*-
"""基础图像浏览控件：缩放、拖拽、像素坐标查询与颗粒点击选中。"""
from PyQt5.QtWidgets import QScrollArea, QLabel, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np

from core.measure import find_particle_at


class ImageViewer(QScrollArea):
    particleClicked = pyqtSignal(int)
    particleDeleted = pyqtSignal(int)
    labelTextChanged = pyqtSignal(str)
    zoomChanged = pyqtSignal(float)  # 缩放比例数值信号（1.0 = 100%）
    mouseMoved = pyqtSignal(int, int, object)  # x, y, pixel_value
    panChanged = pyqtSignal(int, int)  # hScroll, vScroll 滚动位置变化

    def __init__(self):

        super().__init__()

        self.scale_factor = 1.0
        self.pixmap = None
        self._dragging = False
        self._last_pos = None
        self._markers = None
        self._particles = None
        self._highlight_id = None
        self._color_img = None
        self._orig_img = None  # 存储原始 numpy 图像用于像素值查询
        self._fitted_w = 0  # 适应视口后的图像宽度（用于坐标转换）
        self._fitted_h = 0  # 适应视口后的图像高度
        self._display_w = 0  # 当前显示的缩放图像宽度
        self._display_h = 0  # 当前显示的缩放图像高度

        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignCenter)

        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        self.setMinimumSize(200, 200)

        self.inner = QLabel()
        self.inner.setAlignment(Qt.AlignCenter)
        self.inner.setStyleSheet("""
            QLabel{
                border:1px solid #cccccc;
                background:white;
                border-radius: 4px;
            }
        """)
        self.setWidget(self.inner)

    def setParticleData(self, markers, particles, color_img):
        self._markers = markers
        self._particles = particles
        self._color_img = color_img.copy() if color_img is not None else None
        self._highlight_id = None

    def setZoom(self, scale):
        """外部设置缩放比例（不发射信号）。"""
        self.scale_factor = scale
        self.updateImage()

    def setScrollPosition(self, h, v):
        """外部设置滚动位置（不触发 panChanged）。"""
        self.blockSignals(True)
        self.horizontalScrollBar().setValue(h)
        self.verticalScrollBar().setValue(v)
        self.blockSignals(False)

    def showImage(self, img):
        if len(img.shape) == 2:
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, c = img.shape
            bytesPerLine = c * w
            qimg = QImage(img.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)
        self._orig_img = img.copy()  # 保存原始 numpy 图像
        self.scale_factor = 1.0
        self._highlight_id = None
        self.updateImage()

    def clear(self):
        """清空显示的图像和数据。"""
        self.pixmap = None
        self._markers = None
        self._particles = None
        self._highlight_id = None
        self._color_img = None
        self._orig_img = None
        self._fitted_w = 0
        self._fitted_h = 0
        self._display_w = 0
        self._display_h = 0
        self.inner.clear()
        self.scale_factor = 1.0

    def highlightParticle(self, pid):
        self._highlight_id = pid
        self.updateImage()

    def clearHighlight(self):
        self._highlight_id = None
        self.updateImage()

    def updateImage(self):
        if self.pixmap is None:
            return

        vw = self.viewport().width()
        vh = self.viewport().height()

        fitted = self.pixmap.scaled(vw, vh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._fitted_w = fitted.width()
        self._fitted_h = fitted.height()

        w = int(fitted.width() * self.scale_factor)
        h = int(fitted.height() * self.scale_factor)
        self._display_w = w
        self._display_h = h

        scaled = self.pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        if self._highlight_id is not None and self._color_img is not None and self._markers is not None:
            overlay = self._color_img.copy()
            mask = np.uint8(self._markers == self._highlight_id)
            overlay[mask > 0] = [255, 255, 0]
            alpha = 0.45
            display = cv2.addWeighted(overlay, alpha, self._color_img, 1 - alpha, 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (255, 255, 0), 2)

            h2, w2 = display.shape[:2]
            qimg2 = QImage(display.data, w2, h2, w2 * 3, QImage.Format_RGB888)
            hl_pixmap = QPixmap.fromImage(qimg2)
            hl_scaled = hl_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scaled = hl_scaled

        self.inner.setPixmap(scaled)
        self.inner.resize(max(w, vw), max(h, vh))
        self._reposition()

    def _reposition(self):
        if self.pixmap is None:
            return
        vw = self.viewport().width()
        vh = self.viewport().height()
        iw = self.inner.width()
        ih = self.inner.height()
        x = (vw - iw) // 2 if iw < vw else 0
        y = (vh - ih) // 2 if ih < vh else 0
        self.inner.move(x, y)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        vw = self.viewport().width()
        vh = self.viewport().height()
        if vw > 0 and vh > 0:
            self.inner.resize(max(self.inner.width(), vw), max(self.inner.height(), vh))
        self.updateImage()

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        # 发射滚动位置变化信号，用于锁定模式下同步其他 viewer
        if not self.signalsBlocked():
            self.panChanged.emit(
                self.horizontalScrollBar().value(),
                self.verticalScrollBar().value()
            )

    def mouseDoubleClickEvent(self, event):
        self.scale_factor = 1.0
        self.updateImage()
        self._emitLabelChanged()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and self._last_pos:
            delta = event.pos() - self._last_pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            self._last_pos = event.pos()
        # 发射鼠标像素坐标信号
        if self.pixmap is not None:
            self._emitPixelPos(event.pos())
        super().mouseMoveEvent(event)

    def _emitPixelPos(self, pos):
        """计算鼠标位置对应的图像像素坐标和值，并发射信号。
        仅当鼠标在实际显示的图像区域内时才发射有效坐标。"""
        if self.pixmap is None or self._display_w <= 0 or self._display_h <= 0:
            self.mouseMoved.emit(-1, -1, None)
            return

        # 将 QScrollArea 坐标转换为 inner widget 坐标
        inner_pos = self.inner.mapFrom(self, pos)

        # 计算 pixmap 在 inner QLabel 中的居中偏移
        # inner 尺寸 = max(display, viewport)，pixmap(display) 在其中居中
        px = (self.inner.width() - self._display_w) // 2
        py = (self.inner.height() - self._display_h) // 2

        # 鼠标在显示图像上的位置
        dx = inner_pos.x() - px
        dy = inner_pos.y() - py

        # 检查是否在显示图像范围内（空白区域则清除）
        if not (0 <= dx < self._display_w and 0 <= dy < self._display_h):
            self.mouseMoved.emit(-1, -1, None)
            return

        # 显示图像坐标 -> 原始图像像素坐标
        pw = self.pixmap.width()
        ph = self.pixmap.height()
        orig_x = int(dx * pw / self._display_w)
        orig_y = int(dy * ph / self._display_h)

        # 安全检查原始图像范围
        if self._orig_img is not None:
            img_h, img_w = self._orig_img.shape[:2]
            orig_x = max(0, min(orig_x, img_w - 1))
            orig_y = max(0, min(orig_y, img_h - 1))
            # 获取像素值
            if len(self._orig_img.shape) == 2:
                pixel_val = int(self._orig_img[orig_y, orig_x])
            elif len(self._orig_img.shape) == 3:
                pixel_val = tuple(int(v) for v in self._orig_img[orig_y, orig_x])
            else:
                pixel_val = None
            self.mouseMoved.emit(orig_x, orig_y, pixel_val)
        else:
            self.mouseMoved.emit(-1, -1, None)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            was_dragging = self._dragging
            self._dragging = False
            self.setCursor(Qt.ArrowCursor)

            if was_dragging:
                self._last_pos = None
                super().mouseReleaseEvent(event)
                return

            if self._markers is None or self.pixmap is None or self._display_w <= 0 or self._display_h <= 0:
                self._last_pos = None
                super().mouseReleaseEvent(event)
                return

            # 将 QScrollArea 坐标转换为 inner widget 坐标
            inner_pos = self.inner.mapFrom(self, event.pos())

            # pixmap 在 inner QLabel 中的居中偏移
            px = (self.inner.width() - self._display_w) // 2
            py = (self.inner.height() - self._display_h) // 2
            dx = inner_pos.x() - px
            dy = inner_pos.y() - py

            # 检查是否在显示图像范围内
            if not (0 <= dx < self._display_w and 0 <= dy < self._display_h):
                self._last_pos = None
                super().mouseReleaseEvent(event)
                return

            # 显示图像坐标 -> 原始图像像素坐标
            pw = self.pixmap.width()
            ph = self.pixmap.height()
            orig_x = int(dx * pw / self._display_w)
            orig_y = int(dy * ph / self._display_h)

            self._last_pos = None
            pid = find_particle_at(self._markers, orig_x, orig_y)
            if pid is not None:
                self.particleClicked.emit(pid)
            else:
                self._highlight_id = None
                self.updateImage()

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() == Qt.ControlModifier:
            self._zoom_at(event)
            self._emitLabelChanged()

    def _zoom_at(self, event):
        """以鼠标位置为锚点执行 Ctrl+滚轮缩放（保持鼠标所在点不动）。"""
        # 记录缩放前的状态
        old_scale = self.scale_factor
        mouse_vp = event.pos()
        vw = self.viewport().width()
        vh = self.viewport().height()
        iw_old = self.inner.width()
        ih_old = self.inner.height()
        ox_old = (vw - iw_old) // 2 if iw_old < vw else 0
        oy_old = (vh - ih_old) // 2 if ih_old < vh else 0

        # 执行缩放
        if event.angleDelta().y() > 0:
            self.scale_factor *= 1.15
        else:
            self.scale_factor /= 1.15
        self.updateImage()

        # 计算缩放后偏移变化，调整滚动条使鼠标所在点不动
        iw_new = self.inner.width()
        ih_new = self.inner.height()
        ox_new = (vw - iw_new) // 2 if iw_new < vw else 0
        oy_new = (vh - ih_new) // 2 if ih_new < vh else 0

        mx_rel = mouse_vp.x() + self.horizontalScrollBar().value() - ox_old
        my_rel = mouse_vp.y() + self.verticalScrollBar().value() - oy_old
        if old_scale > 0:
            ratio = self.scale_factor / old_scale
            dx = mx_rel * (ratio - 1) + ox_new - ox_old
            dy = my_rel * (ratio - 1) + oy_new - oy_old
            self.horizontalScrollBar().setValue(
                int(self.horizontalScrollBar().value() + dx))
            self.verticalScrollBar().setValue(
                int(self.verticalScrollBar().value() + dy))

    def _emitLabelChanged(self):
        zoom = int(self.scale_factor * 100)
        self.labelTextChanged.emit(f"🔍 {zoom}%")
        self.zoomChanged.emit(self.scale_factor)

    @property
    def labelText(self):
        zoom = int(self.scale_factor * 100)
        return f"🔍 {zoom}%"
