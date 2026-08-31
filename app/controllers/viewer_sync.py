# -*- coding: utf-8 -*-
"""视图同步控制器：缩放控制、多图窗缩放/平移锁定同步与像素信息显示。"""


class ViewerSyncMixin:
    """主视图窗与其他图窗之间的缩放/平移/像素坐标联动。"""

    # ============================
    # 缩放控制
    # ============================
    def zoom_in(self):
        val = min(self.zoomSlider.value() + 20, 500)
        self.zoomSlider.setValue(val)

    def zoom_out(self):
        val = max(self.zoomSlider.value() - 20, 10)
        self.zoomSlider.setValue(val)

    def on_zoom_slider(self, val):
        self.lblZoomPercent.setText(f"{val}%")
        ratio = val / 100
        self.lblZoomRatio.setText(f"{ratio:.1f}:1")
        self.resultImage.scale_factor = ratio
        self.resultImage.updateImage()
        # 锁定模式下同步其他图片框
        if self._zoom_locked:
            for viewer in self._all_viewers:
                if viewer is not self.resultImage:
                    viewer.blockSignals(True)
                    viewer.setZoom(ratio)
                    viewer.blockSignals(False)

    def _on_result_zoom(self, scale):
        """resultImage 滚轮缩放时，同步更新标签和滑块。"""
        zoom = int(scale * 100)
        self.lblZoomPercent.setText(f"{zoom}%")
        self.lblZoomRatio.setText(f"{scale:.1f}:1")
        self.zoomSlider.blockSignals(True)
        self.zoomSlider.setValue(
            max(self.zoomSlider.minimum(), min(self.zoomSlider.maximum(), zoom))
        )
        self.zoomSlider.blockSignals(False)

    # ============================
    # 多图窗缩放/平移锁定同步
    # ============================
    def _on_any_viewer_zoomed(self, text):
        """任意图片框缩放时，若锁定则同步所有图片框。"""
        if not self._zoom_locked:
            return
        sender = self.sender()
        if sender is None:
            return
        scale = sender.scale_factor
        # 同步其他图片框
        for viewer in self._all_viewers:
            if viewer is not sender:
                viewer.blockSignals(True)
                viewer.setZoom(scale)
                viewer.blockSignals(False)

    def _on_any_viewer_panned(self, h, v):
        """任意图片框拖动时，若锁定则同步所有图片框（按比例同步）。"""
        if not self._zoom_locked:
            return
        sender = self.sender()
        if sender is None:
            return
        # 计算发送者的滚动比例
        h_max = sender.horizontalScrollBar().maximum()
        v_max = sender.verticalScrollBar().maximum()
        h_ratio = h / h_max if h_max > 0 else 0
        v_ratio = v / v_max if v_max > 0 else 0
        # 按比例同步其他图片框
        for viewer in self._all_viewers:
            if viewer is not sender:
                vh_max = viewer.horizontalScrollBar().maximum()
                vv_max = viewer.verticalScrollBar().maximum()
                viewer.setScrollPosition(
                    int(h_ratio * vh_max),
                    int(v_ratio * vv_max)
                )

    def _toggle_lock(self):
        """切换缩放锁定状态。"""
        self._zoom_locked = not self._zoom_locked
        if self._zoom_locked:
            self.btnLock.setText("🔒")
            self.btnLock.setToolTip("已锁定，所有图片框同步缩放和拖动（点击解锁）")
        else:
            self.btnLock.setText("🔓")
            self.btnLock.setToolTip("点击锁定，同步所有图片框的缩放和拖动")

    # ============================
    # 像素信息显示
    # ============================
    def _on_any_viewer_mouse_moved(self, x, y, pixel_val):
        """任意图片框鼠标移动时统一更新像素坐标和值标签。"""
        if x < 0 or y < 0:
            self.lblPixelInfo.setText("")
            return
        # 获取发送者名称
        sender = self.sender()
        name = "主视图"
        idx = getattr(sender, "_img_idx", None)
        ns = getattr(sender, "_img_names", [])
        if idx is not None and idx < len(ns):
            name = ns[idx]
        # 格式化像素值
        if pixel_val is None:
            val_str = ""
        elif isinstance(pixel_val, tuple):
            val_str = f"  RGB=({pixel_val[0]}, {pixel_val[1]}, {pixel_val[2]})"
        else:
            val_str = f"  V={pixel_val}"
        self.lblPixelInfo.setText(f"[{name}] ({x}, {y}){val_str}")
