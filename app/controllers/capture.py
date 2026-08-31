# -*- coding: utf-8 -*-
"""图像来源控制器：文件打开与摄像头数据采集。"""
import os
import time

import cv2
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox

from app.ui.camera_dialog import CameraDialog


class CaptureMixin:
    """图像输入入口（菜单/工具栏/拖拽之外的来源）。"""

    def open_image(self):
        """打开文件对话框选择图像并载入处理流程。"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "选择图片", self._last_dir, "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if filename == "":
            return
        self._load_image(filename)

    def capture_image(self):
        """打开相机窗口拍照并载入处理流程（含摄像头不可用的鲁棒性处理）。"""
        dlg = CameraDialog(self)
        if dlg.exec_() != QDialog.Accepted or dlg.frame_bgr is None:
            return
        frame = dlg.frame_bgr
        save_dir = self.workspace_dir or self._last_dir or os.getcwd()
        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError as e:
            self.appendLog(f"保存目录创建失败: {e}", "error")
            return
        path = os.path.join(save_dir, f"capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
        ok = False
        try:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                buf.tofile(path)  # 兼容中文路径
        except Exception as e:
            self.appendLog(f"图像编码失败: {e}", "error")
        if not ok or not os.path.isfile(path):
            QMessageBox.critical(self, "采集失败", "图像保存失败。")
            self.appendLog("采集图像保存失败", "error")
            return
        self.appendLog(f"摄像头拍照完成: {path}", "success")
        self._load_image(path)
