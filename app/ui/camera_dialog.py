# -*- coding: utf-8 -*-
"""相机采集窗口：实时预览 + 拍照，含摄像头不可用的鲁棒性处理。"""
import cv2
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap


class CameraDialog(QDialog):
    """相机采集窗口：实时预览 + 拍照，含摄像头不可用的鲁棒性处理。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("数据采集 - 相机")
        self.resize(760, 580)
        self.frame_bgr = None  # 拍照结果

        layout = QVBoxLayout(self)

        # 摄像头设备选择行（打开前可选择调哪个摄像头）
        devRow = QHBoxLayout()
        devRow.addWidget(QLabel("摄像头："))
        self.deviceCombo = QComboBox()
        self.deviceCombo.setMinimumWidth(160)
        self.btnRefreshDev = QPushButton("🔄 刷新设备")
        self.btnRefreshDev.setObjectName("zoomBtn")
        self.lblDevStatus = QLabel("")
        self.lblDevStatus.setObjectName("infoLabel")
        devRow.addWidget(self.deviceCombo)
        devRow.addWidget(self.btnRefreshDev)
        devRow.addWidget(self.lblDevStatus, 1)
        layout.addLayout(devRow)

        self.videoLabel = QLabel("正在扫描摄像头设备...")
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setMinimumSize(640, 480)
        self.videoLabel.setStyleSheet(
            "background: #1e1e1e; color: #cccccc; border-radius: 4px;")
        layout.addWidget(self.videoLabel, 1)

        row = QHBoxLayout()
        row.addStretch()
        self.btnShoot = QPushButton("📷 拍照")
        self.btnShoot.setObjectName("primaryBtn")
        self.btnCancel = QPushButton("取消")
        self.btnCancel.setObjectName("flatBtn")
        row.addWidget(self.btnShoot)
        row.addWidget(self.btnCancel)
        layout.addLayout(row)
        self.btnShoot.clicked.connect(self.accept)
        self.btnCancel.clicked.connect(self.reject)
        self.btnRefreshDev.clicked.connect(self._scan_devices)
        self.deviceCombo.currentIndexChanged.connect(self._on_device_changed)

        self.cap = None
        self._timer = None
        self._fail_count = 0
        self._opening = False
        self._scan_devices()

    def _scan_devices(self):
        """扫描可用摄像头设备（0~4），填充下拉框并默认打开第一个。"""
        self._opening = True
        self.deviceCombo.clear()
        found = []
        for i in range(5):
            cap = None
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            except Exception:
                cap = None
            if cap is not None and cap.isOpened():
                found.append(i)
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
        for i in found:
            self.deviceCombo.addItem(f"摄像头 {i}", i)
        self._opening = False
        if not found:
            self.lblDevStatus.setText("未发现可用摄像头")
            self.videoLabel.setText(
                "无法打开摄像头：\n请检查设备连接与驱动，或确认未被其他程序占用。")
            self.btnShoot.setEnabled(False)
            return
        self.lblDevStatus.setText(f"发现 {len(found)} 个设备")
        self._open_device(0)

    def _on_device_changed(self, idx):
        """下拉框切换摄像头设备时重新打开。"""
        if idx < 0 or self._opening:
            return
        self._open_device(idx)

    def _open_device(self, combo_idx):
        """打开指定设备并启动预览定时刷新；失败则给出提示。"""
        if self._timer is not None:
            self._timer.stop()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        dev = self.deviceCombo.itemData(combo_idx)
        if dev is None:
            return
        cap = None
        for args in ((dev, cv2.CAP_DSHOW), (dev,)):
            try:
                cap = cv2.VideoCapture(*args)
            except Exception:
                cap = None
            if cap is not None and cap.isOpened():
                break
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
        if cap is None:
            self.videoLabel.setText(
                f"无法打开摄像头 {dev}：\n请检查设备连接与驱动，或确认未被其他程序占用。")
            self.btnShoot.setEnabled(False)
            return
        # 预热几帧，稳定曝光/白平衡，同时验证可读性
        for _ in range(5):
            try:
                cap.read()
            except Exception:
                pass
        self.cap = cap
        self.frame_bgr = None
        self.btnShoot.setEnabled(True)
        self.videoLabel.setText("正在读取画面...")
        self._fail_count = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._grab_frame)
        self._timer.start(33)

    def _grab_frame(self):
        """定时器读取一帧并刷新预览；连续失败则提示并禁用拍照。"""
        try:
            ret, frame = self.cap.read()
        except Exception:
            ret, frame = False, None
        if not ret or frame is None:
            self._fail_count += 1
            if self._fail_count >= 30 and self._timer is not None:
                self._timer.stop()
                self.videoLabel.setText("摄像头画面读取失败，请检查设备后重试。")
                self.btnShoot.setEnabled(False)
            return
        self._fail_count = 0
        self.frame_bgr = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img.copy())
        self.videoLabel.setPixmap(pix.scaled(
            self.videoLabel.size(), Qt.KeepAspectRatio,
            Qt.SmoothTransformation))

    def closeEvent(self, event):
        if self._timer is not None:
            self._timer.stop()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        super().closeEvent(event)
