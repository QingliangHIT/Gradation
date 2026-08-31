# -*- coding: utf-8 -*-
"""统一设置对话框：双页签（算法参数 / 系统与推理）。"""
from PyQt5.QtWidgets import (
    QDialog, QTabWidget, QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit, QPushButton,
    QHBoxLayout, QLabel,
)
from PyQt5.QtCore import Qt

from core import registry as model_registry
from app.config import FontSize


class UnifiedSettingsDialog(QDialog):
    """算法参数 + 系统设置一体化对话框。"""

    def __init__(self, seg_params=None, model_config=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setFixedSize(520, 640)
        self.params = seg_params or {}
        self.model_config = model_config or {}
        self._models = model_registry.list_models()
        self._init_ui()

    # ============================
    # 界面构建
    # ============================
    def _init_ui(self):
        root = QVBoxLayout(self)

        tabs = QTabWidget()
        tabs.addTab(self._build_algo_tab(), "算法参数")
        tabs.addTab(self._build_system_tab(), "系统与推理")
        root.addWidget(tabs, 1)

        # ---- 底部按钮 ----
        btnRow = QHBoxLayout()
        btnRow.addStretch()
        self.btnOk = QPushButton("确定")
        self.btnCancel = QPushButton("取消")
        self.btnOk.setFixedSize(88, 32)
        self.btnCancel.setFixedSize(88, 32)
        self.btnOk.setObjectName("primaryBtn")
        self.btnCancel.setObjectName("flatBtn")
        btnRow.addWidget(self.btnOk)
        btnRow.addWidget(self.btnCancel)
        root.addLayout(btnRow)
        self.btnOk.clicked.connect(self.accept)
        self.btnCancel.clicked.connect(self.reject)

    def _combo(self):
        c = QComboBox()
        c.setStyleSheet(f"font-size: {FontSize.lv(2)}px; padding: 4px;")
        return c

    def _build_algo_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)

        # ---- 分割模型（来自注册表）----
        typeBox = QGroupBox("分割模型")
        typeLayout = QVBoxLayout(typeBox)
        self.segTypeCombo = self._combo()
        cur_key = self.params.get("seg_type", "unet_watershed")
        cur_idx = 0
        for i, spec in enumerate(self._models):
            self.segTypeCombo.addItem(spec.label, spec.key)
            if spec.key == cur_key:
                cur_idx = i
            if spec.description:
                self.segTypeCombo.setItemData(i, spec.description, Qt.ToolTipRole)
        self.segTypeCombo.setCurrentIndex(cur_idx)
        self.segTypeCombo.currentIndexChanged.connect(self._on_model_changed)
        typeLayout.addWidget(self.segTypeCombo)
        self.modelDescLbl = QLabel("")
        self.modelDescLbl.setObjectName("infoLabel")
        self.modelDescLbl.setWordWrap(True)
        typeLayout.addWidget(self.modelDescLbl)
        layout.addWidget(typeBox)

        # ---- 分水岭参数组（传统/UNet 共用）----
        self.watershedBox = QGroupBox("分水岭参数")
        wl = QFormLayout(self.watershedBox)
        wl.setSpacing(8)
        self.distThreshSpin = QDoubleSpinBox()
        self.distThreshSpin.setRange(0.1, 0.9)
        self.distThreshSpin.setSingleStep(0.05)
        self.distThreshSpin.setDecimals(2)
        self.distThreshSpin.setValue(self.params.get("dist_thresh_ratio", 0.4))
        self.distThreshSpin.setToolTip("距离变换阈值比例，越小分割越多颗粒")
        wl.addRow("距离阈值比例:", self.distThreshSpin)

        self.kernelSizeSpin = QSpinBox()
        self.kernelSizeSpin.setRange(1, 9)
        self.kernelSizeSpin.setSingleStep(2)
        self.kernelSizeSpin.setValue(self.params.get("kernel_size", 3))
        wl.addRow("形态学核大小:", self.kernelSizeSpin)

        self.closeIterSpin = QSpinBox()
        self.closeIterSpin.setRange(0, 10)
        self.closeIterSpin.setValue(self.params.get("close_iterations", 2))
        wl.addRow("闭运算迭代:", self.closeIterSpin)

        self.openIterSpin = QSpinBox()
        self.openIterSpin.setRange(0, 10)
        self.openIterSpin.setValue(self.params.get("open_iterations", 2))
        wl.addRow("开运算迭代:", self.openIterSpin)
        layout.addWidget(self.watershedBox)

        # ---- SAM 参数组 ----
        self.samBox = QGroupBox("SAM 参数")
        sl = QFormLayout(self.samBox)
        sl.setSpacing(8)
        self.samPointsPerSideSpin = QSpinBox()
        self.samPointsPerSideSpin.setRange(1, 64)
        self.samPointsPerSideSpin.setValue(self.params.get("sam_points_per_side", 4))
        self.samPointsPerSideSpin.setToolTip("每边采样点数，总点数为 points_per_side^2")
        sl.addRow("每边采样点数:", self.samPointsPerSideSpin)

        self.samPredIouThreshSpin = QDoubleSpinBox()
        self.samPredIouThreshSpin.setRange(0.0, 1.0)
        self.samPredIouThreshSpin.setSingleStep(0.01)
        self.samPredIouThreshSpin.setDecimals(2)
        self.samPredIouThreshSpin.setValue(self.params.get("sam_pred_iou_thresh", 0.86))
        sl.addRow("预测IoU阈值:", self.samPredIouThreshSpin)

        self.samStabilityThreshSpin = QDoubleSpinBox()
        self.samStabilityThreshSpin.setRange(0.0, 1.0)
        self.samStabilityThreshSpin.setSingleStep(0.01)
        self.samStabilityThreshSpin.setDecimals(2)
        self.samStabilityThreshSpin.setValue(self.params.get("sam_stability_score_thresh", 0.92))
        sl.addRow("稳定性分数阈值:", self.samStabilityThreshSpin)

        self.samCropNLayersSpin = QSpinBox()
        self.samCropNLayersSpin.setRange(0, 5)
        self.samCropNLayersSpin.setValue(self.params.get("sam_crop_n_layers", 1))
        sl.addRow("裁剪层数:", self.samCropNLayersSpin)

        self.samCropDownscaleSpin = QSpinBox()
        self.samCropDownscaleSpin.setRange(1, 16)
        self.samCropDownscaleSpin.setValue(
            self.params.get("sam_crop_n_points_downscale_factor", 2))
        sl.addRow("裁剪下采样因子:", self.samCropDownscaleSpin)

        self.samMinMaskAreaSpin = QSpinBox()
        self.samMinMaskAreaSpin.setRange(0, 10000)
        self.samMinMaskAreaSpin.setSingleStep(10)
        self.samMinMaskAreaSpin.setValue(self.params.get("sam_min_mask_region_area", 100))
        sl.addRow("最小掩码面积:", self.samMinMaskAreaSpin)
        layout.addWidget(self.samBox)

        # ---- YOLO 参数组 ----
        self.yoloBox = QGroupBox("YOLO 参数")
        yl = QFormLayout(self.yoloBox)
        yl.setSpacing(8)
        self.yoloConfSpin = QDoubleSpinBox()
        self.yoloConfSpin.setRange(0.05, 0.95)
        self.yoloConfSpin.setSingleStep(0.05)
        self.yoloConfSpin.setDecimals(2)
        self.yoloConfSpin.setValue(self.params.get("yolo_conf", 0.25))
        self.yoloConfSpin.setToolTip("检测置信度阈值，越低保留越多实例")
        yl.addRow("置信度阈值:", self.yoloConfSpin)
        hint = QLabel("权重文件在「系统与推理」页签中配置")
        hint.setObjectName("infoLabel")
        yl.addRow("", hint)
        layout.addWidget(self.yoloBox)

        # ---- 测量参数 ----
        measureBox = QGroupBox("测量参数")
        ml = QFormLayout(measureBox)
        ml.setSpacing(8)
        self.pixelSizeSpin = QDoubleSpinBox()
        self.pixelSizeSpin.setRange(0.001, 10.0)
        self.pixelSizeSpin.setSingleStep(0.01)
        self.pixelSizeSpin.setDecimals(3)
        self.pixelSizeSpin.setValue(self.params.get("pixel_size", 0.05))
        self.pixelSizeSpin.setToolTip("每个像素对应的实际尺寸(mm)")
        ml.addRow("像素尺寸(mm/px):", self.pixelSizeSpin)

        self.minAreaSpin = QSpinBox()
        self.minAreaSpin.setRange(1, 1000)
        self.minAreaSpin.setSingleStep(5)
        self.minAreaSpin.setValue(self.params.get("min_area", 20))
        ml.addRow("最小颗粒面积(像素):", self.minAreaSpin)
        layout.addWidget(measureBox)

        layout.addStretch()
        self._on_model_changed(cur_idx)
        return page

    def _build_system_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)

        # ---- 显示设置 ----
        displayBox = QGroupBox("显示设置")
        dl = QFormLayout(displayBox)
        self.fontSizeCombo = self._combo()
        self.fontSizeCombo.addItems(["大", "中", "小"])
        idx_map = {"large": 0, "medium": 1, "small": 2}
        self.fontSizeCombo.setCurrentIndex(idx_map.get(FontSize.PRESET, 0))
        dl.addRow("字体大小:", self.fontSizeCombo)
        layout.addWidget(displayBox)

        # ---- 推理设置 ----
        inferBox = QGroupBox("推理设置")
        il = QFormLayout(inferBox)
        il.setSpacing(8)
        self.deviceCombo = self._combo()
        self.deviceCombo.addItem("自动（优先 GPU）", "auto")
        self.deviceCombo.addItem("CPU", "cpu")
        self.deviceCombo.addItem("GPU（CUDA）", "cuda")
        dev_map = {"auto": 0, "cpu": 1, "cuda": 2}
        self.deviceCombo.setCurrentIndex(
            dev_map.get(self.model_config.get("device", "auto"), 0))
        il.addRow("计算设备:", self.deviceCombo)
        layout.addWidget(inferBox)

        # ---- 模型权重 ----
        weightBox = QGroupBox("模型权重")
        wl = QFormLayout(weightBox)
        wl.setSpacing(8)
        self.unetPathEdit = QLineEdit(self.model_config.get("unet_model_path", ""))
        self.unetPathEdit.setToolTip("UNet/ResUNet 权重文件路径（.pth），留空或无效路径保持默认")
        wl.addRow("UNet 权重:", self.unetPathEdit)

        self.samPathEdit = QLineEdit(self.model_config.get("sam_checkpoint", ""))
        self.samPathEdit.setToolTip("SAM 检查点文件路径（.pth），留空或无效路径保持默认")
        wl.addRow("SAM 权重:", self.samPathEdit)

        self.yoloPathEdit = QLineEdit(self.model_config.get("yolo_weights", ""))
        self.yoloPathEdit.setToolTip("YOLO-seg 权重文件路径（.pt），留空则 YOLO 模型不可用")
        wl.addRow("YOLO 权重:", self.yoloPathEdit)
        layout.addWidget(weightBox)

        layout.addStretch()
        return page

    # ============================
    # 交互与取值
    # ============================
    def _on_model_changed(self, index):
        """按所选模型的参数组显示对应参数面板。"""
        if index < 0 or index >= len(self._models):
            return
        spec = self._models[index]
        self.modelDescLbl.setText(spec.description)
        self.watershedBox.setVisible(spec.param_group == model_registry.GROUP_WATERSHED)
        self.samBox.setVisible(spec.param_group == model_registry.GROUP_SAM)
        self.yoloBox.setVisible(spec.param_group == model_registry.GROUP_YOLO)

    def get_params(self):
        """返回全部设置：分割/测量参数 + 系统配置键。"""
        preset_map = {0: "large", 1: "medium", 2: "small"}
        return {
            # 分割模型与参数
            "seg_type": self.segTypeCombo.currentData(),
            "dist_thresh_ratio": self.distThreshSpin.value(),
            "kernel_size": self.kernelSizeSpin.value(),
            "close_iterations": self.closeIterSpin.value(),
            "open_iterations": self.openIterSpin.value(),
            "min_area": self.minAreaSpin.value(),
            "pixel_size": self.pixelSizeSpin.value(),
            "sam_points_per_side": self.samPointsPerSideSpin.value(),
            "sam_pred_iou_thresh": self.samPredIouThreshSpin.value(),
            "sam_stability_score_thresh": self.samStabilityThreshSpin.value(),
            "sam_crop_n_layers": self.samCropNLayersSpin.value(),
            "sam_crop_n_points_downscale_factor": self.samCropDownscaleSpin.value(),
            "sam_min_mask_region_area": self.samMinMaskAreaSpin.value(),
            "yolo_conf": self.yoloConfSpin.value(),
            # 系统配置
            "font_preset": preset_map.get(self.fontSizeCombo.currentIndex(), "large"),
            "device": self.deviceCombo.currentData(),
            "unet_model_path": self.unetPathEdit.text().strip(),
            "sam_checkpoint": self.samPathEdit.text().strip(),
            "yolo_weights": self.yoloPathEdit.text().strip(),
        }
