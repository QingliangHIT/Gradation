# -*- coding: utf-8 -*-
"""
主窗口界面结构（专业化停靠布局）：
    - 菜单栏 + 工具栏（含步骤状态指示器）
    - 左侧停靠栏（占整列）：工作区目录树 / 图像信息
    - 中央：主视图窗（图层滚轮切换）+ 缩放控制
    - 右侧停靠栏：级配曲线（可叠加标准级配范围）/ 统计结果表
    - 底部停靠栏：处理日志（位于中央与右侧下方）
"""
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QSlider,
    QTextEdit, QDockWidget, QSplitter, QTreeView, QComboBox, QTabWidget,
    QFileSystemModel, QToolBar, QAction, QSizePolicy, QDialog,
    QAbstractItemView,
)
from PyQt5.QtCore import Qt, QDir, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from algorithms.image_viewer import ImageViewer, ImageViewerEx
from algorithms.grading import SIEVE_SIZE
from font_config import FontSize
from styles import THEMES


# ============================
# 步骤状态指示器（工具栏右侧）
# ============================
class StepIndicator(QWidget):

    def __init__(self, compact=False):
        super().__init__()
        self._compact = compact
        self.steps = ["图像采集与预处理", "集料粒度提取", "结果分析与统计"]
        self.steps_short = ["预处理", "粒度提取", "结果分析"]
        self.current_step = -1
        self._circles = []
        self._labels = []
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.setSpacing(8 if self._compact else 20)
        layout.setContentsMargins(0, 0, 0, 0)
        F = FontSize
        # 紧凑模式（分析面板内）：圆圈/文字更小、短步骤名，与下拉框尺寸协调
        steps = self.steps_short if self._compact else self.steps
        self._c_size = 18 if self._compact else 32
        self._circle_font = max(F.lv(3) - 1, 8) if self._compact else F.lv(2)
        self._lbl_font = F.lv(3) if self._compact else F.lv(1)
        arrow_font = F.lv(3) if self._compact else F.lv(1)

        self._circles = []
        self._labels = []

        for i, step in enumerate(steps):
            container = QHBoxLayout()

            circle = QLabel(str(i + 1))
            circle.setFixedSize(self._c_size, self._c_size)
            circle.setAlignment(Qt.AlignCenter)
            self._circles.append(circle)

            label = QLabel(step)
            label.setStyleSheet(f"font-weight: bold; font-size: {self._lbl_font}px;")
            self._labels.append(label)

            container.addWidget(circle)
            container.addWidget(label)

            if i < len(steps) - 1:
                arrow = QLabel("→")
                arrow.setStyleSheet(f"color: #999999; font-size: {arrow_font}px;")
                container.addWidget(arrow)

            layout.addLayout(container)

        layout.addStretch()
        self.setLayout(layout)
        self.update_ui()

    def update_ui(self):
        """根据 current_step 更新圆圈和标签的样式（已完成/进行中/未开始三态）。"""
        F = FontSize
        for i, (circle, label) in enumerate(zip(self._circles, self._labels)):
            if i <= self.current_step:
                # 已完成：绿色 ✓
                bg, fg, text = "#4CAF50", "#4CAF50", "✓"
            elif i == self.current_step + 1:
                # 当前进行中：蓝色高亮
                bg, fg, text = "#4A90E2", "#4A90E2", str(i + 1)
            else:
                # 未开始：灰色
                bg, fg, text = "#CCCCCC", "#666666", str(i + 1)
            circle.setText(text)
            circle.setStyleSheet(f"""
                QLabel{{
                    background-color: {bg};
                    color: white;
                    border-radius: {self._c_size // 2}px;
                    font-weight: bold;
                    font-size: {self._circle_font}px;
                }}
            """)
            label.setStyleSheet(f"""
                QLabel{{
                    color: {fg};
                    font-weight: bold;
                    font-size: {self._lbl_font}px;
                }}
            """)


# ============================
# 面板小标题
# ============================
def _panel_title(text):
    lbl = QLabel(text)
    lbl.setObjectName("panelTitle")
    return lbl


# ============================
# 主窗口
# ============================
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2PSL - 混凝土骨料颗粒智能筛分")
        self.resize(1280, 800)
        self.workspace_dir = ""
        self._build_menus()
        self._build_toolbar()
        self._build_left_dock()
        self._build_center()
        self._build_right_dock()
        self._build_bottom_dock()
        # 左/右侧停靠栏占整列，底部停靠栏仅位于中央下方
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)

    # ============================
    # 菜单栏
    # ============================
    def _build_menus(self):
        mb = self.menuBar()

        mFile = mb.addMenu("文件(&F)")
        self.actOpenImage = mFile.addAction("打开图像(&O)")
        self.actOpenImage.setShortcut("Ctrl+O")
        self.actOpenDir = mFile.addAction("打开工作区目录(&D)")
        self.actOpenDir.setShortcut("Ctrl+Shift+O")
        mFile.addSeparator()
        self.actSaveProject = mFile.addAction("保存项目(&S)")
        self.actSaveProject.setShortcut("Ctrl+S")
        self.actOpenProject = mFile.addAction("打开项目(&P)")
        self.actOpenProject.setShortcut("Ctrl+Shift+P")
        self.actExport = mFile.addAction("导出结果(&E)")
        self.actExport.setShortcut("Ctrl+E")
        self.actExportChart = mFile.addAction("导出图表")
        mFile.addSeparator()
        self.actExit = mFile.addAction("退出")
        self.actExit.setShortcut("Ctrl+Q")

        mProc = mb.addMenu("处理(&P)")
        self.actRunAll = mProc.addAction("一键执行")
        self.actRunAll.setShortcut("F5")
        self.actNext = mProc.addAction("下一步")
        self.actNext.setShortcut("Ctrl+Return")
        self.actReprocess = mProc.addAction("重新处理")
        self.actBatch = mProc.addAction("批量处理...")
        mProc.addSeparator()
        self.actSettings = mProc.addAction("设置...")

        mWin = mb.addMenu("窗口(&W)")
        self.actResetLayout = mWin.addAction("重置窗口布局")
        mWin.addSeparator()
        # 停靠窗口显示/隐藏（与停靠栏叉号关闭联动）
        self.actShowLeft = mWin.addAction("显示工作区")
        self.actShowRight = mWin.addAction("显示分析结果")
        self.actShowBottom = mWin.addAction("显示输出日志")
        self.actShowExportBar = mWin.addAction("显示导出按钮栏")
        for act in (self.actShowLeft, self.actShowRight, self.actShowBottom):
            act.setCheckable(True)
            act.setChecked(True)
        # 导出按钮栏默认收起（双击工具栏或勾选本菜单项展开）
        self.actShowExportBar.setCheckable(True)
        self.actShowExportBar.setChecked(False)
        mWin.addSeparator()
        # 界面主题（配色风格可设置，互斥勾选，随 THEMES 自动生成）
        mTheme = mWin.addMenu("界面主题")
        self.themeActs = {}
        for name in THEMES:
            act = mTheme.addAction(name)
            act.setCheckable(True)
            self.themeActs[name] = act
        self.themeActs["明亮"].setChecked(True)

        mHelp = mb.addMenu("帮助(&H)")
        self.actAbout = mHelp.addAction("关于")

    # ============================
    # 工具栏
    # ============================
    def _build_toolbar(self):
        tb = QToolBar("主工具栏")
        tb.setObjectName("mainToolbar")
        tb.setMovable(False)
        tb.setIconSize(tb.iconSize())
        self.addToolBar(tb)
        self.mainToolbar = tb

        # 复用菜单动作，保证状态同步
        tb.addAction(self.actOpenImage)
        tb.addAction(self.actOpenDir)
        tb.addSeparator()
        tb.addAction(self.actRunAll)
        tb.addAction(self.actNext)
        tb.addAction(self.actReprocess)
        tb.addAction(self.actBatch)
        tb.addSeparator()
        tb.addAction(self.actSettings)

        # 步骤状态指示器已移至分析结果面板（与标准级配选择同行）
        spacer = QWidget()
        spacer.setObjectName("toolbarSpacer")
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)

        # 第二行工具栏：仅容纳四个导出按钮，默认收起；双击主工具栏或在“窗口”菜单勾选展开
        self.tbExport = QToolBar("导出按钮栏")
        self.tbExport.setObjectName("mainToolbar")
        self.tbExport.setMovable(False)
        self.addToolBar(self.tbExport)
        self._ensure_export_break()
        self.tbExport.setVisible(False)

        # 导出按钮组（QAction 形式，纯文字与第一行工具栏按钮风格/高度一致）
        self.actExportParticles = QAction("颗粒详情", self)
        self.actExportData = QAction("导出结果", self)
        self.actExportChart = QAction("导出图表", self)
        self.actExportLog = QAction("导出日志", self)
        for _a in (self.actExportParticles, self.actExportData,
                   self.actExportChart, self.actExportLog):
            self.tbExport.addAction(_a)

    def _ensure_export_break(self):
        """确保导出按钮栏独占一行（先移除重复断行符再插入，可安全重复调用）。"""
        self.removeToolBarBreak(self.tbExport)
        self.insertToolBarBreak(self.tbExport)

    # ============================
    # 左侧停靠栏：工作区 + 图像信息
    # ============================
    def _build_left_dock(self):
        dock = QDockWidget("工作区", self)
        dock.setObjectName("leftDock")
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)

        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Vertical)
        self.leftSplitter = splitter

        # ---- 工作区目录树 ----
        wsPanel = QWidget()
        wsLayout = QVBoxLayout(wsPanel)
        wsLayout.setContentsMargins(0, 0, 0, 0)
        wsLayout.setSpacing(4)

        wsHead = QHBoxLayout()
        wsHead.addWidget(_panel_title("项目目录"))
        wsHead.addStretch()
        self.btnOpenDir = QPushButton("打开目录")
        self.btnCloseDir = QPushButton("关闭")
        for b in (self.btnOpenDir, self.btnCloseDir):
            b.setObjectName("zoomBtn")
        wsHead.addWidget(self.btnOpenDir)
        wsHead.addWidget(self.btnCloseDir)
        wsLayout.addLayout(wsHead)

        self.fsModel = QFileSystemModel()
        self.fsModel.setRootPath(QDir.rootPath())
        self.treeWorkspace = QTreeView()
        self.treeWorkspace.setModel(self.fsModel)
        self.treeWorkspace.setObjectName("workspaceTree")
        # 仅显示名称列
        for col in (1, 2, 3):
            self.treeWorkspace.hideColumn(col)
        self.treeWorkspace.setHeaderHidden(True)
        self.treeWorkspace.setAnimated(False)
        # 多选支持（Ctrl/Shift）
        self.treeWorkspace.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.lblWorkspace = QLabel("未打开目录（点击“打开目录”加载工作区）")
        self.lblWorkspace.setObjectName("infoLabel")
        self.lblWorkspace.setWordWrap(True)
        wsLayout.addWidget(self.lblWorkspace)
        wsLayout.addWidget(self.treeWorkspace, 1)

        # ---- 原图预览（左 dock 中部）----
        origPanel = QWidget()
        origLayout = QVBoxLayout(origPanel)
        origLayout.setContentsMargins(0, 0, 0, 0)
        origLayout.setSpacing(2)
        origLayout.addWidget(_panel_title("○ 原图"))
        self.origViewer = ImageViewer()
        self.origViewer.setToolTip("当前加载的原图（Ctrl+滚轮缩放）")
        origLayout.addWidget(self.origViewer, 1)

        # ---- 图像信息（左 dock 底部）----
        infoPanel = QWidget()
        infoLayout = QVBoxLayout(infoPanel)
        infoLayout.setContentsMargins(0, 0, 0, 0)
        infoLayout.setSpacing(2)
        infoLayout.addWidget(_panel_title("○ 图像信息"))
        self.lblImageSize = QLabel("图像尺寸：-")
        self.lblCurrentStep = QLabel("当前步骤：-")
        self.lblProcessTime = QLabel("处理时间：-")
        for lbl in [self.lblImageSize,
                    self.lblCurrentStep, self.lblProcessTime]:
            lbl.setObjectName("infoLabel")
            infoLayout.addWidget(lbl)

        splitter.addWidget(wsPanel)
        splitter.addWidget(origPanel)
        splitter.addWidget(infoPanel)
        # splitter.setSizes([1, 1, 1])
        # 默认比例：项目目录与原图各占一半，图像信息栏自适应最小高度
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 0)

        layout.addWidget(splitter)
        dock.setWidget(host)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.leftDock = dock

    def setWorkspace(self, path):
        """设置工作区目录并展开树。"""
        self.workspace_dir = path
        if path:
            idx = self.fsModel.index(path)
            self.treeWorkspace.setRootIndex(idx)
            self.lblWorkspace.setText(path)
        else:
            self.treeWorkspace.setRootIndex(self.fsModel.index(QDir.rootPath()))
            self.lblWorkspace.setText("未打开目录（点击“打开目录”加载工作区）")

    def setOrigImage(self, img_rgb):
        """在左侧原图窗显示原图（RGB）。"""
        self.origViewer.showImage(img_rgb)

    # ============================
    # 中央：主视图窗（图层切换）
    # ============================
    def _build_center(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)
        self.setCentralWidget(central)

        # layout.addWidget(_panel_title("主视图窗（滚轮切换图层 / Ctrl+滚轮缩放 / 点击颗粒查看参数）"))
        layout.addWidget(_panel_title("主视图窗"))

        # 图层切换栏
        layerRow = QHBoxLayout()
        self.btnPrevLayer = QPushButton("◀")
        self.btnNextLayer = QPushButton("▶")
        for b in (self.btnPrevLayer, self.btnNextLayer):
            b.setFixedSize(26, 24)
            b.setObjectName("zoomBtn")
        self.lblLayerName = QLabel("主视图窗")
        self.lblLayerName.setObjectName("imageLabel")
        self.lblLayerZoom = QLabel("🔍 100%")
        self.lblLayerZoom.setObjectName("zoomLabel")
        layerRow.addWidget(self.btnPrevLayer)
        layerRow.addWidget(self.btnNextLayer)
        layerRow.addWidget(self.lblLayerName, 1)
        layerRow.addWidget(self.lblLayerZoom)
        layout.addLayout(layerRow)

        self.resultImage = ImageViewerEx()
        self.resultImage.setToolTip("滚轮切换图层，Ctrl+滚轮缩放，点击颗粒查看参数")
        layout.addWidget(self.resultImage, 1)
        self.btnPrevLayer.clicked.connect(lambda: self.resultImage.switchImage(-1))
        self.btnNextLayer.clicked.connect(lambda: self.resultImage.switchImage(1))
        self.resultImage.imageChanged.connect(self._on_layer_changed)
        self.resultImage.zoomChanged.connect(self._on_layer_zoom)

        self.lblPixelInfo = QLabel("")
        self.lblPixelInfo.setObjectName("pixelInfo")
        self.lblPixelInfo.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.lblPixelInfo)

        # 缩放控制栏
        zoomRow = QHBoxLayout()
        self.btnLock = QPushButton("🔓")
        self.btnZoomIn = QPushButton("🔍+")
        self.btnZoomOut = QPushButton("🔍-")
        self.zoomSlider = QSlider(Qt.Horizontal)
        self.zoomSlider.setRange(10, 500)
        self.zoomSlider.setValue(100)
        self.zoomSlider.setFixedWidth(120)
        self.lblZoomPercent = QLabel("100%")
        self.lblZoomRatio = QLabel("1.0:1")
        self.lblZoomPercent.setObjectName("zoomLabel")
        self.lblZoomRatio.setObjectName("zoomLabel")

        for b in [self.btnLock, self.btnZoomIn, self.btnZoomOut]:
            b.setFixedSize(32, 28)
            b.setObjectName("zoomBtn")

        zoomRow.addWidget(self.btnLock)
        zoomRow.addWidget(self.btnZoomIn)
        zoomRow.addWidget(self.btnZoomOut)
        zoomRow.addWidget(self.zoomSlider)
        zoomRow.addWidget(self.lblZoomPercent)
        zoomRow.addWidget(self.lblZoomRatio)
        zoomRow.addStretch()
        self.btnCapture = QPushButton("📷 数据采集")
        self.btnCapture.setToolTip("通过摄像头拍照采集图像")
        self.btnCapture.setObjectName("successBtn")
        zoomRow.addWidget(self.btnCapture)
        layout.addLayout(zoomRow)

    def _on_layer_changed(self, idx, name):
        """主视图窗图层切换时同步标题。"""
        total = len(self.resultImage._img_list)
        self.lblLayerName.setText(f"{name}  [{idx + 1}/{total}]")

    def _on_layer_zoom(self, scale):
        self.lblLayerZoom.setText(f"🔍 {int(scale * 100)}%")

    def _connect_dock_visibility(self):
        """窗口菜单勾选与停靠栏显隐双向联动（叉号关闭后可从菜单恢复）。"""
        pairs = [
            (self.actShowLeft, self.leftDock),
            (self.actShowRight, self.rightDock),
            (self.actShowBottom, self.bottomDock),
        ]
        for act, dock in pairs:
            act.triggered.connect(lambda checked, d=dock: d.setVisible(checked))
            dock.visibilityChanged.connect(act.setChecked)

    def _redock_all(self):
        """恢复显示所有停靠窗口并同步菜单勾选（布局重置时调用）。"""
        for dock in (self.leftDock, self.rightDock, self.bottomDock):
            dock.setVisible(True)

    def setLayers(self, imgs, names):
        """更新主视图窗的图层内容（各阶段产生的图像汇总，滚轮切换）。"""
        self.resultImage.setImages(imgs, names)
        if not imgs:
            self.lblLayerName.setText("主视图窗")
            self.lblLayerZoom.setText("🔍 100%")

    # ============================
    # 右侧停靠栏：级配曲线 + 统计表
    # ============================
    def _build_right_dock(self):
        dock = QDockWidget("分析结果", self)
        dock.setObjectName("rightDock")
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)

        host = QWidget()
        # 右键菜单：导出结果 / 导出颗粒 / 导出图表 / 智能分析（逻辑在 main.py 连接）
        self.analysisPanel = host
        layout = QVBoxLayout(host)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # 图表区（级配曲线 / 粒径分布共用同一位置，页签切换）
        specRow = QHBoxLayout()
        # 步骤状态指示器（紧凑模式，与标准级配选择同行）
        self.stepIndicator = StepIndicator(compact=True)
        specRow.addWidget(self.stepIndicator)
        specRow.addStretch()
        # specRow.addWidget(QLabel("标准级配："))
        self.specCombo = QComboBox()
        self.specCombo.setToolTip("选择标准级配范围，在曲线上叠加显示通过率上下限区间")
        specRow.addWidget(self.specCombo)

        layout.addLayout(specRow)

        self.chartTabs = QTabWidget()
        self.figure = Figure(figsize=(5, 4.2))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet(
            "border: 1px solid #e0e0e0; border-radius: 4px; background: white;")
        self.histFigure = Figure(figsize=(5, 4.2))
        self.histCanvas = FigureCanvas(self.histFigure)
        self.histCanvas.setStyleSheet(
            "border: 1px solid #e0e0e0; border-radius: 4px; background: white;")
        self.chartTabs.addTab(self.canvas, "级配曲线")
        self.chartTabs.addTab(self.histCanvas, "粒径分布")
        self.table = QTableWidget()
        self.table.setObjectName("resultTable")
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["粒径范围 (mm)", "分计筛余 (%)", "累计筛余 (%)", "累计通过率 (%)", "数量占比 (%)"]
        )
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        # 列宽按内容自适应（Ctrl+滚轮放大字体后表头文字不会被裁切），末列拉伸填满面板
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        self._initTableRanges()

        # 分割器：图表区 / 统计表高度可拖拽自由调整
        self.rightSplitter = QSplitter(Qt.Vertical)
        self.rightSplitter.addWidget(self.chartTabs)
        tablePanel = QWidget()
        tableLayout = QVBoxLayout(tablePanel)
        tableLayout.setContentsMargins(0, 0, 0, 0)
        tableLayout.setSpacing(2)
        tableLayout.addWidget(_panel_title("○ 统计结果"))
        tableLayout.addWidget(self.table)
        self.rightSplitter.addWidget(tablePanel)
        self.rightSplitter.setStretchFactor(0, 1)
        self.rightSplitter.setStretchFactor(1, 1)
        layout.addWidget(self.rightSplitter, 1)

        dock.setWidget(host)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.rightDock = dock

    # ============================
    # 底部停靠栏：处理日志（位于中央与右侧停靠栏下方）
    # ============================
    def _build_bottom_dock(self):
        dock = QDockWidget("输出", self)
        dock.setObjectName("bottomDock")
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)

        # ---- 处理日志（右键：清空日志 / 导出日志，策略在 main.py 连接）----
        self.logText = QTextEdit()
        self.logText.setReadOnly(True)
        self.logText.setObjectName("logText")

        dock.setWidget(self.logText)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.bottomDock = dock

    # ============================
    # 日志辅助
    # ============================
    def appendLog(self, msg, level="info"):
        """向日志追加一条带时间戳的消息。
        level: info/success/error/warn，不同级别不同颜色。"""
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        colors = {
            "info": "#333333",
            "success": "#2e7d32",
            "error": "#c62828",
            "warn": "#ef6c00",
        }
        color = colors.get(level, "#333333")
        self.logText.append(
            f'<span style="color:#999999;">[{ts}]</span> '
            f'<span style="color:{color};">{msg}</span>'
        )

    def setImageInfo(self, w, h, step="", elapsed=""):
        self.lblImageSize.setText(f"图像尺寸：{w} × {h} 像素")
        self.lblCurrentStep.setText(f"当前步骤：{step}")
        self.lblProcessTime.setText(f"处理时间：{elapsed}")

    def _initTableRanges(self):
        """预先填充粒径范围列。"""
        n = len(SIEVE_SIZE) + 1  # 每个筛孔区间 + 最后一行
        self.table.setRowCount(n)
        for i in range(len(SIEVE_SIZE)):
            if i == 0:
                text = f"> {SIEVE_SIZE[i]}"
            else:
                text = f"{SIEVE_SIZE[i - 1]} - {SIEVE_SIZE[i]}"
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 0, item)
        # 最后一行：小于最小筛孔
        item = QTableWidgetItem(f"< {SIEVE_SIZE[-1]}")
        item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(len(SIEVE_SIZE), 0, item)
