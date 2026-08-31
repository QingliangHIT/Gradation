# -*- coding: utf-8 -*-
"""应用组装层：App 主窗口类（界面骨架 + 各功能 Mixin）与 main() 启动入口。"""
import os
import sys

import matplotlib

# matplotlib 图表中文字体（需在创建任何 Figure 之前设置）
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

from PyQt5.QtCore import QByteArray, Qt
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import QApplication, QShortcut

from app.config import APP_ROOT, FontSize, MemSettings, MEM_FILE
from app.controllers.analysis import AnalysisMixin
from app.controllers.appearance import AppearanceMixin
from app.controllers.batch import BatchMixin
from app.controllers.capture import CaptureMixin
from app.controllers.exports import ExportMixin
from app.controllers.pipeline import PipelineMixin
from app.controllers.project_io import ProjectIOMixin
from app.controllers.result_view import ResultViewMixin
from app.controllers.viewer_sync import ViewerSyncMixin
from app.controllers.workspace import WorkspaceMixin
from app.styles import build_app_stylesheet
from app.ui.main_window import MainWindow
from core import config as infer_config
from core.grading import SPEC_RANGES


class App(
    PipelineMixin,
    ResultViewMixin,
    ViewerSyncMixin,
    WorkspaceMixin,
    ProjectIOMixin,
    ExportMixin,
    BatchMixin,
    AnalysisMixin,
    AppearanceMixin,
    CaptureMixin,
    MainWindow,
):
    """主窗口：界面骨架由 MainWindow 构建，业务逻辑由各 Mixin 提供。"""

    def __init__(self):
        super().__init__()

        # ---- 配置持久化（项目根目录下的 state.mem 文件）----
        self.settings = MemSettings(MEM_FILE)
        self._last_dir = self.settings.value("last_dir", "")

        # ---- 窗口图标 ----
        icon_path = os.path.join(APP_ROOT, "common", "image", "icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # 支持拖拽图片打开
        self.setAcceptDrops(True)

        # 分步处理状态
        self.current_step = -1   # -1=未开始, 0=预处理完成, 1=分割完成, 2=分析完成
        self.img_path = None     # 保存图像路径用于重新处理
        self.binary = None       # 二值图缓存
        self.markers = None      # 标记图缓存
        self.particles = None    # 颗粒数据缓存
        self.df = None           # 级配数据缓存
        self._pre_imgs = []      # 预处理结果图列表（主视图窗图层）
        self._pre_names = []     # 预处理结果名称
        self._color_img = None   # 颗粒彩色图缓存
        self._overlay_img = None  # 级配标记图缓存

        # 分割参数
        self.seg_params = {
            "seg_type": "unet_watershed",
            "dist_thresh_ratio": 0.4,
            "kernel_size": 3,
            "close_iterations": 2,
            "open_iterations": 2,
            "min_area": 20,
            "pixel_size": 0.05,
            # SAM 参数
            "sam_points_per_side": 4,
            "sam_pred_iou_thresh": 0.86,
            "sam_stability_score_thresh": 0.92,
            "sam_crop_n_layers": 1,
            "sam_crop_n_points_downscale_factor": 2,
            "sam_min_mask_region_area": 100,
            # YOLO 参数
            "yolo_conf": 0.25,
        }
        # 恢复上次保存的参数
        saved_params = self.settings.value("seg_params", None)
        if isinstance(saved_params, dict):
            for k in self.seg_params:
                if k in saved_params:
                    self.seg_params[k] = saved_params[k]

        # 恢复推理配置（计算设备/模型权重）
        infer_config.configure(
            device=self.settings.value("device_pref", "auto"),
            unet_model_path=self.settings.value("unet_model_path", ""),
            sam_checkpoint=self.settings.value("sam_checkpoint", ""),
            yolo_weights=self.settings.value("yolo_weights", ""),
        )

        # 连接菜单/工具栏动作
        self.actOpenImage.triggered.connect(self.open_image)
        self.actOpenDir.triggered.connect(self.open_workspace_dir)
        self.actSaveProject.triggered.connect(self.save_project)
        self.actOpenProject.triggered.connect(self.open_project)
        self.actExport.triggered.connect(self.export_results)
        self.actExit.triggered.connect(self.close)
        self.actRunAll.triggered.connect(self.run_all)
        self.actNext.triggered.connect(self.go_next_step)
        self.actReprocess.triggered.connect(self.reprocess)
        self.actBatch.triggered.connect(self.run_batch)
        self.actSettings.triggered.connect(self.open_settings)
        self.actAbout.triggered.connect(self.show_about)
        self.actResetLayout.triggered.connect(self.reset_layout)
        # 导出按钮栏（第二行工具栏）
        self.actExportData.triggered.connect(self.export_results)
        self.actExportChart.triggered.connect(self.export_chart)
        self.actExportParticles.triggered.connect(self.export_particles)
        self.actExportLog.triggered.connect(self.export_log)

        # 界面主题（配色风格可设置，持久化）
        self._theme = self.settings.value("theme", "明亮")
        for _name, _act in self.themeActs.items():
            _act.setChecked(_name == self._theme)
            _act.triggered.connect(lambda _c=False, n=_name: self._apply_theme(n))

        # 连接工作区目录树
        self.btnOpenDir.clicked.connect(self.open_workspace_dir)
        self.btnCloseDir.clicked.connect(self.close_workspace_dir)
        self.treeWorkspace.doubleClicked.connect(self._on_tree_double_clicked)
        self.treeWorkspace.setContextMenuPolicy(Qt.CustomContextMenu)
        self.treeWorkspace.customContextMenuRequested.connect(self._on_tree_context_menu)
        self.treeWorkspace.selectionModel().selectionChanged.connect(
            self._on_tree_selection_changed)
        # Del 快捷键删除选中项（支持多选）
        del_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self.treeWorkspace)
        del_shortcut.activated.connect(self._delete_selected)
        ws_dir = self.settings.value("workspace_dir", "")
        if ws_dir and os.path.isdir(ws_dir):
            self.setWorkspace(ws_dir)

        # 连接中间缩放控制
        self.btnZoomIn.clicked.connect(self.zoom_in)
        self.btnZoomOut.clicked.connect(self.zoom_out)
        self.zoomSlider.valueChanged.connect(self.on_zoom_slider)
        self.btnLock.clicked.connect(self._toggle_lock)
        # resultImage 滚轮缩放时同步更新标签和滑块（数值信号）
        self.resultImage.zoomChanged.connect(self._on_result_zoom)
        # 点击结果图中的颗粒查看形态参数
        self.resultImage.particleClicked.connect(self._on_particle_clicked)

        # 收集所有图窗，连接缩放同步信号
        self._all_viewers = [self.resultImage, self.origViewer]
        for viewer in self._all_viewers:
            viewer.labelTextChanged.connect(self._on_any_viewer_zoomed)
            viewer.panChanged.connect(self._on_any_viewer_panned)

        # 标准级配范围选择（曲线叠加上下限）
        self.specCombo.addItem("无")
        self.specCombo.addItems(list(SPEC_RANGES.keys()))
        saved_spec = self.settings.value("spec_range", "无")
        idx = self.specCombo.findText(saved_spec)
        self.specCombo.setCurrentIndex(idx if idx >= 0 else 0)
        self.specCombo.currentIndexChanged.connect(self._on_spec_changed)

        # 级配曲线鼠标悬停交互（显示最近数据点信息）
        self._curve_ax = None
        self._curve_points = None
        self._curve_annot = None
        self.canvas.mpl_connect("motion_notify_event", self._on_curve_hover)

        # 数据采集（摄像头拍照）
        self.btnCapture.clicked.connect(self.capture_image)

        # 日志窗 / 分析面板右键菜单；停靠栏显隐与菜单勾选联动
        self.logText.setContextMenuPolicy(Qt.CustomContextMenu)
        self.logText.customContextMenuRequested.connect(self._on_log_context_menu)
        self.analysisPanel.setContextMenuPolicy(Qt.CustomContextMenu)
        self.analysisPanel.customContextMenuRequested.connect(self._on_analysis_context_menu)
        self._connect_dock_visibility()
        # 第二行导出按钮栏显隐（菜单勾选与双击切换共用同一状态）
        self.actShowExportBar.toggled.connect(self._set_export_bar_visible)
        # 旧的独立记忆废弃（工具栏可见性已含在 window_state 中）
        self.settings.remove("export_collapsed")

        # Ctrl+滚轮字体缩放：日志/表格/图表均为“lv() 层级字号 + 偏移量”，偏移持久化
        _log_saved = int(self.settings.value("log_font_px", 0))
        _tbl_saved = int(self.settings.value("table_font_px", 0))
        # 兼容旧版绝对像素记忆：存值 ≥6 时视为旧格式绝对字号，折算为偏移量
        self._log_font_off = max(-4, min(8, (_log_saved - FontSize.lv(4)) if _log_saved >= 6 else _log_saved))
        self._table_font_off = max(-4, min(8, (_tbl_saved - FontSize.lv(6)) if _tbl_saved >= 6 else _tbl_saved))
        self._chart_font_off = int(self.settings.value("chart_font_off", 0))
        self._apply_log_font(persist=False)
        self._apply_table_font(persist=False)
        self._apply_tab_font()
        self._log_vp = self.logText.viewport()
        self._table_vp = self.table.viewport()
        self._log_vp.installEventFilter(self)
        self._table_vp.installEventFilter(self)
        self.canvas.installEventFilter(self)
        self.histCanvas.installEventFilter(self)
        self.chartTabs.tabBar().installEventFilter(self)
        # 双击工具栏扩展按钮/空白处 -> 两行工具栏切换
        QApplication.instance().installEventFilter(self)

        # 所有图片框鼠标移动 -> 统一显示到中间面板像素坐标标签
        for viewer in self._all_viewers:
            viewer.mouseMoved.connect(self._on_any_viewer_mouse_moved)

        self.img = None
        self._zoom_locked = False  # 缩放锁定状态
        self._worker = None  # 工作线程引用
        self._batch_worker = None  # 批量处理线程引用
        self._batch_dialog = None  # 批量处理进度对话框
        self._update_step_indicator()

        # 快捷键已并入菜单动作（Ctrl+O / F5 / Ctrl+Return）
        self.statusBar().showMessage("就绪（Ctrl+O 打开图像，F5 一键执行）")
        # 记录初始窗口布局（用于重置）与删除撤销状态
        self._initial_state = self.saveState()
        self._trash_last = None  # (原路径, 回收站路径)
        # 恢复上次持久化的面板调整状态（窗口布局/工具栏/分割器）
        _saved_state = self.settings.value("window_state")
        if isinstance(_saved_state, (QByteArray, bytes, bytearray)):
            self.restoreState(_saved_state)
        _rs = self.settings.value("right_splitter")
        if isinstance(_rs, (QByteArray, bytes, bytearray)):
            self.rightSplitter.restoreState(_rs)
        _ls = self.settings.value("left_splitter")
        if isinstance(_ls, (QByteArray, bytes, bytearray)):
            self.leftSplitter.restoreState(_ls)
        self._ensure_export_break()
        self._set_export_bar_visible(self.tbExport.isVisible())


def main():
    """程序启动入口：HighDPI、字体档位恢复、全局样式与应用事件循环。"""
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # 恢复上次保存的字体档位（需在构建 UI 前生效，记忆来自项目 state.mem 文件）
    boot_settings = MemSettings(MEM_FILE)
    FontSize.set_preset(boot_settings.value("font_preset", "large"))
    boot_theme = boot_settings.value("theme", "明亮")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # 全局样式（集中管理于 app/styles）
    app.setStyleSheet(build_app_stylesheet(FontSize, boot_theme))

    win = App()
    win.show()
    sys.exit(app.exec_())
