import sys
import os
import json
import shutil
import time
import cv2
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QTableWidgetItem, QMessageBox,
    QProgressDialog, QMenu, QDialog, QTableWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QShortcut, QTextEdit,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent, QByteArray
from PyQt5.QtGui import QIcon, QKeySequence
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 项目依赖根目录下的 unet_project / samInstance_project 包，
# 将工作区根目录加入搜索路径，保证可直接运行
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from ui_main import MainWindow, CameraDialog
from dialogs import UnifiedSettingsDialog
from font_config import FontSize, MemSettings

# 界面记忆文件（项目级 .mem）
MEM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.mem")
from styles import build_app_stylesheet
from algorithms import segmentation
from algorithms import model_registry
from algorithms.process import preprocess
from algorithms.segmentation import (
    colorize_particles,
    measure_particles,
    overlay_markers,
)
from algorithms.grading import (
    calculate_grading,
    SIEVE_SIZE,
    SPEC_RANGES,
)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# 颗粒参数中文列名映射（用于导出）
_PARTICLE_CN = {
    "id": "编号",
    "area_pixel": "像素面积(px²)",
    "area_mm2": "投影面积(mm²)",
    "diameter_mm": "等效粒径(mm)",
    "length_mm": "长轴(mm)",
    "width_mm": "短轴(mm)",
    "perimeter_mm": "周长(mm)",
    "feret_major_mm": "最大Feret径(mm)",
    "feret_ratio": "Feret长宽比",
    "eccentricity": "偏心率",
    "circularity": "圆形度",
    "roundness": "圆整度",
    "solidity": "实心度",
    "convexity": "凸性",
    "rect_fill": "矩形填充率",
    "radial_cv": "径向变异系数",
    "angularity": "棱角性指数",
    "corner_density": "角点密度(1/px)",
    "cx": "质心X",
    "cy": "质心Y",
}


def calc_dxx(df, percent):
    """在级配曲线上插值求指定通过率对应的特征粒径（D10/D50/D90），
    无法插值时返回 '-'。"""
    if df is None or len(df) == 0:
        return "-"
    sizes = df["筛孔(mm)"].to_numpy(dtype=float)
    passing = df["累计通过率(%)"].to_numpy(dtype=float)
    if passing.min() > percent or passing.max() < percent:
        return "-"
    # 通过率升序排列后对 log(粒径) 插值
    order = np.argsort(passing)
    log_size = np.interp(percent, passing[order], np.log10(sizes[order]))
    return round(float(10 ** log_size), 3)


class SegmentationWorker(QThread):
    """后台线程执行模型推理，避免阻塞主线程。"""
    finished = pyqtSignal(object, object)  # binary, markers
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, img, seg_type, params, parent=None):
        super().__init__(parent)
        self.img = img
        self.seg_type = seg_type
        self.params = params

    def run(self):
        try:
            binary, markers = model_registry.run_model(
                self.seg_type, self.img, self.params, stage=self.progress.emit
            )
            self.finished.emit(binary, markers)
        except Exception as e:
            self.error.emit(str(e))


class BatchWorker(QThread):
    """批量处理：对文件夹内所有图片执行完整流程并汇总统计。"""
    progress = pyqtSignal(int, str)   # (已完成数, 当前文件名)
    finished = pyqtSignal(object)     # 汇总 DataFrame
    error = pyqtSignal(str)

    def __init__(self, files, params, parent=None):
        super().__init__(parent)
        self.files = files
        self.params = params
        self._abort = False

    def stop(self):
        self._abort = True

    def run(self):
        p = self.params
        rows = []
        for i, f in enumerate(self.files):
            if self._abort:
                break
            name = os.path.basename(f)
            try:
                img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    rows.append({"文件": name, "颗粒数": 0, "备注": "读取失败"})
                    self.progress.emit(i + 1, name)
                    continue
                _, markers = model_registry.run_model(p["seg_type"], img, p)
                particles = measure_particles(
                    markers,
                    pixel_size=p["pixel_size"],
                    min_area=p["min_area"],
                )
                row = {"文件": name, "颗粒数": len(particles)}
                if particles:
                    ds = np.array([pt["diameter_mm"] for pt in particles])
                    row["平均粒径(mm)"] = round(float(ds.mean()), 3)
                    row["最大粒径(mm)"] = round(float(ds.max()), 3)
                    row["最小粒径(mm)"] = round(float(ds.min()), 3)
                    gdf = calculate_grading(particles)
                    row["D10(mm)"] = calc_dxx(gdf, 10)
                    row["D50(mm)"] = calc_dxx(gdf, 50)
                    row["D90(mm)"] = calc_dxx(gdf, 90)
                rows.append(row)
            except Exception as e:
                rows.append({"文件": name, "颗粒数": 0, "备注": f"出错: {e}"})
            self.progress.emit(i + 1, name)
        self.finished.emit(pd.DataFrame(rows))


class App(MainWindow):

    def __init__(self):
        super().__init__()

        # ---- 配置持久化（项目目录下的 .mem 文件）----
        self.settings = MemSettings(MEM_FILE)
        self._last_dir = self.settings.value("last_dir", "")

        # ---- 窗口图标 ----
        icon_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "common", "image", "icon.ico",
        )
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
        segmentation.configure(
            device=self.settings.value("device_pref", "auto"),
            unet_model_path=self.settings.value("unet_model_path", ""),
            sam_checkpoint=self.settings.value("sam_checkpoint", ""),
            yolo_weights=self.settings.value("yolo_weights", ""),
        )

        # 连接菜单/工具栏动作
        self.actOpenImage.triggered.connect(self.open_image)
        self.actOpenDir.triggered.connect(self.open_workspace_dir)
        self.actSaveProject.triggered.connect(self.save_project)
        self.actExport.triggered.connect(self.export_results)
        self.actExportChart.triggered.connect(self.export_chart)
        self.actBatch.triggered.connect(self.run_batch)
        self.actExit.triggered.connect(self.close)
        self.actRunAll.triggered.connect(self.run_all)
        self.actNext.triggered.connect(self.go_next_step)
        self.actReprocess.triggered.connect(self.reprocess)
        self.actSettings.triggered.connect(self.open_settings)
        self.actAbout.triggered.connect(self.show_about)
        self.actOpenProject.triggered.connect(self.open_project)
        self.actResetLayout.triggered.connect(self.reset_layout)

        # 界面主题（配色风格可设置，持久化）
        self._theme = self.settings.value("theme", "明亮")
        for _name, _act in self.themeActs.items():
            _act.setChecked(_name == self._theme)
            _act.triggered.connect(lambda _c=False, n=_name: self._apply_theme(n))
        self.actExportChart.triggered.connect(self.export_chart)
        self.actExportData.triggered.connect(self.export_results)
        self.actExportParticles.triggered.connect(self.export_particles)
        self.actExportLog.triggered.connect(self.export_log)

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

    # ============================
    # 当前步骤更新
    # ============================
    def _update_step_indicator(self):
        """根据 current_step 更新工具栏步骤指示器与当前步骤标签。"""
        self.stepIndicator.current_step = self.current_step
        self.stepIndicator.update_ui()
        step_names = {
            -1: "未开始",
            0: "图像预处理完成",
            1: "集料粒度提取完成",
            2: "结果分析与统计完成",
        }
        self.lblCurrentStep.setText(f"当前步骤：{step_names.get(self.current_step, '未开始')}")

    def _clear_step_data(self, from_step=0):
        """清空指定步骤及之后的所有数据。"""
        if from_step <= 0:
            self._pre_imgs = []
            self._pre_names = []
        if from_step <= 1:
            self.binary = None
            self.markers = None
            self._color_img = None
            self.resultImage.clear()
            self.resultImage.setParticleData(None, None, None)
        if from_step <= 2:
            self.particles = None
            self.df = None
            self._overlay_img = None
        self.table.setRowCount(0)
        self.figure.clear()
        self.canvas.draw()
        self.histFigure.clear()
        self.histCanvas.draw()
        self._initTableRanges()
        self._refresh_proc_views()

    def _refresh_proc_views(self, select_last=False):
        """按当前数据重建主视图窗图层（预处理/二值/实例/彩色图/标记图，原图在左侧）。"""
        imgs, names = [], []
        imgs.extend(self._pre_imgs)
        names.extend(self._pre_names)
        if self.binary is not None:
            imgs.append(self.binary)
            names.append("二值掩膜")
        if self.markers is not None:
            imgs.append(self.markers)
            names.append("实例标签")
            if self._color_img is not None:
                imgs.append(self._color_img)
                names.append("颗粒彩色图")
        if self._overlay_img is not None:
            imgs.append(self._overlay_img)
            names.append("级配标记图")
        self.setLayers(imgs, names)
        if imgs and select_last:
            self.resultImage.setCurrentIndex(len(imgs) - 1)

    # ============================
    # 打开图像（仅预处理，步骤0）
    # ============================
    def open_image(self):
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

    def _load_image(self, filename):
        """加载图像（兼容中文路径），成功后重置处理流程。"""
        data = np.fromfile(filename, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if img is None:
            self.appendLog(f"图像读取失败: {filename}", "error")
            QMessageBox.critical(self, "读取失败", "无法读取该图像，文件可能已损坏或格式不支持。")
            return

        self._last_dir = os.path.dirname(filename)
        self.img_path = filename
        self.img = img
        self.setOrigImage(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.logText.clear()
        self.appendLog(f"图像加载完成: {os.path.basename(filename)}", "success")

        # 重置所有步骤
        self.current_step = -1
        self._clear_step_data(0)
        self._update_step_indicator()
        self._refresh_proc_views()
        self.statusBar().showMessage(f"已加载: {filename}")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(IMG_EXTS):
                self._load_image(path)
                return

    def _do_step0_preprocess(self):
        """步骤0：图像预处理。"""
        t0 = time.time()
        h, w = self.img.shape[:2]
        self.setImageInfo(w, h, step="图像预处理完成")

        # ---- 预处理 ----
        img_list, name_list, info1 = preprocess(self.img)
        self._pre_imgs = list(img_list)
        self._pre_names = [f"预处理-{n}" for n in name_list]
        self._refresh_proc_views()
        self.appendLog(info1)

        elapsed = f"{time.time() - t0:.2f} s"
        self.lblProcessTime.setText(f"处理时间：{elapsed}")
        self.appendLog(f"预处理完成，耗时 {elapsed}")

        self.current_step = 0
        self._update_step_indicator()

    # ============================
    # 步骤1：分割
    # ============================
    def _do_step1_segmentation(self):
        """步骤1：颗粒分割（异步执行）。"""
        if self.img is None:
            self.appendLog("请先加载图像")
            return

        # 如果正在处理中，不允许重复启动
        if self._worker is not None and self._worker.isRunning():
            self.appendLog("正在处理中，请稍候...")
            return

        t0 = time.time()
        self.appendLog("开始颗粒分割...")

        p = self.seg_params
        seg_type = p["seg_type"]

        # 禁用按钮防止重复操作
        self._set_buttons_enabled(False)

        # 创建并启动工作线程
        self._worker = SegmentationWorker(self.img.copy(), seg_type, p)
        self._worker.finished.connect(lambda binary, markers: self._on_segmentation_finished(binary, markers, t0))
        self._worker.error.connect(self._on_segmentation_error)
        self._worker.progress.connect(self.appendLog)
        self._worker.start()

    def _on_segmentation_finished(self, binary, markers, t0):
        """分割完成回调（主线程）。"""
        self._set_buttons_enabled(True)
        self.binary = binary
        self.markers = markers

        p = self.seg_params
        seg_type = p["seg_type"]
        elapsed = f"{time.time() - t0:.2f} s"
        self.appendLog(f"分割完成（类型: {seg_type}）", "success")
        self.appendLog(f"分割耗时 {elapsed}")
        self.lblProcessTime.setText(f"处理时间：{elapsed}")
        self.statusBar().showMessage(f"分割完成，耗时 {elapsed}")

        if self.markers is None:
            self.appendLog("分割失败：未生成标记图")
            return

        color = colorize_particles(self.markers)
        self._color_img = color
        self._refresh_proc_views()
        self.resultImage.setParticleData(self.markers, None, color)

        self.current_step = 1
        self._update_step_indicator()

        # 继续执行剩余步骤（一键执行模式）
        self._run_remaining_steps()

    def _on_segmentation_error(self, error_msg):
        """分割错误回调（主线程）。"""
        self._set_buttons_enabled(True)
        self.appendLog(f"分割出错: {error_msg}", "error")
        self.statusBar().showMessage("分割出错")
        QMessageBox.critical(self, "分割出错", f"分割过程中发生错误：\n{error_msg}")

    def _set_buttons_enabled(self, enabled):
        """启用/禁用操作动作。"""
        self.actNext.setEnabled(enabled)
        self.actRunAll.setEnabled(enabled)
        self.actReprocess.setEnabled(enabled)
        self.actOpenImage.setEnabled(enabled)
        if not enabled:
            self.actNext.setText("处理中...")
            self.actRunAll.setText("处理中...")
        else:
            self.actNext.setText("下一步")
            self.actRunAll.setText("一键执行")

    # ============================
    # 步骤2：分析与统计
    # ============================
    def _do_step2_analysis(self):
        """步骤2：尺寸统计与级配计算。"""
        if self.markers is None:
            self.appendLog("请先完成分割步骤")
            return
        self._overlay_img = overlay_markers(self.img, self.markers)
        self._refresh_proc_views(select_last=True)

        t0 = time.time()
        p = self.seg_params

        # ---- 尺寸统计 ----
        self.particles = measure_particles(
            self.markers,
            pixel_size=p["pixel_size"],
            min_area=p["min_area"],
        )
        self.appendLog(f"共识别 {len(self.particles)} 个颗粒")

        # ---- 级配 ----
        self.df = calculate_grading(self.particles)
        if self.df is not None:
            self.updateTable(self.df)
            self.updateCurve(self.df)
            d10 = calc_dxx(self.df, 10)
            d50 = calc_dxx(self.df, 50)
            d90 = calc_dxx(self.df, 90)
            self.updateHistogram(self.df)
            self.appendLog(f"特征粒径: D10={d10} mm, D50={d50} mm, D90={d90} mm")
            self.appendLog("级配计算完成", "success")

        elapsed = f"{time.time() - t0:.2f} s"
        self.appendLog(f"分析耗时 {elapsed}")

        self.current_step = 2
        self._update_step_indicator()

    # ============================
    # 下一步 / 一键执行
    # ============================
    def go_next_step(self):
        """执行下一个待处理步骤。"""
        if self.img is None and self.current_step < 0:
            self.appendLog("请先打开图像")
            return

        next_step = self.current_step + 1
        if next_step == 0:
            if self.img_path:
                self._do_step0_preprocess()
            else:
                self.appendLog("请先打开图像")
        elif next_step == 1:
            self._do_step1_segmentation()
        elif next_step == 2:
            self._do_step2_analysis()
        else:
            self.appendLog("所有步骤已完成")

    def run_all(self):
        """一键执行所有剩余步骤。"""
        if self.img is None and self.img_path is None:
            self.appendLog("请先打开图像")
            return

        # 如果正在处理中，不允许重复启动
        if self._worker is not None and self._worker.isRunning():
            self.appendLog("正在处理中，请稍候...")
            return

        # 如果还没加载图像，先打开
        if self.current_step < 0 and self.img is None:
            if self.img_path:
                self._do_step0_preprocess()
            else:
                self.appendLog("请先打开图像")
                return

        self._run_remaining_steps()

    def _run_remaining_steps(self):
        """依次执行剩余步骤，遇到异步步骤则启动后等待回调继续。"""
        while self.current_step < 2:
            next_step = self.current_step + 1
            if next_step == 0:
                self._do_step0_preprocess()
            elif next_step == 1:
                # 分割是异步的，启动后退出循环，由回调继续
                self._do_step1_segmentation()
                return
            elif next_step == 2:
                self._do_step2_analysis()
            else:
                break

        self.appendLog("一键执行完成")

    # ============================
    # 表格更新
    # ============================
    def updateTable(self, df):
        # 构建粒径范围列
        ranges = []
        for i, sieve in enumerate(SIEVE_SIZE):
            if i == 0:
                ranges.append(f"> {sieve}")
            else:
                ranges.append(f"{SIEVE_SIZE[i - 1]} - {sieve}")
        ranges.append(f"< {SIEVE_SIZE[-1]}")

        n = len(df)
        # 数据行 + 最细粒级行 + 总计行
        self.table.setRowCount(n + 2)

        keys = ["分计筛余(%)", "累计筛余(%)", "累计通过率(%)", "数量占比(%)"]
        for i, row in df.iterrows():
            item0 = QTableWidgetItem(ranges[i] if i < len(ranges) else "")
            item0.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 0, item0)
            for col, key in enumerate(keys, start=1):
                item = QTableWidgetItem(str(round(row[key], 2)))
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, col, item)

        # 最细粒级行（小于最小筛孔部分）
        fine_row = n
        fine_mass = 100 - float(df["累计筛余(%)"].iloc[-1])
        fine_count = 100 - float(df["数量占比(%)"].sum())
        fine_pass = float(df["累计通过率(%)"].iloc[-1])
        fine_vals = [
            ranges[n] if n < len(ranges) else "",
            f"{fine_mass:.2f}",
            "-",
            f"{fine_pass:.2f}",
            f"{fine_count:.2f}",
        ]
        for col, text in enumerate(fine_vals):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter if col == 0 else (Qt.AlignRight | Qt.AlignVCenter))
            self.table.setItem(fine_row, col, item)

        # 总计行
        total_row = n + 1
        for col, text in enumerate(["总计", "100", "-", "-", "100"]):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter if col == 0 else (Qt.AlignRight | Qt.AlignVCenter))
            self.table.setItem(total_row, col, item)

        # 加粗总计行
        for j in range(self.table.columnCount()):
            item = self.table.item(total_row, j)
            if item:
                font = item.font()
                font.setBold(True)
                item.setFont(font)

        # 列宽按内容自适应，保证表头与数据文字完整显示
        self.table.resizeColumnsToContents()

    # ============================
    # 曲线更新
    # ============================
    def updateCurve(self, df):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ax.semilogx(
            df["筛孔(mm)"],
            df["累计通过率(%)"],
            marker="o",
            color="#2196F3",
            linewidth=2,
            markersize=6,
        )

        # 特征粒径参考线 D10/D50/D90
        for pct, color in ((10, "#E91E63"), (50, "#FF9800"), (90, "#9C27B0")):
            dxx = calc_dxx(df, pct)
            if isinstance(dxx, float):
                ax.axvline(dxx, color=color, linestyle="--", linewidth=1, alpha=0.8)
                ax.text(dxx, pct + 4, f"D{pct}={dxx:g}", color=color,
                        fontsize=self._chart_fs(-2), ha="center")

        ax.set_xlabel("粒径 (mm)", fontsize=self._chart_fs())
        ax.set_ylabel("通过率 (%)", fontsize=self._chart_fs())
        # 横坐标范围根据筛孔尺寸自适应，由大到小显示，仅保留标准筛孔主刻度
        lo, hi = float(min(SIEVE_SIZE)) * 0.7, float(max(SIEVE_SIZE)) * 1.3
        ax.set_xlim(lo, hi)
        ax.invert_xaxis()
        ax.minorticks_off()
        ticks = [s for s in SIEVE_SIZE if lo <= s <= hi]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        # 纵轴固定 0~100，每 20 一格；仅主刻度网格，避免坐标杂乱
        ax.set_ylim(-2, 102)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(True, which="major", axis="both", linestyle="--", alpha=0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        # 标准级配范围上下限叠加显示
        self._draw_spec_envelope(ax, lo, hi)

        # 缓存数据点用于鼠标悬停交互
        self._curve_ax = ax
        self._curve_points = (
            np.asarray(df["筛孔(mm)"], dtype=float),
            np.asarray(df["累计通过率(%)"], dtype=float),
            np.asarray(df["累计筛余(%)"], dtype=float),
        )
        self._curve_annot = ax.annotate(
            "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#b0b0b0", alpha=0.95),
            fontsize=self._chart_fs(-2),
        )
        self._curve_annot.set_visible(False)

        self.figure.tight_layout()
        self.canvas.draw()

    # ============================
    # 级配曲线：标准范围叠加与鼠标交互
    # ============================
    def _draw_spec_envelope(self, ax, lo, hi):
        """在当前坐标轴上叠加所选标准级配的通过率上下限区间。"""
        pts = SPEC_RANGES.get(self.specCombo.currentText())
        if not pts:
            return
        pts = sorted(pts, key=lambda t: t[0])
        xs = np.log10([p[0] for p in pts])
        lows = np.array([p[1] for p in pts], dtype=float)
        highs = np.array([p[2] for p in pts], dtype=float)
        grid = np.linspace(np.log10(lo), np.log10(hi), 200)
        gx = 10 ** grid
        low_y = np.interp(grid, xs, lows)
        high_y = np.interp(grid, xs, highs)
        ax.plot(gx, low_y, color="#8BC34A", linewidth=1.2, linestyle="--",
                label=f"{self.specCombo.currentText()} 下限")
        ax.plot(gx, high_y, color="#8BC34A", linewidth=1.2, linestyle="--",
                label=f"{self.specCombo.currentText()} 上限")
        ax.fill_between(gx, low_y, high_y, color="#8BC34A", alpha=0.12)
        # ax.legend(loc="lower left", fontsize=max(FontSize.lv(6) - 2, 7),
        #           framealpha=0.8)

    def _on_spec_changed(self, _idx):
        """切换标准级配范围：持久化并重绘曲线。"""
        self.settings.setValue("spec_range", self.specCombo.currentText())
        if self.df is not None:
            self.updateCurve(self.df)

    def _on_curve_hover(self, event):
        """级配曲线鼠标悬停：显示最近数据点的筛孔/通过率/筛余信息。"""
        annot = self._curve_annot
        if annot is None:
            return
        if event.inaxes is not self._curve_ax or self._curve_points is None:
            if annot.get_visible():
                annot.set_visible(False)
                self.canvas.draw_idle()
            return
        xs, ys, retains = self._curve_points
        disp = self._curve_ax.transData.transform(np.column_stack([xs, ys]))
        d = np.hypot(disp[:, 0] - event.x, disp[:, 1] - event.y)
        i = int(np.argmin(d))
        if d[i] <= 28:
            annot.xy = (xs[i], ys[i])
            annot.set_text(
                f"筛孔 {xs[i]:g} mm\n累计通过率 {ys[i]:.1f}%\n累计筛余 {retains[i]:.1f}%"
            )
            self._adjust_hover_annot(annot)
            annot.set_visible(True)
        else:
            annot.set_visible(False)
        self.canvas.draw_idle()

    def _adjust_hover_annot(self, annot):
        """自适应调整悬停提示框位置，避免溢出坐标轴/面板外。"""
        renderer = self.canvas.get_renderer()
        ax_bbox = self._curve_ax.get_window_extent(renderer)
        ox, oy = 12, 12
        annot.set_position((ox, oy))
        bbox = annot.get_window_extent(renderer)
        if bbox.x1 > ax_bbox.x1:  # 右侧溢出 → 移到数据点左侧
            ox = -(bbox.width + 12)
        if bbox.y1 > ax_bbox.y1:  # 上方溢出 → 移到数据点下方
            oy = -(bbox.height + 12)
        annot.set_position((ox, oy))
        bbox = annot.get_window_extent(renderer)
        if bbox.x0 < ax_bbox.x0:  # 左侧溢出 → 移回右侧
            ox = 12
        if bbox.y0 < ax_bbox.y0:  # 下方溢出 → 移回上方
            oy = 12
        annot.set_position((ox, oy))

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

    # ============================
    # 其他操作
    # ============================
    def reprocess(self):
        """重新处理：清空后续步骤数据，从当前图像重新执行预处理。"""
        if self.img is None or self.img_path is None:
            self.appendLog("没有可重新处理的图像", "warn")
            return

        self.appendLog("重新处理...")
        # 清空所有步骤数据
        self.current_step = -1
        self._clear_step_data(0)
        self.logText.clear()
        self.appendLog("已清空所有处理结果")
        self._update_step_indicator()

        # 重新执行预处理
        self._do_step0_preprocess()

    def open_settings(self):
        """统一设置对话框：算法参数 + 系统与推理。"""
        dlg = UnifiedSettingsDialog(self.seg_params, segmentation.get_config(), self)
        if dlg.exec_() != UnifiedSettingsDialog.Accepted:
            return
        new_params = dlg.get_params()

        # 字体大小
        font_preset = new_params.pop("font_preset", None)
        if font_preset and font_preset != FontSize.PRESET:
            FontSize.set_preset(font_preset)
            self.appendLog(f"字体大小切换为: {FontSize.preset_name()}")
            self._refresh_all_fonts()

        # 应用并持久化推理设置（计算设备/模型权重）
        cfg = segmentation.configure(
            device=new_params.pop("device", None),
            unet_model_path=new_params.pop("unet_model_path", ""),
            sam_checkpoint=new_params.pop("sam_checkpoint", ""),
            yolo_weights=new_params.pop("yolo_weights", ""),
        )
        self.settings.setValue("device_pref", cfg["device"])
        self.settings.setValue("unet_model_path", cfg["unet_model_path"])
        self.settings.setValue("sam_checkpoint", cfg["sam_checkpoint"])
        self.settings.setValue("yolo_weights", cfg["yolo_weights"])
        self.appendLog(f"推理配置已更新: 设备={cfg['device']}")

        # 分割/测量参数（其余键均为分割参数）
        self.seg_params.update(new_params)
        spec = model_registry.get_model(new_params["seg_type"])
        self.appendLog(f"参数已更新: 分割模型={spec.label if spec else new_params['seg_type']}")
        # 如果分割已完成，参数变更需要重新分割
        if self.current_step >= 1:
            self.appendLog("参数已更改，请重新执行分割步骤")
            self.current_step = 0
            self._clear_step_data(1)
            self._update_step_indicator()

    def show_about(self):
        QMessageBox.about(
            self, "关于",
            "2PSL 混凝土骨料颗粒智能筛分系统\n\n"
            "支持传统分水岭 / UNet / SAM / YOLO 分割模型，\n"
            "12 项形态指标、D10/D50/D90 特征粒径与级配分析。",
        )

    def _refresh_all_fonts(self):
        """重新应用所有字体样式：重建全局 QSS、步骤指示器，并同步日志/统计表/页签/图表。
        注：日志/表格/页签设的是控件级样式表（含 Ctrl+滚轮记忆值），优先级高于
        全局 QSS，档位切换时必须按新档位重置并重新应用，否则不会变化。
        """
        self.stepIndicator.init_ui()
        self._log_font_off = 0
        self._table_font_off = 0
        self._chart_font_off = 0
        self._apply_log_font()
        self._apply_table_font()
        self.settings.setValue("chart_font_off", 0)
        self._apply_tab_font()
        QApplication.instance().setStyleSheet(build_app_stylesheet(FontSize, self._theme))
        if self.df is not None:
            self.updateCurve(self.df)
            self.updateHistogram(self.df)

    # ============================
    # 导出 / 保存 / 批量处理
    # ============================
    def _grading_table_rows(self):
        """从表格收集当前显示的数据行。"""
        rows = []
        for r in range(self.table.rowCount()):
            row_data = []
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                row_data.append(item.text() if item else "")
            rows.append(row_data)
        return rows

    def _save_grading_file(self, path):
        """按扩展名保存级配数据为 xlsx 或 csv；xlsx 附带颗粒详情工作表。"""
        header = ["粒径范围 (mm)", "分计筛余 (%)", "累计筛余 (%)", "累计通过率 (%)", "数量占比 (%)"]
        df = pd.DataFrame(self._grading_table_rows(), columns=header)
        if path.lower().endswith(".xlsx"):
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="级配结果", index=False)
                details = self._particles_dataframe()
                if details is not None:
                    details.to_excel(writer, sheet_name="颗粒详情", index=False)
        else:
            df.to_csv(path, index=False, encoding="utf-8-sig")

    def _particles_dataframe(self):
        """将颗粒详情（含 12 项形态指标）整理为中文列名 DataFrame。"""
        if not self.particles:
            return None
        df = pd.DataFrame(self.particles)
        df = df.rename(columns=_PARTICLE_CN)
        cols = [v for v in _PARTICLE_CN.values() if v in df.columns]
        return df[cols]

    def _on_particle_clicked(self, pid):
        """点击结果图中的颗粒时显示其形态参数。"""
        if not self.particles:
            return
        p = next((pt for pt in self.particles if pt["id"] == pid), None)
        if p is None:
            return
        self.resultImage.highlightParticle(pid)
        info = (
            f"颗粒 #{pid}: 等效粒径 {p['diameter_mm']:.2f} mm | 面积 {p['area_mm2']:.1f} mm² | "
            f"圆形度 {p['circularity']:.2f} | Feret长宽比 {p['feret_ratio']:.2f} | 棱角性指数 {p['angularity']:.2f}"
        )
        self.statusBar().showMessage(info)
        self.appendLog(info)

    def export_results(self):
        """导出级配结果到 Excel/CSV。"""
        if self.df is None:
            QMessageBox.information(self, "提示", "暂无结果可导出，请先完成分析。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出结果", self._last_dir, "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        if not path:
            return
        self._save_grading_file(path)
        self.appendLog(f"结果已导出: {path}", "success")

    def export_chart(self):
        if self.df is None:
            QMessageBox.information(self, "提示", "暂无图表可导出，请先完成分析。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出图表", self._last_dir, "PNG (*.png);;PDF (*.pdf)"
        )
        if path:
            self.figure.savefig(path, dpi=150, bbox_inches="tight")
            self.appendLog(f"图表已导出: {path}", "success")

    def export_log(self):
        """导出处理日志为文本文件。"""
        text = self.logText.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "提示", "日志为空，无需导出。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出日志", os.path.join(self._last_dir, "处理日志.txt"),
            "Text Files (*.txt)")
        if not path:
            return
        from datetime import datetime
        header = f"2PSL 处理日志  导出时间: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + text + "\n")
        self.appendLog(f"日志已导出: {path}", "success")

    def export_particles(self):
        """独立导出颗粒详情（12项形态指标）。"""
        details = self._particles_dataframe()
        if details is None or len(details) == 0:
            QMessageBox.information(self, "提示", "暂无颗粒数据可导出，请先完成分析。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出颗粒详情", os.path.join(self._last_dir, "颗粒详情.xlsx"),
            "Excel Files (*.xlsx);;CSV Files (*.csv)")
        if not path:
            return
        if path.lower().endswith(".xlsx"):
            details.to_excel(path, index=False, sheet_name="颗粒详情")
        else:
            details.to_csv(path, index=False, encoding="utf-8-sig")
        self.appendLog(f"颗粒详情已导出: {path}", "success")

    def open_project(self):
        """打开项目文件：恢复图像、分割参数与级配结果。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "打开项目", self._last_dir, "Project Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                project = json.load(f)
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, "打开失败", f"无法读取项目文件：\n{e}")
            return

        # 恢复分割参数
        saved_params = project.get("params") or {}
        for k in self.seg_params:
            if k in saved_params:
                self.seg_params[k] = saved_params[k]

        # 恢复图像（会重置处理流程）
        img_path = project.get("image")
        if img_path and os.path.isfile(img_path):
            self._load_image(img_path)
        else:
            self.appendLog(f"项目中的图像不存在: {img_path}", "warn")

        # 恢复级配结果与图表
        grading = project.get("grading")
        if grading:
            self.df = pd.DataFrame(grading)
            self.updateTable(self.df)
            self.updateCurve(self.df)
            self.updateHistogram(self.df)
            self.current_step = int(project.get("current_step", 2))
            self._update_step_indicator()
        self.appendLog(f"项目已打开: {path}", "success")

    def reset_layout(self):
        """重置窗口布局及全部面板调整（分割器/字体/第二行工具栏）为初始状态。"""
        self.restoreState(self._initial_state)
        self._redock_all()
        self._ensure_export_break()
        # 分割器恢复默认比例（项目目录/原图各半，图表/统计表各半）
        self.rightSplitter.setStretchFactor(0, 1)
        self.rightSplitter.setStretchFactor(1, 1)
        # self.rightSplitter.setSizes([1, 1])
        self.leftSplitter.setStretchFactor(0, 1)
        self.leftSplitter.setStretchFactor(1, 0)
        self.leftSplitter.setStretchFactor(2, 0)
        # 信息栏拉伸系数为 0，尺寸按比例分配但不可为 0，否则会被压没且无法自行恢复
        # self.leftSplitter.setSizes([6, 3, 1])
        # 清除旧的布局记忆，重置立即生效（退出时会以新布局重新保存）
        self.settings.remove("window_state")
        self.settings.remove("right_splitter")
        self.settings.remove("left_splitter")
        # 字体调整恢复为默认档位值（各控件字号 = lv() 层级字号，偏移归零）
        self._log_font_off = 0
        self._table_font_off = 0
        self._chart_font_off = 0
        self._apply_log_font()
        self._apply_table_font()
        self._apply_tab_font()
        self.settings.setValue("chart_font_off", 0)
        if self.df is not None:
            self.updateCurve(self.df)
            self.updateHistogram(self.df)
        # 第二行导出按钮栏：按初始布局恢复可见性，并同步菜单勾选
        self._set_export_bar_visible(self.tbExport.isVisible())
        self.appendLog("窗口布局与面板调整已全部重置")

    def _apply_theme(self, theme):
        """切换界面主题配色并持久化。"""
        self._theme = theme
        self.settings.setValue("theme", theme)
        for name, act in self.themeActs.items():
            act.setChecked(name == theme)
        QApplication.instance().setStyleSheet(build_app_stylesheet(FontSize, theme))
        self.appendLog(f"界面主题已切换: {theme}")

    def _apply_log_font(self, persist=True):
        """应用日志字体（正文层 lv(3) + Ctrl+滚轮偏移；控件级样式，优先于全局 QSS）。"""
        self.logText.setStyleSheet(f"font-size: {self._log_fs()}px;")
        if persist:
            self.settings.setValue("log_font_px", self._log_font_off)

    def _apply_table_font(self, persist=True):
        """应用统计表字体大小（表头同步，保持比表体大 2px 的档位差）。
        注：表头样式直接设在 QHeaderView 自身上——全局样式用了带 #resultTable
        ID 选择器的 QHeaderView::section 规则，优先级高于设在表格上的规则，
        只有控件自身样式表才能覆盖它。
        """
        body_px = self._table_fs()
        head_px = body_px + 2
        self.table.setStyleSheet(f"font-size: {body_px}px;")
        self.table.horizontalHeader().setStyleSheet(
            f"QHeaderView::section {{ font-size: {head_px}px; }}")
        # 字号变化后重新按内容适配列宽，避免表头文字被裁切
        self.table.resizeColumnsToContents()
        if persist:
            self.settings.setValue("table_font_px", self._table_font_off)

    def _log_fs(self):
        """日志字号 = 正文层 lv(3) + Ctrl+滚轮偏移（保底不低于 6px）。"""
        return max(FontSize.lv(4) + self._log_font_off, 6)

    def _table_fs(self):
        """统计表字号 = 正文层 lv(3) + Ctrl+滚轮偏移（保底不低于 6px）。"""
        return max(FontSize.lv(6) + self._table_font_off, 6)

    def _chart_fs(self, delta=0):
        """图表字号 = 图表层 lv(6) + Ctrl+滚轮偏移（保底不低于 5px）。"""
        return max(FontSize.lv(4) + self._chart_font_off + delta, 5)

    def _apply_tab_font(self):
        """页签文字跟随图表字体缩放（widget 级样式，优先于全局 QTabBar::tab 规则）。
        顺序不能颠倒：必须先设样式表并 polish，最后 setFont——样式 polish 会按
        全局样式表重新解析控件字体，把先设的字体覆盖回旧字号，导致页签宽度按
        旧字形度量计算、文字被裁切。另禁掉页签文字省略，空间不足时用滚动按钮。
        """
        px = self._chart_fs()
        tb = self.chartTabs.tabBar()
        tb.setElideMode(Qt.ElideNone)
        tb.setUsesScrollButtons(True)
        tb.setStyleSheet(f"QTabBar::tab {{ font-size: {px}px; }}")
        tb.style().polish(tb)
        f = tb.font()
        f.setPixelSize(px)
        tb.setFont(f)
        for i in range(self.chartTabs.count()):
            self.chartTabs.setTabText(i, self.chartTabs.tabText(i))
        tb.updateGeometry()
        tb.update()

    def eventFilter(self, obj, event):
        """Ctrl+滚轮调整日志/统计表/图表字体；双击工具栏扩展按钮切换两行工具栏。"""
        etype = event.type()
        if etype == QEvent.Wheel and (event.modifiers() & Qt.ControlModifier):
            step = 1 if event.angleDelta().y() > 0 else -1
            if obj is getattr(self, "_log_vp", None):
                self._log_font_off = max(-4, min(8, self._log_font_off + step))
                self._apply_log_font()
                self.statusBar().showMessage(f"日志字体: {self._log_fs()}px", 1500)
                return True
            if obj is getattr(self, "_table_vp", None):
                self._table_font_off = max(-4, min(8, self._table_font_off + step))
                self._apply_table_font()
                self.statusBar().showMessage(f"统计表字体: {self._table_fs()}px", 1500)
                return True
            if obj in (getattr(self, "canvas", None), getattr(self, "histCanvas", None)) \
                    or obj is self.chartTabs.tabBar():
                self._chart_font_off = max(-4, min(8, self._chart_font_off + step))
                self.settings.setValue("chart_font_off", self._chart_font_off)
                self._apply_tab_font()
                if self.df is not None:
                    self.updateCurve(self.df)
                    self.updateHistogram(self.df)
                self.statusBar().showMessage(
                    f"图表字体: {self._chart_fs()}px", 1500)
                return True
        if etype == QEvent.MouseButtonDblClick:
            name = obj.objectName() if hasattr(obj, "objectName") else ""
            if name in ("qt_toolbar_ext_button", "toolbarSpacer") \
                    or obj is getattr(self, "mainToolbar", None):
                self._toggle_export_bar()
                return True
        return super().eventFilter(obj, event)

    def _toggle_export_bar(self):
        """切换第二行导出按钮栏的显示/隐藏（按钮始终留在第二行，不并入第一行）。"""
        self._set_export_bar_visible(not self.tbExport.isVisible())
        self.appendLog("导出按钮栏" + ("已显示（工具栏第二行）" if self.tbExport.isVisible() else "已隐藏"))

    def _set_export_bar_visible(self, visible):
        """显示/隐藏第二行导出按钮栏，并静默同步菜单勾选状态。"""
        self.tbExport.setVisible(visible)
        self.actShowExportBar.blockSignals(True)
        self.actShowExportBar.setChecked(visible)
        self.actShowExportBar.blockSignals(False)

    def _on_log_context_menu(self, pos):
        """日志窗右键菜单：清空日志 / 导出日志。"""
        menu = QMenu(self)
        act_clear = menu.addAction("清空日志")
        act_export = menu.addAction("导出日志")
        act = menu.exec_(self.logText.mapToGlobal(pos))
        if act == act_clear:
            self.logText.clear()
            self.appendLog("日志已清空")
        elif act == act_export:
            self.export_log()

    def _on_analysis_context_menu(self, pos):
        """分析面板右键菜单：导出结果 / 导出颗粒 / 导出图表 / 智能分析。"""
        menu = QMenu(self)
        act_results = menu.addAction("导出结果")
        act_particles = menu.addAction("导出颗粒")
        act_chart = menu.addAction("导出图表")
        menu.addSeparator()
        act_ai = menu.addAction("智能分析")
        act = menu.exec_(self.analysisPanel.mapToGlobal(pos))
        if act == act_results:
            self.export_results()
        elif act == act_particles:
            self.export_particles()
        elif act == act_chart:
            self.export_chart()
        elif act == act_ai:
            self.ai_analysis()

    def ai_analysis(self):
        """智能分析：级配特征、均匀性/曲率系数、细度模数、标准符合性与颗粒形态评价。"""
        if self.df is None or len(self.df) == 0:
            QMessageBox.information(self, "提示", "暂无级配数据，请先完成分析。")
            return
        lines = ["【智能分析报告】"]

        sizes = self.df["筛孔(mm)"].to_numpy(dtype=float)
        passing = self.df["累计通过率(%)"].to_numpy(dtype=float)

        def _fmt(v):
            return f"{v:.3f} mm" if isinstance(v, float) else str(v)

        d10 = calc_dxx(self.df, 10)
        d30 = calc_dxx(self.df, 30)
        d50 = calc_dxx(self.df, 50)
        d60 = calc_dxx(self.df, 60)
        d90 = calc_dxx(self.df, 90)
        lines.append(f"一、特征粒径: D10={_fmt(d10)}, D30={_fmt(d30)}, D50={_fmt(d50)}, D60={_fmt(d60)}, D90={_fmt(d90)}")

        if isinstance(d60, float) and isinstance(d10, float) and d10 > 0:
            cu = d60 / d10
            cc_desc = "无法计算"
            if isinstance(d30, float):
                cc = d30 ** 2 / (d60 * d10)
                cc_desc = f"{cc:.2f}"
            lines.append(f"二、均匀性系数 Cu=D60/D10={cu:.2f}，曲率系数 Cc={cc_desc}")
            if cu < 5:
                lines.append("    → Cu 较小，颗粒粒径较均匀，接近单粒级")
            else:
                lines.append("    → Cu 较大，粒径分布较广，有利于密实堆积")
            if cc_desc != "无法计算":
                cc = d30 ** 2 / (d60 * d10)
                lines.append("    → 级配曲线连续平滑，属良好级配" if 1 <= cc <= 3 else
                             "    → 曲率系数偏离 1~3，中间粒级偏多或存在断档")
        else:
            lines.append("二、通过率范围不足以计算均匀性系数")

        cum_retain = self.df["累计筛余(%)"].to_numpy(dtype=float)
        fm = float(np.sum(cum_retain[sizes >= 0.15]) / 100.0)
        if fm > 3.5:
            fm_cat = "偏粗"
        elif fm >= 2.3:
            fm_cat = "中等"
        else:
            fm_cat = "偏细"
        lines.append(f"三、细度模数 Mx≈{fm:.2f}（{fm_cat}）")

        gaps = [f"{sizes[i]:g}~{sizes[i + 1]:g}" for i in range(len(sizes) - 1)
                if abs(float(passing[i] - passing[i + 1])) < 3]
        if gaps:
            lines.append("四、级配类型: 存在明显断档区间（" + "、".join(gaps) + " mm），偏向间断级配/单粒级")
        else:
            lines.append("四、级配类型: 各粒级过渡平缓，接近连续级配")

        spec = SPEC_RANGES.get(self.specCombo.currentText())
        if spec:
            ok_all = True
            detail = []
            for s, lo, hi in spec:
                idx = int(np.argmin(np.abs(sizes - s)))
                if abs(float(sizes[idx] - s)) > 1e-6:
                    continue
                p = float(passing[idx])
                ok = lo <= p <= hi
                ok_all = ok_all and ok
                detail.append(f"    筛孔 {s:g} mm: 通过率 {p:.1f}%（要求 {lo}~{hi}%）→ {'合格' if ok else '超限'}")
            lines.append(f"五、标准级配符合性（{self.specCombo.currentText()}）:")
            lines.extend(detail)
            lines.append("    结论: 全部指标落在标准范围内，满足该级配要求" if ok_all else
                         "    结论: 存在超限筛孔，不满足该标准级配，建议调整骨料配比")
        else:
            lines.append("五、未选择标准级配范围，跳过符合性对比（可在右侧下拉框选择后重试）")

        if self.particles:
            ps = pd.DataFrame(self.particles)
            n = len(ps)
            d_mean = float(ps["diameter_mm"].mean())
            parts = [f"六、颗粒形态: 共 {n} 颗，等效粒径均值 {d_mean:.2f} mm"]
            if "roundness" in ps.columns:
                r_mean = float(ps["roundness"].mean())
                parts.append(f"圆整度均值 {r_mean:.2f}（{'偏圆滑' if r_mean > 0.7 else '偏棱角'}）")
            if "angularity" in ps.columns:
                a_mean = float(ps["angularity"].mean())
                parts.append(f"棱角性指数均值 {a_mean:.2f}")
            lines.append("，".join(parts))
            lines.append("    建议: 棱角逐多时宜适当提高浆体用量以保证工作性")

        report = "\n".join(lines)
        self.appendLog("智能分析完成", "success")

        dlg = QDialog(self)
        dlg.setWindowTitle("智能分析")
        dlg.resize(620, 460)
        v = QVBoxLayout(dlg)
        view = QTextEdit(dlg)
        view.setReadOnly(True)
        view.setPlainText(report)
        btn = QPushButton("关闭")
        btn.setObjectName("flatBtn")
        btn.clicked.connect(dlg.accept)
        v.addWidget(view)
        v.addWidget(btn, 0, Qt.AlignRight)
        dlg.exec_()

    def updateHistogram(self, df):
        """绘制粒径分布直方图（各筛孔区间的数量占比）。"""
        self.histFigure.clear()
        ax = self.histFigure.add_subplot(111)

        ranges = []
        for i, sieve in enumerate(SIEVE_SIZE):
            ranges.append(f"> {sieve:g}" if i == 0 else f"{SIEVE_SIZE[i - 1]:g}-{sieve:g}")
        ranges.append(f"< {SIEVE_SIZE[-1]:g}")

        pcts = [float(v) for v in df["数量占比(%)"]]
        pcts.append(100 - sum(pcts))  # 小于最小筛孔的颗粒占比
        x = np.arange(len(pcts))
        ax.bar(x, pcts, color="#4A90E2", edgecolor="white", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(ranges, rotation=35, ha="right",
                           fontsize=self._chart_fs(-2))
        ax.set_ylabel("数量占比 (%)", fontsize=self._chart_fs())
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        self.histFigure.tight_layout()
        self.histCanvas.draw()

    def save_project(self):
        """保存项目：图像路径、参数与级配结果。"""
        if self.img_path is None:
            QMessageBox.information(self, "提示", "请先打开图像。")
            return
        default = os.path.splitext(os.path.basename(self.img_path))[0] + ".json"
        path, _ = QFileDialog.getSaveFileName(
            self, "保存项目", os.path.join(self._last_dir, default), "Project Files (*.json)"
        )
        if not path:
            return
        project = {
            "image": self.img_path,
            "params": self.seg_params,
            "current_step": self.current_step,
            "grading": self.df.to_dict(orient="list") if self.df is not None else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(project, f, ensure_ascii=False, indent=2)
        self.appendLog(f"项目已保存: {path}", "success")

    def run_batch(self):
        """批量处理文件夹内所有图片并汇总统计。"""
        if self._batch_worker is not None and self._batch_worker.isRunning():
            QMessageBox.information(self, "提示", "批量处理正在进行中...")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹", self._last_dir)
        if not folder:
            return
        files = [
            os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(IMG_EXTS)
        ]
        if not files:
            QMessageBox.warning(self, "提示", "所选文件夹中没有图片文件。")
            return

        self._last_dir = folder
        self.appendLog(f"批量处理: 共 {len(files)} 张图片")
        self._batch_dialog = QProgressDialog("批量处理中...", "取消", 0, len(files), self)
        self._batch_dialog.setWindowTitle("批量处理")
        self._batch_dialog.setWindowModality(Qt.WindowModal)
        self._batch_dialog.canceled.connect(self._on_batch_cancel)
        self._batch_dialog.show()

        self._batch_worker = BatchWorker(files, dict(self.seg_params))
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.error.connect(
            lambda msg: self.appendLog(f"批量处理出错: {msg}", "error")
        )
        self._batch_worker.start()

    def _on_batch_cancel(self):
        if self._batch_worker is not None:
            self._batch_worker.stop()
            self.appendLog("批量处理已取消", "warn")

    def _on_batch_progress(self, done, name):
        if self._batch_dialog:
            self._batch_dialog.setValue(done)
            self._batch_dialog.setLabelText(f"正在处理: {name}")
        self.statusBar().showMessage(f"批量处理: {name}")

    def _on_batch_finished(self, df):
        if self._batch_dialog:
            self._batch_dialog.close()
            self._batch_dialog = None
        self.statusBar().showMessage("就绪")
        if df is None or len(df) == 0:
            self.appendLog("批量处理无结果", "warn")
            return
        self.appendLog(f"批量处理完成，共 {len(df)} 条记录", "success")
        self._show_batch_preview(df)

    def _show_batch_preview(self, df):
        """弹窗预览批量处理结果，可选择保存为 Excel/CSV。"""
        dlg = QDialog(self)
        dlg.setWindowTitle("批量结果预览")
        dlg.resize(920, 560)
        layout = QVBoxLayout(dlg)

        table = QTableWidget(len(df), len(df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        for r in range(len(df)):
            for c in range(len(df.columns)):
                item = QTableWidgetItem(str(df.iloc[r, c]))
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        layout.addWidget(table, 1)

        btnRow = QHBoxLayout()
        btnRow.addStretch()
        btnSave = QPushButton("保存结果...")
        btnSave.setObjectName("flatBtn")
        btnClose = QPushButton("关闭")
        btnClose.setObjectName("flatBtn")
        btnRow.addWidget(btnSave)
        btnRow.addWidget(btnClose)
        layout.addLayout(btnRow)

        def _save():
            path, _ = QFileDialog.getSaveFileName(
                dlg, "保存批量结果", os.path.join(self._last_dir, "batch_result.xlsx"),
                "Excel Files (*.xlsx);;CSV Files (*.csv)")
            if not path:
                return
            if path.lower().endswith(".xlsx"):
                df.to_excel(path, index=False)
            else:
                df.to_csv(path, index=False, encoding="utf-8-sig")
            self.appendLog(f"批量结果已保存: {path}", "success")

        btnSave.clicked.connect(_save)
        btnClose.clicked.connect(dlg.accept)
        dlg.exec_()

    # ============================
    # 工作区目录树（类 PyCharm 项目栏）
    # ============================
    def open_workspace_dir(self):
        """选择一个目录作为工作区，在左侧目录树中浏览。"""
        folder = QFileDialog.getExistingDirectory(self, "选择工作区目录", self._last_dir)
        if not folder:
            return
        self._last_dir = folder
        self.setWorkspace(folder)
        self.appendLog(f"工作区目录已打开: {folder}", "success")

    def close_workspace_dir(self):
        self.setWorkspace("")
        self.appendLog("工作区目录已关闭")

    def _on_tree_context_menu(self, pos):
        """项目目录树右键菜单：打开/系统程序打开/复制路径/删除/撤销删除。"""
        index = self.treeWorkspace.indexAt(pos)
        if not index.isValid():
            return
        path = self.fsModel.filePath(index)
        is_dir = self.fsModel.isDir(index)
        is_img = (not is_dir) and path.lower().endswith(IMG_EXTS)

        menu = QMenu(self)
        act_open = menu.addAction("打开(&O)") if is_img else None
        act_sys = menu.addAction("用系统程序打开(&S)") if not is_dir else None
        act_folder = menu.addAction("打开所在目录(&F)")
        act_copy_path = menu.addAction("复制目录路径(&C)" if is_dir else "复制路径(&C)")
        act_copy_name = menu.addAction("复制文件名")
        menu.addSeparator()
        act_delete = menu.addAction("删除(&Del)")
        act_undo = menu.addAction("撤销上次删除")
        act_undo.setEnabled(self._trash_last is not None)

        chosen = menu.exec_(self.treeWorkspace.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if act_open is not None and chosen is act_open:
            self._load_image(path)
        elif act_sys is not None and chosen is act_sys:
            try:
                os.startfile(path)
            except OSError as e:
                self.appendLog(f"无法打开文件: {e}", "error")
        elif chosen is act_folder:
            target = path if is_dir else os.path.dirname(path)
            try:
                os.startfile(target)
            except OSError as e:
                self.appendLog(f"无法打开目录: {e}", "error")
        elif chosen is act_copy_path:
            QApplication.clipboard().setText(path)
            self.appendLog(f"已复制路径: {path}")
        elif chosen is act_copy_name:
            QApplication.clipboard().setText(os.path.basename(path))
            self.appendLog(f"已复制文件名: {os.path.basename(path)}")
        elif chosen is act_delete:
            self._delete_selected()
        elif chosen is act_undo:
            self._undo_delete()

    def _trash_root(self):
        """工作区内的回收站目录。"""
        if self.workspace_dir:
            return os.path.join(self.workspace_dir, ".trash")
        return ""

    def _delete_tree_path(self, path, is_dir, confirm=True):
        """删除：工作区内移入 .trash 回收站（可撤销）；其余确认后永久删除。"""
        if not path:
            return
        abs_path = os.path.abspath(path)
        ws_abs = os.path.abspath(self.workspace_dir) if self.workspace_dir else ""
        trash_root = self._trash_root()
        if ws_abs and abs_path == ws_abs:
            QMessageBox.warning(self, "提示", "不允许删除工作区根目录。")
            return
        if trash_root and abs_path == os.path.abspath(trash_root):
            QMessageBox.warning(self, "提示", "不允许删除回收站目录本身。")
            return

        in_ws = bool(ws_abs and abs_path.startswith(ws_abs + os.sep))
        in_trash = bool(
            trash_root and abs_path.startswith(os.path.abspath(trash_root) + os.sep))
        if in_ws and not in_trash:
            try:
                os.makedirs(trash_root, exist_ok=True)
                dst = os.path.join(trash_root, os.path.basename(path))
                if os.path.exists(dst):
                    stamp = time.strftime("%Y%m%d%H%M%S")
                    root, ext = os.path.splitext(os.path.basename(path))
                    dst = os.path.join(trash_root, f"{root}_{stamp}{ext}")
                shutil.move(path, dst)
                self._trash_last = (path, dst)
                self.appendLog(
                    f"已移入回收站: {os.path.basename(path)}（右键菜单可撤销）",
                    "success")
            except (OSError, shutil.Error) as e:
                self.appendLog(f"移入回收站失败: {e}", "error")
            return

        kind = "目录" if is_dir else "文件"
        if confirm:
            ret = QMessageBox.question(
                self, "确认删除",
                f"确定要永久删除该{kind}吗？此操作不可恢复。\n{path}",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ret != QMessageBox.Yes:
                return
        try:
            if is_dir:
                shutil.rmtree(path)
            else:
                os.remove(path)
            self.appendLog(f"已删除{kind}: {path}", "success")
        except OSError as e:
            self.appendLog(f"删除失败: {e}", "error")
            QMessageBox.critical(self, "删除失败", f"无法删除：\n{e}")

    def _undo_delete(self):
        """撤销上次删除：把文件从回收站移回原位置。"""
        if not self._trash_last:
            return
        src, dst = self._trash_last
        try:
            if os.path.exists(src):
                self.appendLog(f"原位置已有同名文件，无法撤销: {src}", "warn")
                return
            os.makedirs(os.path.dirname(src), exist_ok=True)
            shutil.move(dst, src)
            self._trash_last = None
            self.appendLog(f"已恢复: {src}", "success")
        except (OSError, shutil.Error) as e:
            self.appendLog(f"撤销删除失败: {e}", "error")

    def _delete_selected(self):
        """删除目录树当前选中项（支持多选）。"""
        sel = self.treeWorkspace.selectionModel().selectedRows()
        if not sel:
            return
        items = [(self.fsModel.filePath(i), self.fsModel.isDir(i)) for i in sel]
        if len(items) == 1:
            self._delete_tree_path(items[0][0], items[0][1])
            return
        ret = QMessageBox.question(
            self, "确认删除",
            f"确定要删除选中的 {len(items)} 个项目吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret != QMessageBox.Yes:
            return
        for p, d in items:
            self._delete_tree_path(p, d, confirm=False)

    def _on_tree_selection_changed(self, current, _previous):
        """选中目录树中的图片文件时，在左侧原图窗预览。"""
        indexes = current.indexes()
        if not indexes:
            return
        path = self.fsModel.filePath(indexes[0])
        if os.path.isfile(path) and path.lower().endswith(IMG_EXTS):
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
            if img is not None:
                self.setOrigImage(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def _on_tree_double_clicked(self, index):
        """双击目录树中的图像文件直接加载。"""
        path = self.fsModel.filePath(index)
        if os.path.isfile(path) and path.lower().endswith(IMG_EXTS):
            self._load_image(path)

    def closeEvent(self, event):
        """退出时保存配置与面板调整状态（下次启动自动恢复）。"""
        self.settings.setValue("last_dir", self._last_dir)
        self.settings.setValue("font_preset", FontSize.PRESET)
        self.settings.setValue("seg_params", self.seg_params)
        self.settings.setValue("workspace_dir", self.workspace_dir)
        # 面板调整：窗口布局/停靠栏/工具栏、左右分割器尺寸、导出按钮栏显隐
        self.settings.setValue("window_state", self.saveState())
        self.settings.setValue("right_splitter", self.rightSplitter.saveState())
        self.settings.setValue("left_splitter", self.leftSplitter.saveState())
        super().closeEvent(event)


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # 恢复上次保存的字体档位（需在构建 UI 前生效，记忆来自项目 .mem 文件）
    _boot_settings = MemSettings(MEM_FILE)
    FontSize.set_preset(_boot_settings.value("font_preset", "large"))
    _boot_theme = _boot_settings.value("theme", "明亮")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # 全局样式（集中管理于 styles.py）
    app.setStyleSheet(build_app_stylesheet(FontSize, _boot_theme))

    win = App()
    win.show()
    sys.exit(app.exec_())
