# -*- coding: utf-8 -*-
"""处理流程控制器：三步骤执行（预处理 → 分割 → 分析统计）与步骤状态管理。"""
import os
import time

import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from app.workers import SegmentationWorker
from core import registry as model_registry
from core.preprocess import preprocess
from core.measure import colorize_particles, measure_particles, overlay_markers
from core.grading import calculate_grading, calc_dxx

# 支持拖拽打开的图像扩展名
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


class PipelineMixin:
    """图像加载与三步处理流程控制。"""

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
    # 加载图像
    # ============================
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

    # ============================
    # 步骤0：预处理
    # ============================
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
    # 步骤1：分割（异步）
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
    # 下一步 / 一键执行 / 重新处理
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

