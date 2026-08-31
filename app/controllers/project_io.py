# -*- coding: utf-8 -*-
"""项目文件控制器：保存/打开 .json 项目（图像路径 + 分割参数 + 级配结果）。"""
import json
import os

import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QMessageBox


class ProjectIOMixin:
    """项目文件的序列化与恢复。"""

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
