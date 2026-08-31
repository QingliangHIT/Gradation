# -*- coding: utf-8 -*-
"""后台工作线程：模型推理与批量处理（避免阻塞界面主线程）。"""
import os

import cv2
import numpy as np
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal

from core import registry as model_registry
from core.measure import measure_particles
from core.grading import calculate_grading, calc_dxx


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
