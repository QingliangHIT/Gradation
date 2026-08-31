# -*- coding: utf-8 -*-
"""导出控制器：级配结果（Excel/CSV）、颗粒详情、图表与处理日志。"""
import os
from datetime import datetime

import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QMessageBox

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


class ExportMixin:
    """结果导出入口（菜单 / 导出按钮栏 / 右键菜单共用）。"""

    # ============================
    # 数据准备
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

    def _particles_dataframe(self):
        """将颗粒详情（含 12 项形态指标）整理为中文列名 DataFrame。"""
        if not self.particles:
            return None
        df = pd.DataFrame(self.particles)
        df = df.rename(columns=_PARTICLE_CN)
        cols = [v for v in _PARTICLE_CN.values() if v in df.columns]
        return df[cols]

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

    # ============================
    # 导出入口
    # ============================
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
