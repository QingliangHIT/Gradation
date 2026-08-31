# -*- coding: utf-8 -*-
"""结果视图控制器：统计表、级配曲线、粒径分布直方图与曲线鼠标交互。"""
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTableWidgetItem

from core.grading import SIEVE_SIZE, SPEC_RANGES, calc_dxx


class ResultViewMixin:
    """级配结果表 / 图表更新与交互。"""

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
