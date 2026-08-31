# -*- coding: utf-8 -*-
"""智能分析控制器：级配特征评价报告（纯函数 + 界面入口）。"""
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QMessageBox

from core.grading import SPEC_RANGES, calc_dxx
from app.ui.report_dialog import show_report_dialog


def build_grading_report(df, particles, spec_name):
    """构建智能分析报告文本（纯函数，便于单独测试/复用）。

    df: 级配统计 DataFrame；particles: 颗粒形态列表；spec_name: 标准级配名称。
    """
    lines = ["【智能分析报告】"]

    sizes = df["筛孔(mm)"].to_numpy(dtype=float)
    passing = df["累计通过率(%)"].to_numpy(dtype=float)

    def _fmt(v):
        return f"{v:.3f} mm" if isinstance(v, float) else str(v)

    d10 = calc_dxx(df, 10)
    d30 = calc_dxx(df, 30)
    d50 = calc_dxx(df, 50)
    d60 = calc_dxx(df, 60)
    d90 = calc_dxx(df, 90)
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

    cum_retain = df["累计筛余(%)"].to_numpy(dtype=float)
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

    spec = SPEC_RANGES.get(spec_name)
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
        lines.append(f"五、标准级配符合性（{spec_name}）:")
        lines.extend(detail)
        lines.append("    结论: 全部指标落在标准范围内，满足该级配要求" if ok_all else
                     "    结论: 存在超限筛孔，不满足该标准级配，建议调整骨料配比")
    else:
        lines.append("五、未选择标准级配范围，跳过符合性对比（可在右侧下拉框选择后重试）")

    if particles:
        ps = pd.DataFrame(particles)
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

    return "\n".join(lines)


class AnalysisMixin:
    """智能分析入口（右键菜单调用）。"""

    def ai_analysis(self):
        """智能分析：级配特征、均匀性/曲率系数、细度模数、标准符合性与颗粒形态评价。"""
        if self.df is None or len(self.df) == 0:
            QMessageBox.information(self, "提示", "暂无级配数据，请先完成分析。")
            return
        report = build_grading_report(
            self.df, self.particles, self.specCombo.currentText())
        self.appendLog("智能分析完成", "success")
        show_report_dialog(self, "智能分析", report)
