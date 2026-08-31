# -*- coding: utf-8 -*-
"""外观与设置控制器：设置对话框、主题切换、字体缩放、布局管理与事件过滤。"""
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import QApplication, QMenu, QMessageBox

from app.config import FontSize
from app.styles import build_app_stylesheet
from app.ui.settings_dialog import UnifiedSettingsDialog
from core import config as infer_config
from core import registry as model_registry


class AppearanceMixin:
    """主题 / 字体 / 布局 / 设置入口与全局事件过滤。"""

    # ============================
    # 设置与关于
    # ============================
    def open_settings(self):
        """统一设置对话框：算法参数 + 系统与推理。"""
        dlg = UnifiedSettingsDialog(self.seg_params, infer_config.get_config(), self)
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
        cfg = infer_config.configure(
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

    # ============================
    # 主题与字体
    # ============================
    def _apply_theme(self, theme):
        """切换界面主题配色并持久化。"""
        self._theme = theme
        self.settings.setValue("theme", theme)
        for name, act in self.themeActs.items():
            act.setChecked(name == theme)
        QApplication.instance().setStyleSheet(build_app_stylesheet(FontSize, theme))
        self.appendLog(f"界面主题已切换: {theme}")

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

    def _apply_log_font(self, persist=True):
        """应用日志字体（正文层 lv(4) + Ctrl+滚轮偏移；控件级样式，优先于全局 QSS）。"""
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
        """日志字号 = 正文层 lv(4) + Ctrl+滚轮偏移（保底不低于 6px）。"""
        return max(FontSize.lv(4) + self._log_font_off, 6)

    def _table_fs(self):
        """统计表字号 = 表格层 lv(6) + Ctrl+滚轮偏移（保底不低于 6px）。"""
        return max(FontSize.lv(6) + self._table_font_off, 6)

    def _chart_fs(self, delta=0):
        """图表字号 = 图表层 lv(4) + Ctrl+滚轮偏移（保底不低于 5px）。"""
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

    # ============================
    # 全局事件过滤
    # ============================
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

    # ============================
    # 导出按钮栏显隐
    # ============================
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

    # ============================
    # 布局重置
    # ============================
    def reset_layout(self):
        """重置窗口布局及全部面板调整（分割器/字体/第二行工具栏）为初始状态。"""
        self.restoreState(self._initial_state)
        self._redock_all()
        self._ensure_export_break()
        # 分割器恢复默认比例（项目目录/原图各半，图表/统计表各半）
        self.rightSplitter.setStretchFactor(0, 1)
        self.rightSplitter.setStretchFactor(1, 1)
        self.leftSplitter.setStretchFactor(0, 1)
        self.leftSplitter.setStretchFactor(1, 0)
        self.leftSplitter.setStretchFactor(2, 0)
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

    # ============================
    # 右键菜单
    # ============================
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

    # ============================
    # 退出保存
    # ============================
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
