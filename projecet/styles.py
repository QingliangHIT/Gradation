"""
全局样式表配置
集中管理所有 QSS 样式，字体档位/主题切换时重新生成即可全局生效。
控件通过 setObjectName 匹配对应样式。
"""
from font_config import FontSize

# 可选界面主题：明亮（默认）/ 暗黑 / 清新绿 / 海洋蓝 / 暖橙 / 优雅紫
# 注：日志、表格、图表画布内容色固定为浅色，故其背景在所有主题下保持白色
THEMES = {
    "明亮": {
        "window_bg": "#f5f5f5", "separator": "#e0e0e0",
        "bar_bg": "#fafafa", "border": "#e0e0e0", "menu_border": "#d0d0d0",
        "panel": "white", "btn_bg": "white",
        "hover": "#e3f2fd", "accent": "#2196F3", "accent_dark": "#1565C0",
        "sel_bg": "#bbdefb", "sel_fg": "#0d47a1",
        "text_main": "#333", "text_sub": "#555", "text_faint": "#888",
        "tab_bg": "#eeeeee", "dock_title": "#ececec",
        "btn_hover": "#f0f0f0", "btn_pressed": "#e4e4e4",
        "disabled_bg": "#e0e0e0", "disabled_fg": "#9e9e9e",
        "primary": "#2196F3", "primary_hover": "#1976D2", "primary_pressed": "#1565C0",
        "success": "#4CAF50", "success_hover": "#43A047", "success_pressed": "#388E3C",
        "gridline": "#e8e8e8", "table_head": "#f5f5f5", "alternate": "#f7fafd",
    },
    "暗黑": {
        "window_bg": "#2b2b2b", "separator": "#444444",
        "bar_bg": "#3c3f41", "border": "#5a5a5a", "menu_border": "#5a5a5a",
        "panel": "#313335", "btn_bg": "#3c3f41",
        "hover": "#2f4f7f", "accent": "#4A90E2", "accent_dark": "#6fa8dc",
        "sel_bg": "#2d5b94", "sel_fg": "#ffffff",
        "text_main": "#cccccc", "text_sub": "#aaaaaa", "text_faint": "#8a8a8a",
        "tab_bg": "#3c3f41", "dock_title": "#3c3f41",
        "btn_hover": "#45494c", "btn_pressed": "#4d5255",
        "disabled_bg": "#3a3a3a", "disabled_fg": "#777777",
        "primary": "#3d7dcc", "primary_hover": "#356cb3", "primary_pressed": "#2d5b99",
        "success": "#4CAF50", "success_hover": "#43A047", "success_pressed": "#388E3C",
        "gridline": "#444444", "table_head": "#3c3f41", "alternate": "#363839",
    },
    "清新绿": {
        "window_bg": "#f3f7f2", "separator": "#dbe5d9",
        "bar_bg": "#f7faf6", "border": "#d7e3d4", "menu_border": "#cfdccd",
        "panel": "white", "btn_bg": "white",
        "hover": "#e6f4e4", "accent": "#2e7d32", "accent_dark": "#1b5e20",
        "sel_bg": "#c8e6c9", "sel_fg": "#1b5e20",
        "text_main": "#2f3b2f", "text_sub": "#556655", "text_faint": "#7a8a7a",
        "tab_bg": "#eef3ed", "dock_title": "#e9f0e7",
        "btn_hover": "#f0f5ee", "btn_pressed": "#e4ede1",
        "disabled_bg": "#e0e0e0", "disabled_fg": "#9e9e9e",
        "primary": "#2e7d32", "primary_hover": "#27692b", "primary_pressed": "#1b5e20",
        "success": "#43a047", "success_hover": "#388e3c", "success_pressed": "#2e7d32",
        "gridline": "#e2e9df", "table_head": "#eef3ed", "alternate": "#f5f9f4",
    },
    "海洋蓝": {
        "window_bg": "#f2f7fa", "separator": "#d9e6ef",
        "bar_bg": "#f7fafc", "border": "#d5e3ec", "menu_border": "#cfdfe9",
        "panel": "white", "btn_bg": "white",
        "hover": "#e2f0f8", "accent": "#0277bd", "accent_dark": "#01579b",
        "sel_bg": "#b3e5fc", "sel_fg": "#01579b",
        "text_main": "#2c3e50", "text_sub": "#546e7a", "text_faint": "#90a4ae",
        "tab_bg": "#ecf3f7", "dock_title": "#e8f1f6",
        "btn_hover": "#eef5f9", "btn_pressed": "#e0edf4",
        "disabled_bg": "#e0e0e0", "disabled_fg": "#9e9e9e",
        "primary": "#0288d1", "primary_hover": "#0277bd", "primary_pressed": "#01579b",
        "success": "#26a69a", "success_hover": "#00897b", "success_pressed": "#00695c",
        "gridline": "#e0ebf1", "table_head": "#ecf3f7", "alternate": "#f5fafc",
    },
    "暖橙": {
        "window_bg": "#faf6f1", "separator": "#eee3d7",
        "bar_bg": "#fcf9f5", "border": "#eadfd2", "menu_border": "#e4d6c5",
        "panel": "white", "btn_bg": "white",
        "hover": "#fdeee0", "accent": "#e65100", "accent_dark": "#bf360c",
        "sel_bg": "#ffe0b2", "sel_fg": "#bf360c",
        "text_main": "#4e342e", "text_sub": "#6d4c41", "text_faint": "#a1887f",
        "tab_bg": "#f5efe7", "dock_title": "#f3ece2",
        "btn_hover": "#f9f2ea", "btn_pressed": "#f2e8dc",
        "disabled_bg": "#e0e0e0", "disabled_fg": "#9e9e9e",
        "primary": "#ef6c00", "primary_hover": "#e65100", "primary_pressed": "#bf360c",
        "success": "#7cb342", "success_hover": "#689f38", "success_pressed": "#558b2f",
        "gridline": "#efe6da", "table_head": "#f5efe7", "alternate": "#fbf7f2",
    },
    "优雅紫": {
        "window_bg": "#f6f4f9", "separator": "#e5dfee",
        "bar_bg": "#faf8fc", "border": "#e3dced", "menu_border": "#dcd3e8",
        "panel": "white", "btn_bg": "white",
        "hover": "#f0e9f8", "accent": "#6a1b9a", "accent_dark": "#4a148c",
        "sel_bg": "#e1bee7", "sel_fg": "#4a148c",
        "text_main": "#3a3042", "text_sub": "#5e5470", "text_faint": "#8f85a0",
        "tab_bg": "#f0edf4", "dock_title": "#eeeaf3",
        "btn_hover": "#f5f1f9", "btn_pressed": "#ece5f3",
        "disabled_bg": "#e0e0e0", "disabled_fg": "#9e9e9e",
        "primary": "#7b1fa2", "primary_hover": "#6a1b9a", "primary_pressed": "#4a148c",
        "success": "#43a047", "success_hover": "#388e3c", "success_pressed": "#2e7d32",
        "gridline": "#e8e3f0", "table_head": "#f0edf4", "alternate": "#f9f7fb",
    },
}


def build_app_stylesheet(F=None, theme="明亮"):
    """根据字体档位与主题配色生成全局 QSS。"""
    if F is None:
        F = FontSize
    C = THEMES.get(theme, THEMES["明亮"])
    return f"""
    /* ===== 主窗口 ===== */
    QMainWindow {{ background: {C["window_bg"]}; }}
    QMainWindow::separator {{ background: {C["separator"]}; width: 3px; height: 3px; }}

    /* ===== 菜单栏 ===== */
    QMenuBar {{
        background: {C["bar_bg"]};
        border-bottom: 1px solid {C["border"]};
        color: {C["text_main"]};
        font-size: {F.lv(2)}px;
    }}
    QMenuBar::item {{ padding: 5px 10px; background: transparent; }}
    QMenuBar::item:selected {{ background: {C["hover"]}; border-radius: 3px; }}
    QMenu {{
        background: {C["panel"]};
        color: {C["text_main"]};
        border: 1px solid {C["menu_border"]};
        padding: 4px;
        font-size: {F.lv(2)}px;
    }}
    QMenu::item {{ padding: 6px 24px 6px 12px; border-radius: 3px; }}
    QMenu::item:selected {{ background: {C["hover"]}; color: {C["accent_dark"]}; }}
    QMenu::separator {{ height: 1px; background: {C["gridline"]}; margin: 4px 8px; }}

    /* ===== 工具栏 ===== */
    QToolBar#mainToolbar {{
        background: {C["bar_bg"]};
        border: none;
        border-bottom: 1px solid {C["border"]};
        padding: 3px 6px;
        spacing: 4px;
    }}
    QToolBar#mainToolbar QToolButton {{
        border: 1px solid {C["menu_border"]};
        border-radius: 4px;
        padding: 4px 12px;
        background: {C["btn_bg"]};
        color: {C["text_main"]};
        font-size: {F.lv(2)}px;
    }}
    QToolBar#mainToolbar QToolButton:hover {{ background: {C["btn_hover"]}; }}
    QToolBar#mainToolbar QToolButton:pressed {{ background: {C["btn_pressed"]}; }}
    QToolBar#mainToolbar QToolButton:disabled {{ background: {C["disabled_bg"]}; color: {C["disabled_fg"]}; }}
    QToolBar#mainToolbar::separator {{
        width: 1px; background: {C["menu_border"]}; margin: 4px 6px;
    }}

    /* ===== 停靠窗口 ===== */
    QDockWidget {{
        font-weight: bold;
        font-size: {F.lv(1)}px;
        color: {C["text_main"]};
    }}
    QDockWidget::title {{
        background: {C["dock_title"]};
        padding: 5px 10px;
        border-bottom: 1px solid {C["border"]};
    }}

    /* ===== 工作区目录树（白底便于阅读，跨主题一致）===== */
    QTreeView#workspaceTree {{
        border: 1px solid {C["border"]};
        border-radius: 4px;
        background: white;
        color: #333;
        font-size: {F.lv(3)}px;
        alternate-background-color: #fafafa;
    }}
    QTreeView#workspaceTree::item {{ padding: 3px 2px; }}
    QTreeView#workspaceTree::item:hover {{ background: {C["hover"]}; }}
    QTreeView#workspaceTree::item:selected {{ background: {C["sel_bg"]}; color: {C["sel_fg"]}; }}

    /* ===== 页签控件（图表/对话框页签）===== */
    QTabWidget::pane {{ border: 1px solid {C["border"]}; background: {C["panel"]}; border-radius: 4px; }}
    QTabBar::tab {{
        background: {C["tab_bg"]};
        color: {C["text_main"]};
        border: 1px solid {C["border"]};
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        padding: 5px 16px;
        margin-right: 2px;
        font-size: {F.lv(3)}px;
    }}
    QTabBar::tab:selected {{ background: {C["panel"]}; color: {C["accent_dark"]}; font-weight: bold; }}

    /* ===== 按钮通用禁用态 ===== */
    QPushButton:disabled {{
        background: {C["disabled_bg"]};
        color: {C["disabled_fg"]};
        border: 1px solid {C["disabled_bg"]};
    }}

    /* ===== GroupBox 基础样式 ===== */
    QGroupBox {{
        font-weight: bold;
        border: 1px solid {C["border"]};
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 16px;
        background: {C["panel"]};
        color: {C["text_main"]};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 4px;
        color: {C["accent"]};
        font-size: {F.lv(1)}px;
    }}

    /* ===== 信息卡片 ===== */
    QGroupBox#infoCard {{
        border: 1px solid {C["border"]};
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 1px;
        background: {C["bar_bg"]};
    }}
    QLabel#cardTitle {{
        font-weight: bold;
        font-size: {F.lv(1)}px;
        color: {C["accent"]};
    }}

    /* ===== 文本标签 ===== */
    QLabel#imageLabel {{ font-weight: bold; font-size: {F.lv(3)}px; color: {C["text_main"]}; }}
    QLabel#panelTitle {{ font-weight: bold; font-size: {F.lv(1)}px; color: {C["text_main"]}; }}
    QLabel#infoLabel  {{ font-size: {F.lv(3)}px; color: {C["text_sub"]}; }}
    QLabel#zoomLabel  {{ font-size: {F.lv(5)}px; color: {C["text_faint"]}; }}
    QLabel#pixelInfo  {{ font-size: {F.lv(5)}px; color: {C["text_sub"]}; font-family: Consolas; }}

    /* ===== 顶部工具栏按钮 ===== */
    QPushButton#toolbarBtn {{
        border: 1px solid {C["menu_border"]};
        border-radius: 4px;
        padding: 4px 12px;
        background: {C["btn_bg"]};
        color: {C["text_main"]};
        font-size: {F.lv(2)}px;
    }}
    QPushButton#toolbarBtn:hover {{ background: {C["btn_hover"]}; }}
    QPushButton#toolbarBtn:pressed {{ background: {C["btn_pressed"]}; }}

    /* ===== 普通按钮 ===== */
    QPushButton#flatBtn {{
        border: 1px solid {C["menu_border"]};
        border-radius: 4px;
        padding: 6px 16px;
        background: {C["btn_bg"]};
        color: {C["text_main"]};
        font-size: {F.lv(2)}px;
    }}
    QPushButton#flatBtn:hover {{ background: {C["btn_hover"]}; }}
    QPushButton#flatBtn:pressed {{ background: {C["btn_pressed"]}; }}

    /* ===== 主色调按钮 ===== */
    QPushButton#primaryBtn {{
        border: none;
        border-radius: 4px;
        padding: 6px 20px;
        background: {C["primary"]};
        color: white;
        font-size: {F.lv(2)}px;
        font-weight: bold;
    }}
    QPushButton#primaryBtn:hover {{ background: {C["primary_hover"]}; }}
    QPushButton#primaryBtn:pressed {{ background: {C["primary_pressed"]}; }}

    /* ===== 高亮按钮（数据采集）===== */
    QPushButton#successBtn {{
        border: 1px solid {C["success"]};
        border-radius: 4px;
        padding: 6px 16px;
        background: {C["success"]};
        color: white;
        font-size: {F.lv(2)}px;
        font-weight: bold;
    }}
    QPushButton#successBtn:hover {{ background: {C["success_hover"]}; }}
    QPushButton#successBtn:pressed {{ background: {C["success_pressed"]}; }}

    /* ===== 缩放控制小按钮 ===== */
    QPushButton#zoomBtn {{
        border: 1px solid {C["menu_border"]};
        border-radius: 4px;
        background: {C["btn_bg"]};
        color: {C["text_main"]};
        font-size: {F.lv(4)}px;
    }}
    QPushButton#zoomBtn:hover {{ background: {C["btn_hover"]}; }}
    QPushButton#zoomBtn:pressed {{ background: {C["btn_pressed"]}; }}

    /* ===== 处理日志（白底，跨主题一致）===== */
    QTextEdit#logText {{
        border: 1px solid {C["border"]};
        border-radius: 4px;
        background: white;
        color: #333333;
        font-size: {F.lv(3)}px;
        font-family: Consolas, 'Microsoft YaHei';
    }}

    /* ===== 统计表格（白底，跨主题一致）===== */
    QTableWidget#resultTable {{
        border: 1px solid {C["border"]};
        border-radius: 4px;
        gridline-color: #e8e8e8;
        background: white;
        color: #333333;
        alternate-background-color: #f7fafd;
        font-size: {F.lv(3)}px;
    }}
    QTableWidget#resultTable::item {{ padding: 4px; }}
    QTableWidget#resultTable QHeaderView::section {{
        background: {C["table_head"]};
        color: {C["text_main"]};
        border: 1px solid {C["border"]};
        padding: 4px;
        font-weight: bold;
        font-size: {F.lv(2)}px;
    }}
    """
