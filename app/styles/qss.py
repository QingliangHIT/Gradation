# -*- coding: utf-8 -*-
"""全局样式表生成：根据字体档位与主题配色拼装 QSS。

集中管理所有 QSS 样式，字体档位/主题切换时重新生成即可全局生效。
控件通过 setObjectName 匹配对应样式。
"""
from app.config import FontSize
from app.styles.themes import THEMES


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
