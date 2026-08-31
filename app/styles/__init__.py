# -*- coding: utf-8 -*-
"""app.styles —— 主题配色与全局 QSS。"""
from app.styles.themes import THEMES
from app.styles.qss import build_app_stylesheet

__all__ = ["THEMES", "build_app_stylesheet"]
