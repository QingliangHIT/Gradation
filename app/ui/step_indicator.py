# -*- coding: utf-8 -*-
"""步骤状态指示器（已完成/进行中/未开始三态圆圈）与面板小标题。"""
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt

from app.config import FontSize


def panel_title(text):
    """面板小标题标签（配合全局 QSS 的 #panelTitle 样式）。"""
    lbl = QLabel(text)
    lbl.setObjectName("panelTitle")
    return lbl


class StepIndicator(QWidget):
    """工具栏/面板内的三步骤状态指示器。

    compact=True 时使用短步骤名与更小尺寸（分析面板内布局）。
    """

    def __init__(self, compact=False):
        super().__init__()
        self._compact = compact
        self.steps = ["图像采集与预处理", "集料粒度提取", "结果分析与统计"]
        self.steps_short = ["预处理", "粒度提取", "结果分析"]
        self.current_step = -1
        self._circles = []
        self._labels = []
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.setSpacing(8 if self._compact else 20)
        layout.setContentsMargins(0, 0, 0, 0)
        F = FontSize
        # 紧凑模式（分析面板内）：圆圈/文字更小、短步骤名，与下拉框尺寸协调
        steps = self.steps_short if self._compact else self.steps
        self._c_size = 18 if self._compact else 32
        self._circle_font = max(F.lv(3) - 1, 8) if self._compact else F.lv(2)
        self._lbl_font = F.lv(3) if self._compact else F.lv(1)
        arrow_font = F.lv(3) if self._compact else F.lv(1)

        self._circles = []
        self._labels = []

        for i, step in enumerate(steps):
            container = QHBoxLayout()

            circle = QLabel(str(i + 1))
            circle.setFixedSize(self._c_size, self._c_size)
            circle.setAlignment(Qt.AlignCenter)
            self._circles.append(circle)

            label = QLabel(step)
            label.setStyleSheet(f"font-weight: bold; font-size: {self._lbl_font}px;")
            self._labels.append(label)

            container.addWidget(circle)
            container.addWidget(label)

            if i < len(steps) - 1:
                arrow = QLabel("→")
                arrow.setStyleSheet(f"color: #999999; font-size: {arrow_font}px;")
                container.addWidget(arrow)

            layout.addLayout(container)

        layout.addStretch()
        self.setLayout(layout)
        self.update_ui()

    def update_ui(self):
        """根据 current_step 更新圆圈和标签的样式（已完成/进行中/未开始三态）。"""
        for i, (circle, label) in enumerate(zip(self._circles, self._labels)):
            if i <= self.current_step:
                # 已完成：绿色 ✓
                bg, fg, text = "#4CAF50", "#4CAF50", "✓"
            elif i == self.current_step + 1:
                # 当前进行中：蓝色高亮
                bg, fg, text = "#4A90E2", "#4A90E2", str(i + 1)
            else:
                # 未开始：灰色
                bg, fg, text = "#CCCCCC", "#666666", str(i + 1)
            circle.setText(text)
            circle.setStyleSheet(f"""
                QLabel{{
                    background-color: {bg};
                    color: white;
                    border-radius: {self._c_size // 2}px;
                    font-weight: bold;
                    font-size: {self._circle_font}px;
                }}
            """)
            label.setStyleSheet(f"""
                QLabel{{
                    color: {fg};
                    font-weight: bold;
                    font-size: {self._lbl_font}px;
                }}
            """)
