# -*- coding: utf-8 -*-
"""结果展示对话框：智能分析报告预览、批量处理结果预览（含保存）。"""
import os

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QFileDialog,
)
from PyQt5.QtCore import Qt


def show_report_dialog(parent, title, report_text):
    """弹出只读文本报告对话框（智能分析等）。"""
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(620, 460)
    v = QVBoxLayout(dlg)
    view = QTextEdit(dlg)
    view.setReadOnly(True)
    view.setPlainText(report_text)
    btn = QPushButton("关闭")
    btn.setObjectName("flatBtn")
    btn.clicked.connect(dlg.accept)
    v.addWidget(view)
    v.addWidget(btn, 0, Qt.AlignRight)
    dlg.exec_()


def show_batch_preview(parent, df, save_dir, on_saved=None):
    """弹窗预览批量处理结果，可选择保存为 Excel/CSV。

    on_saved: 保存成功后的回调 on_saved(path)，用于写日志。
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle("批量结果预览")
    dlg.resize(920, 560)
    layout = QVBoxLayout(dlg)

    table = QTableWidget(len(df), len(df.columns))
    table.setHorizontalHeaderLabels([str(c) for c in df.columns])
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)
    for r in range(len(df)):
        for c in range(len(df.columns)):
            item = QTableWidgetItem(str(df.iloc[r, c]))
            item.setTextAlignment(Qt.AlignCenter)
            table.setItem(r, c, item)
    table.resizeColumnsToContents()
    layout.addWidget(table, 1)

    btnRow = QHBoxLayout()
    btnRow.addStretch()
    btnSave = QPushButton("保存结果...")
    btnSave.setObjectName("flatBtn")
    btnClose = QPushButton("关闭")
    btnClose.setObjectName("flatBtn")
    btnRow.addWidget(btnSave)
    btnRow.addWidget(btnClose)
    layout.addLayout(btnRow)

    def _save():
        path, _ = QFileDialog.getSaveFileName(
            dlg, "保存批量结果", os.path.join(save_dir, "batch_result.xlsx"),
            "Excel Files (*.xlsx);;CSV Files (*.csv)")
        if not path:
            return
        if path.lower().endswith(".xlsx"):
            df.to_excel(path, index=False)
        else:
            df.to_csv(path, index=False, encoding="utf-8-sig")
        if on_saved:
            on_saved(path)

    btnSave.clicked.connect(_save)
    btnClose.clicked.connect(dlg.accept)
    dlg.exec_()
