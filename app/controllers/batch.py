# -*- coding: utf-8 -*-
"""批量处理控制器：文件夹全量图片执行完整流程并汇总预览/保存。"""
import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressDialog

from app.controllers.pipeline import IMG_EXTS
from app.ui.report_dialog import show_batch_preview
from app.workers import BatchWorker


class BatchMixin:
    """批量处理任务管理与结果预览。"""

    def run_batch(self):
        """批量处理文件夹内所有图片并汇总统计。"""
        if self._batch_worker is not None and self._batch_worker.isRunning():
            QMessageBox.information(self, "提示", "批量处理正在进行中...")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹", self._last_dir)
        if not folder:
            return
        files = [
            os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(IMG_EXTS)
        ]
        if not files:
            QMessageBox.warning(self, "提示", "所选文件夹中没有图片文件。")
            return

        self._last_dir = folder
        self.appendLog(f"批量处理: 共 {len(files)} 张图片")
        self._batch_dialog = QProgressDialog("批量处理中...", "取消", 0, len(files), self)
        self._batch_dialog.setWindowTitle("批量处理")
        self._batch_dialog.setWindowModality(Qt.WindowModal)
        self._batch_dialog.canceled.connect(self._on_batch_cancel)
        self._batch_dialog.show()

        self._batch_worker = BatchWorker(files, dict(self.seg_params))
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.error.connect(
            lambda msg: self.appendLog(f"批量处理出错: {msg}", "error")
        )
        self._batch_worker.start()

    def _on_batch_cancel(self):
        if self._batch_worker is not None:
            self._batch_worker.stop()
            self.appendLog("批量处理已取消", "warn")

    def _on_batch_progress(self, done, name):
        if self._batch_dialog:
            self._batch_dialog.setValue(done)
            self._batch_dialog.setLabelText(f"正在处理: {name}")
        self.statusBar().showMessage(f"批量处理: {name}")

    def _on_batch_finished(self, df):
        if self._batch_dialog:
            self._batch_dialog.close()
            self._batch_dialog = None
        self.statusBar().showMessage("就绪")
        if df is None or len(df) == 0:
            self.appendLog("批量处理无结果", "warn")
            return
        self.appendLog(f"批量处理完成，共 {len(df)} 条记录", "success")
        show_batch_preview(
            self, df, self._last_dir,
            on_saved=lambda path: self.appendLog(f"批量结果已保存: {path}", "success"),
        )
