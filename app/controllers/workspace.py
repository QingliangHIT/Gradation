# -*- coding: utf-8 -*-
"""工作区控制器：目录树浏览、右键操作、回收站删除与撤销。"""
import os
import shutil
import time

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QMenu, QMessageBox

from app.controllers.pipeline import IMG_EXTS


class WorkspaceMixin:
    """左侧工作区目录树（类 PyCharm 项目栏）的交互逻辑。"""

    def open_workspace_dir(self):
        """选择一个目录作为工作区，在左侧目录树中浏览。"""
        folder = QFileDialog.getExistingDirectory(self, "选择工作区目录", self._last_dir)
        if not folder:
            return
        self._last_dir = folder
        self.setWorkspace(folder)
        self.appendLog(f"工作区目录已打开: {folder}", "success")

    def close_workspace_dir(self):
        self.setWorkspace("")
        self.appendLog("工作区目录已关闭")

    # ============================
    # 目录树交互
    # ============================
    def _on_tree_selection_changed(self, current, _previous):
        """选中目录树中的图片文件时，在左侧原图窗预览。"""
        indexes = current.indexes()
        if not indexes:
            return
        path = self.fsModel.filePath(indexes[0])
        if os.path.isfile(path) and path.lower().endswith(IMG_EXTS):
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
            if img is not None:
                self.setOrigImage(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def _on_tree_double_clicked(self, index):
        """双击目录树中的图像文件直接加载。"""
        path = self.fsModel.filePath(index)
        if os.path.isfile(path) and path.lower().endswith(IMG_EXTS):
            self._load_image(path)

    def _on_tree_context_menu(self, pos):
        """项目目录树右键菜单：打开/系统程序打开/复制路径/删除/撤销删除。"""
        index = self.treeWorkspace.indexAt(pos)
        if not index.isValid():
            return
        path = self.fsModel.filePath(index)
        is_dir = self.fsModel.isDir(index)
        is_img = (not is_dir) and path.lower().endswith(IMG_EXTS)

        menu = QMenu(self)
        act_open = menu.addAction("打开(&O)") if is_img else None
        act_sys = menu.addAction("用系统程序打开(&S)") if not is_dir else None
        act_folder = menu.addAction("打开所在目录(&F)")
        act_copy_path = menu.addAction("复制目录路径(&C)" if is_dir else "复制路径(&C)")
        act_copy_name = menu.addAction("复制文件名")
        menu.addSeparator()
        act_delete = menu.addAction("删除(&Del)")
        act_undo = menu.addAction("撤销上次删除")
        act_undo.setEnabled(self._trash_last is not None)

        chosen = menu.exec_(self.treeWorkspace.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if act_open is not None and chosen is act_open:
            self._load_image(path)
        elif act_sys is not None and chosen is act_sys:
            try:
                os.startfile(path)
            except OSError as e:
                self.appendLog(f"无法打开文件: {e}", "error")
        elif chosen is act_folder:
            target = path if is_dir else os.path.dirname(path)
            try:
                os.startfile(target)
            except OSError as e:
                self.appendLog(f"无法打开目录: {e}", "error")
        elif chosen is act_copy_path:
            QApplication.clipboard().setText(path)
            self.appendLog(f"已复制路径: {path}")
        elif chosen is act_copy_name:
            QApplication.clipboard().setText(os.path.basename(path))
            self.appendLog(f"已复制文件名: {os.path.basename(path)}")
        elif chosen is act_delete:
            self._delete_selected()
        elif chosen is act_undo:
            self._undo_delete()

    # ============================
    # 删除与回收站
    # ============================
    def _trash_root(self):
        """工作区内的回收站目录。"""
        if self.workspace_dir:
            return os.path.join(self.workspace_dir, ".trash")
        return ""

    def _delete_tree_path(self, path, is_dir, confirm=True):
        """删除：工作区内移入 .trash 回收站（可撤销）；其余确认后永久删除。"""
        if not path:
            return
        abs_path = os.path.abspath(path)
        ws_abs = os.path.abspath(self.workspace_dir) if self.workspace_dir else ""
        trash_root = self._trash_root()
        if ws_abs and abs_path == ws_abs:
            QMessageBox.warning(self, "提示", "不允许删除工作区根目录。")
            return
        if trash_root and abs_path == os.path.abspath(trash_root):
            QMessageBox.warning(self, "提示", "不允许删除回收站目录本身。")
            return

        in_ws = bool(ws_abs and abs_path.startswith(ws_abs + os.sep))
        in_trash = bool(
            trash_root and abs_path.startswith(os.path.abspath(trash_root) + os.sep))
        if in_ws and not in_trash:
            try:
                os.makedirs(trash_root, exist_ok=True)
                dst = os.path.join(trash_root, os.path.basename(path))
                if os.path.exists(dst):
                    stamp = time.strftime("%Y%m%d%H%M%S")
                    root, ext = os.path.splitext(os.path.basename(path))
                    dst = os.path.join(trash_root, f"{root}_{stamp}{ext}")
                shutil.move(path, dst)
                self._trash_last = (path, dst)
                self.appendLog(
                    f"已移入回收站: {os.path.basename(path)}（右键菜单可撤销）",
                    "success")
            except (OSError, shutil.Error) as e:
                self.appendLog(f"移入回收站失败: {e}", "error")
            return

        kind = "目录" if is_dir else "文件"
        if confirm:
            ret = QMessageBox.question(
                self, "确认删除",
                f"确定要永久删除该{kind}吗？此操作不可恢复。\n{path}",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ret != QMessageBox.Yes:
                return
        try:
            if is_dir:
                shutil.rmtree(path)
            else:
                os.remove(path)
            self.appendLog(f"已删除{kind}: {path}", "success")
        except OSError as e:
            self.appendLog(f"删除失败: {e}", "error")
            QMessageBox.critical(self, "删除失败", f"无法删除：\n{e}")

    def _undo_delete(self):
        """撤销上次删除：把文件从回收站移回原位置。"""
        if not self._trash_last:
            return
        src, dst = self._trash_last
        try:
            if os.path.exists(src):
                self.appendLog(f"原位置已有同名文件，无法撤销: {src}", "warn")
                return
            os.makedirs(os.path.dirname(src), exist_ok=True)
            shutil.move(dst, src)
            self._trash_last = None
            self.appendLog(f"已恢复: {src}", "success")
        except (OSError, shutil.Error) as e:
            self.appendLog(f"撤销删除失败: {e}", "error")

    def _delete_selected(self):
        """删除目录树当前选中项（支持多选）。"""
        sel = self.treeWorkspace.selectionModel().selectedRows()
        if not sel:
            return
        items = [(self.fsModel.filePath(i), self.fsModel.isDir(i)) for i in sel]
        if len(items) == 1:
            self._delete_tree_path(items[0][0], items[0][1])
            return
        ret = QMessageBox.question(
            self, "确认删除",
            f"确定要删除选中的 {len(items)} 个项目吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret != QMessageBox.Yes:
            return
        for p, d in items:
            self._delete_tree_path(p, d, confirm=False)
