# -*- coding: utf-8 -*-
"""应用级配置：路径常量、项目级 .mem 记忆（MemSettings）与三档字体（FontSize）。"""
import base64
import json
import os

from PyQt5.QtCore import QByteArray

# 项目根目录（app/ 的上一级）
APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 界面记忆文件（项目根目录 state.mem）
MEM_FILE = os.path.join(APP_ROOT, "state.mem")


class MemSettings:
    """项目级记忆：以 JSON 文本保存在项目目录下的 .mem 文件中，不使用注册表。
    接口兼容 QSettings 常用方法（value/setValue/remove/sync）。
    首次运行（.mem 不存在）时，自动从旧注册表迁移已有配置一次。"""

    def __init__(self, path):
        self._path = path
        self._data = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self._data = loaded
            except (OSError, ValueError):
                self._data = {}  # 文件损坏时退回默认，避免启动失败
        else:
            # self._migrate_from_registry()
            self._flush()

    def _migrate_from_registry(self):
        """从旧版 QSettings（注册表）导入一次已有配置，保证升级后设置不丢失。"""
        try:
            from PyQt5.QtCore import QSettings
            legacy = QSettings("2PSL", "AggregateGrading")
            for key in legacy.allKeys():
                v = legacy.value(key)
                if isinstance(v, QByteArray):
                    self._data[key] = self._encode_binary(bytes(v))
                elif isinstance(v, str) and v.startswith("@ByteArray(") and v.endswith(")"):
                    # Qt 文本格式的二进制记忆（布局/分割器状态），还原为字节后存储
                    try:
                        self._data[key] = self._encode_binary(v[11:-1].encode("latin-1"))
                    except UnicodeEncodeError:
                        continue
                elif isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                    self._data[key] = v
        except Exception:
            pass

    @staticmethod
    def _encode_binary(raw):
        """二进制记忆（窗口布局/分割器状态）以 base64 字典存入 JSON。"""
        return {"__qb64__": base64.b64encode(raw).decode("ascii")}

    def _flush(self):
        """将当前记忆写盘（先写临时文件再替换，避免写入中断导致损坏）。"""
        try:
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._path)
        except (OSError, TypeError, ValueError):
            pass  # 写入失败不影响程序运行

    def value(self, key, default=None):
        v = self._data.get(key, default)
        if isinstance(v, dict) and "__qb64__" in v:
            try:
                return QByteArray(base64.b64decode(v["__qb64__"]))
            except (ValueError, TypeError):
                return default
        return v

    def setValue(self, key, value):
        if isinstance(value, (bytes, bytearray, QByteArray)):
            value = self._encode_binary(bytes(value))
        self._data[key] = value
        self._flush()

    def remove(self, key):
        self._data.pop(key, None)
        self._flush()

    def path(self):
        return self._path

    def sync(self):
        self._flush()


class FontSize:
    """三档字体配置：每档用一组 6 个层级字号（L1 最大 … L6 最小），
    所有界面文字直接按层级编号 FontSize.lv(1)~lv(6) 取号，不再使用语义键名。
    层级含义：L1 窗口/面板标题，L2 标题与按钮，L3 正文内容，
    L4 小按钮，L5 次要信息，L6 图表。
    调整字体只需改 _LEVELS 里对应档位的那 6 个数字，即可一次性调整整个软件的所有字体。"""

    # 当前档位: "large" / "medium" / "small"
    PRESET = "large"

    # ========== 三档 × 六层级字号（唯一的调整入口）==========
    _LEVELS = {
        "large":  [15, 14, 13, 12, 11, 10],
        "medium": [13, 12, 11, 10, 9, 8],
        "small":  [11, 10, 9, 8, 7, 6],
    }

    @classmethod
    def lv(cls, n):
        """按层级取字号，n = 1~6（1 最大，6 最小）。"""
        return cls._LEVELS[cls.PRESET][n - 1]

    # ========== 切换档位 ==========
    @classmethod
    def set_preset(cls, preset):
        """preset: 'large' / 'medium' / 'small'"""
        if preset in cls._LEVELS:
            cls.PRESET = preset

    @classmethod
    def next_preset(cls):
        """循环切换: small -> medium -> large -> small"""
        order = ["small", "medium", "large"]
        idx = order.index(cls.PRESET)
        cls.PRESET = order[(idx + 1) % len(order)]
        return cls.PRESET

    @classmethod
    def preset_name(cls):
        names = {"large": "大", "medium": "中", "small": "小"}
        return names.get(cls.PRESET, cls.PRESET)
