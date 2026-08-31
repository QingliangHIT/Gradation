# -*- coding: utf-8 -*-
"""app.controllers —— 主窗口功能控制器（Mixin，按职责拆分业务逻辑）。

App 类通过多继承组装这些 Mixin；各 Mixin 仅通过 self 访问主窗口的
控件与状态，不自行持有状态。
"""
