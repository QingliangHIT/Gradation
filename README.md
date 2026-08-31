# 混凝土骨料智能筛分系统（Gradation）

基于多尺度图像处理的混凝土骨料颗粒智能实时筛分系统。系统通过图像采集/导入 → 图像预处理 → 实例分割模型推理 → 颗粒形态参数计算 → 粒径分布与级配分析的完整流程，实现骨料颗粒的非接触式快速检测，并支持批次管理、批量处理与报表导出。

## 功能特性

- **图像采集与浏览**：支持打开图片文件夹、拖拽图片、目录树多选管理（Ctrl/Shift 多选、Del 批量删除、右键打开所在目录），滚轮切换图片；内置独立相机窗口实现实时预览与拍照采集。
- **多种分割模型可插拔**：通过统一的模型注册机制（`model_registry`）管理分割算法，新模型注册后自动出现在界面下拉框中。当前内置：
  - 传统分水岭（传统二值化 + 距离变换分水岭）
  - UNet + 分水岭（UNet 语义分割 + 分水岭实例化，默认模型）
  - SAM 实例分割（Segment Anything 自动分割）
  - YOLO 实例分割（ultralytics，需在设置中配置 `.pt` 权重）
- **颗粒参数计算**：提取 20 余项形态指标，包括等效粒径、长短轴、周长、最大 Feret 径、偏心率、圆形度、圆整度、实心度、凸性、矩形填充率、径向变异系数、棱角性指数、角点密度等。
- **粒径与级配分析**：按标准筛孔（53 ~ 4.75 mm）计算分计筛余、累计筛余与累计通过率，生成级配曲线；内置 **GB/T 14685-2022** 连续级配标准区间（5~16 / 5~20 / 5~25 / 5~31.5 / 5~40）对照，支持插值求 D10 / D50 / D90 特征粒径。
- **批量处理**：对整个文件夹的图片执行完整流程，汇总每张图的颗粒数、平均/最大/最小粒径与特征粒径。
- **数据管理与导出**：颗粒参数表与级配结果导出为 Excel（中文列名映射），支持标定校准（像素当量）。
- **界面配置持久化**：窗口布局、字体档位、界面配色、上次打开目录等状态保存于项目级 `state.mem` 文件；字体采用"基数 + 层级偏移"自动推导机制，提供大/中/小三档。

## 项目结构

```
Gradation829/
├── projecet/                     # 主程序（PyQt5 图形界面）
│   ├── main.py                   # 程序入口与业务逻辑（App、后台工作线程）
│   ├── ui_main.py                # 主窗口 / 相机对话框界面布局
│   ├── dialogs.py                # 统一设置对话框
│   ├── font_config.py            # 字体配置与 MemSettings 持久化
│   ├── styles.py                 # 全局 QSS 样式
│   ├── state.mem                 # 界面状态记忆文件
│   └── algorithms/               # 算法模块
│       ├── model_registry.py     # 分割模型注册表（模块化管理中心）
│       ├── segmentation.py       # 分水岭 / UNet / SAM 分割实现
│       ├── process.py            # 图像预处理与传统二值化
│       ├── grading.py            # 级配计算与标准级配区间
│       ├── calibration.py        # 像素标定（像素当量标定）
│       ├── export.py             # 结果导出
│       ├── parameter_dialog.py   # 分割参数对话框
│       └── image_viewer.py       # 图像浏览组件
├── unet_project/                 # UNet 训练与推理
│   ├── train_unet.py             # 训练脚本（UNet / ResUNet / NNUNet 等）
│   ├── predict_unet.py           # 独立推理脚本
│   └── unet_model.py             # 模型定义
├── samInstance_project/          # SAM（segment-anything）本地源码
└── yolo_project/                 # YOLO 训练脚本
```

## 环境要求

- Python 3.10+（开发环境为 Python 3.13）
- 建议使用 conda 虚拟环境

### 安装依赖

基础运行依赖（主程序）：

```bash
pip install PyQt5 opencv-python numpy matplotlib pandas openpyxl scikit-image
```

模型训练 / 深度学习推理另需：

```bash
pip install torch torchvision tqdm   # UNet 训练与推理
pip install ultralytics              # YOLO 实例分割（可选）
pip install torch                    # SAM 分割依赖
```

## 运行

```bash
python projecet/main.py
```

### 使用流程

1. **打开图像**：打开文件夹或拖拽图片进入，也可通过"数据采集"连接相机拍照。
2. **像素标定**：在标定设置中输入像素当量（mm/px），保证粒径计算的物理单位正确。
3. **选择模型与参数**：在分割模型下拉框中选择模型，按需调整分割参数（距离变换阈值、形态学核大小等；SAM / YOLO 有各自的参数面板）。
4. **执行分析**：预处理 → 分割 → 参数测量 → 级配计算，结果以图层方式在图像区展示（原图、二值图、彩色实例图、级配标记图可滚轮/页签切换），颗粒参数表与级配曲线同步更新。
5. **导出结果**：将颗粒参数与级配数据导出为 Excel 报表。

## 模型扩展

新增分割模型只需在 `projecet/algorithms/model_registry.py` 中：

1. 实现 `run(img, params, stage) -> (binary, markers)` 函数（约定：`markers` 为实例标签图，背景 ≤ 1，实例从 2 开始）；
2. 调用 `register(ModelSpec(key=..., label=..., param_group=..., run=...))` 注册。

注册完成后，新模型自动出现在界面模型下拉框中，并按 `param_group` 显示对应参数面板。

## 标准依据

- 级配标准区间参照 **GB/T 14685-2022**《建设用卵石、碎石》连续粒级要求。
- 筛孔序列：53 / 37.5 / 31.5 / 26.5 / 19 / 16 / 9.5 / 4.75 mm（可在 `grading.py` 中调整）。

## 说明

- 级配计算以颗粒投影面积近似质量（`weights = d²`），后续可替换为密度修正。
- YOLO 与 SAM 模型需自行准备权重/模型文件，并在"设置 → 系统与推理"中配置路径。
- 界面状态保存于 `projecet/state.mem`，删除该文件可重置界面布局与配置。
