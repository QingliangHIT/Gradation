# 混凝土骨料智能筛分系统（Gradation）

基于多尺度图像处理的混凝土骨料颗粒智能实时筛分系统。系统通过图像采集/导入 → 图像预处理 → 实例分割模型推理 → 颗粒形态参数计算 → 粒径分布与级配分析的完整流程，实现骨料颗粒的非接触式快速检测，并支持批次管理、批量处理与报表导出。

## 功能特性

- **图像采集与浏览**：支持打开图片文件夹、拖拽图片、目录树多选管理（Ctrl/Shift 多选、Del 批量删除、右键打开所在目录），滚轮切换图片；内置独立相机窗口实现实时预览与拍照采集。
- **多种分割模型可插拔**：通过统一的模型注册机制（`core/registry.py`）管理分割算法，新模型注册后自动出现在界面下拉框中。当前内置：
  - 传统分水岭（传统二值化 + 距离变换分水岭）
  - UNet + 分水岭（UNet 语义分割 + 分水岭实例化，默认模型）
  - SAM 实例分割（Segment Anything 自动分割）
  - YOLO 实例分割（ultralytics，需在设置中配置 `.pt` 权重）
- **颗粒参数计算**：提取 20 余项形态指标，包括等效粒径、长短轴、周长、最大 Feret 径、偏心率、圆形度、圆整度、实心度、凸性、矩形填充率、径向变异系数、棱角性指数、角点密度等。
- **粒径与级配分析**：按标准筛孔（53 ~ 4.75 mm）计算分计筛余、累计筛余与累计通过率，生成级配曲线；内置 **GB/T 14685-2022** 连续级配标准区间（5~16 / 5~20 / 5~25 / 5~31.5 / 5~40）对照，支持插值求 D10 / D50 / D90 特征粒径。
- **智能分析报告**：一键生成特征粒径、均匀性/曲率系数、细度模数、标准符合性与颗粒形态评价报告。
- **批量处理**：对整个文件夹的图片执行完整流程，汇总每张图的颗粒数、平均/最大/最小粒径与特征粒径。
- **数据管理与导出**：颗粒参数表与级配结果导出为 Excel（中文列名映射），支持标定校准（像素当量）。
- **界面配置持久化**：窗口布局、字体档位、界面配色、上次打开目录等状态保存于项目级 `state.mem` 文件；字体采用"层级字号 + Ctrl+滚轮偏移"机制，提供大/中/小三档。

## 项目结构

```
Gradation829/
├── run.py                        # 程序入口（python run.py）
├── common/                       # 通用工具层（无 Qt / 算法依赖）
│   └── image_io.py               # 中文路径安全图像读写、缩放/切分/拼接
├── core/                         # 算法核心层（无 Qt 依赖，可独立复用）
│   ├── config.py                 # 推理运行配置（设备/权重路径，签名比对缓存）
│   ├── preprocess.py             # 图像预处理与传统二值化
│   ├── watershed.py              # 距离变换分水岭实例化
│   ├── measure.py                # 颗粒形态参数测量、着色与标记叠加
│   ├── grading.py                # 级配计算、标准级配区间与 D10/D50/D90
│   ├── unet_infer.py             # UNet 推理适配器（模型缓存）
│   ├── sam_infer.py              # SAM 推理适配器
│   └── registry.py               # 分割模型注册表（ModelSpec，模块化管理中心）
├── app/                          # PyQt5 界面层
│   ├── application.py            # App 主窗口组装（各 Mixin）与 main() 启动入口
│   ├── config.py                 # 路径常量、state.mem 记忆（MemSettings）、三档字体
│   ├── workers.py                # 后台工作线程（分割 / 批量处理）
│   ├── styles/                   # 主题配色（themes）与全局 QSS（qss）
│   ├── ui/                       # 界面组件
│   │   ├── main_window.py        # 主窗口骨架（菜单/工具栏/停靠栏布局）
│   │   ├── image_viewer.py       # 图像查看器基类（缩放/像素信息/颗粒点击）
│   │   ├── layer_viewer.py       # 多图层主视图窗（滚轮切换图层）
│   │   ├── step_indicator.py     # 步骤状态指示器
│   │   ├── camera_dialog.py      # 相机采集对话框
│   │   ├── settings_dialog.py    # 统一设置对话框（算法参数 / 系统与推理）
│   │   └── report_dialog.py      # 智能分析报告 / 批量结果预览对话框
│   └── controllers/              # 业务控制器（Mixin，按职责拆分）
│       ├── pipeline.py           # 三步处理流程（预处理 → 分割 → 分析）
│       ├── result_view.py        # 统计表 / 级配曲线 / 直方图更新与交互
│       ├── viewer_sync.py        # 多图窗缩放/平移锁定同步与像素信息
│       ├── workspace.py          # 工作区目录树 / 回收站删除
│       ├── project_io.py         # 项目文件保存与打开（.json）
│       ├── exports.py            # 结果 / 颗粒详情 / 图表 / 日志导出
│       ├── batch.py              # 批量处理任务管理
│       ├── analysis.py           # 智能分析报告（纯函数 + 界面入口）
│       ├── appearance.py         # 设置 / 主题 / 字体 / 布局管理
│       └── capture.py            # 图像打开与相机采集
├── models/                       # 模型训练工程（与界面解耦，可独立运行）
│   ├── unet/                     # UNet 系列网络与训练 / 预测
│   │   ├── blocks.py             # 基础卷积块（DoubleConv / ResConv / nnUNet 块）
│   │   ├── unet.py               # UNet / ResUNet
│   │   ├── nnunet.py             # NNUNet / NNUNetv2（深度监督）
│   │   ├── dataset.py            # 颗粒分割数据集
│   │   ├── metrics.py            # IoU / Dice / Acc（tensor 与 numpy 两版）
│   │   ├── train.py              # 训练脚本（python -m models.unet.train）
│   │   ├── predict.py            # 推理与结果叠加
│   │   └── viewer.py             # 交互式预测查看器（python -m models.unet.viewer）
│   └── yolo/
│       └── train.py              # YOLO 分割预测演示（python -m models.yolo.train）
├── third_party/                  # 第三方源码
│   ├── segment_anything/         # SAM（segment-anything）官方源码
│   └── setup.py
└── state.mem                     # 界面状态记忆文件（自动生成）
```

## 环境要求

- Python 3.10+（开发环境为 Python 3.13）
- 建议使用 conda 虚拟环境

### 安装依赖

```bash
pip install -r requirements.txt
```

模型训练 / 深度学习推理另需：

```bash
pip install torch torchvision tqdm   # UNet 训练与推理
pip install ultralytics              # YOLO 实例分割（可选）
```

## 运行

```bash
python run.py
```

### 使用流程

1. **打开图像**：打开文件夹或拖拽图片进入，也可通过"数据采集"连接相机拍照。
2. **像素标定**：在设置中输入像素当量（mm/px），保证粒径计算的物理单位正确。
3. **选择模型与参数**：在分割模型下拉框中选择模型，按需调整分割参数（距离变换阈值、形态学核大小等；SAM / YOLO 有各自的参数面板）。
4. **执行分析**：预处理 → 分割 → 参数测量 → 级配计算，结果以图层方式在图像区展示（原图、二值图、彩色实例图、级配标记图可滚轮切换），颗粒参数表与级配曲线同步更新。
5. **导出结果**：将颗粒参数与级配数据导出为 Excel 报表，或生成智能分析报告。

## 模型扩展

新增分割模型只需在 `core/registry.py` 中：

1. 实现 `run(img, params, stage) -> (binary, markers)` 函数（约定：`markers` 为实例标签图，背景 ≤ 1，实例从 2 开始；无结果时 `binary` 返回 `None`，不得伪造空掩膜）；
2. 调用 `register(ModelSpec(key=..., label=..., param_group=..., run=...))` 注册。

注册完成后，新模型自动出现在界面模型下拉框中，并按 `param_group` 显示对应参数面板。

## 标准依据

- 级配标准区间参照 **GB/T 14685-2022**《建设用卵石、碎石》连续粒级要求。
- 筛孔序列：53 / 37.5 / 31.5 / 26.5 / 19 / 16 / 9.5 / 4.75 mm（可在 `core/grading.py` 中调整）。

## 说明

- 级配计算以颗粒投影面积近似质量（`weights = d²`），后续可替换为密度修正。
- YOLO 与 SAM 模型需自行准备权重/模型文件，并在"设置 → 系统与推理"中配置路径。
- 界面状态保存于根目录 `state.mem`，删除该文件可重置界面布局与配置。
