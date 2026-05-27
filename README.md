<div align="center">
  <h1>🏗️ OpenBIM FloorPlan AI</h1>
  <p><strong>建筑平面图语义分割系统 | Semantic Segmentation for Architectural Floor Plans</strong></p>

  <p>
    <a href="https://github.com/Chunling1/OpenBIM-FloorPlan-AI"><img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
    <a href="https://github.com/Chunling1/OpenBIM-FloorPlan-AI"><img alt="Python" src="https://img.shields.io/badge/Python-3.9+-green.svg"></a>
    <a href="https://github.com/Chunling1/OpenBIM-FloorPlan-AI"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg"></a>
    <a href="https://github.com/Chunling1/OpenBIM-FloorPlan-AI"><img alt="Model" src="https://img.shields.io/badge/mIoU-0.787-orange.svg"></a>
  </p>

  <p>
    <a href="#-快速开始">快速开始</a> •
    <a href="#-模型性能">模型性能</a> •
    <a href="#-标注工具">标注工具</a> •
    <a href="#-部署">部署</a> •
    <a href="#english">English</a>
  </p>
</div>

---

## 📖 项目简介

**OpenBIM FloorPlan AI** 是一个将 2D 建筑平面图（图片/扫描 PDF）自动转换为结构化语义数据的深度学习系统。系统自动识别 **墙体、窗户、门** 三大建筑构件，支持后续 BIM 建模和建筑能耗仿真。

### 核心特性

- 🎯 **高精度分割** — 基于 UNet + ResNet34，CubiCasa5K 数据集 mIoU 达 **0.787**
- 🔄 **域适应** — 专门优化了黑底 CAD 施工图的识别效果，解决学术数据集→实际工程图的域迁移问题
- ⚡ **CPU 部署** — 提供 ONNX Runtime 推理方案，无需 GPU，单张推理 < 1.5s
- 🌐 **Web 平台** — 开箱即用的 FastAPI 服务 + Glassmorphism 前端界面
- 🏢 **BIM 集成** — 分割结果可自动转换为 IFC 文件，接入建筑能耗计算引擎
- 🖌️ **标注工具** — 内置浏览器端语义标注工具，支持自定义数据集扩展

### 语义类别

| 类别 ID | 名称 | 颜色 | 说明 |
|---------|------|------|------|
| 0 | Background | — | 背景 |
| 1 | Wall | 🔴 红色 | 墙体 + 栏杆 |
| 2 | Window | 🔵 蓝色 | 窗户 |
| 3 | Door | 🟢 绿色 | 门（单开门/双开门/推拉门） |

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/Chunling1/OpenBIM-FloorPlan-AI.git
cd OpenBIM-FloorPlan-AI
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载预训练模型

从 [GitHub Releases](https://github.com/Chunling1/OpenBIM-FloorPlan-AI/releases) 下载模型权重，放入 `models/` 目录：

| 模型 | 文件 | 大小 | mIoU |
|------|------|------|------|
| M2-DA (推荐) | `M2_DA_FT_v2_best.pt` | 93 MB | **0.787** |
| M1-Light | `M1_LightUNet_best.pt` | 30 MB | 0.685 |
| M3-DeepLab | `M3_DeepLabV3p_EffB4_best.pt` | 72 MB | — |

### 4. 启动 Web 服务

```bash
python server.py
# 访问 http://localhost:8070
```

---

## 📊 模型性能

### 模型对比

| 模型 | 架构 | 骨干网络 | 输入尺寸 | 参数量 | mIoU | 推理速度 |
|------|------|---------|---------|--------|------|---------|
| M1 | LightUNet | 自定义 (5层) | 256×256 | ~1.5M | 0.685 | ~0.3s |
| **M2-DA** | **UNet** | **ResNet34** | **512×512** | **~24M** | **0.787** | **~0.8s** |
| M3 | DeepLabV3+ | EfficientNet-B4 | 512×512 | ~19M | — | ~1.0s |

### 分类 IoU 详情 (M2-DA)

| Background | Wall | Window | Door | **Mean** |
|-----------|------|--------|------|----------|
| 0.970 | 0.739 | 0.765 | 0.661 | **0.787** |

### 域适应效果

| 配置 | CubiCasa5K mIoU | CAD 黑底图效果 |
|------|-----------------|---------------|
| M2 原始 | 0.742 | 差 |
| M2 + 域随机化 | **0.787** | 好 |

---

## 🏋️ 训练

### 数据准备

```bash
# 1. 下载 CubiCasa5K 数据集
python download_dataset.py

# 2. 预计算精确 Mask（从 SVG 标注生成）
python precompute_masks.py
```

### 开始训练

```bash
# 训练 M1 (轻量版, ~3h on RTX 4050)
python train_all.py --model m1

# 训练 M2 (推荐, ~11h on RTX 4050)
python train_all.py --model m2

# 训练 M3 (DeepLabV3+)
python train_all.py --model m3
```

### 域适应微调

```bash
# 对 M2 做域适应训练，提升 CAD 施工图识别能力
python finetune_domain.py
```

### 评估

```bash
# CubiCasa5K 测试集评估
python eval_test_set.py

# 完整 Benchmark (含消融实验)
python benchmark_eval.py
```

---

## 🖌️ 标注工具

内置浏览器端语义标注工具，支持对自定义建筑平面图进行像素级标注。

```bash
cd label_tool
python server.py
# 访问 http://localhost:8099
```

**功能特性：**
- 画笔/橡皮擦工具，支持笔刷大小调节
- 墙体(红)/窗户(蓝)/门(绿) 三类标注
- 标注结果自动保存为 `.npy` 格式
- 实时标注进度统计

---

## 🚀 部署

### ONNX CPU 部署（推荐）

无需 PyTorch 和 GPU，适合服务器生产环境：

```bash
pip install onnxruntime opencv-python-headless numpy
```

```python
from deploy.floorplan_onnx import get_segmenter

segmenter = get_segmenter('models/M2_DA_best.onnx')
result = segmenter.predict(image, use_preprocessing=True)

# result['mask']     - 语义分割 Mask
# result['overlay']  - 可视化叠加图
# result['geometry'] - 墙/窗/门几何信息
```

### FastAPI 服务

```bash
python server.py  # http://localhost:8070

# API 端点：
# POST /api/predict          - 图片分割
# POST /energy/ai_recognize  - AI 识别建筑构件
# POST /energy/ai_simulate   - 建筑能耗仿真
```

---

## 🏗️ 扩展功能

### Mask → IFC 自动建模

将分割结果自动转换为 IFC 2x3 BIM 文件：

```python
from mask_to_ifc import mask_to_ifc
mask_to_ifc('segmentation_mask.npy', 'output.ifc')
```

### 建筑能耗计算

基于度日数法，支持 5 个中国城市气候区：

```python
from energy_calc import calculate_energy
result = calculate_energy(geometry_data, city='北京')
# 输出: 供暖/制冷/照明/设备/通风/热水 6 项 EUI
```

---

## 📁 项目结构

```
OpenBIM-FloorPlan-AI/
├── train_all.py              # 统一训练脚本 (M1/M2/M3)
├── train.py / train_v2.py    # 历史训练版本
├── finetune_domain.py        # 域适应微调
├── precompute_masks.py       # CubiCasa5K Mask 预计算
├── download_dataset.py       # 数据集下载
├── inference_api.py          # PyTorch 推理 API
├── server.py                 # FastAPI Web 服务
├── energy_calc.py            # 建筑能耗计算引擎
├── mask_to_ifc.py            # Mask → IFC 转换
├── eval_test_set.py          # 测试集评估
├── benchmark_eval.py         # 完整 Benchmark
│
├── models/                   # 模型权重 (从 Release 下载)
├── deploy/                   # ONNX 部署脚本
│   └── floorplan_onnx.py
├── web/                      # Web 前端
│   ├── index.html
│   ├── app.js
│   └── style.css
├── label_tool/               # 标注工具
│   ├── server.py
│   └── static/index.html
└── data/                     # 数据目录
```

---

## 🤝 Contributing

欢迎贡献！可以参与的方向：

- 🏢 **商业建筑数据集** — 收集标注办公楼、酒店、商场等商业建筑平面图
- 📐 **新语义类别** — 扩展识别楼梯、电梯、家具等更多构件
- 🧠 **模型优化** — 尝试更好的骨干网络或分割架构
- 🌍 **数据增强** — 更多域适应策略
- 📱 **前端优化** — React/Vue 重构前端

```bash
# Fork 仓库后
git checkout -b feature/your-feature
# 修改代码...
git commit -m "Add: your feature"
git push origin feature/your-feature
# 提交 Pull Request
```

---

## 📄 License

[MIT License](LICENSE)

---

<a name="english"></a>

## English

**OpenBIM FloorPlan AI** is a deep learning system that automatically converts 2D architectural floor plans into structured semantic data. It detects **walls, windows, and doors** with a domain-adapted UNet+ResNet34 model achieving **0.787 mIoU** on CubiCasa5K. Features include ONNX CPU deployment, a built-in web annotation tool, IFC export, and building energy simulation.

See the Chinese documentation above for full details, or check the [docs/](docs/) directory.
