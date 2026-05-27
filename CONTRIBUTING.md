# 贡献指南 / Contributing Guide

感谢你对 **OpenBIM-FloorPlan-AI** 项目的关注！我们欢迎各种形式的贡献，包括标注数据和代码改进。

Thank you for your interest in **OpenBIM-FloorPlan-AI**! We welcome all forms of contribution, including annotation data and code improvements.

---

## 📋 目录 / Table of Contents

- [标注贡献 / Annotation Contribution](#标注贡献--annotation-contribution)
- [代码贡献 / Code Contribution](#代码贡献--code-contribution)
- [行为准则 / Code of Conduct](#行为准则--code-of-conduct)

---

## 标注贡献 / Annotation Contribution

我们最需要的贡献是**高质量的户型图语义分割标注数据**。即使你不会写代码，也可以通过标注来帮助改进模型！

Our most needed contribution is **high-quality semantic segmentation annotations for floor plan images**. Even if you don't code, you can help improve the model by annotating!

### 前置要求 / Prerequisites

- Python 3.10+
- Git
- 浏览器（Chrome/Firefox 推荐）/ A modern browser (Chrome/Firefox recommended)

### 标注步骤 / Step-by-Step Guide

#### 1️⃣ Fork 本仓库 / Fork this repo

在 GitHub 页面右上角点击 **Fork** 按钮，将仓库复制到你的账号下。

Click the **Fork** button in the top-right corner of the GitHub page to copy the repo to your account.

#### 2️⃣ 克隆你的 Fork / Clone your fork

```bash
git clone https://github.com/YOUR_USERNAME/OpenBIM-FloorPlan-AI.git
cd OpenBIM-FloorPlan-AI
pip install -r requirements.txt
```

#### 3️⃣ 获取待标注图片 / Get images to annotate

**方式 A：自动下载 / Option A: Auto download**

```bash
python label_tool/download_batch.py
```

**方式 B：使用自己的 CAD 图 / Option B: Use your own CAD images**

将你的 CAD 户型图放入 `community_annotations/images/` 目录：

Place your CAD floor plan images in `community_annotations/images/`:

```bash
cp /path/to/your/cad_images/*.png community_annotations/images/
```

#### 4️⃣ 启动标注工具 / Run the labeling tool

```bash
python label_tool/server.py \
    --data-dir community_annotations/images \
    --mask-dir community_annotations/annotations
```

#### 5️⃣ 在浏览器中标注 / Annotate in browser

打开浏览器访问：

Open your browser and go to:

```
http://localhost:8099
```

**标注类别 / Label Classes:**

| 快捷键 / Key | 类别 ID | 颜色 | 名称 | Description |
|:---:|:---:|:---:|------|-------------|
| `W` | 1 | 🔴 | 墙体 | Wall |
| `E` | 2 | 🔵 | 窗户 | Window |
| `D` | 3 | 🟢 | 门 | Door |
| `X` | — | ⚪ | 橡皮擦 | Eraser |

**标注技巧 / Annotation Tips:**

- 🖱️ 使用画笔工具涂抹区域 / Use the brush tool to paint regions
- ⌨️ 数字键快速切换类别 / Number keys to switch classes quickly
- 🔍 滚轮缩放查看细节 / Scroll to zoom for details
- ↩️ Ctrl+Z 撤销 / Ctrl+Z to undo
- 💾 每张图标注完成后会自动保存 / Auto-saves after each image

#### 6️⃣ 导出你的贡献 / Export your contribution

标注完成后，运行导出脚本：

After annotating, run the export script:

```bash
python label_tool/export_contribution.py --annotator-name "你的名字"
```

脚本会自动将标注数据整理到 `community_annotations/` 目录，并更新贡献者记录。

The script will organize your annotations into `community_annotations/` and update the contributor records.

#### 7️⃣ 提交并推送 / Commit and push

```bash
git add community_annotations/
git commit -m "Add annotations by YourName"
git push origin main
```

#### 8️⃣ 创建 Pull Request / Open a Pull Request

1. 访问你的 Fork 仓库的 GitHub 页面 / Go to your fork on GitHub
2. 点击 **"Contribute"** → **"Open pull request"** / Click **"Contribute"** → **"Open pull request"**
3. 填写 PR 说明，包括：/ Fill in the PR description, including:
   - 标注了多少张图片 / How many images you annotated
   - 图片来源（下载/自有 CAD）/ Image source (downloaded / own CAD)
   - 任何特殊情况说明 / Any special notes

> 💡 **提示 / Tip**: PR 标题建议格式 / Suggested PR title format:
> `Add N annotations by YourName`

---

## 代码贡献 / Code Contribution

我们同样欢迎代码方面的贡献，包括但不限于：

We also welcome code contributions, including but not limited to:

- 🐛 Bug 修复 / Bug fixes
- ✨ 新功能 / New features
- 📝 文档改进 / Documentation improvements
- ⚡ 性能优化 / Performance optimizations
- 🧪 测试用例 / Test cases

### 代码贡献流程 / Code Contribution Workflow

#### 1️⃣ 创建 Issue / Create an Issue

在开始编码之前，请先创建一个 Issue 描述你想做的改动，以便讨论方案。

Before coding, please create an Issue describing the change you'd like to make, so we can discuss the approach.

#### 2️⃣ Fork 并创建分支 / Fork and create a branch

```bash
git clone https://github.com/YOUR_USERNAME/OpenBIM-FloorPlan-AI.git
cd OpenBIM-FloorPlan-AI
git checkout -b feature/your-feature-name
```

**分支命名约定 / Branch naming convention:**
- `feature/xxx` — 新功能 / New features
- `fix/xxx` — Bug 修复 / Bug fixes
- `docs/xxx` — 文档更新 / Documentation updates

#### 3️⃣ 编写代码 / Write code

请遵循以下规范 / Please follow these guidelines:

- 代码风格遵循 PEP 8 / Follow PEP 8 style
- 添加必要的注释（中英文均可）/ Add necessary comments (Chinese or English)
- 新功能请附带测试 / Include tests for new features
- 确保现有测试通过 / Ensure existing tests pass

#### 4️⃣ 提交并推送 / Commit and push

```bash
git add .
git commit -m "feat: 简要描述你的改动"
git push origin feature/your-feature-name
```

**Commit 消息格式 / Commit message format:**
- `feat: xxx` — 新功能 / New feature
- `fix: xxx` — Bug 修复 / Bug fix
- `docs: xxx` — 文档 / Documentation
- `refactor: xxx` — 重构 / Refactoring
- `test: xxx` — 测试 / Tests

#### 5️⃣ 创建 Pull Request / Create a Pull Request

- 在 PR 中详细描述你的改动 / Describe your changes in detail in the PR
- 关联相关 Issue / Link related Issues
- 等待代码审查 / Wait for code review

---

## 行为准则 / Code of Conduct

- 🤝 保持友善和尊重 / Be friendly and respectful
- 💬 使用中文或英文交流均可 / Chinese and English are both welcome
- 📖 遵循开源社区最佳实践 / Follow open-source community best practices

---

## 🙏 致谢 / Acknowledgments

感谢每一位贡献者！你的标注数据和代码改进直接推动了户型图 AI 分析技术的进步。

Thank you to every contributor! Your annotations and code improvements directly advance floor plan AI analysis technology.

所有贡献者都会被记录在 `community_annotations/contributors.json` 中。

All contributors are tracked in `community_annotations/contributors.json`.
