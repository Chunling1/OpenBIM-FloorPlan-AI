# 社区标注数据 / Community Annotation Data

本目录用于收集社区贡献的户型图语义分割标注数据。

This directory collects community-contributed semantic segmentation annotations for floor plan images.

## 目录结构 / Directory Structure

```
community_annotations/
├── README.md              # 本说明文件 / This README
├── images/                # 待标注的原始图片 / Unlabeled source images
├── annotations/           # 提交的标注掩码 (.npy) / Submitted annotation masks (.npy)
└── contributors.json      # 贡献者追踪记录 / Track who annotated what
```

## 各目录说明 / Directory Descriptions

### `images/` — 原始图片 / Source Images

存放待标注或已标注的户型图原始图片。支持的格式：`.png`, `.jpg`, `.jpeg`。

Contains raw floor plan images (labeled or unlabeled). Supported formats: `.png`, `.jpg`, `.jpeg`.

- 你可以使用 `label_tool/download_batch.py` 自动下载图片
- 也可以手动放入你自己的 CAD 户型图

### `annotations/` — 标注掩码 / Annotation Masks

存放与图片对应的语义分割标注掩码，格式为 NumPy `.npy` 文件。每个掩码文件的命名应与对应图片文件名一致（去掉图片扩展名，加 `.npy`）。

Contains semantic segmentation masks corresponding to images, stored as NumPy `.npy` files. Each mask file should match the corresponding image filename (replace image extension with `.npy`).

例如 / Example:
```
images/floor_001.png  →  annotations/floor_001.npy
images/cad_002.jpg    →  annotations/cad_002.npy
```

**标注类别 / Label Classes:**

| 类别 ID | 名称 | Description |
|---------|------|-------------|
| 0 | 背景 | Background |
| 1 | 墙体 | Wall |
| 2 | 门 | Door |
| 3 | 窗户 | Window |
| 4 | 房间 | Room |

### `contributors.json` — 贡献者记录 / Contributor Records

自动生成的 JSON 文件，记录每次标注贡献的元信息：

Auto-generated JSON file tracking metadata for each annotation contribution:

- **annotator**: 贡献者姓名 / Contributor name
- **date**: 提交日期 / Submission date
- **files**: 标注的文件列表 / List of annotated files

## 如何贡献标注 / How to Contribute Annotations

请参阅项目根目录下的 [CONTRIBUTING.md](../CONTRIBUTING.md) 了解完整的贡献流程。

Please refer to [CONTRIBUTING.md](../CONTRIBUTING.md) in the project root for the full contribution workflow.

简要步骤 / Quick Steps:

1. Fork 并克隆本仓库 / Fork and clone this repo
2. 下载图片或使用自己的 CAD 图 / Download images or use your own CAD drawings
3. 运行标注工具 / Run the labeling tool:
   ```bash
   python label_tool/server.py --data-dir community_annotations/images --mask-dir community_annotations/annotations
   ```
4. 在浏览器中标注 / Annotate in browser at `http://localhost:8099`
5. 导出贡献 / Export contribution:
   ```bash
   python label_tool/export_contribution.py --annotator-name "你的名字"
   ```
6. 提交 Pull Request / Submit a Pull Request

## 数据质量要求 / Data Quality Requirements

- 掩码尺寸必须与对应图片尺寸一致 / Mask dimensions must match the corresponding image
- 所有像素值必须在有效类别范围内（0-4）/ All pixel values must be within valid class range (0-4)
- 墙体和门窗的边界应尽可能准确 / Wall, door, and window boundaries should be as precise as possible
- 优先标注清晰、高质量的户型图 / Prioritize clear, high-quality floor plan images

## 许可证 / License

贡献的标注数据将遵循项目主许可证（MIT）。提交即表示您同意以此许可证共享数据。

Contributed annotations are licensed under the project's main license (MIT). By submitting, you agree to share your data under this license.
