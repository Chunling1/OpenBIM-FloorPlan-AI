#!/usr/bin/env python3
"""
导出标注贡献 / Export Annotation Contribution

将标注好的掩码和对应的原始图片复制到 community_annotations/ 目录，
并更新 contributors.json 记录贡献信息。

Copies annotated masks and their corresponding source images to the
community_annotations/ directory, and updates contributors.json with
contribution metadata.

用法 / Usage:
    python label_tool/export_contribution.py --annotator-name "张三"
    python label_tool/export_contribution.py --annotator-name "Zhang San" \
        --data-dir my_images --mask-dir my_masks
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# 项目根目录 / Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMMUNITY_DIR = PROJECT_ROOT / "community_annotations"
COMMUNITY_IMAGES = COMMUNITY_DIR / "images"
COMMUNITY_MASKS = COMMUNITY_DIR / "annotations"
CONTRIBUTORS_FILE = COMMUNITY_DIR / "contributors.json"

# 支持的图片格式 / Supported image formats
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def find_source_image(image_name: str, data_dir: Path) -> Path | None:
    """
    根据掩码文件名（不含 .npy 扩展名）在 data_dir 中查找对应的原始图片。
    Find the source image in data_dir matching the mask filename (without .npy).
    """
    for ext in IMAGE_EXTENSIONS:
        candidate = data_dir / f"{image_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_contributors() -> dict:
    """加载 contributors.json，不存在则返回初始结构。"""
    if CONTRIBUTORS_FILE.exists():
        with open(CONTRIBUTORS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "annotations": [],
        "stats": {
            "total_images": 0,
            "total_annotated": 0,
            "contributors": [],
        },
    }


def save_contributors(data: dict) -> None:
    """保存 contributors.json。"""
    with open(CONTRIBUTORS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_contribution(
    annotator_name: str, data_dir: Path, mask_dir: Path
) -> None:
    """
    主导出逻辑：
    1. 扫描 mask_dir 下的 .npy 文件
    2. 复制掩码到 community_annotations/annotations/
    3. 复制对应原始图片到 community_annotations/images/
    4. 更新 contributors.json
    """
    # 确保目标目录存在 / Ensure target directories exist
    COMMUNITY_IMAGES.mkdir(parents=True, exist_ok=True)
    COMMUNITY_MASKS.mkdir(parents=True, exist_ok=True)

    # 扫描掩码文件 / Scan for mask files
    mask_files = sorted(mask_dir.glob("*.npy"))
    if not mask_files:
        print(f"❌ 在 {mask_dir} 中未找到 .npy 掩码文件。")
        print(f"❌ No .npy mask files found in {mask_dir}.")
        print("   请先使用标注工具标注图片。/ Please annotate images first.")
        sys.exit(1)

    print(f"📦 找到 {len(mask_files)} 个掩码文件 / Found {len(mask_files)} mask files")
    print(f"👤 标注者 / Annotator: {annotator_name}")
    print()

    exported_files = []
    skipped_no_image = []

    for mask_path in mask_files:
        image_name = mask_path.stem  # e.g., "floor_001"

        # 查找对应的原始图片 / Find corresponding source image
        source_image = find_source_image(image_name, data_dir)
        if source_image is None:
            skipped_no_image.append(mask_path.name)
            continue

        # 复制掩码 / Copy mask
        dst_mask = COMMUNITY_MASKS / mask_path.name
        shutil.copy2(mask_path, dst_mask)

        # 复制原始图片 / Copy source image
        dst_image = COMMUNITY_IMAGES / source_image.name
        shutil.copy2(source_image, dst_image)

        exported_files.append(
            {
                "image": source_image.name,
                "mask": mask_path.name,
            }
        )
        print(f"  ✅ {source_image.name} → {mask_path.name}")

    if not exported_files:
        print()
        print("❌ 没有成功导出任何文件。")
        print("❌ No files were exported successfully.")
        if skipped_no_image:
            print(f"   跳过了 {len(skipped_no_image)} 个掩码（找不到对应图片）:")
            for name in skipped_no_image:
                print(f"     - {name}")
        sys.exit(1)

    # 更新 contributors.json / Update contributors.json
    contributors_data = load_contributors()

    contribution_record = {
        "annotator": annotator_name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "num_files": len(exported_files),
        "files": exported_files,
    }
    contributors_data["annotations"].append(contribution_record)

    # 更新统计信息 / Update stats
    all_images = set()
    all_masks = set()
    all_contributors = set()
    for record in contributors_data["annotations"]:
        all_contributors.add(record["annotator"])
        for f in record["files"]:
            all_images.add(f["image"])
            all_masks.add(f["mask"])

    contributors_data["stats"] = {
        "total_images": len(all_images),
        "total_annotated": len(all_masks),
        "contributors": sorted(all_contributors),
    }

    save_contributors(contributors_data)

    # 打印总结 / Print summary
    print()
    print("=" * 60)
    print(f"📊 导出完成 / Export Complete")
    print(f"   导出掩码数: {len(exported_files)}")
    print(f"   Exported masks: {len(exported_files)}")
    if skipped_no_image:
        print(f"   跳过（无对应图片）: {len(skipped_no_image)}")
        print(f"   Skipped (no source image): {len(skipped_no_image)}")
    print(f"   总贡献者数: {len(all_contributors)}")
    print(f"   Total contributors: {len(all_contributors)}")
    print("=" * 60)

    # 打印提交 PR 的说明 / Print PR submission instructions
    print()
    print("📋 下一步：提交 Pull Request / Next: Submit a Pull Request")
    print("-" * 60)
    print()
    print("1. 添加文件到 Git / Add files to Git:")
    print("   git add community_annotations/")
    print()
    print(f'2. 提交更改 / Commit changes:')
    print(f'   git commit -m "Add annotations by {annotator_name}"')
    print()
    print("3. 推送到你的 Fork / Push to your fork:")
    print("   git push origin main")
    print()
    print("4. 在 GitHub 上创建 Pull Request / Create a Pull Request on GitHub:")
    print("   - 访问你的 Fork 仓库页面 / Go to your fork's page")
    print("   - 点击 'Contribute' → 'Open pull request'")
    print(f"   - 标题建议 / Suggested title:")
    print(f'     "Add {len(exported_files)} annotations by {annotator_name}"')
    print()
    print("🙏 感谢你的贡献！/ Thank you for your contribution!")


def main():
    parser = argparse.ArgumentParser(
        description="导出标注贡献到 community_annotations/ 目录\n"
        "Export annotation contributions to community_annotations/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--annotator-name",
        required=True,
        help="标注者姓名 / Annotator name (e.g., '张三' or 'Zhang San')",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=COMMUNITY_IMAGES,
        help="原始图片目录 / Source images directory "
        f"(default: {COMMUNITY_IMAGES.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=COMMUNITY_MASKS,
        help="掩码输出目录 / Mask output directory "
        f"(default: {COMMUNITY_MASKS.relative_to(PROJECT_ROOT)})",
    )

    args = parser.parse_args()

    # 解析相对路径 / Resolve relative paths
    data_dir = args.data_dir.resolve()
    mask_dir = args.mask_dir.resolve()

    if not data_dir.exists():
        print(f"❌ 图片目录不存在 / Image directory does not exist: {data_dir}")
        sys.exit(1)
    if not mask_dir.exists():
        print(f"❌ 掩码目录不存在 / Mask directory does not exist: {mask_dir}")
        sys.exit(1)

    export_contribution(args.annotator_name, data_dir, mask_dir)


if __name__ == "__main__":
    main()
