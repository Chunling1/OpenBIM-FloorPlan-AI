"""
只评估外围轮廓的 Wall/Window IoU
策略：
1. 从GT mask找到建筑物最外围轮廓（最大连通域的外边界）
2. 在外围轮廓附近创建一个 band mask（向内外各扩N像素）
3. 只在 band 内计算 Wall/Window IoU
"""
import os
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import torch
import segmentation_models_pytorch as smp
import albumentations as A
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "cubicasa5k"
REPORT_DIR = BASE_DIR / "output_paper" / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 512
NUM_CLASSES = 4
BAND_WIDTH = 30  # 外围band宽度（像素），在原图分辨率上

CLASS_COLORS = [(40,40,40), (231,76,60), (52,152,219), (46,204,113)]


def load_model(path):
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=NUM_CLASSES)
    ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def predict(model, img_rgb, device):
    aug = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    t = aug(image=img_rgb)
    tensor = torch.from_numpy(t['image'].transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).argmax(dim=1).squeeze().cpu().numpy()
    return pred


def extract_exterior_band(mask, band_width=30):
    """
    从GT mask提取最外围轮廓的band区域。
    返回 band_mask: 布尔数组，True = 外围区域
    """
    # 1. 二值化：所有非背景像素
    building = (mask > 0).astype(np.uint8)

    # 2. 填充内部空洞 → 得到实心建筑区域
    # 用 floodFill 从边角填充背景，剩下的就是建筑内部
    h, w = building.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    building_filled = building.copy()
    # 从四个角flood fill（确保外部背景被标记）
    for seed in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]:
        if building_filled[seed[1], seed[0]] == 0:
            cv2.floodFill(building_filled, flood_mask, seed, 2)
    # 外部背景=2，内部空洞=0，建筑=1
    # 建筑实心区域 = 建筑(1) + 内部空洞(0 that's not external)
    exterior_bg = (building_filled == 2)
    solid_building = ~exterior_bg  # 建筑物实心区域（含内部空洞）

    # 3. 提取外轮廓
    solid_u8 = solid_building.astype(np.uint8)
    contours, _ = cv2.findContours(solid_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros_like(mask, dtype=bool)

    # 取最大轮廓
    largest = max(contours, key=cv2.contourArea)

    # 4. 创建轮廓线mask
    contour_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(contour_mask, [largest], -1, 255, thickness=1)

    # 5. 膨胀 → 得到band
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_width*2+1, band_width*2+1))
    band = cv2.dilate(contour_mask, kernel, iterations=1) > 0

    return band


def compute_exterior_iou(pred, gt, band_mask, cls_id):
    """只在band_mask区域内计算特定类别的IoU"""
    pred_cls = (pred == cls_id) & band_mask
    gt_cls = (gt == cls_id) & band_mask
    inter = (pred_cls & gt_cls).sum()
    union = (pred_cls | gt_cls).sum()
    if union == 0:
        return float('nan')
    return inter / union


def mask_to_color(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        color[mask == c] = CLASS_COLORS[c]
    return color


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    models = {}
    model_files = {
        'M2-DA': BASE_DIR / "models" / "M2_UNet_ResNet34_DA_best.pt",
        'M2-Orig': BASE_DIR / "models" / "M2_UNet_ResNet34_best.pt",
    }
    for name, path in model_files.items():
        if path.exists():
            models[name] = load_model(path).to(device)
            print(f"Loaded {name}")

    # 读验证集
    val_file = DATA_DIR / "val.txt"
    all_samples = [l.strip().strip('/') for l in val_file.read_text().strip().split('\n') if l.strip()]
    valid_samples = [s for s in all_samples if (DATA_DIR / s / "mask.npy").exists()]
    print(f"Val samples: {len(valid_samples)}")

    # 全量评估
    results = {name: {'wall_ext': [], 'window_ext': [], 'wall_all': [], 'window_all': [],
                       'wall_int': [], 'window_int': []}
               for name in models}

    for i, sample_rel in enumerate(valid_samples):
        sample_dir = DATA_DIR / sample_rel
        pil_img = Image.open(str(sample_dir / "F1_scaled.png")).convert('RGB')
        img_rgb = np.array(pil_img)
        pil_img.close()

        gt = np.load(str(sample_dir / "mask.npy"))
        if gt.shape[:2] != img_rgb.shape[:2]:
            gt = cv2.resize(gt, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt = np.clip(gt, 0, 3)

        # 提取外围band
        band = extract_exterior_band(gt, band_width=BAND_WIDTH)
        interior = ~band  # 内部区域

        if band.sum() == 0:
            continue

        for name, model in models.items():
            pred_raw = predict(model, img_rgb, device)
            pred = cv2.resize(pred_raw.astype(np.uint8),
                              (img_rgb.shape[1], img_rgb.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

            # 外围IoU
            w_ext = compute_exterior_iou(pred, gt, band, 1)  # wall
            wi_ext = compute_exterior_iou(pred, gt, band, 2)  # window

            # 全图IoU
            full = np.ones_like(gt, dtype=bool)
            w_all = compute_exterior_iou(pred, gt, full, 1)
            wi_all = compute_exterior_iou(pred, gt, full, 2)

            # 内部IoU
            w_int = compute_exterior_iou(pred, gt, interior, 1)
            wi_int = compute_exterior_iou(pred, gt, interior, 2)

            results[name]['wall_ext'].append(w_ext)
            results[name]['window_ext'].append(wi_ext)
            results[name]['wall_all'].append(w_all)
            results[name]['window_all'].append(wi_all)
            results[name]['wall_int'].append(w_int)
            results[name]['window_int'].append(wi_int)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(valid_samples)}")

    # 汇总
    print("\n" + "=" * 80)
    print(f"外围轮廓评估结果 (band_width={BAND_WIDTH}px)")
    print("=" * 80)
    print(f"\n{'Model':<12} {'区域':<10} {'Wall IoU':>10} {'Window IoU':>10}")
    print("-" * 50)

    summary_data = {}
    for name in models:
        r = results[name]
        for region, w_key, wi_key in [
            ('外围', 'wall_ext', 'window_ext'),
            ('内部', 'wall_int', 'window_int'),
            ('全图', 'wall_all', 'window_all'),
        ]:
            w = np.nanmean(r[w_key])
            wi = np.nanmean(r[wi_key])
            print(f"{name:<12} {region:<10} {w:10.4f} {wi:10.4f}")
            summary_data[f"{name}_{region}"] = (w, wi)
        print()

    # ========== 可视化 ==========

    # 1. 柱状图对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'外围轮廓 vs 内部 vs 全图 IoU (band={BAND_WIDTH}px)', fontsize=16, fontweight='bold')

    model_names = list(models.keys())
    regions = ['外围', '内部', '全图']
    region_colors = ['#e74c3c', '#3498db', '#95a5a6']

    for ax_idx, (cls_name, cls_keys) in enumerate([
        ('Wall', ['wall_ext', 'wall_int', 'wall_all']),
        ('Window', ['window_ext', 'window_int', 'window_all']),
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(model_names))
        width = 0.25

        for r_idx, (region, key) in enumerate(zip(regions, cls_keys)):
            vals = [np.nanmean(results[m][key]) for m in model_names]
            bars = ax.bar(x + (r_idx - 1) * width, vals, width, label=region,
                          color=region_colors[r_idx], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel('IoU', fontsize=12)
        ax.set_title(f'{cls_name} IoU', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)

    plt.tight_layout()
    save_path = REPORT_DIR / "05_exterior_iou.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {save_path}")

    # 2. 选几个样本可视化 band + 预测
    np.random.seed(123)
    vis_indices = np.random.choice(len(valid_samples), 6, replace=False)

    fig, axes = plt.subplots(6, 4, figsize=(20, 30))
    fig.suptitle('外围轮廓区域可视化: 原图 / GT+Band / 全图预测 / 外围区域预测',
                 fontsize=16, fontweight='bold', y=1.0)

    model_name = 'M2-DA'
    model = models[model_name]

    for row, idx in enumerate(vis_indices):
        sample_rel = valid_samples[idx]
        sample_dir = DATA_DIR / sample_rel
        pil_img = Image.open(str(sample_dir / "F1_scaled.png")).convert('RGB')
        img_rgb = np.array(pil_img)
        pil_img.close()

        gt = np.load(str(sample_dir / "mask.npy"))
        if gt.shape[:2] != img_rgb.shape[:2]:
            gt = cv2.resize(gt, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt = np.clip(gt, 0, 3)

        band = extract_exterior_band(gt, band_width=BAND_WIDTH)

        pred_raw = predict(model, img_rgb, device)
        pred = cv2.resize(pred_raw.astype(np.uint8),
                          (img_rgb.shape[1], img_rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

        # col 0: 原图
        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title(f'Original: {Path(sample_rel).name}', fontsize=9)
        axes[row, 0].axis('off')

        # col 1: GT + band overlay
        gt_color = mask_to_color(gt)
        band_overlay = gt_color.copy()
        # 在band外区域变暗
        band_overlay[~band] = (gt_color[~band] * 0.25).astype(np.uint8)
        # band边界画黄线
        band_u8 = band.astype(np.uint8) * 255
        band_contours, _ = cv2.findContours(band_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(band_overlay, band_contours, -1, (255, 255, 0), 2)
        axes[row, 1].imshow(band_overlay)
        axes[row, 1].set_title('GT + Exterior Band (yellow)', fontsize=9)
        axes[row, 1].axis('off')

        # col 2: 全图预测
        pred_color = mask_to_color(pred)
        axes[row, 2].imshow(pred_color)
        w_all = compute_exterior_iou(pred, gt, np.ones_like(gt, dtype=bool), 1)
        wi_all = compute_exterior_iou(pred, gt, np.ones_like(gt, dtype=bool), 2)
        axes[row, 2].set_title(f'Full Pred | W:{w_all:.3f} Wi:{wi_all:.3f}', fontsize=9)
        axes[row, 2].axis('off')

        # col 3: 外围区域预测（非band区域置灰）
        ext_pred_color = pred_color.copy()
        ext_pred_color[~band] = 30  # 暗灰
        cv2.drawContours(ext_pred_color, band_contours, -1, (255, 255, 0), 2)
        w_ext = compute_exterior_iou(pred, gt, band, 1)
        wi_ext = compute_exterior_iou(pred, gt, band, 2)
        axes[row, 3].imshow(ext_pred_color)
        axes[row, 3].set_title(f'Exterior Only | W:{w_ext:.3f} Wi:{wi_ext:.3f}', fontsize=9)
        axes[row, 3].axis('off')

    legend_elements = [
        Patch(facecolor=np.array(c)/255, label=n)
        for c, n in zip(CLASS_COLORS, ['Background', 'Wall', 'Window', 'Door'])
    ]
    legend_elements.append(Patch(facecolor='yellow', edgecolor='yellow', label='Exterior Band'))
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=11)

    plt.tight_layout()
    save_path2 = REPORT_DIR / "06_exterior_visual.png"
    plt.savefig(str(save_path2), dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path2}")

    del models
    torch.cuda.empty_cache()
    print("\nDone!")


if __name__ == "__main__":
    main()
