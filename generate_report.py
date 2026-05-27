"""
M2 Domain Adaptation 训练报告生成脚本
生成:
1. 训练曲线图（loss + mIoU + 各类别IoU + LR）
2. 验证集 GT vs Prediction 对比图
3. 真实CAD图推理效果对比（DA模型 vs 原版模型）
4. 所有模型横向对比
"""
import os
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

import torch
import segmentation_models_pytorch as smp
import albumentations as A

BASE_DIR = Path(__file__).parent
REPORT_DIR = BASE_DIR / "output_paper" / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 512
NUM_CLASSES = 4
CLASS_NAMES = ["Background", "Wall", "Window", "Door"]
# matplotlib用RGB
CLASS_COLORS_RGB = [(0.15, 0.15, 0.15), (0.91, 0.30, 0.24), (0.20, 0.60, 0.86), (0.18, 0.80, 0.44)]
CLASS_COLORS_UINT8 = [(40, 40, 40), (231, 76, 60), (52, 152, 219), (46, 204, 113)]

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================
# 1. 训练曲线
# ============================

def plot_training_curves():
    """从 JSON 历史绘制完整训练曲线"""
    hist_path = BASE_DIR / "output_paper" / "M2_UNet_ResNet34_DA_history.json"
    if not hist_path.exists():
        print("No history JSON found, skipping curves")
        return

    with open(str(hist_path)) as f:
        hist = json.load(f)

    epochs = list(range(1, len(hist['train_loss']) + 1))
    best_epoch = hist.get('best_epoch', 0)
    best_miou = hist.get('best_miou', 0)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('M2 UNet-ResNet34 Domain Adaptation Training Report',
                 fontsize=18, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

    # --- Loss ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, hist['train_loss'], color='#e74c3c', linewidth=1.5, alpha=0.8, label='Train Loss')
    ax1.plot(epochs, hist['val_loss'], color='#3498db', linewidth=1.5, alpha=0.8, label='Val Loss')
    ax1.axvline(x=best_epoch, color='#2ecc71', linestyle='--', alpha=0.6, label=f'Best Epoch ({best_epoch})')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (CE + Dice)', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, len(epochs))

    # --- mIoU ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, hist['train_miou'], color='#e74c3c', linewidth=1.5, alpha=0.8, label='Train mIoU')
    ax2.plot(epochs, hist['val_miou'], color='#3498db', linewidth=1.5, alpha=0.8, label='Val mIoU')
    ax2.axhline(y=best_miou, color='#2ecc71', linestyle='--', alpha=0.6,
                label=f'Best Val mIoU ({best_miou:.4f})')
    ax2.axvline(x=best_epoch, color='#2ecc71', linestyle=':', alpha=0.3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('mIoU', fontsize=12)
    ax2.set_title('Mean IoU Over Training', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(epochs))
    ax2.set_ylim(0.3, 0.85)

    # --- Per-class IoU ---
    ax3 = fig.add_subplot(gs[1, 0])
    class_ious = np.array(hist['val_class_iou'])  # (epochs, 4)
    colors_cls = ['#95a5a6', '#e74c3c', '#3498db', '#2ecc71']
    for c in range(NUM_CLASSES):
        ax3.plot(epochs, class_ious[:, c], color=colors_cls[c], linewidth=1.5,
                 alpha=0.8, label=f'{CLASS_NAMES[c]}')
    ax3.axvline(x=best_epoch, color='#f39c12', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('IoU', fontsize=12)
    ax3.set_title('Per-Class IoU (Validation)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1, len(epochs))

    # --- Summary Stats ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Best epoch class IoU
    best_idx = min(best_epoch - 1, len(class_ious) - 1)
    best_class_iou = class_ious[best_idx]
    final_class_iou = class_ious[-1]

    summary_text = f"""Training Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model:         U-Net + ResNet34 (ImageNet)
Input Size:    512 × 512
Batch Size:    2 (× 4 accum = eff. 8)
Optimizer:     AdamW (encoder LR: 3e-5, decoder LR: 3e-4)
Scheduler:     LinearWarmup(5ep) + CosineAnnealing
Loss:          Weighted CE + Dice
Class Weights: [0.5, 2.0, 3.0, 3.0]
GPU:           RTX 4050 Laptop (6GB)
Training Time: 926.4 min (~15.4 hours)

Domain Adaptation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
InvertImg:     p=0.3 (simulate dark CAD)
ToGray:        p=0.2 (simulate line drawings)
ColorJitter:   p=0.3 (B=0.4, C=0.4, S=0.3, H=0.1)

Results (Best @ Epoch {best_epoch})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Best mIoU:     {best_miou:.4f}
Background:    {best_class_iou[0]:.4f}
Wall:          {best_class_iou[1]:.4f}
Window:        {best_class_iou[2]:.4f}
Door:          {best_class_iou[3]:.4f}"""

    ax4.text(0.05, 0.98, summary_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#2c3e50', edgecolor='#34495e', alpha=0.9),
             color='white')

    save_path = REPORT_DIR / "01_training_curves.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[1/4] Training curves -> {save_path.name}")


# ============================
# 2. 验证集 GT vs Prediction
# ============================

def load_m2_model(model_path):
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=NUM_CLASSES)
    ckpt = torch.load(str(model_path), map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def predict_single(model, img_rgb, device):
    aug = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    processed = aug(image=img_rgb)
    tensor = torch.from_numpy(processed['image'].transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).squeeze().cpu().numpy()
    return pred


def mask_to_color(mask, alpha_bg=True):
    """将分割 mask 转为彩色图"""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        color[mask == c] = CLASS_COLORS_UINT8[c]
    return color


def compute_iou(pred, gt, num_classes):
    ious = []
    for c in range(num_classes):
        pc = (pred == c)
        gc = (gt == c)
        inter = (pc & gc).sum()
        union = (pc | gc).sum()
        if union > 0:
            ious.append(inter / union)
        else:
            ious.append(float('nan'))
    valid = [x for x in ious if not np.isnan(x)]
    miou = np.mean(valid) if valid else 0
    return ious, miou


def plot_val_predictions():
    """在验证集上选几个样本，展示 原图 / GT mask / Prediction 对比"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = BASE_DIR / "data" / "cubicasa5k"

    # 加载 DA 模型
    model_path = BASE_DIR / "models" / "M2_UNet_ResNet34_DA_best.pt"
    if not model_path.exists():
        print("No DA model found, skipping val predictions")
        return
    model = load_m2_model(model_path).to(device)

    # 读验证集
    val_file = data_dir / "val.txt"
    if not val_file.exists():
        print("No val.txt found")
        return

    all_samples = [l.strip().strip('/') for l in val_file.read_text().strip().split('\n') if l.strip()]
    valid_samples = [s for s in all_samples if (data_dir / s / "mask.npy").exists()]

    # 选8个均匀分布的样本
    np.random.seed(42)
    indices = np.linspace(0, len(valid_samples)-1, 8, dtype=int)
    selected = [valid_samples[i] for i in indices]

    fig, axes = plt.subplots(8, 3, figsize=(18, 48))
    fig.suptitle('Validation Set: Original → Ground Truth → M2-DA Prediction',
                 fontsize=18, fontweight='bold', y=1.0)

    for row, sample_rel in enumerate(selected):
        sample_dir = data_dir / sample_rel

        # 加载图片
        img_path = sample_dir / "F1_scaled.png"
        pil_img = Image.open(str(img_path)).convert('RGB')
        img_rgb = np.array(pil_img)
        pil_img.close()

        # 加载 GT mask
        gt_mask = np.load(str(sample_dir / "mask.npy"))
        if gt_mask.shape[:2] != img_rgb.shape[:2]:
            gt_mask = cv2.resize(gt_mask, (img_rgb.shape[1], img_rgb.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        gt_mask = np.clip(gt_mask, 0, 3)

        # 预测
        pred_mask = predict_single(model, img_rgb, device)
        pred_mask_full = cv2.resize(pred_mask.astype(np.uint8),
                                     (img_rgb.shape[1], img_rgb.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

        # 计算IoU
        ious, miou = compute_iou(pred_mask_full, gt_mask, NUM_CLASSES)

        # 显示
        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title(f'Original: {Path(sample_rel).name}', fontsize=10)
        axes[row, 0].axis('off')

        gt_color = mask_to_color(gt_mask)
        axes[row, 1].imshow(gt_color)
        axes[row, 1].set_title('Ground Truth', fontsize=10)
        axes[row, 1].axis('off')

        pred_color = mask_to_color(pred_mask_full)
        iou_str = f'mIoU={miou:.3f} | W={ious[1]:.3f} Wi={ious[2]:.3f} D={ious[3]:.3f}'
        axes[row, 2].imshow(pred_color)
        axes[row, 2].set_title(f'Prediction ({iou_str})', fontsize=9)
        axes[row, 2].axis('off')

    # 图例
    legend_elements = [Patch(facecolor=np.array(c)/255 if max(c)>1 else c, label=n)
                       for c, n in zip(CLASS_COLORS_UINT8, CLASS_NAMES)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=12,
               frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    save_path = REPORT_DIR / "02_val_gt_vs_pred.png"
    plt.savefig(str(save_path), dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[2/4] Val GT vs Pred -> {save_path.name}")
    del model
    torch.cuda.empty_cache()


# ============================
# 3. CAD图 DA vs 原版 对比
# ============================

def preprocess_dark_cad(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(inverted)
    _, binary = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
    result = cv2.addWeighted(enhanced, 0.4, binary, 0.6, 0)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result_rgb


def plot_cad_comparison():
    """在真实CAD图上，对比 DA模型 vs 原版模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    viz_dir = BASE_DIR / "output_paper" / "visualizations"

    da_path = BASE_DIR / "models" / "M2_UNet_ResNet34_DA_best.pt"
    orig_path = BASE_DIR / "models" / "M2_UNet_ResNet34_best.pt"

    if not da_path.exists() or not orig_path.exists():
        print("Missing model files, skipping CAD comparison")
        return

    model_da = load_m2_model(da_path).to(device)
    model_orig = load_m2_model(orig_path).to(device)

    cad_images = sorted(viz_dir.glob("test_china_cad_*"))
    if not cad_images:
        print("No CAD test images found")
        return

    n = len(cad_images)
    fig, axes = plt.subplots(n, 4, figsize=(24, 6*n))
    if n == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Real CAD Drawings: DA Model vs Original Model',
                 fontsize=18, fontweight='bold', y=0.99)

    for row, img_path in enumerate(cad_images):
        orig_bgr = cv2.imread(str(img_path))
        if orig_bgr is None:
            continue
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

        # 预处理版
        preprocessed_rgb = preprocess_dark_cad(orig_bgr)

        # DA模型直接推理（黑底原图）
        pred_da_direct = predict_single(model_da, orig_rgb, device)
        pred_da_direct_full = cv2.resize(pred_da_direct.astype(np.uint8),
                                          (orig_rgb.shape[1], orig_rgb.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)

        # 原版模型+预处理
        pred_orig = predict_single(model_orig, preprocessed_rgb, device)
        pred_orig_full = cv2.resize(pred_orig.astype(np.uint8),
                                     (preprocessed_rgb.shape[1], preprocessed_rgb.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

        # DA模型+预处理
        pred_da_preproc = predict_single(model_da, preprocessed_rgb, device)
        pred_da_preproc_full = cv2.resize(pred_da_preproc.astype(np.uint8),
                                           (preprocessed_rgb.shape[1], preprocessed_rgb.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)

        # 显示
        axes[row, 0].imshow(orig_rgb)
        axes[row, 0].set_title(f'Original: {img_path.name}', fontsize=10)
        axes[row, 0].axis('off')

        da_color = mask_to_color(pred_da_direct_full)
        # 叠加到原图
        overlay_da = orig_rgb.copy().astype(float)
        fg = pred_da_direct_full > 0
        overlay_da[fg] = overlay_da[fg] * 0.4 + np.array(da_color, dtype=float)[fg] * 0.6
        axes[row, 1].imshow(overlay_da.astype(np.uint8))
        s1 = {c: (pred_da_direct_full==i).sum()/pred_da_direct_full.size*100 for i,c in enumerate(CLASS_NAMES)}
        axes[row, 1].set_title(f'DA + Direct\nW:{s1["Wall"]:.1f}% Wi:{s1["Window"]:.1f}% D:{s1["Door"]:.1f}%', fontsize=9)
        axes[row, 1].axis('off')

        orig_color = mask_to_color(pred_orig_full)
        overlay_orig = preprocessed_rgb.copy().astype(float)
        fg2 = pred_orig_full > 0
        overlay_orig[fg2] = overlay_orig[fg2] * 0.4 + np.array(orig_color, dtype=float)[fg2] * 0.6
        axes[row, 2].imshow(overlay_orig.astype(np.uint8))
        s2 = {c: (pred_orig_full==i).sum()/pred_orig_full.size*100 for i,c in enumerate(CLASS_NAMES)}
        axes[row, 2].set_title(f'Orig + Preprocess\nW:{s2["Wall"]:.1f}% Wi:{s2["Window"]:.1f}% D:{s2["Door"]:.1f}%', fontsize=9)
        axes[row, 2].axis('off')

        da_pp_color = mask_to_color(pred_da_preproc_full)
        overlay_da_pp = preprocessed_rgb.copy().astype(float)
        fg3 = pred_da_preproc_full > 0
        overlay_da_pp[fg3] = overlay_da_pp[fg3] * 0.4 + np.array(da_pp_color, dtype=float)[fg3] * 0.6
        axes[row, 3].imshow(overlay_da_pp.astype(np.uint8))
        s3 = {c: (pred_da_preproc_full==i).sum()/pred_da_preproc_full.size*100 for i,c in enumerate(CLASS_NAMES)}
        axes[row, 3].set_title(f'DA + Preprocess\nW:{s3["Wall"]:.1f}% Wi:{s3["Window"]:.1f}% D:{s3["Door"]:.1f}%', fontsize=9)
        axes[row, 3].axis('off')

    legend_elements = [Patch(facecolor=np.array(c)/255, label=n)
                       for c, n in zip(CLASS_COLORS_UINT8, CLASS_NAMES)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=12,
               frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    save_path = REPORT_DIR / "03_cad_da_vs_orig.png"
    plt.savefig(str(save_path), dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[3/4] CAD DA vs Orig -> {save_path.name}")
    del model_da, model_orig
    torch.cuda.empty_cache()


# ============================
# 4. 所有模型横向对比
# ============================

def plot_model_comparison():
    """3个模型训练曲线对比 + 最终指标对比"""
    models_info = {
        'M1_LightUNet': ('M1: LightUNet', '#e74c3c'),
        'M2_UNet_ResNet34': ('M2: UNet+ResNet34', '#f39c12'),
        'M2_UNet_ResNet34_DA': ('M2-DA: UNet+ResNet34+DA', '#3498db'),
        'M3_DeepLabV3p_EffB4': ('M3: DeepLabV3+EfficientNet', '#2ecc71'),
    }

    histories = {}
    for key in models_info:
        hist_path = BASE_DIR / "output_paper" / f"{key}_history.json"
        if hist_path.exists():
            with open(str(hist_path)) as f:
                histories[key] = json.load(f)

    if not histories:
        print("No model histories found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle('All Models Comparison', fontsize=18, fontweight='bold')

    # Val mIoU curves
    ax = axes[0]
    for key, hist in histories.items():
        label, color = models_info[key]
        epochs = range(1, len(hist['val_miou']) + 1)
        ax.plot(epochs, hist['val_miou'], color=color, linewidth=1.5, alpha=0.8, label=label)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Val mIoU', fontsize=12)
    ax.set_title('Validation mIoU', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Best mIoU bar chart
    ax = axes[1]
    names = []
    mious = []
    colors = []
    for key, hist in histories.items():
        label, color = models_info[key]
        names.append(label.split(': ')[1] if ': ' in label else label)
        mious.append(hist['best_miou'])
        colors.append(color)
    bars = ax.barh(names, mious, color=colors, alpha=0.8, height=0.5)
    for bar, val in zip(bars, mious):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('Best mIoU', fontsize=12)
    ax.set_title('Best Validation mIoU', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(mious) * 1.15)
    ax.grid(True, alpha=0.3, axis='x')

    # Per-class IoU at best epoch (grouped bar)
    ax = axes[2]
    x = np.arange(NUM_CLASSES)
    width = 0.18
    for i, (key, hist) in enumerate(histories.items()):
        label, color = models_info[key]
        best_ep = hist.get('best_epoch', len(hist['val_class_iou']))
        idx = min(best_ep - 1, len(hist['val_class_iou']) - 1)
        class_ious = hist['val_class_iou'][idx]
        offset = (i - len(histories)/2 + 0.5) * width
        ax.bar(x + offset, class_ious, width, label=label.split(': ')[1] if ': ' in label else label,
               color=color, alpha=0.8)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('IoU', fontsize=12)
    ax.set_title('Per-Class IoU @ Best Epoch', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = REPORT_DIR / "04_model_comparison.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[4/4] Model comparison -> {save_path.name}")


# ============================
# Main
# ============================

if __name__ == "__main__":
    print("=" * 60)
    print("Generating M2 DA Training Report")
    print("=" * 60)
    print()

    plot_training_curves()
    plot_val_predictions()
    plot_cad_comparison()
    plot_model_comparison()

    print()
    print(f"All reports saved to: {REPORT_DIR}")
    print("Done!")
