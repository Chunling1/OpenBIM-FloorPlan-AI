"""
生成对比可视化图：左边原图+GT标注，右边模型预测结果
"""
import os
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"
import sys
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
import warnings
warnings.filterwarnings('ignore')

# ---- 配置 ----
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "cubicasa5k"
MODEL_PATH = BASE_DIR / "models" / "M2_UNet_ResNet34_best.pt"
OUTPUT_DIR = BASE_DIR / "output_paper" / "visualizations"
IMG_SIZE = 512
NUM_CLASSES = 4
NUM_SAMPLES = 6  # 生成几张对比图

CLASS_NAMES = ["Background", "Wall", "Window", "Door"]
# 鲜明、专业的配色 (BGR for OpenCV)
CLASS_COLORS_BGR = [
    (40, 40, 40),       # Background - 深灰
    (60, 76, 231),      # Wall - 红色 (BGR)
    (219, 152, 52),     # Window - 蓝色 (BGR)  
    (113, 204, 46),     # Door - 绿色 (BGR)
]
# 用于图例的 RGB 色
CLASS_COLORS_RGB = [
    (40, 40, 40),
    (231, 76, 60),      # Wall - 红
    (52, 152, 219),     # Window - 蓝
    (46, 204, 113),     # Door - 绿
]

ALPHA = 0.55  # 叠加透明度


def load_model():
    """加载最优 M2 模型"""
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=NUM_CLASSES)
    ckpt = torch.load(str(MODEL_PATH), map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded model: best mIoU = {ckpt.get('best_miou', 'N/A')}")
    return model


def preprocess(img_np, sz=IMG_SIZE):
    """图片预处理"""
    aug = A.Compose([
        A.Resize(sz, sz),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    result = aug(image=img_np)
    tensor = torch.from_numpy(result['image'].transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


def mask_to_color(mask, colors=CLASS_COLORS_BGR):
    """将类别 mask 转为彩色图"""
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(colors):
        color_img[mask == cls_id] = color
    return color_img


def overlay_mask(img_bgr, mask, alpha=ALPHA):
    """将彩色 mask 半透明叠加在原图上"""
    color_mask = mask_to_color(mask)
    # 只叠加非背景区域
    fg = mask > 0
    overlay = img_bgr.copy()
    overlay[fg] = cv2.addWeighted(img_bgr, 1 - alpha, color_mask, alpha, 0)[fg]
    # 给构件画轮廓线
    for cls_id in [1, 2, 3]:
        cls_mask = (mask == cls_id).astype(np.uint8)
        contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, CLASS_COLORS_BGR[cls_id], 1)
    return overlay


def add_legend(img, x_start, y_start, class_names=CLASS_NAMES, colors=CLASS_COLORS_BGR):
    """在图片上添加图例"""
    for i, (name, color) in enumerate(zip(class_names[1:], colors[1:])):  # 跳过 background
        y = y_start + i * 28
        cv2.rectangle(img, (x_start, y), (x_start + 18, y + 18), color, -1)
        cv2.rectangle(img, (x_start, y), (x_start + 18, y + 18), (255, 255, 255), 1)
        cv2.putText(img, name, (x_start + 25, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def compute_iou_per_class(pred, gt, num_classes=NUM_CLASSES):
    """计算每类 IoU"""
    ious = {}
    for c in range(num_classes):
        pc = (pred == c)
        tc = (gt == c)
        inter = (pc & tc).sum()
        union = (pc | tc).sum()
        if union > 0:
            ious[CLASS_NAMES[c]] = inter / union
    return ious


def create_comparison(img_rgb, gt_mask, pred_mask, sample_name, save_path):
    """生成单张对比图"""
    H, W = IMG_SIZE, IMG_SIZE

    # Resize 原图到目标尺寸
    img_display = cv2.resize(img_rgb, (W, H))
    img_bgr = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)

    # Resize masks
    gt_resized = cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    pred_resized = cv2.resize(pred_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # 计算 IoU
    ious = compute_iou_per_class(pred_resized, gt_resized)
    miou = np.mean([v for k, v in ious.items() if k != "Background"])

    # 创建左右两张叠加图
    left = overlay_mask(img_bgr.copy(), gt_resized)
    right = overlay_mask(img_bgr.copy(), pred_resized)

    # 拼接画布 (左图 + 间隔 + 右图)
    gap = 4
    title_h = 50
    bottom_h = 35
    canvas_w = W * 2 + gap
    canvas_h = H + title_h + bottom_h
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 30  # 深色背景

    # 放置图片
    canvas[title_h:title_h + H, 0:W] = left
    canvas[title_h:title_h + H, W + gap:W * 2 + gap] = right

    # 中间分隔线
    canvas[title_h:title_h + H, W:W + gap] = (80, 80, 80)

    # 标题
    cv2.putText(canvas, "Ground Truth", (W // 2 - 70, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Model Prediction (U-Net + ResNet34)", (W + gap + W // 2 - 170, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

    # 底部信息
    iou_text = f"mIoU: {miou:.1%}  |  Wall: {ious.get('Wall', 0):.1%}  Window: {ious.get('Window', 0):.1%}  Door: {ious.get('Door', 0):.1%}"
    cv2.putText(canvas, iou_text, (W + gap + 10, canvas_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Sample: {sample_name}", (10, canvas_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)

    # 图例
    add_legend(canvas, W * 2 + gap - 100, title_h + 10)

    cv2.imwrite(str(save_path), canvas)
    print(f"  Saved: {save_path.name}  (mIoU={miou:.3f})")


def create_grid_figure(all_data, save_path):
    """生成多样本网格大图"""
    n = len(all_data)
    cols = 2
    rows = (n + cols - 1) // cols

    cell_w = IMG_SIZE * 2 + 4  # 左右图+间隔
    cell_h = IMG_SIZE + 50 + 35
    margin = 10
    header = 70

    total_w = cols * cell_w + (cols + 1) * margin
    total_h = rows * cell_h + (rows + 1) * margin + header

    big_canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 25

    # 大标题
    title = "Floorplan Semantic Segmentation: Ground Truth vs Prediction"
    cv2.putText(big_canvas, title, (total_w // 2 - 340, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 230, 230), 2, cv2.LINE_AA)

    for idx, (cell_img, _) in enumerate(all_data):
        r = idx // cols
        c = idx % cols
        x = margin + c * (cell_w + margin)
        y = header + margin + r * (cell_h + margin)
        ch, cw = cell_img.shape[:2]
        big_canvas[y:y + ch, x:x + cw] = cell_img

    cv2.imwrite(str(save_path), big_canvas,
                [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"\nGrid figure saved: {save_path}")


def main():
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model().to(device)

    # 读取验证集样本
    val_file = DATA_DIR / "val.txt"
    samples = [l.strip().strip('/') for l in val_file.read_text().strip().split('\n') if l.strip()]

    # 筛选有 mask 的样本，并优先选择构件丰富的
    valid_samples = []
    for s in samples:
        mask_path = DATA_DIR / s / "mask.npy"
        if mask_path.exists():
            mask = np.load(str(mask_path))
            # 计算非背景像素比例,选构件多的
            fg_ratio = (mask > 0).sum() / mask.size
            n_classes = len(np.unique(mask))
            valid_samples.append((s, fg_ratio, n_classes))

    # 按构件丰富度排序，取 top
    valid_samples.sort(key=lambda x: (x[2], x[1]), reverse=True)
    selected = valid_samples[:NUM_SAMPLES]

    print(f"\nGenerating {len(selected)} comparison images...")
    print(f"Output: {OUTPUT_DIR}\n")

    all_cells = []

    for s, fg_ratio, n_cls in selected:
        sample_dir = DATA_DIR / s
        sample_name = s.replace('/', '_')

        # 加载原图
        img_path = sample_dir / "F1_scaled.png"
        img_rgb = np.array(Image.open(str(img_path)).convert('RGB'))

        # 加载 GT mask
        gt_mask = np.load(str(sample_dir / "mask.npy"))
        if gt_mask.shape[:2] != img_rgb.shape[:2]:
            gt_mask = cv2.resize(gt_mask, (img_rgb.shape[1], img_rgb.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

        # 模型推理
        input_tensor = preprocess(img_rgb).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(dim=1).squeeze().cpu().numpy()

        # 生成单张对比图
        save_path = OUTPUT_DIR / f"compare_{sample_name}.png"
        create_comparison(img_rgb, gt_mask, pred, s, save_path)

        # 读回用于网格图
        cell_img = cv2.imread(str(save_path))
        all_cells.append((cell_img, s))

    # 生成网格大图
    grid_path = OUTPUT_DIR / "comparison_grid.png"
    create_grid_figure(all_cells, grid_path)

    print(f"\nDone! {len(selected)} comparisons generated.")


if __name__ == "__main__":
    main()
