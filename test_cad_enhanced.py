"""
路线1：启发式过滤增强
- 预处理：移除标注色（绿、品红、紫、黄、橙、红）
- 后处理：连通域分析去除边缘假墙体
- 窗户增强：降低Window类阈值
"""
import os
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"
import numpy as np
import cv2
from pathlib import Path
import torch
import segmentation_models_pytorch as smp
import albumentations as A
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
MODEL_DA = BASE_DIR / "models" / "M2_UNet_ResNet34_DA_best.pt"
OUTPUT_DIR = BASE_DIR / "output_paper" / "visualizations"
RESULT_DIR = BASE_DIR / "output_cad_test"
RESULT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 512
NUM_CLASSES = 4
COLORS_BGR = [(40,40,40), (60,76,231), (219,152,52), (113,204,46)]


def load_model(model_path):
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=NUM_CLASSES)
    ckpt = torch.load(str(model_path), map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


# ============ 预处理：移除标注色 ============

def remove_annotations(img_bgr):
    """
    移除CAD图中的标注线颜色（绿、品红、紫、黄、橙、红）。
    保留白/灰色（墙体线）和青色（可能含窗户）。
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 需要移除的标注颜色范围（HSV）
    annotation_ranges = [
        ((50, 60, 60), (85, 255, 255)),    # 绿色 - 标注线
        ((155, 60, 60), (175, 255, 255)),   # 品红 - 图层
        ((130, 60, 60), (155, 255, 255)),   # 紫色 - 图层
        ((10, 80, 80), (35, 255, 255)),     # 橙色+黄色 - 装饰/标注
        ((0, 120, 100), (10, 255, 255)),    # 纯红 - 标注
        ((170, 120, 100), (180, 255, 255)), # 纯红2
    ]
    
    combined_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for lo, hi in annotation_ranges:
        mask = cv2.inRange(img_hsv, np.array(lo), np.array(hi))
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # 膨胀一点，消除标注线周围的碎片
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    
    # 用黑色填充标注区域
    result = img_bgr.copy()
    result[combined_mask > 0] = 0
    
    removed_pct = combined_mask.sum() / 255 / combined_mask.size * 100
    print(f"    Removed annotation pixels: {removed_pct:.1f}%")
    
    return result, combined_mask


# ============ 推理（带概率输出） ============

def predict_with_probs(model, img_rgb, device):
    """返回原始概率图和 argmax 预测"""
    h_orig, w_orig = img_rgb.shape[:2]
    aug = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    processed = aug(image=img_rgb)
    tensor = torch.from_numpy(processed['image'].transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()  # (4, H, W)
    
    # resize 回原尺寸
    probs_full = np.zeros((NUM_CLASSES, h_orig, w_orig), dtype=np.float32)
    for c in range(NUM_CLASSES):
        probs_full[c] = cv2.resize(probs[c], (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    
    pred = probs_full.argmax(axis=0).astype(np.uint8)
    return pred, probs_full


# ============ 后处理：去除假墙体 + 窗户增强 ============

def postprocess_mask(pred, probs, img_shape, annotation_mask=None):
    """
    1. 去除边缘附近的小面积假墙体（标注误检）
    2. 去除极细长的假墙体连通域
    3. 窗户概率增强
    """
    h, w = img_shape[:2]
    result = pred.copy()
    
    # === 1. 边缘区域的假墙体过滤 ===
    # 定义边缘区域（图像外围15%）
    margin = 0.12
    edge_mask = np.zeros((h, w), dtype=bool)
    mh, mw = int(h * margin), int(w * margin)
    edge_mask[:mh, :] = True
    edge_mask[-mh:, :] = True
    edge_mask[:, :mw] = True
    edge_mask[:, -mw:] = True
    
    # 在边缘区域，只保留大面积的墙体连通域
    wall_mask = (result == 1).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(wall_mask)
    
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        comp_mask = labels == i
        
        # 该连通域在边缘区域的像素占比
        edge_ratio = comp_mask[edge_mask].sum() / max(comp_mask.sum(), 1)
        
        # 计算长宽比（用bounding box）
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        
        # 紧凑度: 面积 / bbox面积
        compactness = area / max(bw * bh, 1)
        
        # 过滤条件：
        # 1) 主要在边缘 且 面积小
        # 2) 极细长（标注线特征）
        # 3) 如果annotation_mask存在，检查是否大量重叠标注区域
        should_remove = False
        
        if edge_ratio > 0.7 and area < (h * w * 0.01):
            should_remove = True  # 边缘小碎片
        
        if aspect > 15 and compactness < 0.1:
            should_remove = True  # 极细长线条
        
        if annotation_mask is not None:
            annot_overlap = (comp_mask & (annotation_mask > 0)).sum() / max(comp_mask.sum(), 1)
            if annot_overlap > 0.3:
                should_remove = True  # 大量重叠标注区域
        
        if should_remove:
            result[comp_mask] = 0  # 设为背景
    
    # === 2. 窗户概率增强 ===
    # 对于当前被判为背景或墙体的像素，如果Window概率 > 阈值，强制设为Window
    window_threshold = 0.15  # 降低Window的判定阈值
    window_boost = (probs[2] > window_threshold) & (result != 3)  # 不覆盖门
    # 只在原本是背景的地方增强（不把墙变成窗）
    window_boost = window_boost & (result == 0)
    result[window_boost] = 2
    
    # === 3. 门概率增强 ===
    door_threshold = 0.12
    door_boost = (probs[3] > door_threshold) & (result == 0)
    result[door_boost] = 3
    
    return result


# ============ 预处理黑底→白底 ============

def preprocess_dark_cad(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(inverted)
    _, binary = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
    result = cv2.addWeighted(enhanced, 0.4, binary, 0.6, 0)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)


# ============ 可视化 ============

def make_overlay(img_bgr, pred_mask):
    seg_color = np.zeros_like(img_bgr)
    for cls_id, color in enumerate(COLORS_BGR):
        seg_color[pred_mask == cls_id] = color
    overlay = img_bgr.copy()
    fg = pred_mask > 0
    overlay[fg] = cv2.addWeighted(img_bgr, 0.4, seg_color, 0.6, 0)[fg]
    for cls_id in [1, 2, 3]:
        cls_mask = (pred_mask == cls_id).astype(np.uint8)
        contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, COLORS_BGR[cls_id], 2)
    return overlay


def get_stats_str(pred):
    t = pred.size
    return f"W:{(pred==1).sum()/t*100:.1f}% Wi:{(pred==2).sum()/t*100:.1f}% D:{(pred==3).sum()/t*100:.1f}%"


def create_comparison(orig_bgr, panels, save_path):
    """生成多栏对比图"""
    h, w = orig_bgr.shape[:2]
    max_w = 380
    if w > max_w:
        scale = max_w / w
        orig_bgr = cv2.resize(orig_bgr, None, fx=scale, fy=scale)
        h, w = orig_bgr.shape[:2]
        for p in panels:
            p['img'] = cv2.resize(p['img'], (w, h))
    
    n = len(panels) + 1
    gap = 3
    title_h = 65
    legend_h = 45
    cw = w * n + gap * (n - 1)
    ch = h + title_h + legend_h
    canvas = np.ones((ch, cw, 3), dtype=np.uint8) * 20
    
    # 原图
    canvas[title_h:title_h+h, 0:w] = orig_bgr
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Original", (w//2-30, 20), font, 0.5, (200,200,200), 1, cv2.LINE_AA)
    
    for i, p in enumerate(panels):
        x = w*(i+1) + gap*(i+1)
        canvas[title_h:title_h+h, x:x+w] = p['img']
        canvas[title_h:title_h+h, x-gap:x] = (50,50,50)
        
        lines = p['title'].split('\n')
        for j, line in enumerate(lines):
            ts = cv2.getTextSize(line, font, 0.4, 1)[0]
            tx = x + w//2 - ts[0]//2
            cv2.putText(canvas, line, (tx, 18 + j*18), font, 0.4, (200,200,200), 1, cv2.LINE_AA)
    
    # 图例
    ly = title_h + h + 12
    items = [("Wall", COLORS_BGR[1]), ("Window", COLORS_BGR[2]), ("Door", COLORS_BGR[3])]
    lx = cw//2 - 170
    for name, color in items:
        cv2.rectangle(canvas, (lx, ly), (lx+16, ly+16), color, -1)
        cv2.putText(canvas, name, (lx+22, ly+13), font, 0.45, (200,200,200), 1, cv2.LINE_AA)
        lx += 110
    
    cv2.imwrite(str(save_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"  Saved: {save_path.name}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading model...")
    model = load_model(MODEL_DA).to(device)
    print(f"Model loaded on {device}\n")
    
    cad_images = sorted(OUTPUT_DIR.glob("test_china_cad_*.jpg")) + sorted(OUTPUT_DIR.glob("test_china_cad_*.png"))
    
    for img_path in cad_images:
        print(f"\n=== {img_path.name} ===")
        orig_bgr = cv2.imread(str(img_path))
        
        # ---- 方案A: DA直接推理（基线） ----
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        pred_a, probs_a = predict_with_probs(model, orig_rgb, device)
        print(f"  [A] DA baseline:        {get_stats_str(pred_a)}")
        
        # ---- 方案B: 去标注色 + 推理 + 后处理 ----
        cleaned_bgr, annot_mask = remove_annotations(orig_bgr)
        cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
        pred_b_raw, probs_b = predict_with_probs(model, cleaned_rgb, device)
        pred_b = postprocess_mask(pred_b_raw, probs_b, orig_bgr.shape, annot_mask)
        print(f"  [B] Clean+PostProc:     {get_stats_str(pred_b)}")
        
        # ---- 方案C: 去标注色 + 白底预处理 + 推理 + 后处理 ----
        preprocessed_rgb = preprocess_dark_cad(cleaned_bgr)
        pred_c_raw, probs_c = predict_with_probs(model, preprocessed_rgb, device)
        pred_c = postprocess_mask(pred_c_raw, probs_c, orig_bgr.shape, annot_mask)
        print(f"  [C] Clean+White+Post:   {get_stats_str(pred_c)}")
        
        # ---- 方案D: 去标注色 + 白底 + 推理（不后处理，做对照） ----
        print(f"  [D] Clean+White(noPost):{get_stats_str(pred_c_raw)}")
        
        # 生成对比图
        panels = [
            {'img': make_overlay(orig_bgr.copy(), pred_a),
             'title': f"[A] DA Baseline\n{get_stats_str(pred_a)}"},
            {'img': make_overlay(cleaned_bgr.copy(), pred_b),
             'title': f"[B] DeAnnot+PostProc\n{get_stats_str(pred_b)}"},
            {'img': make_overlay(cv2.cvtColor(preprocessed_rgb, cv2.COLOR_RGB2BGR), pred_c),
             'title': f"[C] DeAnnot+White+Post\n{get_stats_str(pred_c)}"},
        ]
        
        save_path = RESULT_DIR / f"enhanced_{img_path.stem}.png"
        create_comparison(orig_bgr, panels, save_path)
    
    print("\nAll done!")


if __name__ == "__main__":
    main()
