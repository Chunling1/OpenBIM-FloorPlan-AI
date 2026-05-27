"""
商业建筑户型图测试脚本
- 对 test_commercial/ 目录下的商业建筑平面图进行全面测试
- 输出：分割overlay、房间提取、热工分区、定量统计
- 支持多层建筑分层处理

测试维度：
  1. 语义分割效果 (wall/window/door 检出率)
  2. 房间提取准确性 (连通域数量、面积分布)  
  3. 热工分区合理性 (内外区比例、窗墙比)
  4. 域泛化能力 (对比有无预处理的效果差异)
"""
import os
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import cv2
from pathlib import Path
import torch
import segmentation_models_pytorch as smp
import albumentations as A
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
import json
import time
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
TEST_DIR = BASE_DIR / "test_commercial"
OUTPUT_DIR = BASE_DIR / "output_commercial_test"
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 512
NUM_CLASSES = 4
CLASS_NAMES = ["Background", "Wall", "Window", "Door"]
COLORS_BGR = [(40,40,40), (60,76,231), (219,152,52), (113,204,46)]
COLORS_RGB = [(40,40,40), (231,76,60), (52,152,219), (46,204,113)]

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============ 模型加载 ============

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


# ============ 预处理管线 ============

def remove_annotations(img_bgr):
    """移除CAD标注色"""
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    annotation_ranges = [
        ((50, 60, 60), (85, 255, 255)),
        ((155, 60, 60), (175, 255, 255)),
        ((130, 60, 60), (155, 255, 255)),
        ((10, 80, 80), (35, 255, 255)),
        ((0, 120, 100), (10, 255, 255)),
        ((170, 120, 100), (180, 255, 255)),
    ]
    combined_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for lo, hi in annotation_ranges:
        mask = cv2.inRange(img_hsv, np.array(lo), np.array(hi))
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    result = img_bgr.copy()
    result[combined_mask > 0] = 0
    return result, combined_mask


def preprocess_dark_cad(img_bgr):
    """黑底→白底增强"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(inverted)
    _, binary = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
    result = cv2.addWeighted(enhanced, 0.4, binary, 0.6, 0)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)


def auto_detect_dark_bg(img_bgr, threshold=0.55):
    """自动检测是否为深色背景图"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dark_ratio = (gray < 80).sum() / gray.size
    return dark_ratio > threshold


# ============ 推理 ============

def predict_with_probs(model, img_rgb, device):
    h_orig, w_orig = img_rgb.shape[:2]
    aug = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    processed = aug(image=img_rgb)
    tensor = torch.from_numpy(processed['image'].transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
    probs_full = np.zeros((NUM_CLASSES, h_orig, w_orig), dtype=np.float32)
    for c in range(NUM_CLASSES):
        probs_full[c] = cv2.resize(probs[c], (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    pred = probs_full.argmax(axis=0).astype(np.uint8)
    return pred, probs_full


# ============ 后处理 ============

def postprocess_mask(pred, probs, img_shape, annotation_mask=None):
    h, w = img_shape[:2]
    result = pred.copy()
    
    margin = 0.10
    edge_mask = np.zeros((h, w), dtype=bool)
    mh, mw = int(h * margin), int(w * margin)
    edge_mask[:mh, :] = True
    edge_mask[-mh:, :] = True
    edge_mask[:, :mw] = True
    edge_mask[:, -mw:] = True
    
    wall_mask = (result == 1).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(wall_mask)
    
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        comp_mask = labels == i
        edge_ratio = comp_mask[edge_mask].sum() / max(comp_mask.sum(), 1)
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        compactness = area / max(bw * bh, 1)
        
        should_remove = False
        if edge_ratio > 0.7 and area < (h * w * 0.008):
            should_remove = True
        if aspect > 15 and compactness < 0.08:
            should_remove = True
        if annotation_mask is not None:
            annot_overlap = (comp_mask & (annotation_mask > 0)).sum() / max(comp_mask.sum(), 1)
            if annot_overlap > 0.3:
                should_remove = True
        if should_remove:
            result[comp_mask] = 0
    
    window_boost = (probs[2] > 0.15) & (result == 0)
    result[window_boost] = 2
    door_boost = (probs[3] > 0.12) & (result == 0)
    result[door_boost] = 3
    
    return result


# ============ 房间提取 ============

def extract_rooms(mask_pred):
    """分水岭算法提取独立房间"""
    obstacles = ((mask_pred == 1) | (mask_pred == 2) | (mask_pred == 3)).astype(np.uint8) * 255
    free_space = 255 - obstacles
    dist_transform = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.25 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[obstacles == 255] = 0
    dummy_rgb = cv2.cvtColor(free_space, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(dummy_rgb, markers)
    return markers


def get_room_stats(markers):
    """统计房间面积分布"""
    unique = np.unique(markers)
    room_areas = []
    for r_id in unique:
        if r_id <= 1:
            continue
        area = (markers == r_id).sum()
        room_areas.append(area)
    room_areas.sort(reverse=True)
    return room_areas


# ============ 热工分区 ============

def split_thermal_zones(mask_pred, rooms_markers, perimeter_depth_px=45):
    """ASHRAE 内外区划分"""
    windows = (mask_pred == 2).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (perimeter_depth_px*2+1, perimeter_depth_px*2+1))
    perimeter_influence = cv2.dilate(windows, kernel, iterations=1)
    
    thermal_zones = np.zeros_like(rooms_markers)
    unique_rooms = np.unique(rooms_markers)
    zone_id = 1
    perimeter_count = 0
    core_count = 0
    
    for r_id in unique_rooms:
        if r_id <= 1:
            continue
        room_mask = (rooms_markers == r_id).astype(np.uint8)
        if room_mask.sum() < perimeter_depth_px * perimeter_depth_px:
            thermal_zones[room_mask == 1] = zone_id
            zone_id += 1
            continue
        p_zone = cv2.bitwise_and(room_mask, perimeter_influence)
        c_zone = room_mask - p_zone
        if p_zone.sum() > 0:
            thermal_zones[p_zone == 1] = zone_id
            zone_id += 1
            perimeter_count += 1
        if c_zone.sum() > 0:
            thermal_zones[c_zone == 1] = zone_id
            zone_id += 1
            core_count += 1
    
    return thermal_zones, perimeter_count, core_count


# ============ 指标计算 ============

def compute_segmentation_stats(pred_mask, img_shape):
    """计算分割统计指标"""
    total_pixels = pred_mask.size
    stats = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        cls_pixels = (pred_mask == cls_id).sum()
        stats[cls_name] = {
            'pixels': int(cls_pixels),
            'ratio': float(cls_pixels / total_pixels),
        }
    
    # 墙体连通域分析
    wall_mask = (pred_mask == 1).astype(np.uint8)
    n_wall, _, wall_stats, _ = cv2.connectedComponentsWithStats(wall_mask)
    stats['wall_components'] = n_wall - 1
    
    # 窗墙比 (WWR)
    wall_pixels = (pred_mask == 1).sum()
    window_pixels = (pred_mask == 2).sum()
    if wall_pixels + window_pixels > 0:
        stats['WWR'] = float(window_pixels / (wall_pixels + window_pixels))
    else:
        stats['WWR'] = 0.0
    
    return stats


def compute_confidence_stats(probs):
    """计算各类别平均置信度"""
    pred = probs.argmax(axis=0)
    conf_stats = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        cls_mask = pred == cls_id
        if cls_mask.sum() > 0:
            conf_stats[cls_name] = float(probs[cls_id][cls_mask].mean())
        else:
            conf_stats[cls_name] = 0.0
    return conf_stats


# ============ 可视化 ============

def make_overlay(img_bgr, pred_mask, alpha=0.6):
    seg_color = np.zeros_like(img_bgr)
    for cls_id, color in enumerate(COLORS_BGR):
        seg_color[pred_mask == cls_id] = color
    overlay = img_bgr.copy()
    fg = pred_mask > 0
    overlay[fg] = cv2.addWeighted(img_bgr, 1-alpha, seg_color, alpha, 0)[fg]
    for cls_id in [1, 2, 3]:
        cls_mask = (pred_mask == cls_id).astype(np.uint8)
        contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, COLORS_BGR[cls_id], 2)
    return overlay


def visualize_rooms(rooms_markers, orig_bgr):
    overlay = orig_bgr.copy()
    unique = np.unique(rooms_markers)
    np.random.seed(42)
    colors = np.random.randint(60, 255, size=(max(unique)+2, 3))
    for z_id in unique:
        if z_id <= 1:
            continue
        mask = (rooms_markers == z_id)
        color = colors[z_id].tolist()
        overlay[mask] = (overlay[mask].astype(float) * 0.4 + np.array(color) * 0.6).astype(np.uint8)
    return overlay


def visualize_thermal_zones(thermal_zones, orig_bgr, pred_mask):
    """热工分区可视化 - 外区暖色，内区冷色"""
    overlay = orig_bgr.copy()
    windows = (pred_mask == 2).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91))
    perimeter_influence = cv2.dilate(windows, kernel, iterations=1)
    
    unique = np.unique(thermal_zones)
    for z_id in unique:
        if z_id <= 0:
            continue
        zmask = (thermal_zones == z_id)
        # 判断该zone是外区还是内区
        overlap_with_perimeter = (zmask & (perimeter_influence > 0)).sum() / max(zmask.sum(), 1)
        if overlap_with_perimeter > 0.3:
            # 外区 - 暖色 (橙红)
            color = [60, 100, 230]  # BGR
        else:
            # 内区 - 冷色 (蓝绿)
            color = [200, 180, 60]  # BGR
        overlay[zmask] = (overlay[zmask].astype(float) * 0.45 + np.array(color) * 0.55).astype(np.uint8)
    
    return overlay


def create_comprehensive_report(results, save_path):
    """生成综合PDF级报告图"""
    n_images = len(results)
    if n_images == 0:
        return
    
    fig, axes = plt.subplots(n_images, 5, figsize=(28, 6 * n_images))
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('商业建筑平面图分割测试报告', fontsize=20, fontweight='bold', y=0.995)
    
    for row, res in enumerate(results):
        name = res['name']
        orig_rgb = cv2.cvtColor(res['orig_bgr'], cv2.COLOR_BGR2RGB)
        
        # Col 0: 原图
        axes[row, 0].imshow(orig_rgb)
        axes[row, 0].set_title(f'原图: {name}', fontsize=10, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Col 1: 分割结果 overlay
        overlay_rgb = cv2.cvtColor(res['overlay'], cv2.COLOR_BGR2RGB)
        axes[row, 1].imshow(overlay_rgb)
        stats = res['seg_stats']
        title_str = f"分割结果\nW:{stats['Wall']['ratio']*100:.1f}% Wi:{stats['Window']['ratio']*100:.1f}% D:{stats['Door']['ratio']*100:.1f}%"
        axes[row, 1].set_title(title_str, fontsize=9)
        axes[row, 1].axis('off')
        
        # Col 2: 房间提取
        rooms_rgb = cv2.cvtColor(res['rooms_vis'], cv2.COLOR_BGR2RGB)
        axes[row, 2].imshow(rooms_rgb)
        room_areas = res['room_areas']
        axes[row, 2].set_title(f'房间提取: {len(room_areas)}个房间', fontsize=10)
        axes[row, 2].axis('off')
        
        # Col 3: 热工分区
        thermal_rgb = cv2.cvtColor(res['thermal_vis'], cv2.COLOR_BGR2RGB)
        axes[row, 3].imshow(thermal_rgb)
        axes[row, 3].set_title(
            f"热工分区: {res['perimeter_zones']}外区 / {res['core_zones']}内区\nWWR={stats['WWR']*100:.1f}%",
            fontsize=9
        )
        axes[row, 3].axis('off')
        
        # Col 4: 置信度+统计
        axes[row, 4].axis('off')
        conf = res['confidence']
        info_text = f"""文件: {name}
分辨率: {res['orig_bgr'].shape[1]}×{res['orig_bgr'].shape[0]}
预处理: {'深色背景增强' if res.get('is_dark') else '直接推理'}

▸ 分割统计
  墙体: {stats['Wall']['ratio']*100:.2f}%  ({stats['wall_components']}个连通域)
  窗户: {stats['Window']['ratio']*100:.2f}%
  门:   {stats['Door']['ratio']*100:.2f}%
  WWR:  {stats['WWR']*100:.1f}%

▸ 平均置信度
  背景: {conf['Background']:.3f}
  墙体: {conf['Wall']:.3f}
  窗户: {conf['Window']:.3f}
  门:   {conf['Door']:.3f}

▸ 房间分析
  房间数: {len(room_areas)}
  最大: {room_areas[0]:,}px² 
  中位: {room_areas[len(room_areas)//2]:,}px²

▸ 热工分区
  外区: {res['perimeter_zones']}
  内区: {res['core_zones']}

▸ 推理耗时: {res['inference_time']*1000:.0f}ms"""
        
        axes[row, 4].text(0.05, 0.95, info_text, transform=axes[row, 4].transAxes,
                         fontsize=8, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', 
                                   edgecolor='#16213e', alpha=0.9),
                         color='#e0e0e0')
    
    # 图例
    legend_elements = [
        Patch(facecolor=np.array(COLORS_RGB[1])/255, label='Wall'),
        Patch(facecolor=np.array(COLORS_RGB[2])/255, label='Window'),
        Patch(facecolor=np.array(COLORS_RGB[3])/255, label='Door'),
        Patch(facecolor=[0.9, 0.4, 0.2], label='外区(Perimeter)'),
        Patch(facecolor=[0.2, 0.7, 0.8], label='内区(Core)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=11)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n[REPORT] Saved: {save_path}")


def create_domain_comparison(results, save_path):
    """对比有/无预处理的分割差异"""
    n = len(results)
    if n == 0:
        return
    
    fig, axes = plt.subplots(n, 4, figsize=(22, 5.5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('域泛化对比: 直接推理 vs 预处理增强', fontsize=18, fontweight='bold')
    
    for row, res in enumerate(results):
        name = res['name']
        orig_rgb = cv2.cvtColor(res['orig_bgr'], cv2.COLOR_BGR2RGB)
        
        axes[row, 0].imshow(orig_rgb)
        axes[row, 0].set_title(f'原图: {name}', fontsize=10)
        axes[row, 0].axis('off')
        
        # 无预处理
        overlay_raw = cv2.cvtColor(res['overlay_raw'], cv2.COLOR_BGR2RGB)
        stats_raw = res['seg_stats_raw']
        axes[row, 1].imshow(overlay_raw)
        axes[row, 1].set_title(
            f"直接推理 (无预处理)\nW:{stats_raw['Wall']['ratio']*100:.1f}% Wi:{stats_raw['Window']['ratio']*100:.1f}% D:{stats_raw['Door']['ratio']*100:.1f}%",
            fontsize=9
        )
        axes[row, 1].axis('off')
        
        # 有预处理
        overlay_pp = cv2.cvtColor(res['overlay'], cv2.COLOR_BGR2RGB)
        stats_pp = res['seg_stats']
        axes[row, 2].imshow(overlay_pp)
        axes[row, 2].set_title(
            f"去标注+白底+后处理\nW:{stats_pp['Wall']['ratio']*100:.1f}% Wi:{stats_pp['Window']['ratio']*100:.1f}% D:{stats_pp['Door']['ratio']*100:.1f}%",
            fontsize=9
        )
        axes[row, 2].axis('off')
        
        # 差异图
        diff = (res['pred_raw'] != res['pred_final']).astype(np.uint8) * 255
        diff_color = cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)
        diff_color[diff > 0] = [255, 255, 0]  # 黄色标记差异
        axes[row, 3].imshow(diff_color)
        diff_pct = diff.sum() / 255 / diff.size * 100
        axes[row, 3].set_title(f'差异区域 ({diff_pct:.1f}% 像素变化)', fontsize=10)
        axes[row, 3].axis('off')
    
    legend_elements = [
        Patch(facecolor=np.array(COLORS_RGB[1])/255, label='Wall'),
        Patch(facecolor=np.array(COLORS_RGB[2])/255, label='Window'),
        Patch(facecolor=np.array(COLORS_RGB[3])/255, label='Door'),
        Patch(facecolor=[1, 1, 0], label='Difference'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[DOMAIN] Saved: {save_path}")


# ============ 主流程 ============

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model_da_path = BASE_DIR / "models" / "M2_UNet_ResNet34_DA_best.pt"
    model_orig_path = BASE_DIR / "models" / "M2_UNet_ResNet34_best.pt"
    
    print("=" * 70)
    print("商业建筑平面图分割 - 全面测试")
    print("=" * 70)
    print(f"Device: {device}")
    
    model_da = load_model(model_da_path).to(device)
    print(f"✓ DA模型已加载: {model_da_path.name}")
    
    model_orig = None
    if model_orig_path.exists():
        model_orig = load_model(model_orig_path).to(device)
        print(f"✓ 原版模型已加载: {model_orig_path.name}")
    
    # 扫描测试图片
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    images = []
    for ext in extensions:
        images.extend(TEST_DIR.glob(ext))
    images.sort()
    
    if not images:
        print(f"\n[ERROR] 未找到测试图片，请将商业建筑平面图放入: {TEST_DIR}")
        return
    
    print(f"\n找到 {len(images)} 张测试图片")
    
    all_results = []
    all_stats = {}
    
    for img_path in images:
        print(f"\n{'='*60}")
        print(f"处理: {img_path.name}")
        print(f"{'='*60}")
        
        orig_bgr = cv2.imread(str(img_path))
        if orig_bgr is None:
            print(f"  [SKIP] 无法读取")
            continue
        
        h, w = orig_bgr.shape[:2]
        print(f"  分辨率: {w}×{h}")
        
        # 检测深色背景
        is_dark = auto_detect_dark_bg(orig_bgr)
        print(f"  背景类型: {'深色 (需预处理)' if is_dark else '浅色'}")
        
        # ---- 方案A: 直接推理 (无预处理) ----
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        pred_raw, probs_raw = predict_with_probs(model_da, orig_rgb, device)
        t_raw = time.time() - t0
        
        stats_raw = compute_segmentation_stats(pred_raw, orig_bgr.shape)
        overlay_raw = make_overlay(orig_bgr, pred_raw)
        print(f"  [RAW]  W:{stats_raw['Wall']['ratio']*100:.1f}% Wi:{stats_raw['Window']['ratio']*100:.1f}% D:{stats_raw['Door']['ratio']*100:.1f}% ({t_raw*1000:.0f}ms)")
        
        # ---- 方案B: 全管线 (去标注+白底+后处理) ----
        cleaned_bgr, annot_mask = remove_annotations(orig_bgr)
        preprocessed_rgb = preprocess_dark_cad(cleaned_bgr)
        
        t0 = time.time()
        pred_pp, probs_pp = predict_with_probs(model_da, preprocessed_rgb, device)
        pred_final = postprocess_mask(pred_pp, probs_pp, orig_bgr.shape, annot_mask)
        t_pp = time.time() - t0
        
        stats_pp = compute_segmentation_stats(pred_final, orig_bgr.shape)
        conf_pp = compute_confidence_stats(probs_pp)
        overlay_pp = make_overlay(orig_bgr, pred_final)
        print(f"  [FULL] W:{stats_pp['Wall']['ratio']*100:.1f}% Wi:{stats_pp['Window']['ratio']*100:.1f}% D:{stats_pp['Door']['ratio']*100:.1f}% ({t_pp*1000:.0f}ms)")
        
        # ---- 房间提取 ----
        room_markers = extract_rooms(pred_final)
        room_areas = get_room_stats(room_markers)
        rooms_vis = visualize_rooms(room_markers, orig_bgr)
        print(f"  房间: {len(room_areas)}个")
        
        # ---- 热工分区 ----
        thermal_zones, perimeter_n, core_n = split_thermal_zones(pred_final, room_markers)
        thermal_vis = visualize_thermal_zones(thermal_zones, orig_bgr, pred_final)
        print(f"  热工分区: {perimeter_n}外区 / {core_n}内区")
        print(f"  WWR: {stats_pp['WWR']*100:.1f}%")
        
        # 保存单张结果
        cv2.imwrite(str(OUTPUT_DIR / f"{img_path.stem}_overlay.png"), overlay_pp)
        cv2.imwrite(str(OUTPUT_DIR / f"{img_path.stem}_rooms.png"), rooms_vis)
        cv2.imwrite(str(OUTPUT_DIR / f"{img_path.stem}_thermal.png"), thermal_vis)
        
        result = {
            'name': img_path.name,
            'orig_bgr': orig_bgr,
            'is_dark': is_dark,
            'pred_raw': pred_raw,
            'pred_final': pred_final,
            'overlay_raw': overlay_raw,
            'overlay': overlay_pp,
            'seg_stats': stats_pp,
            'seg_stats_raw': stats_raw,
            'confidence': conf_pp,
            'rooms_vis': rooms_vis,
            'room_areas': room_areas,
            'thermal_vis': thermal_vis,
            'perimeter_zones': perimeter_n,
            'core_zones': core_n,
            'inference_time': t_pp,
        }
        all_results.append(result)
        
        all_stats[img_path.name] = {
            'resolution': f"{w}x{h}",
            'is_dark_bg': bool(is_dark),
            'wall_ratio': stats_pp['Wall']['ratio'],
            'window_ratio': stats_pp['Window']['ratio'],
            'door_ratio': stats_pp['Door']['ratio'],
            'WWR': stats_pp['WWR'],
            'wall_components': stats_pp['wall_components'],
            'room_count': len(room_areas),
            'perimeter_zones': perimeter_n,
            'core_zones': core_n,
            'inference_ms': round(t_pp * 1000),
            'confidence': conf_pp,
        }
    
    # ---- 生成综合报告 ----
    print("\n\n生成报告图...")
    create_comprehensive_report(all_results, OUTPUT_DIR / "comprehensive_report.png")
    create_domain_comparison(all_results, OUTPUT_DIR / "domain_comparison.png")
    
    # ---- 保存JSON统计 ----
    json_path = OUTPUT_DIR / "test_results.json"
    with open(str(json_path), 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"[JSON] Saved: {json_path}")
    
    # ---- 打印汇总表 ----
    print("\n\n" + "=" * 90)
    print("商业建筑测试结果汇总")
    print("=" * 90)
    print(f"{'文件':<25} {'分辨率':<12} {'Wall%':>6} {'Win%':>6} {'Door%':>6} {'WWR%':>5} {'房间':>4} {'外区':>4} {'内区':>4} {'ms':>5}")
    print("-" * 90)
    for name, s in all_stats.items():
        print(f"{name:<25} {s['resolution']:<12} {s['wall_ratio']*100:5.1f}% {s['window_ratio']*100:5.1f}% {s['door_ratio']*100:5.1f}% {s['WWR']*100:4.1f} {s['room_count']:4d} {s['perimeter_zones']:4d} {s['core_zones']:4d} {s['inference_ms']:5d}")
    
    # 清理
    del model_da
    if model_orig:
        del model_orig
    torch.cuda.empty_cache()
    
    print(f"\n所有结果已保存至: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
