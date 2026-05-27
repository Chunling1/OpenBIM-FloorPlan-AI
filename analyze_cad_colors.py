"""
分析黑底CAD图中各颜色的分布，找出标注线的颜色特征
"""
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output_paper" / "visualizations"
RESULT_DIR = BASE_DIR / "output_cad_test"
RESULT_DIR.mkdir(exist_ok=True)

def analyze_colors(img_path):
    img_bgr = cv2.imread(str(img_path))
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 忽略纯黑背景
    non_black = np.any(img_bgr > 20, axis=2)
    
    h, w = img_bgr.shape[:2]
    print(f"\n=== {img_path.name} ({w}x{h}) ===")
    print(f"非黑像素占比: {non_black.sum() / non_black.size * 100:.1f}%")
    
    # 提取非黑像素的HSV
    hsv_pixels = img_hsv[non_black]
    bgr_pixels = img_bgr[non_black]
    
    # 按颜色区间统计
    color_ranges = {
        "红色(Red)":       ((0, 100, 100), (10, 255, 255)),
        "红色2(Red2)":     ((170, 100, 100), (180, 255, 255)),
        "橙色(Orange)":    ((10, 100, 100), (25, 255, 255)),
        "黄色(Yellow)":    ((25, 100, 100), (35, 255, 255)),
        "黄绿(YellowGreen)": ((35, 100, 100), (50, 255, 255)),
        "绿色(Green)":     ((50, 100, 100), (85, 255, 255)),
        "青色(Cyan)":      ((85, 100, 100), (100, 255, 255)),
        "蓝色(Blue)":      ((100, 100, 100), (130, 255, 255)),
        "紫色(Purple)":    ((130, 100, 100), (160, 255, 255)),
        "品红(Magenta)":   ((160, 100, 100), (170, 255, 255)),
        "白/灰(White/Gray)": None,  # 特殊处理
    }
    
    total_non_black = non_black.sum()
    
    for name, hrange in color_ranges.items():
        if hrange is None:
            # 白/灰: 低饱和度
            mask = (hsv_pixels[:, 1] < 100) & (hsv_pixels[:, 2] > 100)
        else:
            lo, hi = hrange
            mask = (hsv_pixels[:, 0] >= lo[0]) & (hsv_pixels[:, 0] <= hi[0]) & \
                   (hsv_pixels[:, 1] >= lo[1]) & (hsv_pixels[:, 2] >= lo[2])
        
        count = mask.sum()
        pct = count / total_non_black * 100
        if pct > 0.5:  # 只显示占比>0.5%的
            avg_bgr = bgr_pixels[mask].mean(axis=0) if count > 0 else [0,0,0]
            print(f"  {name:25s}: {pct:6.1f}% ({count:7d}px)  avg_BGR=({avg_bgr[0]:.0f},{avg_bgr[1]:.0f},{avg_bgr[2]:.0f})")
    
    # 可视化：生成颜色分布图
    vis = np.zeros_like(img_bgr)
    
    # 标注类颜色（通常是绿、黄、品红、红 - CAD标注常用色）
    annotation_colors = {
        "green": ((50, 80, 80), (85, 255, 255)),
        "yellow": ((25, 80, 80), (35, 255, 255)),
        "magenta": ((155, 80, 80), (175, 255, 255)),
        "red1": ((0, 80, 80), (10, 255, 255)),
        "red2": ((170, 80, 80), (180, 255, 255)),
    }
    
    annot_mask = np.zeros((h, w), dtype=bool)
    for cname, (lo, hi) in annotation_colors.items():
        m = cv2.inRange(img_hsv, np.array(lo), np.array(hi))
        annot_mask |= (m > 0)
    
    # 白/灰色（通常是墙线）
    gray_mask = (img_hsv[:,:,1] < 80) & (img_hsv[:,:,2] > 120)
    
    # 青蓝色（通常含窗户线）
    cyan_mask = cv2.inRange(img_hsv, np.array([85, 60, 60]), np.array([105, 255, 255])) > 0
    
    vis[annot_mask] = (0, 0, 255)    # 标注 → 红色标记
    vis[gray_mask] = (200, 200, 200)  # 白灰 → 白
    vis[cyan_mask] = (255, 200, 0)    # 青色 → 蓝标记
    
    # 拼接：原图 | 颜色分析
    max_w = 700
    if w > max_w:
        scale = max_w / w
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale)
        vis = cv2.resize(vis, None, fx=scale, fy=scale)
    
    canvas = np.hstack([img_bgr, np.ones((img_bgr.shape[0], 4, 3), dtype=np.uint8)*50, vis])
    save_path = RESULT_DIR / f"color_analysis_{img_path.stem}.png"
    cv2.imwrite(str(save_path), canvas)
    print(f"  Color analysis saved: {save_path.name}")


cad_images = sorted(OUTPUT_DIR.glob("test_china_cad_*.jpg")) + sorted(OUTPUT_DIR.glob("test_china_cad_*.png"))
for p in cad_images:
    analyze_colors(p)
