import os
import cv2
import numpy as np
from pathlib import Path
from inference_api import FloorplanSegmenter

def extract_rooms(mask_pred):
    """
    通过墙体和门的掩码提取独立房间 (基于距离变换和分水岭算法)
    :param mask_pred: 分割模型的输出掩码 (0:bg, 1:wall, 2:window, 3:door)
    """
    # 背景为可通行区域 (我们把Window也当作障碍物或墙的一部分，门可以作为隔离或连通，这里作为隔离以便区分房间)
    # 构造障碍物掩码：墙(1), 窗(2), 门(3) 都可以当作房间的物理边界
    obstacles = ((mask_pred == 1) | (mask_pred == 2) | (mask_pred == 3)).astype(np.uint8) * 255
    
    # 可通行区域即为全零的非障碍物区域
    free_space = 255 - obstacles

    # 1. 距离变换 (找到房间中心)
    dist_transform = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
    
    # 2. 寻找局部最大值点（房间种子）
    # 阈值可以根据实际图片比例调整，这里假设大房间中心距离墙体较远
    _, sure_fg = cv2.threshold(dist_transform, 0.25 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 3. 连通域标记种子
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1 # 保证背景是1，而非0
    
    # 将障碍区域设为0，以便分水岭算法作为边界停止
    markers[obstacles == 255] = 0

    # 4. 分水岭算法
    # cv2.watershed 需要 3通道 uint8 输入
    # 我们用伪造的RGB背景图
    dummy_rgb = cv2.cvtColor(free_space, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(dummy_rgb, markers)
    
    return markers

def split_thermal_zones(mask_pred, rooms_markers, perimeter_depth_px=50):
    """
    商业建筑热工分区规则 (ASHRAE 内外区划分逻辑)
    :param perimeter_depth_px: 外区深度(像素)，需要根据CAD比例尺换算(一般约5米)
    """
    # 提取窗户 (只有靠窗才算外区)
    windows = (mask_pred == 2).astype(np.uint8)
    
    if int(cv2.__version__.split('.')[0]) >= 4:
        # 膨胀窗户形成外围带影响区 (Perimeter Zone)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (perimeter_depth_px*2+1, perimeter_depth_px*2+1))
        perimeter_influence = cv2.dilate(windows, kernel, iterations=1)
    else:
        dist_from_win = cv2.distanceTransform(255 - windows*255, cv2.DIST_L2, 5)
        perimeter_influence = (dist_from_win <= perimeter_depth_px).astype(np.uint8)
        
    thermal_zones = np.zeros_like(rooms_markers)
    
    # 对于每个识别出的房间进行内外区切割
    unique_rooms = np.unique(rooms_markers)
    zone_id = 1
    for r_id in unique_rooms:
        if r_id <= 1: continue # 0是边界, 1是墙/背景
        
        room_mask = (rooms_markers == r_id).astype(np.uint8)
        
        # 房间面积过小（如小隔间、卫生间），不拆分
        if room_mask.sum() < perimeter_depth_px * perimeter_depth_px:
            thermal_zones[room_mask == 1] = zone_id
            zone_id += 1
            continue
            
        # 与窗户影响区相交的部分为外区，其余为内区 (Core)
        p_zone = cv2.bitwise_and(room_mask, perimeter_influence)
        c_zone = room_mask - p_zone
        
        if p_zone.sum() > 0:
            thermal_zones[p_zone == 1] = zone_id
            zone_id += 1
        if c_zone.sum() > 0:
            thermal_zones[c_zone == 1] = zone_id
            zone_id += 1
            
    return thermal_zones

def visualize_zones(zones, orig_bgr):
    # 为不同分区生成随机颜色
    overlay = orig_bgr.copy()
    unique_zones = np.unique(zones)
    colors = np.random.randint(0, 255, size=(max(unique_zones)+1, 3))
    
    for z_id in unique_zones:
        if z_id <= 1: continue # 保留边界
        mask = (zones == z_id)
        color = colors[z_id].tolist()
        # 半透明覆盖
        overlay[mask] = overlay[mask] * 0.4 + np.array(color) * 0.6
        
    return overlay.astype(np.uint8)

if __name__ == "__main__":
    test_dir = Path("test_commercial")
    out_dir = Path("output_commercial_zoning")
    out_dir.mkdir(exist_ok=True)
    
    # 1. 加载模型
    print("Loading segmentation model...")
    segmenter = FloorplanSegmenter()
    
    # 2. 对所有商业建筑CAD进行测试
    images = list(test_dir.glob("*.jpg"))
    for img_path in images:
        print(f"Processing {img_path.name}...")
        orig = cv2.imread(str(img_path))
        
        # 降采样如果图太大，可以加速
        # orig = cv2.resize(orig, (1024, 1024))
        
        res = segmenter.predict(orig, use_preprocessing=True)
        pred_mask = res['mask']
        
        # 提取房间
        room_markers = extract_rooms(pred_mask)
        room_vis = visualize_zones(room_markers, orig)
        
        # 提取内外热区
        # 商业建筑常用深度比例尺，当前盲设50像素（可依据CAD真实比例尺调整）
        thermal_zones = split_thermal_zones(pred_mask, room_markers, perimeter_depth_px=45)
        thermal_vis = visualize_zones(thermal_zones, orig)
        
        # 连图保存
        h, w = orig.shape[:2]
        combined = np.hstack([orig, room_vis, thermal_vis])
        
        save_path = out_dir / f"zoning_{img_path.name}"
        cv2.imwrite(str(save_path), combined)
        print(f"Saved to {save_path}")
