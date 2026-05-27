"""
=== 集成补丁 ===
将以下代码添加到 web_server.py 中，实现图纸AI识别功能。

使用方法:
1. 将 floorplan_onnx.py 和 M2_DA_best.onnx 上传到服务器
2. 在 web_server.py 中导入并注册路由
3. 在前端 energy.html 中添加 AI 识别 UI

部署步骤:
  scp deploy/floorplan_onnx.py ubuntu@SERVER:/home/ubuntu/.gemini/antigravity/scratch/
  scp models/M2_DA_best.onnx  ubuntu@SERVER:/home/ubuntu/.gemini/antigravity/scratch/models/
  ssh ubuntu@SERVER "pip install onnxruntime opencv-python-headless"
"""

# ==========================================
# 以下代码添加到 web_server.py 的 import 区域
# ==========================================

# --- AI 图纸识别模块 ---
"""
# 在 web_server.py 头部添加:

try:
    from floorplan_onnx import get_segmenter
    ONNX_MODEL_PATH = '/home/ubuntu/.gemini/antigravity/scratch/models/M2_DA_best.onnx'
    _floorplan_segmenter = get_segmenter(ONNX_MODEL_PATH)
    HAS_FLOORPLAN_AI = True
except Exception as e:
    HAS_FLOORPLAN_AI = False
    print(f"FloorPlan AI not available: {e}")
"""


# ==========================================
# 以下路由代码添加到 web_server.py 的路由区域
# ==========================================

# --- 完整的路由代码，直接复制到 web_server.py ---

ROUTE_CODE = '''
import base64
import cv2
import numpy as np
import time as _time

# ==========================================
# 路由 - AI 图纸识别 (语义分割)
# ==========================================

@app.route('/energy/ai_recognize', methods=['POST'])
@login_required
def ai_recognize():
    """
    AI 识别建筑平面图中的墙体、窗户、门
    支持 raster_file (PNG/JPG) 上传
    与 DXF 上传同步：用户可选择 DXF 矢量 或 图片AI识别
    """
    if not HAS_FLOORPLAN_AI:
        return jsonify({'error': 'AI recognition module not available. Install: pip install onnxruntime opencv-python-headless'}), 501

    try:
        t0 = _time.time()
        report_number = request.form.get('report_number', 'default')
        report_number = secure_filename(report_number) or 'default'
        target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_number)
        os.makedirs(target_dir, exist_ok=True)

        preprocessing = request.form.get('preprocessing', 'auto')
        use_preprocessing = preprocessing != 'none'

        # 读取上传图片
        if 'raster_file' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        f = request.files['raster_file']
        if not f or not f.filename:
            return jsonify({'error': 'Empty file'}), 400

        # 保存原图
        ext = f.filename.rsplit('.', 1)[1].lower() if '.' in f.filename else 'png'
        raster_path = os.path.join(target_dir, f'building_plan_ai.{ext}')
        f.save(raster_path)

        # PDF → PNG 转换
        if ext == 'pdf':
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(raster_path)
                if images:
                    png_path = os.path.join(target_dir, 'building_plan_ai.png')
                    images[0].save(png_path, 'PNG')
                    raster_path = png_path
            except Exception as e:
                return jsonify({'error': f'PDF conversion failed: {e}'}), 500

        # AI 推理
        img_bgr = cv2.imread(raster_path)
        if img_bgr is None:
            return jsonify({'error': 'Cannot decode image'}), 400

        result = _floorplan_segmenter.predict(img_bgr, use_preprocessing=use_preprocessing)

        mask = result['mask']
        overlay = result['overlay']
        stats = result['stats']
        geometry = result['geometry']
        h, w = mask.shape

        # 保存结果
        overlay_path = os.path.join(target_dir, 'ai_overlay.jpg')
        mask_path = os.path.join(target_dir, 'ai_mask.png')
        cv2.imwrite(overlay_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # mask 转彩色保存
        mask_color = np.zeros((h, w, 3), dtype=np.uint8)
        mask_color[mask == 1] = (60, 76, 231)   # wall - red
        mask_color[mask == 2] = (219, 152, 52)   # window - blue
        mask_color[mask == 3] = (113, 204, 46)   # door - green
        cv2.imwrite(mask_path, mask_color)

        # 编码为base64用于前端展示
        _, overlay_buf = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        overlay_b64 = base64.b64encode(overlay_buf).decode('utf-8')

        _, orig_buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        orig_b64 = base64.b64encode(orig_buf).decode('utf-8')

        elapsed = round(_time.time() - t0, 3)

        # 构建几何数据，可直接用于能耗模拟
        geo_summary = {
            'walls': len(geometry['walls']),
            'windows': len(geometry['windows']),
            'doors': len(geometry['doors']),
        }

        # 计算墙体/窗户轮廓的总长度 (像素单位，需乘以比例尺得到实际长度)
        wall_total_length = 0
        for wall in geometry['walls']:
            pts = wall['pts']
            for i in range(len(pts) - 1):
                dx = pts[i+1][0] - pts[i][0]
                dy = pts[i+1][1] - pts[i][1]
                wall_total_length += (dx*dx + dy*dy) ** 0.5

        win_total_length = 0
        for win in geometry['windows']:
            pts = win['pts']
            for i in range(len(pts) - 1):
                dx = pts[i+1][0] - pts[i][0]
                dy = pts[i+1][1] - pts[i][1]
                win_total_length += (dx*dx + dy*dy) ** 0.5

        return jsonify({
            'status': 'success',
            'source': 'AI',
            'model': 'M2_UNet_ResNet34_DA',
            'mIoU': 0.787,
            'elapsed_sec': elapsed,
            'image_size': [w, h],
            'stats': stats,
            'geometry_summary': geo_summary,
            'geometry': geometry,
            'pixel_lengths': {
                'wall_px': round(wall_total_length, 1),
                'window_px': round(win_total_length, 1),
            },
            'images': {
                'original': orig_b64,
                'overlay': overlay_b64,
            },
        })

    except Exception as e:
        logger.error(f"AI recognize error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/energy/ai_status')
def ai_status():
    """检查 AI 识别模块状态"""
    return jsonify({
        'available': HAS_FLOORPLAN_AI,
        'model': 'M2_UNet_ResNet34_DA' if HAS_FLOORPLAN_AI else None,
        'mIoU': 0.787 if HAS_FLOORPLAN_AI else None,
        'classes': ['background', 'wall', 'window', 'door'] if HAS_FLOORPLAN_AI else [],
    })


@app.route('/energy/ai_simulate', methods=['POST'])
@login_required
def ai_simulate():
    """
    使用 AI 识别结果直接进行能耗模拟
    合并 AI识别 + 简单模型计算
    """
    if not HAS_FLOORPLAN_AI:
        return jsonify({'error': 'AI module not available'}), 501

    try:
        data = request.get_json() or {}
        report_number = data.get('report_number', 'default')
        scale = float(data.get('scale', 0.01))  # 像素→米的比例尺，默认1px=1cm
        height = float(data.get('height', 3.0))
        floors = int(data.get('floors', 1))

        target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_number)

        # 查找已有的 AI 识别结果图
        for ext in ['png', 'jpg', 'jpeg']:
            raster_path = os.path.join(target_dir, f'building_plan_ai.{ext}')
            if os.path.exists(raster_path):
                break
        else:
            return jsonify({'error': 'No AI-recognized image found. Upload an image first.'}), 404

        # 推理
        img_bgr = cv2.imread(raster_path)
        result = _floorplan_segmenter.predict(img_bgr, use_preprocessing=True)
        geometry = result['geometry']

        # 计算几何量 (像素→实际尺寸)
        wall_pixels = sum(w['area'] for w in geometry['walls'])
        window_pixels = sum(w['area'] for w in geometry['windows'])
        door_pixels = sum(d['area'] for d in geometry['doors'])

        # 利用 scale 转换
        wall_area_m2 = wall_pixels * (scale ** 2) * floors
        window_area_m2 = window_pixels * (scale ** 2) * floors
        # 估算建筑面积: 总像素减去构件像素
        total_px = img_bgr.shape[0] * img_bgr.shape[1]
        floor_area_m2 = total_px * (scale ** 2)

        # 用简单模型计算
        u_wall = float(data.get('u_wall', 0.6))
        u_win = float(data.get('u_win', 2.5))
        u_roof = float(data.get('u_roof', 0.4))
        u_floor = float(data.get('u_floor', 0.3))

        total_ua = (u_wall * wall_area_m2) + (u_win * window_area_m2) + (u_roof * floor_area_m2) + (u_floor * floor_area_m2)

        city = data.get('city', 'Beijing')
        weather_map = {
            'Beijing': {'hdd': 2800, 'cdd': 500},
            'Shanghai': {'hdd': 1500, 'cdd': 1100},
            'Shenzhen': {'hdd': 500, 'cdd': 1800},
        }
        constants = weather_map.get(city, weather_map['Beijing'])

        heating = total_ua * constants['hdd'] * 24 / 1000
        cooling = total_ua * constants['cdd'] * 24 / 1000
        internal = floor_area_m2 * 25
        total_cons = heating + cooling + internal
        eui = total_cons / floor_area_m2 if floor_area_m2 > 0 else 0

        return jsonify({
            'status': 'success',
            'source': 'AI + Simple Model',
            'geometry': {
                'floor_area_m2': round(floor_area_m2, 2),
                'wall_area_m2': round(wall_area_m2, 2),
                'window_area_m2': round(window_area_m2, 2),
                'wwr': round(window_area_m2 / max(wall_area_m2, 1), 3),
            },
            'loads': {
                'heating_kwh': round(heating, 0),
                'cooling_kwh': round(cooling, 0),
                'internal_kwh': round(internal, 0),
                'total_kwh': round(total_cons, 0),
                'eui': round(eui, 2),
            },
        })

    except Exception as e:
        logger.error(f"AI simulate error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
'''

if __name__ == '__main__':
    print("=" * 60)
    print("  BIM FloorPlan AI - 服务器集成补丁")
    print("=" * 60)
    print()
    print("部署步骤:")
    print()
    print("1. 上传文件到服务器:")
    print("   scp deploy/floorplan_onnx.py ubuntu@SERVER:/home/ubuntu/.gemini/antigravity/scratch/")
    print("   scp models/M2_DA_best.onnx   ubuntu@SERVER:/home/ubuntu/.gemini/antigravity/scratch/models/")
    print()
    print("2. 在服务器安装依赖:")
    print("   pip install onnxruntime opencv-python-headless")
    print()
    print("3. 在 web_server.py 头部添加 import:")
    print("   from floorplan_onnx import get_segmenter")
    print("   _floorplan_segmenter = get_segmenter('models/M2_DA_best.onnx')")
    print("   HAS_FLOORPLAN_AI = True")
    print()
    print("4. 将路由代码复制到 web_server.py")
    print()
    print("5. 重启 gunicorn:")
    print("   sudo systemctl restart gunicorn")
    print()
    print("新增API端点:")
    print("   POST /energy/ai_recognize    - 上传图片进行AI分割")
    print("   GET  /energy/ai_status       - 检查AI模块状态")
    print("   POST /energy/ai_simulate     - 用AI结果做能耗模拟")
