# -*- coding: utf-8 -*-
"""
BIM 户型图语义分割与能耗计算 Web 平台 - 后端服务
基于 M2_UNet_ResNet34_DA 构件识别引擎 + 物理度日数负荷求解引擎
"""

import os
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"
import io
import json
import time
import base64
import sqlite3
import traceback
from pathlib import Path

import numpy as np
import cv2
import torch

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from inference_api import FloorplanSegmenter
from energy_calc import calculate_building_energy

# ======== 全局 ========
BASE_DIR = Path(__file__).parent
WEB_DIR = BASE_DIR / "web"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
DB_PATH = BASE_DIR / "energy_reports.db"

app = FastAPI(title="BIM Floor Plan Segmentation & Energy Calc Platform")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 内存几何缓存
ai_geometry_cache = {}

# 初始化数据库
def init_db():
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            report_number TEXT PRIMARY KEY,
            floor_area REAL,
            eui REAL,
            total_energy REAL,
            rating TEXT,
            params_json TEXT,
            results_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        conn.close()
        print("[+] SQLite database initialized successfully.")
    except Exception as e:
        print(f"[-] Database initialization failed: {e}")

init_db()

# 加载模型（全局单例）
print("[*] Loading M2_DA model...")
segmenter = FloorplanSegmenter()
print("[+] Model ready.")


# ======== 工具函数 ========

def cv2_to_base64(img_bgr, fmt=".png"):
    """OpenCV BGR图像转base64"""
    _, buf = cv2.imencode(fmt, img_bgr)
    return base64.b64encode(buf).decode("utf-8")


def mask_to_color(mask, alpha_bg=True):
    """分割mask转RGBA彩色图"""
    h, w = mask.shape
    colors = {
        0: (40, 40, 40, 0 if alpha_bg else 255),
        1: (231, 76, 60, 200),   # wall - 红
        2: (52, 152, 219, 200),  # window - 蓝
        3: (46, 204, 113, 200),  # door - 绿
    }
    result = np.zeros((h, w, 4), dtype=np.uint8)
    for cls_id, color in colors.items():
        result[mask == cls_id] = color
    return result


def compute_geometry(mask, scale=1.0):
    """从mask提取矢量化几何"""
    h, w = mask.shape
    result = {"walls": [], "windows": [], "doors": []}
    class_map = {1: "walls", 2: "windows", 3: "doors"}

    for cls_id, key in class_map.items():
        cls_mask = (mask == cls_id).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        cls_mask = cv2.morphologyEx(cls_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            pts = [[float(p[0][0]) * scale, float(p[0][1]) * scale] for p in approx]
            x, y, bw, bh = cv2.boundingRect(cnt)
            result[key].append({
                "pts": pts,
                "area": float(area) * scale * scale,
                "bbox": [float(x)*scale, float(y)*scale, float(bw)*scale, float(bh)*scale],
            })

    return result


def compute_pixel_perimeter(mask, class_id):
    """计算特定类别像素的总周长/长度"""
    cls_mask = (mask == class_id).astype(np.uint8) * 255
    contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_len = 0.0
    for cnt in contours:
        total_len += cv2.arcLength(cnt, True)
    return total_len


def compute_stats(mask):
    """计算分割统计"""
    total = mask.size
    stats = {}
    names = {0: "background", 1: "wall", 2: "window", 3: "door"}
    colors = {0: "#282828", 1: "#e74c3c", 2: "#3498db", 3: "#2ecc71"}
    for cls_id, name in names.items():
        count = int((mask == cls_id).sum())
        stats[name] = {
            "pixels": count,
            "percentage": round(count / total * 100, 2),
            "color": colors[cls_id],
        }
    return stats


# ======== API Endpoints ========

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    preprocessing: str = Form("auto"),
):
    """
    平面图语义分割推理
    """
    t0 = time.time()
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return JSONResponse({"error": "无法解码图片"}, status_code=400)

        h, w = img_bgr.shape[:2]
        use_preprocessing = preprocessing != "none"

        result = segmenter.predict(img_bgr, use_preprocessing=use_preprocessing)
        mask = result["mask"]
        overlay = result["overlay"]

        stats = compute_stats(mask)
        geometry = compute_geometry(mask)
        geo_summary = {
            "walls": len(geometry["walls"]),
            "windows": len(geometry["windows"]),
            "doors": len(geometry["doors"]),
        }

        color_mask = mask_to_color(mask)
        _, mask_buf = cv2.imencode(".png", color_mask)
        mask_b64 = base64.b64encode(mask_buf).decode("utf-8")
        overlay_b64 = cv2_to_base64(overlay, ".jpg")
        orig_b64 = cv2_to_base64(img_bgr, ".jpg")

        elapsed = round(time.time() - t0, 3)

        return JSONResponse({
            "success": True,
            "elapsed_sec": elapsed,
            "image_size": [w, h],
            "stats": stats,
            "geometry_summary": geo_summary,
            "geometry": geometry,
            "images": {
                "original": orig_b64,
                "overlay": overlay_b64,
                "mask": mask_b64,
            }
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# ======== BIM 能耗计算新接口 ========

@app.post("/energy/ai_recognize")
async def energy_ai_recognize(
    raster_file: UploadFile = File(...),
    preprocessing: str = Form("auto"),
    report_number: str = Form(""),
):
    """
    能耗计算平面图智能识别，提取几何形状并缓存
    """
    try:
        contents = await raster_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return JSONResponse({"error": "图片解码失败"}, status_code=400)

        h, w = img_bgr.shape[:2]
        use_preprocessing = preprocessing != "none"

        result = segmenter.predict(img_bgr, use_preprocessing=use_preprocessing)
        mask = result["mask"]
        overlay = result["overlay"]

        # 计算构件的像素长度
        wall_px_len = compute_pixel_perimeter(mask, 1)
        window_px_len = compute_pixel_perimeter(mask, 2)
        door_px_len = compute_pixel_perimeter(mask, 3)

        color_mask = mask_to_color(mask)
        overlay_b64 = cv2_to_base64(overlay, ".jpg")
        orig_b64 = cv2_to_base64(img_bgr, ".jpg")

        # 缓存几何结果
        cache_data = {
            "image_size": [w, h],
            "pixel_lengths": {
                "wall_px": float(wall_px_len),
                "window_px": float(window_px_len),
                "door_px": float(door_px_len),
            }
        }
        
        if report_number:
            ai_geometry_cache[report_number] = cache_data

        return JSONResponse({
            "success": True,
            "image_size": [w, h],
            "pixel_lengths": {
                "wall_px": float(wall_px_len),
                "window_px": float(window_px_len),
                "door_px": float(door_px_len),
            },
            "images": {
                "original": orig_b64,
                "overlay": overlay_b64
            }
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/energy/ai_simulate")
async def energy_ai_simulate(params: dict):
    """
    能耗计算仿真引擎接口，执行物理计算并持久化到 SQLite
    """
    try:
        report_number = params.get("report_number", "")
        if not report_number:
            # 自动生成一个
            report_number = f"BIM-SIM-{int(time.time())}"
            params["report_number"] = report_number

        # 检索 AI 像素长度
        geo_cache = ai_geometry_cache.get(report_number, {})
        pixel_lengths = geo_cache.get("pixel_lengths", {})
        
        # 将像素长度按照 scale 折算到物理长度中
        scale = float(params.get("scale", 0.05))
        
        wall_px = float(pixel_lengths.get("wall_px", 0))
        win_px = float(pixel_lengths.get("window_px", 0))
        
        # 物理周长 (m)
        wall_length_m = wall_px * scale
        win_length_m = win_px * scale
        
        # 如果缓存中存在 AI 数据，覆盖表单默认计算长度
        params["wall_length_m"] = wall_length_m
        params["window_length_m"] = win_length_m
        
        # 物理度日数负荷计算
        results = calculate_building_energy(params, geo_cache)
        
        # 存入数据库
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO reports (report_number, floor_area, eui, total_energy, rating, params_json, results_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            report_number,
            float(params.get("floor_area_m2", 120)),
            float(results["summary"]["eui"]),
            float(results["summary"]["total_energy_kwh"]),
            results["summary"]["rating"],
            json.dumps(params),
            json.dumps(results)
        ))
        conn.commit()
        conn.close()

        return JSONResponse(results)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/energy/reports")
async def get_energy_reports():
    """
    获取能耗报告简要历史列表
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("""
        SELECT report_number, floor_area, eui, total_energy, rating, created_at
        FROM reports
        ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()
        conn.close()

        reports = []
        for r in rows:
            reports.append({
                "report_number": r[0],
                "floor_area": r[1],
                "eui": r[2],
                "total_energy": r[3],
                "rating": r[4],
                "created_at": r[5]
            })
        return reports
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/energy/report/{report_number}")
async def get_energy_report_details(report_number: str):
    """
    获取指定报告的参数和计算结果详情
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("""
        SELECT report_number, params_json, results_json
        FROM reports
        WHERE report_number = ?
        """, (report_number,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return JSONResponse({"error": f"未找到报告编号为 {report_number} 的记录"}, status_code=404)

        return {
            "report_number": row[0],
            "params": json.loads(row[1]),
            "results": json.loads(row[2])
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "M2_UNet_ResNet34_DA", "device": str(segmenter.device)}


@app.get("/api/model-info")
async def model_info():
    return {
        "name": "M2_UNet_ResNet34_DA",
        "architecture": "UNet + ResNet34 Encoder",
        "training": {
            "dataset": "CubiCasa5K (5000 floor plans)",
            "best_mIoU": 0.787,
            "epochs": 100,
            "domain_adaptation": True,
        },
        "classes": [
            {"id": 0, "name": "Background", "color": "#282828"},
            {"id": 1, "name": "Wall", "color": "#e74c3c"},
            {"id": 2, "name": "Window", "color": "#3498db"},
            {"id": 3, "name": "Door", "color": "#2ecc71"},
        ]
    }


# ======== 静态文件 & 网页挂载 ========

# 能耗计算前端入口
@app.get("/energy")
async def energy_calc_page():
    view_path = WEB_DIR / "energy_calc_view.html"
    return HTMLResponse(view_path.read_text(encoding="utf-8"))

# 分割主平台前端入口
@app.get("/")
async def root():
    index_path = WEB_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))

# 静态文件服务
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  BIM Floor Plan AI Segmentation & Energy Calc Platform")
    print(f"  http://localhost:8070")
    print(f"  http://localhost:8070/energy")
    print(f"{'='*60}\n")
    uvicorn.run(app, host="0.0.0.0", port=8070, log_level="info")
