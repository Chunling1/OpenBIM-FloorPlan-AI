"""
BIM 平面图语义标注工具 — 后端 (纯内置 http.server 版本，无需安装 Flask)
标注类别: 0=背景, 1=墙体, 2=窗户, 3=门

用法:
    python server.py [--data-dir DIR] [--mask-dir DIR] [--port PORT]

默认数据目录为脚本所在目录下的 ./data/images 和 ./data/annotations。
可通过命令行参数覆盖。
"""
import os
import json
import base64
import argparse
import getpass
import platform
from datetime import datetime, timezone

import numpy as np
import cv2
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse

# ---- 脚本相对默认路径 ----
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA_DIR = _SCRIPT_DIR / "data" / "images"
_DEFAULT_MASK_DIR = _SCRIPT_DIR / "data" / "annotations"

# 运行时由 main() 设置，模块级变量供 handler / helper 函数使用
DATA_DIR: Path = _DEFAULT_DATA_DIR
MASK_DIR: Path = _DEFAULT_MASK_DIR

SUPPORTED_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def get_image_list():
    files = []
    if DATA_DIR.exists():
        for f in sorted(DATA_DIR.iterdir()):
            if f.suffix.lower() in SUPPORTED_EXT and f.is_file():
                files.append(f.name)
    return files


class LabelToolHTTPHandler(BaseHTTPRequestHandler):
    
    # 禁用日志，使控制台保持干净，仅打印标注保存日志
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed_url.path)
        
        if path == '/' or path == '/index.html':
            index_path = Path(__file__).parent / "static" / "index.html"
            if index_path.exists():
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                with open(index_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "index.html not found")
            return
            
        elif path == '/api/images':
            images = get_image_list()
            result = []
            for name in images:
                stem = Path(name).stem
                mask_path = MASK_DIR / (stem + '_mask.npy')
                result.append({
                    'name': name,
                    'labeled': mask_path.exists(),
                })
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
            return
            
        elif path.startswith('/api/image/'):
            name = path[len('/api/image/'):]
            img_path = DATA_DIR / name
            if img_path.exists():
                self.send_response(200)
                ext = img_path.suffix.lower()
                content_type = 'image/png'
                if ext in {'.jpg', '.jpeg'}:
                    content_type = 'image/jpeg'
                elif ext == '.bmp':
                    content_type = 'image/bmp'
                self.send_header('Content-Type', content_type)
                self.end_headers()
                with open(img_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, f"Image {name} not found")
            return
            
        elif path.startswith('/api/mask/'):
            name = path[len('/api/mask/'):]
            stem = Path(name).stem
            mask_path = MASK_DIR / (stem + '_mask.npy')
            if not mask_path.exists():
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'exists': False}).encode('utf-8'))
                return
                
            try:
                mask = np.load(str(mask_path))
                color_map = {
                    0: [0, 0, 0, 0],       # 背景 - 透明
                    1: [231, 76, 60, 160],  # 墙体 - 红
                    2: [52, 152, 219, 160], # 窗户 - 蓝
                    3: [46, 204, 113, 160], # 门 - 绿
                }
                h, w = mask.shape
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                for cls_id, color in color_map.items():
                    rgba[mask == cls_id] = color

                _, buf = cv2.imencode('.png', cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
                b64 = base64.b64encode(buf).decode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'exists': True, 'mask_b64': b64, 'shape': [h, w]}).encode('utf-8'))
            except Exception as e:
                self.send_error(500, str(e))
            return
            
        elif path == '/api/stats':
            images = get_image_list()
            labeled = sum(1 for n in images if (MASK_DIR / (Path(n).stem + '_mask.npy')).exists())
            result = {
                'total': len(images),
                'labeled': labeled,
                'remaining': len(images) - labeled,
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
            return

        elif path == '/api/export':
            # 返回贡献者信息及所有已标注 mask 文件名，方便准备 PR 提交
            images = get_image_list()
            mask_files = []
            for name in images:
                stem = Path(name).stem
                npy_name = stem + '_mask.npy'
                if (MASK_DIR / npy_name).exists():
                    mask_files.append(npy_name)
            result = {
                'contributor': getpass.getuser(),
                'hostname': platform.node(),
                'exported_at': datetime.now(timezone.utc).isoformat(),
                'data_dir': str(DATA_DIR),
                'mask_dir': str(MASK_DIR),
                'total_images': len(images),
                'labeled_count': len(mask_files),
                'mask_files': mask_files,
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))
            return

        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed_url.path)
        
        if path == '/api/save_mask':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                image_name = data['image_name']
                mask_data = data['mask_data']  # base64 PNG
                stem = Path(image_name).stem
                
                # 解码 PNG
                png_bytes = base64.b64decode(mask_data)
                arr = np.frombuffer(png_bytes, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                
                if img is None:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'Failed to decode mask'}).encode('utf-8'))
                    return
                    
                # 从 RGBA 颜色反推类别
                if len(img.shape) == 3 and img.shape[2] >= 3:
                    if img.shape[2] == 4:
                        b, g, r, a = img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3]
                    else:
                        b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
                        a = np.full_like(r, 255)
                        
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    has_paint = a > 30
                    wall = has_paint & (r > 100) & (r > b) & (r > g)
                    win = has_paint & (b > 100) & (b > r) & (b > g)
                    door = has_paint & (g > 100) & (g > r) & (g > b)
                    
                    mask[wall] = 1
                    mask[win] = 2
                    mask[door] = 3
                else:
                    mask = img
                    
                # 保存
                npy_path = MASK_DIR / (stem + '_mask.npy')
                np.save(str(npy_path), mask)
                
                # 同时保存一份可视化 PNG 供检查
                vis_path = MASK_DIR / (stem + '_mask_vis.png')
                vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
                vis[mask == 1] = [0, 0, 255]    # 墙 - 红 (BGR)
                vis[mask == 2] = [255, 150, 50]  # 窗 - 蓝 (BGR)
                vis[mask == 3] = [0, 200, 0]     # 门 - 绿 (BGR)
                
                _, buf = cv2.imencode('.png', vis)
                buf.tofile(str(vis_path))
                
                stats = {
                    'wall_px': int((mask == 1).sum()),
                    'window_px': int((mask == 2).sum()),
                    'door_px': int((mask == 3).sum()),
                    'bg_px': int((mask == 0).sum()),
                    'shape': list(mask.shape),
                }
                print(f"[SAVED] {npy_path.name} | wall={stats['wall_px']} win={stats['window_px']} door={stats['door_px']}")
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': True, 'stats': stats}).encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
            return
            
        else:
            self.send_error(404, "Not Found")


def run(server_class=HTTPServer, handler_class=LabelToolHTTPHandler, port=8099):
    server_address = ('127.0.0.1', port)
    httpd = server_class(server_address, handler_class)
    print(f"标注工具服务器已启动: http://127.0.0.1:{port}")
    print(f"原图目录: {DATA_DIR}")
    print(f"标注目录: {MASK_DIR}")
    print(f"可标注图片数量: {len(get_image_list())}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='BIM 平面图语义标注工具后端服务器',
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help=f'原图目录 (默认: {_DEFAULT_DATA_DIR})',
    )
    parser.add_argument(
        '--mask-dir',
        type=Path,
        default=_DEFAULT_MASK_DIR,
        help=f'标注输出目录 (默认: {_DEFAULT_MASK_DIR})',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8099,
        help='监听端口 (默认: 8099)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 将命令行参数写入模块级变量
    DATA_DIR = args.data_dir.resolve()
    MASK_DIR = args.mask_dir.resolve()
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    run(port=args.port)
