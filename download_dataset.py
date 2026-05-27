"""
CubiCasa5K 数据集下载与预处理
用于建筑平面图语义分割（墙体/窗户/门识别）
"""

import os
import sys
import zipfile
import urllib.request
import json
from pathlib import Path

# === 配置 ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CUBICASA_DIR = DATA_DIR / "cubicasa5k"

# CubiCasa5K GitHub repo
REPO_URL = "https://github.com/CubiCasa/CubiCasa5k"
# 数据集通常从这个链接下载（~1.5GB）
DATASET_URL = "https://zenodo.org/records/2613548/files/cubicasa5k.zip"
# 备用：Kaggle 上也有


import requests

def download_dataset():
    """下载 CubiCasa5K 数据集 (支持断点续传)"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    zip_path = DATA_DIR / "cubicasa5k.zip"
    
    if CUBICASA_DIR.exists() and len(list(CUBICASA_DIR.iterdir())) > 10:
        print(f"[OK] 数据集已存在: {CUBICASA_DIR}")
        return True
    
    # 获取已下载大小，用于断点续传
    downloaded_size = zip_path.stat().st_size if zip_path.exists() else 0
    
    headers = {}
    if downloaded_size > 0:
        headers['Range'] = f'bytes={downloaded_size}-'
        print(f"[+] 发现未完成的下载，重修开始自 {downloaded_size/1024/1024:.1f} MB")
        mode = 'ab'
    else:
        mode = 'wb'

    print(f"[↓] 正在下载 CubiCasa5K 数据集...")
    print(f"    来源: {DATASET_URL}")
    print(f"    注意这需要一些时间，请保持网络连接...")
    
    try:
        response = requests.get(DATASET_URL, headers=headers, stream=True, timeout=10)
        
        # 检查是否支持断点续传 (206 Partial Content) 或重新下载 (200 OK)
        if response.status_code not in [200, 206]:
            if response.status_code == 416: # Range Not Satisfiable (已经下载完了)
                print("[OK] 文件似乎已经下载完成。")
                pass
            else:
                print(f"[ERROR] 下载失败: HTTP {response.status_code}")
                return False
                
        if response.status_code == 200 and mode == 'ab':
            print("[!] 服务端不支持断点续传，将重新下载。")
            mode = 'wb'
            downloaded_size = 0
            
        total_size = int(response.headers.get('content-length', 0)) + downloaded_size
        
        if response.status_code in [200, 206]:
            with open(zip_path, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            pct = min(100.0, downloaded_size * 100 / total_size)
                            mb = downloaded_size / 1024 / 1024
                            total_mb = total_size / 1024 / 1024
                            sys.stdout.write(f"\r    [{pct:5.1f}%] {mb:.1f}/{total_mb:.1f} MB")
                            sys.stdout.flush()
        print("\n[OK] 下载结束 (或已确认完整)")
        
    except Exception as e:
        print(f"\n[ERROR] 下载中断: {e}")
        return False
    
    # 解压
    print(f"[↻] 正在解压...")
    try:
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(DATA_DIR))
        print(f"[✓] 解压完成: {CUBICASA_DIR}")
    except Exception as e:
        print(f"[✗] 解压失败: {e}")
        return False
    
    return True


def create_split_files():
    """
    创建 train/val/test 分割文件
    如果 CubiCasa5K 自带分割文件则直接使用
    """
    for split_name in ['train', 'val', 'test']:
        split_file = CUBICASA_DIR / f"{split_name}.txt"
        if split_file.exists():
            lines = split_file.read_text().strip().split('\n')
            print(f"[✓] {split_name}: {len(lines)} 个样本")
    
    # 检查数据完整性
    sample_dirs = [d for d in CUBICASA_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
    print(f"\n[i] 共发现 {len(sample_dirs)} 个样本目录")
    
    # 检查一个样本的结构
    if sample_dirs:
        sample = sample_dirs[0]
        files = list(sample.iterdir())
        print(f"[i] 样本结构 ({sample.name}/):")
        for f in sorted(files):
            print(f"    - {f.name} ({f.stat().st_size / 1024:.0f} KB)")


def verify_dataset():
    """验证数据集完整性"""
    if not CUBICASA_DIR.exists():
        print("[✗] 数据集目录不存在")
        return False
    
    # 统计
    png_count = len(list(CUBICASA_DIR.rglob("*.png")))
    svg_count = len(list(CUBICASA_DIR.rglob("*.svg")))
    
    print(f"\n[i] 数据集统计:")
    print(f"    PNG 图片: {png_count}")
    print(f"    SVG 标注: {svg_count}")
    
    if png_count >= 4000:
        print("[✓] 数据集完整")
        return True
    else:
        print("[⚠] 数据集可能不完整")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("CubiCasa5K 数据集准备工具")
    print("=" * 60)
    
    if download_dataset():
        create_split_files()
        verify_dataset()
    
    print("\n完成！可以开始训练了。")
