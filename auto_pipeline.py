"""
全自动流程: 等待下载完成 -> 解压 -> 训练
放后台跑即可
"""
import os
import sys
import time
import zipfile
from pathlib import Path

BASE = Path(__file__).parent
ZIP_PATH = BASE / "data" / "cubicasa5k.zip"
EXTRACT_DIR = BASE / "data" / "cubicasa5k"
TRAIN_SCRIPT = BASE / "train.py"

EXPECTED_MIN_SIZE = 5_000_000_000  # 5GB 最低

def wait_for_download():
    """等待 robust_download.py 完成"""
    print("[1/3] 等待数据集下载完成...")
    
    last_size = 0
    stale_count = 0
    
    while True:
        if not ZIP_PATH.exists():
            print("  ZIP 文件不存在，等待...")
            time.sleep(30)
            continue
        
        size = ZIP_PATH.stat().st_size
        print(f"  当前: {size/1024/1024:.0f} MB", flush=True)
        
        # 检查是否已足够大
        if size >= EXPECTED_MIN_SIZE:
            print(f"  文件大小 {size/1024/1024:.0f} MB >= {EXPECTED_MIN_SIZE/1024/1024:.0f} MB")
            break
        
        # 检查是否停滞
        if size == last_size:
            stale_count += 1
            if stale_count > 10:
                print("  下载似乎停滞了。尝试验证 ZIP 完整性...")
                try:
                    with zipfile.ZipFile(str(ZIP_PATH), 'r') as zf:
                        result = zf.testzip()
                        if result is None:
                            print("  ZIP 完整且有效！继续处理。")
                            break
                except zipfile.BadZipFile:
                    print("  ZIP 不完整，继续等待...")
                    stale_count = 0
        else:
            stale_count = 0
        
        last_size = size
        time.sleep(30)
    
    return True


def extract():
    """解压"""
    if EXTRACT_DIR.exists() and len(list(EXTRACT_DIR.iterdir())) > 10:
        print("[2/3] 数据集已解压，跳过")
        return True
    
    print("[2/3] 解压数据集...", flush=True)
    try:
        with zipfile.ZipFile(str(ZIP_PATH), 'r') as zf:
            zf.extractall(str(BASE / "data"))
        print(f"  解压完成: {EXTRACT_DIR}")
        return True
    except Exception as e:
        print(f"  解压失败: {e}")
        return False


def train():
    """启动训练"""
    print("[3/3] 启动训练...", flush=True)
    os.system(f'python "{TRAIN_SCRIPT}"')


if __name__ == "__main__":
    print("=" * 60)
    print("全自动流程: 下载 -> 解压 -> 训练")
    print("=" * 60)
    
    if wait_for_download():
        if extract():
            train()
        else:
            print("[FAIL] 解压失败，退出")
    else:
        print("[FAIL] 下载未完成，退出")
