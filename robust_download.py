"""
稳健的 CubiCasa5K 下载器 - 自动重试断点续传
解决 Zenodo SSL 连接不稳定问题
"""
import os
import sys
import time
import zipfile
import requests
from pathlib import Path

URL = "https://zenodo.org/records/2613548/files/cubicasa5k.zip"
DATA_DIR = Path(__file__).parent / "data"
ZIP_PATH = DATA_DIR / "cubicasa5k.zip"
EXTRACT_DIR = DATA_DIR / "cubicasa5k"
EXPECTED_SIZE = 5_470_000_000  # ~5.2GB 估算

MAX_RETRIES = 50
CHUNK_SIZE = 1024 * 1024  # 1MB chunks (larger = fewer writes)
TIMEOUT = 30


def download_with_resume():
    """断点续传下载，自动重试"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for attempt in range(1, MAX_RETRIES + 1):
        current_size = ZIP_PATH.stat().st_size if ZIP_PATH.exists() else 0
        
        if current_size > EXPECTED_SIZE * 0.95:
            print(f"\n[OK] 文件大小 {current_size/1024/1024:.0f} MB，看起来已完成")
            return True
        
        print(f"\n--- 尝试 {attempt}/{MAX_RETRIES} | 已下载 {current_size/1024/1024:.1f} MB ---")
        
        headers = {}
        if current_size > 0:
            headers['Range'] = f'bytes={current_size}-'
        
        try:
            resp = requests.get(URL, headers=headers, stream=True, timeout=TIMEOUT)
            
            if resp.status_code == 416:
                print("[OK] 服务端说文件已完整下载")
                return True
            
            if resp.status_code == 200 and current_size > 0:
                print("[!] 服务端不支持续传，重新开始")
                current_size = 0
                mode = 'wb'
            elif resp.status_code == 206:
                mode = 'ab'
            elif resp.status_code == 200:
                mode = 'wb'
            else:
                print(f"[!] HTTP {resp.status_code}，5秒后重试")
                time.sleep(5)
                continue
            
            total = int(resp.headers.get('content-length', 0)) + current_size
            
            with open(ZIP_PATH, mode) as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        current_size += len(chunk)
                        pct = current_size * 100 / total if total else 0
                        sys.stdout.write(
                            f"\r    [{pct:5.1f}%] {current_size/1024/1024:.0f}/{total/1024/1024:.0f} MB"
                        )
                        sys.stdout.flush()
            
            print(f"\n[OK] 下载流结束，当前 {current_size/1024/1024:.0f} MB")
            
            # 检查是否完整
            if total > 0 and current_size >= total:
                return True
                
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
                Exception) as e:
            err_type = type(e).__name__
            print(f"\n[!] {err_type}: {str(e)[:100]}")
            wait = min(5 * attempt, 30)
            print(f"    {wait}秒后自动重试...")
            time.sleep(wait)
    
    print(f"\n[FAIL] {MAX_RETRIES} 次尝试后仍未完成")
    return False


def extract():
    """解压"""
    if EXTRACT_DIR.exists() and len(list(EXTRACT_DIR.iterdir())) > 10:
        print(f"[OK] 已解压: {EXTRACT_DIR}")
        return True
    
    print(f"[..] 解压中 (这需要几分钟)...")
    try:
        with zipfile.ZipFile(str(ZIP_PATH), 'r') as zf:
            zf.extractall(str(DATA_DIR))
        print(f"[OK] 解压完成")
        return True
    except zipfile.BadZipFile:
        print(f"[FAIL] ZIP 文件损坏，删除后请重新下载")
        ZIP_PATH.unlink(missing_ok=True)
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("CubiCasa5K 稳健下载器")
    print("=" * 60)
    
    if download_with_resume():
        if extract():
            print("\n[OK] 数据集准备完毕！可以开始训练。")
        else:
            print("\n[FAIL] 解压失败")
    else:
        print("\n[FAIL] 下载失败")
