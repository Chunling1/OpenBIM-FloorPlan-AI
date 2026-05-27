"""
最终版下载器 - 从 Zenodo 下载 CubiCasa5K
关键改进: 
- 不自动删除未完成的ZIP
- 下载完成后校验大小
- 完成后自动解压+训练
"""
import os, sys, time, zipfile, requests
from pathlib import Path

URL = "https://zenodo.org/records/2613548/files/cubicasa5k.zip"
BASE = Path(__file__).parent
ZIP_PATH = BASE / "data" / "cubicasa5k.zip"
EXTRACT_DIR = BASE / "data" / "cubicasa5k"
EXPECTED_SIZE = 5_469_495_706
MAX_RETRIES = 100
CHUNK = 512 * 1024  # 512KB chunks

os.makedirs(ZIP_PATH.parent, exist_ok=True)

def download():
    for attempt in range(1, MAX_RETRIES + 1):
        cur = ZIP_PATH.stat().st_size if ZIP_PATH.exists() else 0
        
        if cur >= EXPECTED_SIZE:
            print(f"\n[DONE] {cur/1e6:.0f} MB - complete!")
            return True
        
        remaining = EXPECTED_SIZE - cur
        print(f"\n--- Try {attempt} | {cur/1e6:.0f}/{EXPECTED_SIZE/1e6:.0f} MB | {remaining/1e6:.0f} MB left ---")
        
        try:
            headers = {'Range': f'bytes={cur}-'} if cur > 0 else {}
            r = requests.get(URL, headers=headers, stream=True, timeout=30)
            
            if r.status_code == 416:
                print("Server says file is complete")
                return True
            if r.status_code == 200 and cur > 0:
                print("No resume support, restart")
                cur = 0
                mode = 'wb'
            elif r.status_code == 206:
                mode = 'ab'
            elif r.status_code == 200:
                mode = 'wb'
            else:
                print(f"HTTP {r.status_code}")
                time.sleep(5)
                continue
            
            bytes_this_session = 0
            with open(ZIP_PATH, mode) as f:
                for chunk in r.iter_content(CHUNK):
                    if chunk:
                        f.write(chunk)
                        cur += len(chunk)
                        bytes_this_session += len(chunk)
                        pct = cur * 100 / EXPECTED_SIZE
                        speed = ""  
                        sys.stdout.write(f"\r  [{pct:5.1f}%] {cur/1e6:.0f}/{EXPECTED_SIZE/1e6:.0f} MB")
                        sys.stdout.flush()
            
            print(f"\n  Session: +{bytes_this_session/1e6:.0f} MB")
            if cur >= EXPECTED_SIZE:
                return True
                
        except Exception as e:
            print(f"\n  {type(e).__name__}: {str(e)[:80]}")
            wait = min(3 + attempt, 15)
            print(f"  Retry in {wait}s...")
            time.sleep(wait)
    
    # Even if not fully done, check if usable
    if ZIP_PATH.exists():
        actual = ZIP_PATH.stat().st_size
        if actual > EXPECTED_SIZE * 0.99:
            print(f"\n[WARN] {actual/1e6:.0f} MB - close enough, trying extract")
            return True
    return False

def extract():
    if EXTRACT_DIR.exists() and len(list(EXTRACT_DIR.iterdir())) > 10:
        print("[OK] Already extracted")
        return True
    print("[..] Extracting (takes a few minutes)...")
    try:
        with zipfile.ZipFile(str(ZIP_PATH), 'r') as zf:
            zf.extractall(str(BASE / "data"))
        print("[OK] Extracted!")
        return True
    except zipfile.BadZipFile:
        print("[BAD] ZIP is corrupt. Will NOT delete - retry download.")
        return False

def train():
    print("\n[TRAIN] Starting training...")
    os.system(f'python "{BASE / "train.py"}"')

if __name__ == "__main__":
    print("="*60)
    print("CubiCasa5K: Download -> Extract -> Train")
    print("="*60)
    
    if download():
        if extract():
            train()
