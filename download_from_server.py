"""从 BIMweb 服务器下载 cubicasa5k.zip（断点续传）"""
import os, sys, time, requests
from pathlib import Path

SERVER_URL = "http://43.134.227.60:8888/cubicasa5k.zip"
ZIP_PATH = Path(__file__).parent / "data" / "cubicasa5k.zip"
EXPECTED = 5_469_495_706

os.makedirs(ZIP_PATH.parent, exist_ok=True)

for attempt in range(20):
    current = ZIP_PATH.stat().st_size if ZIP_PATH.exists() else 0
    if current >= EXPECTED:
        print(f"\n[OK] Download complete: {current/1024/1024:.0f} MB")
        break

    print(f"\nAttempt {attempt+1} | {current/1024/1024:.0f}/{EXPECTED/1024/1024:.0f} MB")
    headers = {'Range': f'bytes={current}-'} if current > 0 else {}
    
    try:
        r = requests.get(SERVER_URL, headers=headers, stream=True, timeout=30)
        if r.status_code == 416:
            print("[OK] Already complete")
            break
        mode = 'ab' if r.status_code == 206 else 'wb'
        if r.status_code == 200 and current > 0:
            current = 0
            mode = 'wb'
        
        with open(ZIP_PATH, mode) as f:
            for chunk in r.iter_content(1024*1024):
                if chunk:
                    f.write(chunk)
                    current += len(chunk)
                    pct = current * 100 / EXPECTED
                    sys.stdout.write(f"\r  [{pct:5.1f}%] {current/1024/1024:.0f}/{EXPECTED/1024/1024:.0f} MB")
                    sys.stdout.flush()
    except Exception as e:
        print(f"\n  Error: {e}")
        time.sleep(3)

print(f"\nFinal size: {ZIP_PATH.stat().st_size/1024/1024:.0f} MB")
