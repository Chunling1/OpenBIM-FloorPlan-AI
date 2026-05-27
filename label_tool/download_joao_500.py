import os
import json
import urllib.request
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Configuration ===
OUTPUT_DIR = Path("D:/我的坚果云/工作/博士/bim/标注/原图")
API_URL = "https://huggingface.co/api/datasets/JoaoMigSilva/floorplans"
BASE_URL = "https://huggingface.co/datasets/JoaoMigSilva/floorplans/resolve/main/"
NUM_IMAGES = 500
MAX_WORKERS = 32

def clean_output_dir():
    print(f"Cleaning existing PNG files in {OUTPUT_DIR}...", flush=True)
    if OUTPUT_DIR.exists():
        for f in OUTPUT_DIR.glob("*.png"):
            try:
                f.unlink()
            except Exception as e:
                print(f"Failed to delete {f.name}: {e}", flush=True)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_image_list():
    print("Fetching image list from HuggingFace API...", flush=True)
    req = urllib.request.Request(API_URL, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            siblings = data.get('siblings', [])
            filenames = [x['rfilename'] for x in siblings if x.get('rfilename', '').endswith('.png')]
            print(f"Found {len(filenames)} total PNG files in dataset.", flush=True)
            return filenames
    except Exception as e:
        print(f"Error fetching image list: {e}", flush=True)
        return []

def process_and_save_image(filename):
    url = BASE_URL + filename
    final_path = OUTPUT_DIR / filename
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        # Download
        with urllib.request.urlopen(req, timeout=15) as response:
            img_data = response.read()
            
        # Decode image using numpy (supports Chinese paths)
        img_np = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            return False, "Failed to decode image"
            
        h, w = img.shape[:2]
        
        # Determine background composite and inversion
        if len(img.shape) == 4 or (len(img.shape) == 3 and img.shape[2] == 4):
            alpha = img[:, :, 3]
            rgb = img[:, :, :3]
            
            mask = alpha > 0
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray[mask] if np.any(mask) else gray)
            
            if mean_brightness > 127:
                # Invert to dark background
                inverted_rgb = 255 - rgb
                alpha_norm = (alpha / 255.0)[:, :, np.newaxis]
                bg = np.zeros((h, w, 3), dtype=np.uint8)
                composite = (inverted_rgb * alpha_norm + bg * (1.0 - alpha_norm)).astype(np.uint8)
            else:
                alpha_norm = (alpha / 255.0)[:, :, np.newaxis]
                bg = np.zeros((h, w, 3), dtype=np.uint8)
                composite = (rgb * alpha_norm + bg * (1.0 - alpha_norm)).astype(np.uint8)
        else:
            # 3 channels (RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            if mean_brightness > 127:
                composite = 255 - img
            else:
                composite = img
                
        # Save composite image supporting Chinese paths
        _, buf = cv2.imencode('.png', composite)
        buf.tofile(str(final_path))
        
        return True, "Success"
    except Exception as e:
        return False, str(e)

def main():
    clean_output_dir()
    
    filenames = fetch_image_list()
    if not filenames:
        print("No files found. Exiting.", flush=True)
        return
        
    # Select the first 500 images
    selected_filenames = filenames[:NUM_IMAGES]
    print(f"Selected {len(selected_filenames)} images for download.", flush=True)
    
    success_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(process_and_save_image, fname): fname for fname in selected_filenames}
        
        for i, future in enumerate(as_completed(future_to_file)):
            fname = future_to_file[future]
            try:
                success, msg = future.result()
                if success:
                    success_count += 1
                else:
                    failed_count += 1
                    print(f"Failed to process {fname}: {msg}", flush=True)
            except Exception as exc:
                failed_count += 1
                print(f"{fname} generated an exception: {exc}", flush=True)
                
            if (i + 1) % 10 == 0 or (i + 1) == len(selected_filenames):
                print(f"Progress: {i+1}/{len(selected_filenames)} downloaded and processed. (Success: {success_count}, Failed: {failed_count})", flush=True)
                
    print("\n=== Download Complete ===")
    print(f"Total: {len(selected_filenames)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")

if __name__ == "__main__":
    main()
