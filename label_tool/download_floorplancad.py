import os
import json
import urllib.request
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def is_good_floorplan(img_bytes):
    try:
        # Decode image
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, None
            
        # Calculate percentage of non-black pixels
        # CAD background is black (value near 0)
        # We check pixels > 10 (to allow some compression noise)
        non_black = np.sum(img > 10)
        total_pixels = img.shape[0] * img.shape[1]
        pct = non_black / total_pixels
        
        # If too empty (< 1% pixels), or too full (mostly white background, > 95% pixels), reject
        if pct < 0.015 or pct > 0.95:
            return False, None
            
        # Also check standard deviation to ensure contrast
        std = np.std(img)
        if std < 15.0:
            return False, None
            
        return True, img
    except Exception:
        return False, None

def download_file(url, filename, output_dir, filter_empty=True):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = response.read()
            
        if filter_empty:
            good, img = is_good_floorplan(data)
            if not good:
                return False, "Empty or low quality floorplan"
                
            filepath = os.path.join(output_dir, filename)
            # Save using imencode to support Unicode paths
            ext = os.path.splitext(filename)[1] or '.png'
            res, encoded_img = cv2.imencode(ext, img)
            if res:
                encoded_img.tofile(filepath)
                return True, filepath
            else:
                return False, "Failed to encode image"
        else:
            # Just save directly
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return False, "Not a valid image"
            filepath = os.path.join(output_dir, filename)
            ext = os.path.splitext(filename)[1] or '.png'
            res, encoded_img = cv2.imencode(ext, img)
            if res:
                encoded_img.tofile(filepath)
                return True, filepath
            else:
                return False, "Failed to encode image"
            
    except Exception as e:
        return False, str(e)

if __name__ == '__main__':
    output_dir = r"D:\我的坚果云\工作\博士\bim\标注\原图"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # We keep 1.png
    current_idx = 2
    
    # 1. Download Baidu images first (since they are highly targeted CAD drawings & hand-drawn plans)
    print("--- Downloading Baidu CAD and Hand-Drawn Images ---")
    baidu_urls = []
    try:
        with open(r"C:\Users\chunge\.gemini\antigravity\scratch\baidu_cad_urls.json", 'r', encoding='utf-8') as f:
            baidu_urls.extend(json.load(f))
        with open(r"C:\Users\chunge\.gemini\antigravity\scratch\baidu_hand_drawn_urls.json", 'r', encoding='utf-8') as f:
            baidu_urls.extend(json.load(f))
    except Exception as e:
        print("Error loading Baidu URLs:", e)
        
    # Remove duplicates
    baidu_urls = list(set(baidu_urls))
    print(f"Total Baidu URLs: {len(baidu_urls)}")
    
    success_baidu = 0
    # Download Baidu images without hard filtering (or with soft check)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for idx, url in enumerate(baidu_urls):
            filename = f"{current_idx}.png"
            futures[executor.submit(download_file, url, filename, output_dir, filter_empty=False)] = current_idx
            current_idx += 1
            
        for fut in as_completed(futures):
            c_idx = futures[fut]
            success, msg = fut.result()
            if success:
                success_baidu += 1
                print(f"Baidu download success: {msg}")
            else:
                print(f"Baidu download fail for index {c_idx}: {msg}")
                # Remove file if failed
                filepath = os.path.join(output_dir, f"{c_idx}.png")
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except:
                        pass

    print(f"Completed Baidu downloads. Success count: {success_baidu}")
    
    # Update current index based on actual files in folder
    existing_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    indices = []
    for f in existing_files:
        try:
            indices.append(int(os.path.splitext(f)[0]))
        except ValueError:
            pass
    current_idx = max(indices) + 1 if indices else 2
    print(f"Next image index will be: {current_idx}")
    
    # 2. Download FloorPlanCAD images and filter
    print("--- Downloading and Filtering FloorPlanCAD Images ---")
    metadata_path = r'C:\Users\chunge\.gemini\antigravity\brain\3090097c-bc3b-4d39-abee-d293171b39f9\.system_generated\steps\1816\content.md'
    with open(metadata_path, 'r', encoding='utf-8') as f:
        text = f.read()
    idx = text.find('{')
    js = json.loads(text[idx:])
    png_files = [s['rfilename'] for s in js['siblings'] if s['rfilename'].startswith('data/') and s['rfilename'].endswith('.png')]
    
    # We want to download enough good floorplans to reach around 150-200 images in total.
    target_total = 200
    current_count = len(os.listdir(output_dir))
    needed = target_total - current_count
    print(f"Current images in directory: {current_count}. Needed: {needed}")
    
    if needed > 0:
        success_hf = 0
        lock = time.time()
        # We will submit requests in batches
        with ThreadPoolExecutor(max_workers=8) as executor:
            batch_size = 60
            for i in range(0, len(png_files), batch_size):
                if success_hf >= needed:
                    break
                batch = png_files[i:i+batch_size]
                futures = {}
                for filename_hf in batch:
                    url = f"https://huggingface.co/datasets/Voxel51/FloorPlanCAD/resolve/main/{filename_hf}"
                    # Try to use next index
                    futures[executor.submit(download_file, url, f"temp_{filename_hf.replace('/', '_')}", output_dir, filter_empty=True)] = url
                    
                for fut in as_completed(futures):
                    url = futures[fut]
                    success, filepath = fut.result()
                    if success:
                        # Rename temp file to current index
                        final_filename = f"{current_idx}.png"
                        final_filepath = os.path.join(output_dir, final_filename)
                        try:
                            if os.path.exists(final_filepath):
                                os.remove(final_filepath)
                            os.rename(filepath, final_filepath)
                            print(f"FloorPlanCAD download success: {final_filepath}")
                            current_idx += 1
                            success_hf += 1
                            if success_hf >= needed:
                                break
                        except Exception as rename_err:
                            print(f"Rename error: {rename_err}")
                    else:
                        pass
                print(f"Batch progress: downloaded {success_hf}/{needed} good CAD images.")
                time.sleep(1)
                
    print(f"Finished! Total files in original directory: {len(os.listdir(output_dir))}")
