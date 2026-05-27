import os
import json
import urllib.request
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Configuration ===
JSON_PATH = Path("C:/Users/chunge/.gemini/antigravity/scratch/floorplan_segmentation/label_tool/samples.json")
OUTPUT_DIR = Path("D:/我的坚果云/工作/博士/bim/标注/原图")
BASE_URL = "https://huggingface.co/datasets/Voxel51/FloorPlanCAD/resolve/main/data/"
NUM_IMAGES = 500
MAX_WORKERS = 32

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def process_image(filename):
    url = BASE_URL + filename
    temp_path = OUTPUT_DIR / f"_temp_{filename}"
    final_path = OUTPUT_DIR / filename
    
    if final_path.exists():
        return True, "Already exists"
        
    try:
        # Download
        urllib.request.urlretrieve(url, str(temp_path))
        
        # Read image supporting Chinese paths
        img_data = np.fromfile(str(temp_path), dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)
        if img is None:
            if temp_path.exists():
                temp_path.unlink()
            return False, "Failed to decode image"
            
        # Composite on black background
        h, w = img.shape[:2]
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        
        if len(img.shape) == 4 or (len(img.shape) == 3 and img.shape[2] == 4):
            # Alpha composite
            alpha = img[:, :, 3:4] / 255.0
            rgb = img[:, :, :3]
            composite = (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)
        else:
            # If no alpha, just invert if white background, or keep as is
            composite = img
            
        # Save as standard RGB PNG supporting Chinese paths
        _, buf = cv2.imencode('.png', composite)
        buf.tofile(str(final_path))
        
        # Clean temp file
        if temp_path.exists():
            temp_path.unlink()
            
        return True, "Downloaded and processed"
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        return False, str(e)

def main():
    print(f"Loading samples from {JSON_PATH}...", flush=True)
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    samples = data.get("samples", [])
    print(f"Total samples available: {len(samples)}", flush=True)
    
    # Filter samples that have detections we care about (wall, window, door)
    target_labels = {"wall", "single_door", "double_door", "sliding_door", "window", "bay_window"}
    filtered_samples = []
    
    for s in samples:
        filepath = s.get("filepath", "")
        if not filepath:
            continue
        filename = os.path.basename(filepath)
        
        # Check detections
        detections = s.get("ground_truth", {}).get("detections", [])
        has_target = False
        for det in detections:
            if det.get("label") in target_labels:
                has_target = True
                break
        
        if has_target:
            filtered_samples.append(filename)
            
    print(f"Filtered samples with target labels: {len(filtered_samples)}", flush=True)
    
    # Select first 500 samples
    selected_samples = filtered_samples[:NUM_IMAGES]
    print(f"Selected {len(selected_samples)} images to download.", flush=True)
    
    # Download concurrently
    success_count = 0
    already_exists_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(process_image, fname): fname for fname in selected_samples}
        
        for i, future in enumerate(as_completed(future_to_file)):
            fname = future_to_file[future]
            try:
                success, msg = future.result()
                if success:
                    success_count += 1
                    if msg == "Already exists":
                        already_exists_count += 1
                else:
                    failed_count += 1
                    print(f"Failed to download {fname}: {msg}", flush=True)
            except Exception as exc:
                failed_count += 1
                print(f"{fname} generated an exception: {exc}", flush=True)
                
            if (i + 1) % 10 == 0 or (i + 1) == len(selected_samples):
                print(f"Progress: {i+1}/{len(selected_samples)} processed. (Success: {success_count}, Failed: {failed_count})", flush=True)
                
    print("\n=== Download Report ===")
    print(f"Total processed: {len(selected_samples)}")
    print(f"Successfully downloaded/composited: {success_count - already_exists_count}")
    print(f"Already existed: {already_exists_count}")
    print(f"Failed: {failed_count}")

if __name__ == "__main__":
    main()
