import urllib.request
import cv2
import numpy as np

url = "https://huggingface.co/datasets/Voxel51/FloorPlanCAD/resolve/main/data/0000-0003.png"
out_path = "test_0000-0003.png"

try:
    print("Downloading test image...")
    urllib.request.urlretrieve(url, out_path)
    print("Downloaded successfully!")
    img = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        print(f"Image shape: {img.shape}")
        # Check if transparent or background color
        if len(img.shape) == 3:
            print("RGB image")
        elif len(img.shape) == 4:
            print("RGBA image")
    else:
        print("Failed to read image with OpenCV")
except Exception as e:
    print(f"Error: {e}")
