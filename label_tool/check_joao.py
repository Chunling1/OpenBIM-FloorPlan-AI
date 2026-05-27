import urllib.request
import cv2
import numpy as np

url = "https://huggingface.co/datasets/JoaoMigSilva/floorplans/resolve/main/floorplan_10.png"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

try:
    with urllib.request.urlopen(req) as response:
        data = response.read()
        print(f"Downloaded {len(data)} bytes")
        img_np = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
        print("Image shape:", img.shape)
        # Check values
        unique_vals = np.unique(img)
        print("Unique values in image:", unique_vals[:20])
except Exception as e:
    print("Error:", e)
