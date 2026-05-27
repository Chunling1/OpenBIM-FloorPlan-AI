import urllib.request
import cv2
import numpy as np
from pathlib import Path

img_id = 10
url = f"https://huggingface.co/datasets/JoaoMigSilva/floorplans/resolve/main/floorplan_{img_id}.png"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

try:
    with urllib.request.urlopen(req) as response:
        data = response.read()
        img_np = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
        
        # Save raw downloaded image
        raw_path = Path("raw_sample.png")
        _, buf = cv2.imencode('.png', img)
        buf.tofile(str(raw_path))
        print("Raw saved.")
        
        # Check shape and channels
        h, w = img.shape[:2]
        print(f"Shape: {img.shape}")
        
        # Let's create a black background composite.
        # If it has 4 channels (RGBA)
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            rgb = img[:, :, :3]
            
            # Print some pixel values where alpha > 0
            mask = alpha > 0
            if np.any(mask):
                sample_pixels = rgb[mask][:5]
                print("Sample RGB pixels under mask:", sample_pixels)
            
            # Let's see if the lines are black on white or white on black
            # Usually these are white background (255, 255, 255) with black lines (0, 0, 0)
            # If so, the alpha channel might be 255 everywhere, or it might be transparent.
            # Let's do a composite.
            # If the background is white and lines are black, we want to invert the colors (so lines are white/colored on black background).
            # Let's calculate mean brightness
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray[mask] if np.any(mask) else gray)
            print("Mean brightness of active pixels:", mean_brightness)
            
            # If the mean brightness is high (> 127), it's a light background image.
            # We want to invert the RGB colors to make it dark, and composite it on a black background.
            if mean_brightness > 127:
                print("Inverting light background image to dark...")
                inverted_rgb = 255 - rgb
                # Composite on black background: where alpha is present, use inverted RGB, else black
                # Normalize alpha to 0-1
                alpha_norm = (alpha / 255.0)[:, :, np.newaxis]
                bg = np.zeros((h, w, 3), dtype=np.uint8)
                composite = (inverted_rgb * alpha_norm + bg * (1.0 - alpha_norm)).astype(np.uint8)
            else:
                print("Image is already dark background...")
                alpha_norm = (alpha / 255.0)[:, :, np.newaxis]
                bg = np.zeros((h, w, 3), dtype=np.uint8)
                composite = (rgb * alpha_norm + bg * (1.0 - alpha_norm)).astype(np.uint8)
        else:
            # 3 channels (RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            print("Mean brightness:", mean_brightness)
            if mean_brightness > 127:
                print("Inverting light background RGB image...")
                composite = 255 - img
            else:
                composite = img
                
        # Save composite image
        comp_path = Path("composite_sample.png")
        _, buf = cv2.imencode('.png', composite)
        buf.tofile(str(comp_path))
        print("Composite saved.")
        
except Exception as e:
    print("Error:", e)
