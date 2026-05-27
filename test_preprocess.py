"""
预处理黑底CAD图 → 白底黑线，然后推理
"""
import os
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import torch
import segmentation_models_pytorch as smp
import albumentations as A
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "M2_UNet_ResNet34_best.pt"
OUTPUT_DIR = BASE_DIR / "output_paper" / "visualizations"
IMG_SIZE = 512
NUM_CLASSES = 4

CLASS_NAMES = ["Background", "Wall", "Window", "Door"]
COLORS_BGR = [(40,40,40), (60,76,231), (219,152,52), (113,204,46)]


def load_model():
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=NUM_CLASSES)
    ckpt = torch.load(str(MODEL_PATH), map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def preprocess_dark_cad(img_bgr):
    """黑底CAD → 白底黑线"""
    # 1. 转灰度
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. 反色：黑底白线 → 白底黑线
    inverted = 255 - gray
    
    # 3. 增强对比度，让线条更清晰
    # CLAHE自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(inverted)
    
    # 4. 轻微二值化增强：让背景更白、线条更黑
    _, binary = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
    
    # 5. 混合：保留一些灰度细节
    result = cv2.addWeighted(enhanced, 0.4, binary, 0.6, 0)
    
    # 6. 转回3通道RGB
    result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result_rgb


def predict(model, img_rgb, device):
    aug = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    processed = aug(image=img_rgb)
    tensor = torch.from_numpy(processed['image'].transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).squeeze().cpu().numpy()
    return cv2.resize(pred.astype(np.uint8), (img_rgb.shape[1], img_rgb.shape[0]),
                      interpolation=cv2.INTER_NEAREST)


def create_triple_comparison(orig_bgr, preprocessed_rgb, pred_mask, save_path):
    """三栏对比图：原图 | 预处理后 | 分割结果"""
    h, w = orig_bgr.shape[:2]
    preprocessed_bgr = cv2.cvtColor(preprocessed_rgb, cv2.COLOR_RGB2BGR)
    
    # 分割叠加图
    overlay = preprocessed_bgr.copy()
    seg_color = np.zeros_like(overlay)
    for cls_id, color in enumerate(COLORS_BGR):
        seg_color[pred_mask == cls_id] = color
    fg = pred_mask > 0
    overlay[fg] = cv2.addWeighted(preprocessed_bgr, 0.45, seg_color, 0.55, 0)[fg]
    for cls_id in [1, 2, 3]:
        cls_mask = (pred_mask == cls_id).astype(np.uint8)
        contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, COLORS_BGR[cls_id], 2)
    
    # 缩放
    max_w = 520
    if w > max_w:
        scale = max_w / w
        orig_bgr = cv2.resize(orig_bgr, None, fx=scale, fy=scale)
        preprocessed_bgr = cv2.resize(preprocessed_bgr, None, fx=scale, fy=scale)
        overlay = cv2.resize(overlay, None, fx=scale, fy=scale)
        h, w = orig_bgr.shape[:2]
    
    gap = 4
    title_h = 55
    legend_h = 45
    canvas_w = w * 3 + gap * 2
    canvas_h = h + title_h + legend_h
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 25
    
    # 放置三张图
    canvas[title_h:title_h+h, 0:w] = orig_bgr
    canvas[title_h:title_h+h, w+gap:w*2+gap] = preprocessed_bgr
    canvas[title_h:title_h+h, w*2+gap*2:w*3+gap*2] = overlay
    
    # 分隔线
    canvas[title_h:title_h+h, w:w+gap] = (60,60,60)
    canvas[title_h:title_h+h, w*2+gap:w*2+gap*2] = (60,60,60)
    
    # 标题
    titles = [
        ("Original (Black BG)", w//2 - 100),
        ("Preprocessed (White BG)", w + gap + w//2 - 120),
        ("Segmentation Result", w*2 + gap*2 + w//2 - 100),
    ]
    for text, x in titles:
        cv2.putText(canvas, text, (x, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1, cv2.LINE_AA)
    
    # 图例
    ly = title_h + h + 12
    items = [("Wall", COLORS_BGR[1]), ("Window", COLORS_BGR[2]), ("Door", COLORS_BGR[3])]
    lx = canvas_w // 2 - 180
    for name, color in items:
        cv2.rectangle(canvas, (lx, ly), (lx+18, ly+18), color, -1)
        cv2.rectangle(canvas, (lx, ly), (lx+18, ly+18), (200,200,200), 1)
        cv2.putText(canvas, name, (lx+25, ly+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)
        lx += 120
    
    # 统计
    total = pred_mask.size
    stats = f"Wall:{(pred_mask==1).sum()/total*100:.1f}%  Win:{(pred_mask==2).sum()/total*100:.1f}%  Door:{(pred_mask==3).sum()/total*100:.1f}%"
    cv2.putText(canvas, stats, (canvas_w - 300, canvas_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1, cv2.LINE_AA)
    
    cv2.imwrite(str(save_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"  Saved: {save_path.name}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model().to(device)
    print(f"Model loaded on {device}\n")
    
    # 处理所有中国CAD图
    cad_images = sorted(OUTPUT_DIR.glob("test_china_cad_*.jpg")) + sorted(OUTPUT_DIR.glob("test_china_cad_*.png"))
    
    for img_path in cad_images:
        print(f"Processing: {img_path.name}")
        orig_bgr = cv2.imread(str(img_path))
        
        # 预处理：黑底 → 白底
        preprocessed_rgb = preprocess_dark_cad(orig_bgr)
        
        # 推理
        pred = predict(model, preprocessed_rgb, device)
        
        wall_pct = (pred==1).sum() / pred.size * 100
        win_pct = (pred==2).sum() / pred.size * 100
        door_pct = (pred==3).sum() / pred.size * 100
        print(f"  Wall: {wall_pct:.1f}%, Window: {win_pct:.1f}%, Door: {door_pct:.1f}%")
        
        # 生成三栏对比图
        save_path = OUTPUT_DIR / f"preprocessed_{img_path.stem}.png"
        create_triple_comparison(orig_bgr, preprocessed_rgb, pred, save_path)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
