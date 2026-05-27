"""
对外部下载的平面图进行推理测试，生成原图 + 分割结果的对比图
"""
import os
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"
import sys
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
# BGR colors for OpenCV
COLORS_BGR = [
    (40, 40, 40),       # Background
    (60, 76, 231),      # Wall - 红
    (219, 152, 52),     # Window - 蓝
    (113, 204, 46),     # Door - 绿
]

def load_model():
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=NUM_CLASSES)
    ckpt = torch.load(str(MODEL_PATH), map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def predict(model, img_rgb, device):
    """对图片做推理，返回原始尺寸的预测 mask"""
    h_orig, w_orig = img_rgb.shape[:2]
    
    # 预处理
    aug = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    processed = aug(image=img_rgb)
    tensor = torch.from_numpy(processed['image'].transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).squeeze().cpu().numpy()
    
    # Resize 回原尺寸
    pred_full = cv2.resize(pred.astype(np.uint8), (w_orig, h_orig), 
                           interpolation=cv2.INTER_NEAREST)
    return pred_full


def create_result_image(img_rgb, pred_mask, save_path):
    """生成完整的对比图：原图(左) + 叠加分割结果(右)"""
    h, w = img_rgb.shape[:2]
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # 创建纯色分割图（用于中间显示）
    seg_color = np.zeros_like(img_bgr)
    for cls_id, color in enumerate(COLORS_BGR):
        seg_color[pred_mask == cls_id] = color
    
    # 创建叠加图
    alpha = 0.5
    overlay = img_bgr.copy()
    fg = pred_mask > 0  # 非背景区域
    overlay[fg] = cv2.addWeighted(img_bgr, 1 - alpha, seg_color, alpha, 0)[fg]
    
    # 画轮廓
    for cls_id in [1, 2, 3]:
        cls_mask = (pred_mask == cls_id).astype(np.uint8)
        contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, COLORS_BGR[cls_id], 2)
    
    # 拼接：原图 | 叠加结果
    gap = 6
    title_h = 60
    legend_h = 50
    
    # 如果图太大，缩放
    max_w = 800
    if w > max_w:
        scale = max_w / w
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale)
        overlay = cv2.resize(overlay, None, fx=scale, fy=scale)
        h, w = img_bgr.shape[:2]
    
    canvas_w = w * 2 + gap
    canvas_h = h + title_h + legend_h
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 25
    
    # 放置图片
    canvas[title_h:title_h + h, 0:w] = img_bgr
    canvas[title_h:title_h + h, w + gap:w * 2 + gap] = overlay
    
    # 分隔线
    canvas[title_h:title_h + h, w:w + gap] = (60, 60, 60)
    
    # 标题
    cv2.putText(canvas, "Original Floorplan", (w // 2 - 100, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Segmentation Result (U-Net + ResNet34)", (w + gap + w // 2 - 210, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    
    # 图例（底部）
    legend_y = title_h + h + 15
    legend_items = [
        ("Wall", COLORS_BGR[1]),
        ("Window", COLORS_BGR[2]),
        ("Door", COLORS_BGR[3]),
    ]
    x_pos = canvas_w // 2 - 200
    for name, color in legend_items:
        cv2.rectangle(canvas, (x_pos, legend_y), (x_pos + 22, legend_y + 22), color, -1)
        cv2.rectangle(canvas, (x_pos, legend_y), (x_pos + 22, legend_y + 22), (200, 200, 200), 1)
        cv2.putText(canvas, name, (x_pos + 30, legend_y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        x_pos += 130
    
    # 统计信息
    total_px = pred_mask.size
    wall_pct = (pred_mask == 1).sum() / total_px * 100
    win_pct = (pred_mask == 2).sum() / total_px * 100
    door_pct = (pred_mask == 3).sum() / total_px * 100
    stats = f"Detected: Wall {wall_pct:.1f}%  Window {win_pct:.1f}%  Door {door_pct:.1f}%"
    cv2.putText(canvas, stats, (w + gap + 10, title_h + h + legend_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    
    cv2.imwrite(str(save_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"Saved: {save_path}")
    print(f"  Wall: {wall_pct:.1f}%, Window: {win_pct:.1f}%, Door: {door_pct:.1f}%")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model().to(device)
    print(f"Model loaded on {device}")
    
    # 查找所有外部测试图片
    test_images = list(OUTPUT_DIR.glob("test_*.jpg")) + list(OUTPUT_DIR.glob("test_*.png"))
    
    if not test_images:
        print("No test images found! Place images named test_*.jpg/png in output_paper/visualizations/")
        return
    
    print(f"Found {len(test_images)} test images\n")
    
    for img_path in test_images:
        print(f"Processing: {img_path.name}")
        img_rgb = np.array(Image.open(str(img_path)).convert('RGB'))
        print(f"  Size: {img_rgb.shape[1]}x{img_rgb.shape[0]}")
        
        pred_mask = predict(model, img_rgb, device)
        
        save_name = f"result_{img_path.stem}.png"
        save_path = OUTPUT_DIR / save_name
        create_result_image(img_rgb, pred_mask, save_path)
        print()


if __name__ == "__main__":
    main()
