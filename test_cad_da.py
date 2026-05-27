"""
用DA增强版M2模型测试黑底CAD图，对比：
1. 直接推理（黑底原图）
2. 预处理后推理（黑底→白底）
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
# 使用DA增强版模型
MODEL_DA = BASE_DIR / "models" / "M2_UNet_ResNet34_DA_best.pt"
MODEL_ORIG = BASE_DIR / "models" / "M2_UNet_ResNet34_best.pt"
OUTPUT_DIR = BASE_DIR / "output_paper" / "visualizations"
RESULT_DIR = BASE_DIR / "output_cad_test"
RESULT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 512
NUM_CLASSES = 4

CLASS_NAMES = ["Background", "Wall", "Window", "Door"]
COLORS_BGR = [(40,40,40), (60,76,231), (219,152,52), (113,204,46)]


def load_model(model_path):
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=NUM_CLASSES)
    ckpt = torch.load(str(model_path), map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def preprocess_dark_cad(img_bgr):
    """黑底CAD → 白底黑线"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(inverted)
    _, binary = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
    result = cv2.addWeighted(enhanced, 0.4, binary, 0.6, 0)
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


def make_overlay(img_bgr, pred_mask):
    """生成分割叠加图"""
    seg_color = np.zeros_like(img_bgr)
    for cls_id, color in enumerate(COLORS_BGR):
        seg_color[pred_mask == cls_id] = color
    overlay = img_bgr.copy()
    fg = pred_mask > 0
    overlay[fg] = cv2.addWeighted(img_bgr, 0.45, seg_color, 0.55, 0)[fg]
    for cls_id in [1, 2, 3]:
        cls_mask = (pred_mask == cls_id).astype(np.uint8)
        contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, COLORS_BGR[cls_id], 2)
    return overlay


def get_stats(pred_mask):
    total = pred_mask.size
    return {
        'wall': (pred_mask == 1).sum() / total * 100,
        'window': (pred_mask == 2).sum() / total * 100,
        'door': (pred_mask == 3).sum() / total * 100,
    }


def create_comparison(orig_bgr, results, save_path, img_name):
    """
    生成4栏对比图：
    原图 | DA直接推理 | 原版预处理推理 | DA预处理推理
    """
    h, w = orig_bgr.shape[:2]

    # 缩放
    max_w = 420
    if w > max_w:
        scale = max_w / w
        orig_bgr = cv2.resize(orig_bgr, None, fx=scale, fy=scale)
        h, w = orig_bgr.shape[:2]
        for key in results:
            results[key]['overlay'] = cv2.resize(results[key]['overlay'], (w, h))

    gap = 3
    n_cols = len(results) + 1  # 原图 + N个结果
    title_h = 70
    legend_h = 50
    canvas_w = w * n_cols + gap * (n_cols - 1)
    canvas_h = h + title_h + legend_h
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 20

    # 放置原图
    canvas[title_h:title_h+h, 0:w] = orig_bgr

    # 放置结果
    col_idx = 1
    for key, data in results.items():
        x_start = w * col_idx + gap * col_idx
        canvas[title_h:title_h+h, x_start:x_start+w] = data['overlay']
        # 分隔线
        sep_x = x_start - gap
        canvas[title_h:title_h+h, sep_x:sep_x+gap] = (50, 50, 50)
        col_idx += 1

    # 标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    titles = [f"Original: {img_name}"]
    for key, data in results.items():
        s = data['stats']
        titles.append(f"{key}\nW:{s['wall']:.1f}% Wi:{s['window']:.1f}% D:{s['door']:.1f}%")

    for i, title_lines in enumerate(titles):
        x_center = w * i + gap * i + w // 2
        lines = title_lines.split('\n')
        for j, line in enumerate(lines):
            text_size = cv2.getTextSize(line, font, 0.45, 1)[0]
            tx = x_center - text_size[0] // 2
            ty = 22 + j * 22
            cv2.putText(canvas, line, (tx, ty), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # 图例（底部）
    ly = title_h + h + 15
    items = [("Wall", COLORS_BGR[1]), ("Window", COLORS_BGR[2]), ("Door", COLORS_BGR[3])]
    lx = canvas_w // 2 - 170
    for name, color in items:
        cv2.rectangle(canvas, (lx, ly), (lx+18, ly+18), color, -1)
        cv2.rectangle(canvas, (lx, ly), (lx+18, ly+18), (200, 200, 200), 1)
        cv2.putText(canvas, name, (lx+24, ly+14), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        lx += 120

    cv2.imwrite(str(save_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"  Saved: {save_path.name}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading models...")
    model_da = load_model(MODEL_DA).to(device)
    model_orig = load_model(MODEL_ORIG).to(device)
    print(f"Models loaded on {device}\n")

    # 处理所有中国CAD图
    cad_images = sorted(OUTPUT_DIR.glob("test_china_cad_*.jpg")) + sorted(OUTPUT_DIR.glob("test_china_cad_*.png"))

    if not cad_images:
        print("No CAD test images found!")
        return

    print(f"Found {len(cad_images)} CAD images\n")

    for img_path in cad_images:
        print(f"=== {img_path.name} ===")
        orig_bgr = cv2.imread(str(img_path))
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

        # 预处理版本
        preprocessed_rgb = preprocess_dark_cad(orig_bgr)
        preprocessed_bgr = cv2.cvtColor(preprocessed_rgb, cv2.COLOR_RGB2BGR)

        results = {}

        # 1. DA模型 + 直接推理（黑底原图）
        pred1 = predict(model_da, orig_rgb, device)
        stats1 = get_stats(pred1)
        print(f"  DA Direct:      Wall:{stats1['wall']:.1f}% Win:{stats1['window']:.1f}% Door:{stats1['door']:.1f}%")
        results['DA+Direct'] = {
            'overlay': make_overlay(orig_bgr.copy(), pred1),
            'stats': stats1,
        }

        # 2. 原版模型 + 预处理
        pred2 = predict(model_orig, preprocessed_rgb, device)
        stats2 = get_stats(pred2)
        print(f"  Orig+Preproc:   Wall:{stats2['wall']:.1f}% Win:{stats2['window']:.1f}% Door:{stats2['door']:.1f}%")
        results['Orig+Preproc'] = {
            'overlay': make_overlay(preprocessed_bgr.copy(), pred2),
            'stats': stats2,
        }

        # 3. DA模型 + 预处理
        pred3 = predict(model_da, preprocessed_rgb, device)
        stats3 = get_stats(pred3)
        print(f"  DA+Preproc:     Wall:{stats3['wall']:.1f}% Win:{stats3['window']:.1f}% Door:{stats3['door']:.1f}%")
        results['DA+Preproc'] = {
            'overlay': make_overlay(preprocessed_bgr.copy(), pred3),
            'stats': stats3,
        }

        save_path = RESULT_DIR / f"cad_compare_{img_path.stem}.png"
        create_comparison(orig_bgr, results, save_path, img_path.stem)
        print()

    print("All done!")


if __name__ == "__main__":
    main()
