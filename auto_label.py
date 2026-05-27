"""
半自动标注工具：对中国CAD图生成初始分割mask
用户只需少量修正即可用于fine-tune

用法:
  python auto_label.py --input data/chinese_cad/images/ --output data/chinese_cad/masks/
  python auto_label.py --input path/to/single_image.jpg
"""
import os
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"
import argparse
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
IMG_SIZE = 512
NUM_CLASSES = 4
CLASS_NAMES = ["background", "wall", "window", "door"]
COLORS_BGR = [(0,0,0), (0,0,255), (255,0,0), (0,255,0)]


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
    if gray.mean() < 100:  # 黑底图
        inverted = 255 - gray
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(inverted)
        _, binary = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
        result = cv2.addWeighted(enhanced, 0.4, binary, 0.6, 0)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    else:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def predict(model, img_rgb, device):
    aug = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    processed = aug(image=img_rgb)
    tensor = torch.from_numpy(processed['image'].transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1).squeeze().cpu().numpy()  # (C, H, W)
        pred = output.argmax(dim=1).squeeze().cpu().numpy()
    return pred, prob


def generate_label(model, img_path, device):
    """双路预测 + 融合"""
    img_bgr = cv2.imread(str(img_path))
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 路径1: 原图直接推理
    pred1, prob1 = predict(model, img_rgb, device)

    # 路径2: 预处理后推理（黑底→白底）
    preprocessed = preprocess_dark_cad(img_bgr)
    pred2, prob2 = predict(model, preprocessed, device)

    # 融合: 取置信度更高的结果
    fused_prob = np.maximum(prob1, prob2)  # 逐像素取max概率
    fused_pred = fused_prob.argmax(axis=0)

    # resize回原尺寸
    fused_full = cv2.resize(fused_pred.astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST)
    return fused_full


def save_visualization(img_bgr, mask, save_path):
    """保存可视化用于人工检查"""
    h, w = img_bgr.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = img_bgr.copy()
    for cls_id, color in enumerate(COLORS_BGR):
        if cls_id == 0:
            continue
        region = mask_resized == cls_id
        overlay[region] = cv2.addWeighted(
            img_bgr[region].reshape(-1, 3), 0.5,
            np.full_like(img_bgr[region].reshape(-1, 3), color), 0.5, 0
        )

    # 拼接: 原图 | 叠加
    gap = np.ones((h, 4, 3), dtype=np.uint8) * 128
    vis = np.hstack([img_bgr, gap, overlay])
    cv2.imwrite(str(save_path), vis)


def save_labelme_json(img_path, mask, json_path):
    """转为labelme JSON格式（方便用labelme修正）"""
    import json
    import base64

    h, w = mask.shape
    shapes = []
    for cls_id in [1, 2, 3]:
        cls_mask = (mask == cls_id).astype(np.uint8)
        contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 20:
                continue
            # 简化轮廓
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 3:
                continue
            points = approx.squeeze().tolist()
            if len(points) >= 3:
                shapes.append({
                    "label": CLASS_NAMES[cls_id],
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                })

    # 读取图像数据
    with open(str(img_path), "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")

    labelme_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": str(Path(img_path).name),
        "imageData": img_data,
        "imageHeight": h,
        "imageWidth": w,
    }

    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(labelme_data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="半自动标注工具")
    parser.add_argument("--input", required=True, help="图片文件或目录路径")
    parser.add_argument("--output", default=None, help="mask输出目录（默认同目录下masks/）")
    parser.add_argument("--model", default=None, help="模型路径（默认自动选择最新DA模型）")
    parser.add_argument("--labelme", action="store_true", help="同时输出labelme JSON格式")
    args = parser.parse_args()

    # 自动选择模型
    if args.model:
        model_path = Path(args.model)
    else:
        # 优先用DA模型
        da_path = BASE_DIR / "models" / "M2_UNet_ResNet34_DA_best.pt"
        orig_path = BASE_DIR / "models" / "M2_UNet_ResNet34_best.pt"
        model_path = da_path if da_path.exists() else orig_path

    print(f"Using model: {model_path.name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path).to(device)

    # 收集图片
    input_path = Path(args.input)
    if input_path.is_file():
        images = [input_path]
        output_dir = input_path.parent / "masks" if args.output is None else Path(args.output)
    else:
        images = sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
                        + list(input_path.glob("*.jpeg")))
        output_dir = input_path.parent / "masks" if args.output is None else Path(args.output)

    os.makedirs(str(output_dir), exist_ok=True)
    vis_dir = output_dir.parent / "visualizations"
    os.makedirs(str(vis_dir), exist_ok=True)

    print(f"Found {len(images)} images")
    print(f"Output: {output_dir}\n")

    for img_path in images:
        print(f"Processing: {img_path.name}", end="")
        img_bgr = cv2.imread(str(img_path))
        mask = generate_label(model, img_path, device)

        # 保存mask.npy
        mask_path = output_dir / f"{img_path.stem}_mask.npy"
        np.save(str(mask_path), mask)

        # 保存可视化
        vis_path = vis_dir / f"{img_path.stem}_preview.png"
        save_visualization(img_bgr, mask, vis_path)

        # 保存labelme JSON 
        if args.labelme:
            json_path = output_dir / f"{img_path.stem}.json"
            save_labelme_json(img_path, mask, json_path)

        stats = {CLASS_NAMES[i]: f"{(mask==i).sum()/mask.size*100:.1f}%" for i in range(1, 4)}
        print(f"  → {stats}")

    print(f"\n完成! mask保存在: {output_dir}")
    print(f"可视化预览: {vis_dir}")
    if args.labelme:
        print(f"Labelme JSON已生成，可用 `labelme {output_dir}` 打开修正")


if __name__ == "__main__":
    main()
