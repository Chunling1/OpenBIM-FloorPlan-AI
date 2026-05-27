"""
在CubiCasa5K官方测试集上评估所有模型
输出最终论文级结果
"""
import os
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import torch
import segmentation_models_pytorch as smp
import albumentations as A
import json
import time
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "cubicasa5k"
OUTPUT_DIR = BASE_DIR / "output_paper"
IMG_SIZE = 512
NUM_CLASSES = 4
CLASS_NAMES = ["Background", "Wall", "Window", "Door"]


def load_model(path):
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=NUM_CLASSES)
    ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def predict(model, img_rgb, device):
    aug = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    t = aug(image=img_rgb)
    tensor = torch.from_numpy(t['image'].transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).argmax(dim=1).squeeze().cpu().numpy()
    return pred


def compute_iou(pred, gt, num_classes):
    ious = []
    for c in range(num_classes):
        pc = (pred == c)
        gc = (gt == c)
        inter = (pc & gc).sum()
        union = (pc | gc).sum()
        ious.append(float(inter / union) if union > 0 else float('nan'))
    valid = [x for x in ious if not np.isnan(x)]
    miou = float(np.mean(valid)) if valid else 0.0
    return ious, miou


def eval_on_split(model, split, device):
    split_file = DATA_DIR / f"{split}.txt"
    samples = [l.strip().strip('/') for l in split_file.read_text().strip().split('\n') if l.strip()]
    valid = [s for s in samples if (DATA_DIR / s / "mask.npy").exists()]
    
    all_ious = []
    all_mious = []
    
    for i, rel in enumerate(valid):
        sd = DATA_DIR / rel
        pil = Image.open(str(sd / "F1_scaled.png")).convert('RGB')
        img = np.array(pil)
        pil.close()
        
        gt = np.load(str(sd / "mask.npy"))
        if gt.shape[:2] != img.shape[:2]:
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt = np.clip(gt, 0, 3)
        
        pred = predict(model, img, device)
        pred_full = cv2.resize(pred.astype(np.uint8), (img.shape[1], img.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        ious, miou = compute_iou(pred_full, gt, NUM_CLASSES)
        all_ious.append(ious)
        all_mious.append(miou)
        
        if (i+1) % 100 == 0:
            print(f"    {i+1}/{len(valid)}")
    
    avg_ious = np.nanmean(all_ious, axis=0).tolist()
    avg_miou = float(np.mean(all_mious))
    return avg_miou, avg_ious, len(valid)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models_config = {
        'M2-Orig': BASE_DIR / "models" / "M2_UNet_ResNet34_best.pt",
        'M2-DA': BASE_DIR / "models" / "M2_UNet_ResNet34_DA_best.pt",
    }
    
    print("=" * 70)
    print("CubiCasa5K Official Test Set Evaluation")
    print("=" * 70)
    
    results = {}
    
    for name, path in models_config.items():
        if not path.exists():
            continue
        print(f"\n--- {name} ---")
        model = load_model(path).to(device)
        
        for split in ['val', 'test']:
            print(f"  [{split}]")
            t0 = time.time()
            miou, ious, n_samples = eval_on_split(model, split, device)
            elapsed = time.time() - t0
            
            key = f"{name}_{split}"
            results[key] = {
                'mIoU': miou,
                'class_iou': {CLASS_NAMES[i]: ious[i] for i in range(NUM_CLASSES)},
                'n_samples': n_samples,
                'time_sec': round(elapsed, 1),
            }
            print(f"    mIoU={miou:.4f} | BG={ious[0]:.4f} W={ious[1]:.4f} Wi={ious[2]:.4f} D={ious[3]:.4f} ({elapsed:.0f}s, {n_samples} samples)")
        
        del model
        torch.cuda.empty_cache()
    
    # Summary table
    print("\n\n" + "=" * 90)
    print("Final Results (Val + Test)")
    print("=" * 90)
    print(f"{'Config':<20} {'Split':<6} {'mIoU':>8} {'BG':>8} {'Wall':>8} {'Win':>8} {'Door':>8}")
    print("-" * 70)
    for key, r in results.items():
        parts = key.rsplit('_', 1)
        name, split = parts[0], parts[1]
        c = r['class_iou']
        print(f"{name:<20} {split:<6} {r['mIoU']:8.4f} {c['Background']:8.4f} {c['Wall']:8.4f} {c['Window']:8.4f} {c['Door']:8.4f}")
    
    # Save
    json_path = OUTPUT_DIR / "final_test_results.json"
    with open(str(json_path), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
