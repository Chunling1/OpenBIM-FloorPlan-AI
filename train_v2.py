"""
建筑平面图语义分割 V2 - 高精度版本
ResNet34 pretrained encoder + U-Net decoder (via smp)
对比 baseline LightUNet (train.py, mIoU=0.424)

改进点:
  1. ImageNet 预训练 ResNet34 backbone
  2. 512px 输入分辨率
  3. AMP 混合精度训练
  4. albumentations 高级数据增强
  5. OneCycleLR + 梯度累积
  6. Focal Loss + Dice Loss
"""

import os
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"
import sys
import json
import time
import random
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import segmentation_models_pytorch as smp
import albumentations as A

warnings.filterwarnings('ignore')


# ============================
# 配置
# ============================

class Config:
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data" / "cubicasa5k"
    OUTPUT_DIR = BASE_DIR / "output_v2"
    MODEL_DIR = BASE_DIR / "models"

    NUM_CLASSES = 4
    IMG_SIZE = 512

    # 训练
    BATCH_SIZE = 2
    GRAD_ACCUM = 4           # 梯度累积 -> 等效 batch=8
    NUM_EPOCHS = 100
    LR = 1e-3                # decoder LR
    ENCODER_LR = 1e-4        # encoder LR (预训练部分用更小的LR)
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 0

    # 类别权重
    CLASS_WEIGHTS = [0.5, 2.0, 3.0, 3.0]  # 和 baseline 一致

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    CLASS_NAMES = ["background", "wall", "window", "door"]
    CLASS_COLORS = [
        (0, 0, 0),
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 0),
    ]


# ============================
# 数据增强 (albumentations)
# ============================

def get_train_transforms(img_size):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.5, 1.0), ratio=(0.75, 1.33), p=0.5),
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=30, sigma=5, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.1, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=1.0),
            A.CLAHE(clip_limit=4.0, p=1.0),
        ], p=0.4),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                        min_holes=1, min_height=8, min_width=8,
                        fill_value=0, mask_fill_value=0, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============================
# 数据集 (复用 SVG 解析逻辑)
# ============================

class FloorplanDatasetV2(Dataset):
    CATEGORY_MAP = {
        'Wall': 1,
        'Railing': 1,
        'Window': 2,
        'Door': 3,
    }

    def __init__(self, root_dir, split='train', img_size=512, transform=None):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.transform = transform

        split_file = self.root_dir / f"{split}.txt"
        if split_file.exists():
            self.samples = [
                line.strip() for line in split_file.read_text().strip().split('\n')
                if line.strip()
            ]
        else:
            raise FileNotFoundError(f"Split file not found: {split_file}")

        print(f"[{split}] {len(self.samples)} samples loaded")

    def __len__(self):
        return len(self.samples)

    def _parse_svg_mask(self, svg_path, target_h, target_w):
        """SVG -> mask, 直接生成目标尺寸"""
        mask = np.zeros((target_h, target_w), dtype=np.uint8)

        if not os.path.exists(svg_path):
            return mask

        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            ns = {'svg': 'http://www.w3.org/2000/svg'}

            viewbox = root.get('viewBox', '')
            if viewbox:
                parts = viewbox.split()
                svg_w = float(parts[2]) if len(parts) >= 3 else target_w
                svg_h = float(parts[3]) if len(parts) >= 4 else target_h
            else:
                svg_w = float(root.get('width', target_w))
                svg_h = float(root.get('height', target_h))

            sx = target_w / svg_w if svg_w > 0 else 1.0
            sy = target_h / svg_h if svg_h > 0 else 1.0

            for group in root.findall('.//svg:g', ns):
                cls_name = group.get('class', '') or group.get('id', '')

                mapped = 0
                for key, cid in self.CATEGORY_MAP.items():
                    if key.lower() in cls_name.lower():
                        mapped = cid
                        break
                if mapped == 0:
                    continue

                for polygon in group.findall('.//svg:polygon', ns):
                    pts_str = polygon.get('points', '')
                    if not pts_str:
                        continue
                    try:
                        raw = pts_str.strip().replace(',', ' ').split()
                        pts = []
                        for i in range(0, len(raw) - 1, 2):
                            pts.append([int(float(raw[i]) * sx), int(float(raw[i+1]) * sy)])
                        if len(pts) >= 3:
                            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], mapped)
                    except (ValueError, IndexError):
                        continue

                for rect in group.findall('.//svg:rect', ns):
                    try:
                        x = float(rect.get('x', 0)) * sx
                        y = float(rect.get('y', 0)) * sy
                        w = float(rect.get('width', 0)) * sx
                        h = float(rect.get('height', 0)) * sy
                        if w > 0 and h > 0:
                            pts = np.array([
                                [int(x), int(y)], [int(x+w), int(y)],
                                [int(x+w), int(y+h)], [int(x), int(y+h)]
                            ], dtype=np.int32)
                            cv2.fillPoly(mask, [pts], mapped)
                    except (ValueError, TypeError):
                        continue

                for line in group.findall('.//svg:line', ns):
                    try:
                        x1 = float(line.get('x1', 0)) * sx
                        y1 = float(line.get('y1', 0)) * sy
                        x2 = float(line.get('x2', 0)) * sx
                        y2 = float(line.get('y2', 0)) * sy
                        thickness = max(2, int(3 * min(sx, sy)))
                        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), mapped, thickness)
                    except (ValueError, TypeError):
                        continue

        except ET.ParseError:
            pass

        return mask

    def __getitem__(self, idx):
        rel = self.samples[idx].strip().strip('/')
        sample_dir = self.root_dir / rel

        try:
            # 加载图片
            img_path = sample_dir / "F1_scaled.png"
            if not img_path.exists():
                for alt_name in ["F1_original.png", "image.png"]:
                    alt = sample_dir / alt_name
                    if alt.exists():
                        img_path = alt
                        break

            pil_img = Image.open(str(img_path)).convert('RGB')
            img = np.array(pil_img)
            pil_img.close()

            # 生成 mask (先按原图尺寸解析 SVG, 精度更高)
            svg_path = sample_dir / "model.svg"
            mask = self._parse_svg_mask(str(svg_path), img.shape[0], img.shape[1])

            # albumentations 增强 (同时处理 img 和 mask)
            if self.transform:
                transformed = self.transform(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask']
            else:
                img = cv2.resize(img, (self.img_size, self.img_size))
                mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                img = img.astype(np.float32) / 255.0
                img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

            # to tensor
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img.transpose(2, 0, 1)).float()
            else:
                img = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float()

            mask = np.clip(mask, 0, Config.NUM_CLASSES - 1)
            mask = torch.from_numpy(mask.astype(np.int64)).long()

            return img, mask

        except Exception:
            img = torch.zeros(3, self.img_size, self.img_size)
            mask = torch.zeros(self.img_size, self.img_size, dtype=torch.long)
            return img, mask


# ============================
# 损失函数
# ============================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # class weights tensor

    def forward(self, pred, target):
        w = self.alpha.to(pred.device) if self.alpha is not None else None
        ce = nn.functional.cross_entropy(pred, target, weight=w, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):
        pred_soft = torch.softmax(pred, dim=1)
        target_oh = nn.functional.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        inter = (pred_soft * target_oh).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLossV2(nn.Module):
    def __init__(self, num_classes, class_weights=None, focal_gamma=0.5):
        super().__init__()
        w = torch.FloatTensor(class_weights) if class_weights else None
        self.focal = FocalLoss(alpha=w, gamma=focal_gamma)
        self.dice = DiceLoss(num_classes)

    def forward(self, pred, target):
        return self.focal(pred, target) + self.dice(pred, target)


# ============================
# 评估
# ============================

def compute_metrics(pred, target, num_classes):
    pred_cls = pred.argmax(dim=1)
    ious = []
    for c in range(num_classes):
        pc = (pred_cls == c)
        tc = (target == c)
        inter = (pc & tc).sum().float()
        union = (pc | tc).sum().float()
        ious.append((inter / union).item() if union > 0 else float('nan'))
    valid = [x for x in ious if not np.isnan(x)]
    miou = np.mean(valid) if valid else 0.0
    return ious, miou


def compute_pixel_accuracy(pred, target):
    pred_cls = pred.argmax(dim=1)
    correct = (pred_cls == target).sum().float()
    total = target.numel()
    return (correct / total).item()


# ============================
# 训练器
# ============================

class TrainerV2:
    def __init__(self, config=Config):
        self.cfg = config
        self.device = torch.device(config.DEVICE)

        print("=" * 60)
        print("V2 Model: ResNet34 + U-Net (pretrained ImageNet)")
        print("=" * 60)
        print(f"Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.MODEL_DIR, exist_ok=True)

        # 模型: smp U-Net + ResNet34
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=config.NUM_CLASSES,
        ).to(self.device)

        params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Params: {params:,} ({params/1e6:.1f}M), Trainable: {trainable:,}")

        # 损失
        self.criterion = CombinedLossV2(
            config.NUM_CLASSES,
            class_weights=config.CLASS_WEIGHTS,
            focal_gamma=0.5,
        ).to(self.device)

        # 优化器 - 差分学习率: encoder 用小 LR, decoder 用大 LR
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = [p for p in self.model.parameters() if id(p) not in {id(ep) for ep in encoder_params}]
        self.optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': config.ENCODER_LR},
            {'params': decoder_params, 'lr': config.LR},
        ], weight_decay=config.WEIGHT_DECAY)

        # AMP
        self.scaler = GradScaler('cuda')
        self.use_amp = (self.device.type == 'cuda')

        # 记录
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_miou': [], 'val_miou': [],
            'val_class_iou': [], 'val_pixel_acc': [],
            'best_miou': 0.0, 'best_epoch': 0,
            'model_name': 'UNet-ResNet34-ImageNet',
        }

    def train(self):
        cfg = self.cfg

        # 数据
        train_ds = FloorplanDatasetV2(
            cfg.DATA_DIR, 'train', cfg.IMG_SIZE,
            transform=get_train_transforms(cfg.IMG_SIZE))
        val_ds = FloorplanDatasetV2(
            cfg.DATA_DIR, 'val', cfg.IMG_SIZE,
            transform=get_val_transforms(cfg.IMG_SIZE))

        train_loader = DataLoader(
            train_ds, batch_size=cfg.BATCH_SIZE,
            shuffle=True, num_workers=cfg.NUM_WORKERS,
            pin_memory=True, drop_last=True)
        val_loader = DataLoader(
            val_ds, batch_size=cfg.BATCH_SIZE,
            shuffle=False, num_workers=cfg.NUM_WORKERS,
            pin_memory=True)

        print(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
        print(f"Val:   {len(val_ds)} samples, {len(val_loader)} batches")
        print(f"Effective batch size: {cfg.BATCH_SIZE * cfg.GRAD_ACCUM}")
        print(f"AMP: {self.use_amp}")

        # 调度器 - CosineAnnealing (和 baseline 一致, 已验证有效)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.NUM_EPOCHS,
        )

        start = time.time()
        no_improve = 0

        for epoch in range(1, cfg.NUM_EPOCHS + 1):
            t_loss, t_miou = self._train_epoch(train_loader, epoch)
            v_loss, v_miou, v_ious, v_acc = self._validate(val_loader)
            self.scheduler.step()

            self.history['train_loss'].append(t_loss)
            self.history['val_loss'].append(v_loss)
            self.history['train_miou'].append(t_miou)
            self.history['val_miou'].append(v_miou)
            self.history['val_class_iou'].append(v_ious)
            self.history['val_pixel_acc'].append(v_acc)

            lr = self.optimizer.param_groups[0]['lr']

            if v_miou > self.history['best_miou']:
                self.history['best_miou'] = v_miou
                self.history['best_epoch'] = epoch
                self._save('v2_best_model.pt')
                marker = " * BEST"
                no_improve = 0
            else:
                marker = ""
                no_improve += 1

            if epoch % 10 == 0:
                self._save(f'v2_checkpoint_e{epoch}.pt')

            elapsed = time.time() - start
            eta = elapsed / epoch * (cfg.NUM_EPOCHS - epoch)

            iou_str = " | ".join([
                f"{cfg.CLASS_NAMES[i]}={v_ious[i]:.3f}"
                for i in range(cfg.NUM_CLASSES)
                if not np.isnan(v_ious[i])
            ])

            print(
                f"E{epoch:3d}/{cfg.NUM_EPOCHS} | "
                f"L:{t_loss:.4f}/{v_loss:.4f} | "
                f"mIoU:{t_miou:.3f}/{v_miou:.3f} | "
                f"Acc:{v_acc:.3f} | "
                f"LR:{lr:.6f} | "
                f"ETA:{eta/60:.0f}m{marker}",
                flush=True
            )
            if epoch % 5 == 0:
                print(f"  IoU: {iou_str}", flush=True)

            # Early stopping (patience=20)
            if no_improve >= 20:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for 20 epochs)")
                break

        self._save('v2_final_model.pt')
        self._save_history()

        total = time.time() - start
        print(f"\n[DONE] V2 Training complete! {total/60:.1f} min")
        print(f"  Best mIoU: {self.history['best_miou']:.4f} (Epoch {self.history['best_epoch']})")
        print(f"  Models saved to: {cfg.MODEL_DIR}")

    def _train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0
        total_miou = 0
        n = 0
        self.optimizer.zero_grad()

        for i, (imgs, masks) in enumerate(loader):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)

            if self.use_amp:
                with autocast('cuda'):
                    out = self.model(imgs)
                    loss = self.criterion(out, masks) / self.cfg.GRAD_ACCUM
                self.scaler.scale(loss).backward()
            else:
                out = self.model(imgs)
                loss = self.criterion(out, masks) / self.cfg.GRAD_ACCUM
                loss.backward()

            if (i + 1) % self.cfg.GRAD_ACCUM == 0 or (i + 1) == len(loader):
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()
                pass  # scheduler stepped per epoch below

            with torch.no_grad():
                _, miou = compute_metrics(out, masks, self.cfg.NUM_CLASSES)

            total_loss += loss.item() * self.cfg.GRAD_ACCUM
            total_miou += miou
            n += 1

        return total_loss / max(n, 1), total_miou / max(n, 1)

    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()
        total_loss = 0
        all_ious = []
        total_acc = 0
        n = 0

        for imgs, masks in loader:
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)

            if self.use_amp:
                with autocast('cuda'):
                    out = self.model(imgs)
                    loss = self.criterion(out, masks)
            else:
                out = self.model(imgs)
                loss = self.criterion(out, masks)

            ious, _ = compute_metrics(out, masks, self.cfg.NUM_CLASSES)
            acc = compute_pixel_accuracy(out, masks)

            total_loss += loss.item()
            all_ious.append(ious)
            total_acc += acc
            n += 1

        avg_ious = np.nanmean(all_ious, axis=0).tolist() if all_ious else [0.0] * self.cfg.NUM_CLASSES
        valid = [x for x in avg_ious if not np.isnan(x)]
        miou = np.mean(valid) if valid else 0.0

        return total_loss / max(n, 1), miou, avg_ious, total_acc / max(n, 1)

    def _save(self, filename):
        path = self.cfg.MODEL_DIR / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'num_classes': self.cfg.NUM_CLASSES,
                'img_size': self.cfg.IMG_SIZE,
                'class_names': self.cfg.CLASS_NAMES,
                'encoder': 'resnet34',
                'architecture': 'Unet',
            },
            'best_miou': self.history['best_miou'],
        }, str(path))

    def _save_history(self):
        path = self.cfg.OUTPUT_DIR / "training_history_v2.json"
        with open(str(path), 'w') as f:
            json.dump(self.history, f, indent=2)


# ============================
# V2 推理器
# ============================

class FloorplanPredictorV2:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        cfg = ckpt['config']

        self.model = smp.Unet(
            encoder_name=cfg.get('encoder', 'resnet34'),
            encoder_weights=None,
            in_channels=3,
            classes=cfg['num_classes'],
        )
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.img_size = cfg['img_size']
        self.class_names = cfg['class_names']

        print(f"[OK] V2 model loaded (mIoU: {ckpt.get('best_miou', '?'):.4f})")

    @torch.no_grad()
    def predict(self, image_path):
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        resized = cv2.resize(img, (self.img_size, self.img_size))
        norm = resized.astype(np.float32) / 255.0
        norm = (norm - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        out = self.model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        mask = probs.argmax(dim=0).cpu().numpy()
        mask = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return mask, probs.cpu().numpy()

    def visualize(self, image_path, output_path=None):
        mask, _ = self.predict(image_path)
        img = cv2.imread(str(image_path))
        overlay = np.zeros_like(img)
        for cid in range(1, Config.NUM_CLASSES):
            overlay[mask == cid] = Config.CLASS_COLORS[cid]
        result = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        if output_path:
            cv2.imwrite(str(output_path), result)
        return result


# ============================
# 入口
# ============================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='V2 - ResNet34 U-Net')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    parser.add_argument('--image', type=str)
    parser.add_argument('--model', type=str, default='models/v2_best_model.pt')
    args = parser.parse_args()

    if args.mode == 'train':
        trainer = TrainerV2()
        trainer.train()

    elif args.mode == 'predict':
        if not args.image:
            print("Usage: --image <path>")
            sys.exit(1)
        pred = FloorplanPredictorV2(args.model)
        mask, _ = pred.predict(args.image)
        for cid, name in [(1, 'wall'), (2, 'window'), (3, 'door')]:
            cnt = len(cv2.findContours((mask == cid).astype(np.uint8) * 255,
                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
            print(f"  {name}: {cnt} regions")
        out = Path(args.image).stem + "_v2_result.png"
        pred.visualize(args.image, out)
        print(f"  Saved: {out}")
