"""
路线2：域适应微调（Domain Adaptation Fine-tuning）

核心思路：不需要手工标注！
用 CubiCasa5K 现有标签数据，加入域随机化增强，
让模型学会同时处理白底和黑底平面图。

新增增强：
1. 随机颜色反转（白底→黑底）
2. 随机背景色
3. 随机前景色变换（模拟不同CAD颜色方案）
4. 随机添加标注线噪声

起点：M2_UNet_ResNet34_DA_best.pt
训练：30 epochs（~3.5h）
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
import segmentation_models_pytorch as smp
import albumentations as A

warnings.filterwarnings('ignore')


# ============================
# 配置
# ============================

class Config:
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data" / "cubicasa5k"
    OUTPUT_DIR = BASE_DIR / "output_finetune"
    MODEL_DIR = BASE_DIR / "models"

    # 起始模型
    PRETRAINED_MODEL = MODEL_DIR / "M2_UNet_ResNet34_DA_best.pt"

    NUM_CLASSES = 4
    IMG_SIZE = 512

    # 微调参数（保守）
    BATCH_SIZE = 2
    GRAD_ACCUM = 4           # 等效 batch=8
    NUM_EPOCHS = 30
    ENCODER_LR = 5e-6        # encoder 非常小的LR
    DECODER_LR = 5e-5        # decoder 稍大
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 0

    # 类别权重 - 加大窗户和门的权重
    CLASS_WEIGHTS = [0.3, 2.0, 4.0, 4.0]

    # 域随机化概率
    DOMAIN_AUG_PROB = 0.5     # 50%概率应用域变换

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CLASS_NAMES = ["background", "wall", "window", "door"]


# ============================
# 域随机化增强（核心创新）
# ============================

class DomainRandomization:
    """
    将白底平面图随机变换为黑底风格，模拟中式CAD图的外观。
    关键变换：
    1. 颜色反转 → 白底变黑底
    2. 前景线条颜色随机化 → 模拟CAD不同颜色图层
    3. 添加随机标注线 → 模拟尺寸标注
    4. 背景色随机化
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask, **kwargs):
        if random.random() > self.prob:
            return image, mask

        h, w = image.shape[:2]
        result = image.copy()

        # 检测前景/背景
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

        # 平面图通常白底黑线，背景是亮色像素
        bg_thresh = 200
        fg_mask = gray < bg_thresh

        transform_type = random.choice(['invert', 'dark_bg', 'color_lines', 'full_cad'])

        if transform_type == 'invert':
            # 简单反转
            result = 255 - result

        elif transform_type == 'dark_bg':
            # 黑色背景 + 保留前景
            bg_color = random.choice([
                (0, 0, 0),        # 纯黑
                (10, 10, 15),     # 深蓝黑
                (15, 10, 10),     # 深红黑
                (10, 15, 10),     # 深绿黑
            ])
            result[~fg_mask] = bg_color
            # 前景变亮
            result[fg_mask] = np.clip(result[fg_mask].astype(np.int16) + 80, 0, 255).astype(np.uint8)

        elif transform_type == 'color_lines':
            # 将黑色线条变成随机彩色（模拟CAD图层颜色）
            line_colors = [
                (255, 255, 255),  # 白
                (200, 200, 200),  # 灰
                (0, 255, 0),      # 绿
                (0, 255, 255),    # 青
                (255, 255, 0),    # 黄
                (0, 200, 200),    # 青绿
            ]
            # 黑背景
            bg = np.zeros_like(result)
            # 线条用随机颜色
            color = random.choice(line_colors)
            bg[fg_mask] = color
            # 添加一些颜色变化
            noise = np.random.randint(-20, 20, bg.shape, dtype=np.int16)
            bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            result = bg

        elif transform_type == 'full_cad':
            # 完整模拟CAD风格：
            # 黑背景 + 白色墙线 + 随机颜色标注
            bg = np.zeros_like(result)

            # 墙线用白/灰色
            wall_mask_pixels = (mask == 1)
            if wall_mask_pixels.any():
                wall_color = random.choice([(255,255,255), (200,200,200), (180,180,180)])
                bg[wall_mask_pixels] = wall_color

            # 窗户用青色（模拟CAD窗户图层）
            win_mask_pixels = (mask == 2)
            if win_mask_pixels.any():
                win_color = random.choice([(0,255,255), (100,200,255), (0,200,200)])
                bg[win_mask_pixels] = win_color

            # 门用绿色或黄色
            door_mask_pixels = (mask == 3)
            if door_mask_pixels.any():
                door_color = random.choice([(0,255,0), (255,255,0), (0,200,100)])
                bg[door_mask_pixels] = door_color

            # 其他前景像素（文字、家具等）用灰色
            other_fg = fg_mask & ~wall_mask_pixels & ~win_mask_pixels & ~door_mask_pixels
            if other_fg.any():
                bg[other_fg] = random.choice([(150,150,150), (100,100,100), (0,180,0)])

            result = bg

        # 可选：添加随机标注线噪声（不影响mask）
        if random.random() < 0.3:
            result = self._add_annotation_noise(result, h, w)

        return result, mask

    def _add_annotation_noise(self, img, h, w):
        """添加模拟标注线（尺寸线、引线等）"""
        result = img.copy()
        n_lines = random.randint(2, 8)

        annot_colors = [
            (0, 255, 0),      # 绿
            (255, 0, 255),    # 品红
            (255, 255, 0),    # 黄
            (0, 200, 200),    # 青
        ]

        for _ in range(n_lines):
            color = random.choice(annot_colors)
            thickness = random.randint(1, 2)

            if random.random() < 0.5:
                # 水平标注线
                y = random.randint(0, h-1)
                x1 = random.randint(0, w//3)
                x2 = random.randint(2*w//3, w-1)
                cv2.line(result, (x1, y), (x2, y), color, thickness)
                # 端部小竖线
                cv2.line(result, (x1, y-5), (x1, y+5), color, thickness)
                cv2.line(result, (x2, y-5), (x2, y+5), color, thickness)
            else:
                # 垂直标注线
                x = random.randint(0, w-1)
                y1 = random.randint(0, h//3)
                y2 = random.randint(2*h//3, h-1)
                cv2.line(result, (x, y1), (x, y2), color, thickness)
                cv2.line(result, (x-5, y1), (x+5, y1), color, thickness)
                cv2.line(result, (x-5, y2), (x+5, y2), color, thickness)

        return result


# ============================
# 数据集
# ============================

class FloorplanDatasetFT(Dataset):
    """微调数据集：CubiCasa5K + 预计算mask + 域随机化"""

    CATEGORY_MAP = {
        'Wall': 1, 'Railing': 1,
        'Window': 2, 'Door': 3,
    }

    def __init__(self, root_dir, split='train', img_size=512,
                 transform=None, domain_aug=None):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.transform = transform
        self.domain_aug = domain_aug
        self.mask_cache_dir = self.root_dir / "masks_cache"

        split_file = self.root_dir / f"{split}.txt"
        self.samples = [
            line.strip() for line in split_file.read_text().strip().split('\n')
            if line.strip()
        ]
        print(f"[{split}] {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def _get_mask(self, sample_dir, h, w):
        """尝试加载预计算mask，否则从SVG解析"""
        # 预计算mask路径
        rel = sample_dir.relative_to(self.root_dir)
        cache_path = self.mask_cache_dir / f"{str(rel).replace(os.sep, '_')}.npy"

        if cache_path.exists():
            mask = np.load(str(cache_path))
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            return mask

        # 从SVG解析
        svg_path = sample_dir / "model.svg"
        return self._parse_svg_mask(str(svg_path), h, w)

    def _parse_svg_mask(self, svg_path, target_h, target_w):
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
            img_path = sample_dir / "F1_scaled.png"
            if not img_path.exists():
                for alt in ["F1_original.png", "image.png"]:
                    p = sample_dir / alt
                    if p.exists():
                        img_path = p
                        break

            pil_img = Image.open(str(img_path)).convert('RGB')
            img = np.array(pil_img)
            pil_img.close()

            mask = self._get_mask(sample_dir, img.shape[0], img.shape[1])

            # 域随机化（在标准增强之前）
            if self.domain_aug is not None:
                img, mask = self.domain_aug(img, mask)

            # 标准albumentations增强
            if self.transform:
                transformed = self.transform(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask']
            else:
                img = cv2.resize(img, (self.img_size, self.img_size))
                mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                img = img.astype(np.float32) / 255.0
                img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img.transpose(2, 0, 1)).float()
            else:
                img = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float()

            mask = np.clip(mask, 0, Config.NUM_CLASSES - 1)
            mask = torch.from_numpy(mask.astype(np.int64)).long()

            return img, mask

        except Exception as e:
            img = torch.zeros(3, self.img_size, self.img_size)
            mask = torch.zeros(self.img_size, self.img_size, dtype=torch.long)
            return img, mask


# ============================
# 损失函数
# ============================

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.weight = torch.FloatTensor(class_weights) if class_weights else None

    def forward(self, pred, target):
        w = self.weight.to(pred.device) if self.weight is not None else None
        ce = nn.functional.cross_entropy(pred, target, weight=w)

        # Dice
        pred_soft = torch.softmax(pred, dim=1)
        target_oh = nn.functional.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        inter = (pred_soft * target_oh).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
        dice = (2 * inter + 1e-5) / (union + 1e-5)
        dice_loss = 1.0 - dice.mean()

        return ce + dice_loss


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


# ============================
# 增强配置
# ============================

def get_finetune_transforms(img_size):
    """微调用增强：保留原有增强 + 更强的颜色变换"""
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.5, 1.0), ratio=(0.75, 1.33), p=0.5),
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=30, sigma=5, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
        ], p=0.2),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=40, p=1.0),
            A.CLAHE(clip_limit=4.0, p=1.0),
        ], p=0.5),
        A.RandomGamma(gamma_limit=(60, 140), p=0.3),
        A.ChannelShuffle(p=0.1),
        A.CoarseDropout(max_holes=6, max_height=28, max_width=28,
                        min_holes=1, min_height=8, min_width=8,
                        fill_value=0, mask_fill_value=0, p=0.15),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============================
# 微调训练器
# ============================

class FineTuner:
    def __init__(self, config=Config):
        self.cfg = config
        self.device = torch.device(config.DEVICE)

        print("=" * 60)
        print("Domain Adaptation Fine-tuning")
        print("=" * 60)
        print(f"Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"VRAM: {vram:.1f} GB")

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.MODEL_DIR, exist_ok=True)

        # 加载预训练模型
        print(f"\nLoading pretrained model: {config.PRETRAINED_MODEL.name}")
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=config.NUM_CLASSES,
        )
        ckpt = torch.load(str(config.PRETRAINED_MODEL), map_location='cpu', weights_only=False)
        if 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt)
        self.model.to(self.device)
        print(f"Loaded! Previous mIoU: {ckpt.get('best_miou', 'N/A')}")

        params = sum(p.numel() for p in self.model.parameters())
        print(f"Params: {params:,} ({params/1e6:.1f}M)")

        # 损失
        self.criterion = CombinedLoss(
            config.NUM_CLASSES,
            class_weights=config.CLASS_WEIGHTS,
        ).to(self.device)

        # 优化器 - 差分学习率
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = [p for p in self.model.parameters()
                         if id(p) not in {id(ep) for ep in encoder_params}]
        self.optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': config.ENCODER_LR},
            {'params': decoder_params, 'lr': config.DECODER_LR},
        ], weight_decay=config.WEIGHT_DECAY)

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_miou': [], 'val_miou': [],
            'val_class_iou': [],
            'best_miou': 0.0, 'best_epoch': 0,
        }

    def train(self):
        cfg = self.cfg

        # 域随机化
        domain_aug = DomainRandomization(prob=cfg.DOMAIN_AUG_PROB)

        train_ds = FloorplanDatasetFT(
            cfg.DATA_DIR, 'train', cfg.IMG_SIZE,
            transform=get_finetune_transforms(cfg.IMG_SIZE),
            domain_aug=domain_aug,
        )
        val_ds = FloorplanDatasetFT(
            cfg.DATA_DIR, 'val', cfg.IMG_SIZE,
            transform=get_val_transforms(cfg.IMG_SIZE),
            domain_aug=None,  # 验证集不做域变换
        )

        train_loader = DataLoader(
            train_ds, batch_size=cfg.BATCH_SIZE,
            shuffle=True, num_workers=cfg.NUM_WORKERS,
            pin_memory=True, drop_last=True)
        val_loader = DataLoader(
            val_ds, batch_size=cfg.BATCH_SIZE,
            shuffle=False, num_workers=cfg.NUM_WORKERS,
            pin_memory=True)

        print(f"\nTrain: {len(train_ds)} samples, {len(train_loader)} batches")
        print(f"Val:   {len(val_ds)} samples")
        print(f"Effective batch size: {cfg.BATCH_SIZE * cfg.GRAD_ACCUM}")
        print(f"Domain augmentation prob: {cfg.DOMAIN_AUG_PROB}")
        print(f"Encoder LR: {cfg.ENCODER_LR}, Decoder LR: {cfg.DECODER_LR}")

        # Cosine调度
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.NUM_EPOCHS)

        # Warmup: 前3个epoch线性warmup
        warmup_epochs = 3

        start = time.time()
        no_improve = 0

        for epoch in range(1, cfg.NUM_EPOCHS + 1):
            # warmup
            if epoch <= warmup_epochs:
                warmup_factor = epoch / warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg['lr'] = pg['lr'] * warmup_factor / (warmup_factor if epoch > 1 else 1)

            t_loss, t_miou = self._train_epoch(train_loader, epoch)
            v_loss, v_miou, v_ious = self._validate(val_loader)
            scheduler.step()

            self.history['train_loss'].append(t_loss)
            self.history['val_loss'].append(v_loss)
            self.history['train_miou'].append(t_miou)
            self.history['val_miou'].append(v_miou)
            self.history['val_class_iou'].append(v_ious)

            marker = ""
            if v_miou > self.history['best_miou']:
                self.history['best_miou'] = v_miou
                self.history['best_epoch'] = epoch
                self._save('M2_DomainAdapt_best.pt')
                marker = " ★ BEST"
                no_improve = 0
            else:
                no_improve += 1

            elapsed = time.time() - start
            eta = elapsed / epoch * (cfg.NUM_EPOCHS - epoch)

            iou_str = " ".join([
                f"{cfg.CLASS_NAMES[i][:3]}={v_ious[i]:.3f}"
                for i in range(cfg.NUM_CLASSES)
                if not np.isnan(v_ious[i])
            ])
            enc_lr = self.optimizer.param_groups[0]['lr']

            print(
                f"E{epoch:2d}/{cfg.NUM_EPOCHS} | "
                f"L:{t_loss:.4f}/{v_loss:.4f} | "
                f"mIoU:{t_miou:.3f}/{v_miou:.3f} | "
                f"{iou_str} | "
                f"lr:{enc_lr:.1e} | "
                f"ETA:{eta/60:.0f}m{marker}",
                flush=True
            )

            if no_improve >= 15:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        self._save('M2_DomainAdapt_final.pt')
        self._save_history()

        total = time.time() - start
        print(f"\n{'='*60}")
        print(f"Fine-tuning complete! {total/60:.1f} min")
        print(f"Best mIoU: {self.history['best_miou']:.4f} (Epoch {self.history['best_epoch']})")
        print(f"Model saved: M2_DomainAdapt_best.pt")

    def _train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0
        total_miou = 0
        n = 0
        self.optimizer.zero_grad()

        for i, (imgs, masks) in enumerate(loader):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)

            out = self.model(imgs)
            loss = self.criterion(out, masks) / self.cfg.GRAD_ACCUM
            loss.backward()

            if (i + 1) % self.cfg.GRAD_ACCUM == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()

            with torch.no_grad():
                _, miou = compute_metrics(out, masks, self.cfg.NUM_CLASSES)

            total_loss += loss.item() * self.cfg.GRAD_ACCUM
            total_miou += miou
            n += 1

            # 进度打印
            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(loader)}] loss:{total_loss/n:.4f} miou:{total_miou/n:.3f}", flush=True)

        return total_loss / max(n, 1), total_miou / max(n, 1)

    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()
        total_loss = 0
        all_ious = []
        n = 0

        for imgs, masks in loader:
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)
            out = self.model(imgs)
            loss = self.criterion(out, masks)
            ious, _ = compute_metrics(out, masks, self.cfg.NUM_CLASSES)
            total_loss += loss.item()
            all_ious.append(ious)
            n += 1

        avg_ious = np.nanmean(all_ious, axis=0).tolist() if all_ious else [0.0] * self.cfg.NUM_CLASSES
        valid = [x for x in avg_ious if not np.isnan(x)]
        miou = np.mean(valid) if valid else 0.0
        return total_loss / max(n, 1), miou, avg_ious

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
        path = self.cfg.OUTPUT_DIR / "finetune_history.json"
        with open(str(path), 'w') as f:
            json.dump(self.history, f, indent=2)


if __name__ == "__main__":
    tuner = FineTuner()
    tuner.train()
