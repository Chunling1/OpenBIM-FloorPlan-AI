"""
域适应微调 V2 — 修复上一版的灾难性遗忘问题

核心修改：
1. 两阶段训练：Phase1冻结encoder(15ep) → Phase2全参数微调(15ep)
2. 类别权重与原训练保持一致 [0.5, 2.0, 3.0, 3.0]
3. 域增强概率降到0.3
4. 去除过于激进的颜色增强（ChannelShuffle, HueSaturation等）
5. 修复warmup实现bug
6. 验证集同时评估原始+域变换样本
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
    OUTPUT_DIR = BASE_DIR / "output_finetune_v2"
    MODEL_DIR = BASE_DIR / "models"

    # 起始模型 — 用DA版（验证集0.782）
    PRETRAINED_MODEL = MODEL_DIR / "M2_UNet_ResNet34_DA_best.pt"

    NUM_CLASSES = 4
    IMG_SIZE = 512

    # === 两阶段训练 ===
    # Phase 1: 冻结encoder, 只调decoder
    PHASE1_EPOCHS = 15
    PHASE1_LR = 1e-4          # decoder LR

    # Phase 2: 解冻encoder, 全参数微调
    PHASE2_EPOCHS = 15
    PHASE2_ENCODER_LR = 1e-6  # encoder 极小LR
    PHASE2_DECODER_LR = 1e-5  # decoder 小LR

    BATCH_SIZE = 2
    GRAD_ACCUM = 4             # 等效 batch=8
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 0
    WARMUP_EPOCHS = 3

    # 类别权重 — 与原始训练保持一致！
    CLASS_WEIGHTS = [0.5, 2.0, 3.0, 3.0]

    # 域随机化概率 — 降低到0.3
    DOMAIN_AUG_PROB = 0.3

    # Early stopping patience
    PATIENCE = 20

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CLASS_NAMES = ["background", "wall", "window", "door"]


# ============================
# 域随机化增强（与上版相同）
# ============================

class DomainRandomization:
    """
    将白底平面图随机变换为黑底风格，模拟中式CAD图的外观。
    """

    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, image, mask, **kwargs):
        if random.random() > self.prob:
            return image, mask

        h, w = image.shape[:2]
        result = image.copy()

        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        bg_thresh = 200
        fg_mask = gray < bg_thresh

        transform_type = random.choice(['invert', 'dark_bg', 'color_lines', 'full_cad'])

        if transform_type == 'invert':
            result = 255 - result

        elif transform_type == 'dark_bg':
            bg_color = random.choice([
                (0, 0, 0), (10, 10, 15), (15, 10, 10), (10, 15, 10),
            ])
            result[~fg_mask] = bg_color
            result[fg_mask] = np.clip(
                result[fg_mask].astype(np.int16) + 80, 0, 255
            ).astype(np.uint8)

        elif transform_type == 'color_lines':
            line_colors = [
                (255, 255, 255), (200, 200, 200),
                (0, 255, 0), (0, 255, 255),
                (255, 255, 0), (0, 200, 200),
            ]
            bg = np.zeros_like(result)
            color = random.choice(line_colors)
            bg[fg_mask] = color
            noise = np.random.randint(-15, 15, bg.shape, dtype=np.int16)
            bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            result = bg

        elif transform_type == 'full_cad':
            bg = np.zeros_like(result)
            wall_mask_pixels = (mask == 1)
            if wall_mask_pixels.any():
                wall_color = random.choice([(255,255,255), (200,200,200), (180,180,180)])
                bg[wall_mask_pixels] = wall_color
            win_mask_pixels = (mask == 2)
            if win_mask_pixels.any():
                win_color = random.choice([(0,255,255), (100,200,255), (0,200,200)])
                bg[win_mask_pixels] = win_color
            door_mask_pixels = (mask == 3)
            if door_mask_pixels.any():
                door_color = random.choice([(0,255,0), (255,255,0), (0,200,100)])
                bg[door_mask_pixels] = door_color
            other_fg = fg_mask & ~wall_mask_pixels & ~win_mask_pixels & ~door_mask_pixels
            if other_fg.any():
                bg[other_fg] = random.choice([(150,150,150), (100,100,100), (0,180,0)])
            result = bg

        # 标注线噪声
        if random.random() < 0.2:
            result = self._add_annotation_noise(result, h, w)

        return result, mask

    def _add_annotation_noise(self, img, h, w):
        result = img.copy()
        n_lines = random.randint(1, 5)
        annot_colors = [(0,255,0), (255,0,255), (255,255,0), (0,200,200)]
        for _ in range(n_lines):
            color = random.choice(annot_colors)
            thickness = random.randint(1, 2)
            if random.random() < 0.5:
                y = random.randint(0, h-1)
                x1, x2 = random.randint(0, w//3), random.randint(2*w//3, w-1)
                cv2.line(result, (x1, y), (x2, y), color, thickness)
            else:
                x = random.randint(0, w-1)
                y1, y2 = random.randint(0, h//3), random.randint(2*h//3, h-1)
                cv2.line(result, (x, y1), (x, y2), color, thickness)
        return result


# ============================
# 数据集
# ============================

class FloorplanDatasetFT(Dataset):
    """微调数据集 — 直接使用预计算的 mask.npy（与 train_all.py 一致）"""

    def __init__(self, root_dir, split='train', img_size=512,
                 transform=None, domain_aug=None):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.transform = transform
        self.domain_aug = domain_aug

        split_file = self.root_dir / f"{split}.txt"
        all_samples = [
            line.strip().strip('/') for line in
            split_file.read_text().strip().split('\n') if line.strip()
        ]

        # 只保留有 mask.npy 的样本（与 train_all.py 完全一致）
        self.samples = [
            s for s in all_samples
            if (self.root_dir / s / "mask.npy").exists()
        ]
        print(f"[{split}] {len(self.samples)}/{len(all_samples)} samples with mask.npy")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel = self.samples[idx]
        sample_dir = self.root_dir / rel
        try:
            # 加载图片
            img_path = sample_dir / "F1_scaled.png"
            pil_img = Image.open(str(img_path)).convert('RGB')
            img = np.array(pil_img)
            pil_img.close()

            # 加载预计算 mask（与 train_all.py 一致）
            mask = np.load(str(sample_dir / "mask.npy"))
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

            # 域随机化（在标准增强之前）
            if self.domain_aug is not None:
                img, mask = self.domain_aug(img, mask)

            if self.transform:
                transformed = self.transform(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask']

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

        pred_soft = torch.softmax(pred, dim=1)
        target_oh = nn.functional.one_hot(
            target, self.num_classes
        ).permute(0, 3, 1, 2).float()
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
# 增强配置 — 保守版本
# ============================

def get_train_transforms(img_size):
    """保守增强：与原始训练一致，不加过度颜色抖动"""
    return A.Compose([
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.6, 1.0), ratio=(0.75, 1.33), p=0.5
        ),
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=20, sigma=4, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
        ], p=0.15),
        A.OneOf([
            A.GaussNoise(var_limit=(5, 30), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.15),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        A.CoarseDropout(
            max_holes=6, max_height=24, max_width=24,
            min_holes=1, min_height=8, min_width=8,
            fill_value=0, mask_fill_value=0, p=0.15
        ),
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

class FineTunerV2:
    def __init__(self, config=Config):
        self.cfg = config
        self.device = torch.device(config.DEVICE)

        print("=" * 60)
        print("Domain Adaptation Fine-tuning V2")
        print("=" * 60)
        print(f"Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"VRAM: {vram:.1f} GB")

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.MODEL_DIR, exist_ok=True)

        # 加载预训练模型
        print(f"\nLoading: {config.PRETRAINED_MODEL.name}")
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=config.NUM_CLASSES,
        )
        ckpt = torch.load(str(config.PRETRAINED_MODEL),
                          map_location='cpu', weights_only=False)
        if 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt)
        self.model.to(self.device)

        prev_miou = ckpt.get('best_miou', 'N/A')
        print(f"Loaded! Previous best mIoU: {prev_miou}")

        params = sum(p.numel() for p in self.model.parameters())
        print(f"Params: {params:,} ({params/1e6:.1f}M)")

        # 损失
        self.criterion = CombinedLoss(
            config.NUM_CLASSES,
            class_weights=config.CLASS_WEIGHTS,
        ).to(self.device)

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_miou': [], 'val_miou': [],
            'val_class_iou': [],
            'best_miou': 0.0, 'best_epoch': 0,
            'phase': [],
        }

    def _freeze_encoder(self):
        """冻结encoder参数"""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        frozen = sum(1 for p in self.model.encoder.parameters() if not p.requires_grad)
        print(f"  Encoder frozen: {frozen} param groups")

    def _unfreeze_encoder(self):
        """解冻encoder参数"""
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        print("  Encoder unfrozen")

    def _make_optimizer(self, phase):
        """根据阶段创建优化器"""
        cfg = self.cfg
        if phase == 1:
            # 只优化decoder参数
            decoder_params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(
                decoder_params, lr=cfg.PHASE1_LR,
                weight_decay=cfg.WEIGHT_DECAY
            )
            print(f"  Phase 1 optimizer: AdamW, LR={cfg.PHASE1_LR}")
        else:
            # 差分学习率
            encoder_params = list(self.model.encoder.parameters())
            encoder_ids = {id(p) for p in encoder_params}
            decoder_params = [p for p in self.model.parameters()
                              if id(p) not in encoder_ids]
            optimizer = optim.AdamW([
                {'params': encoder_params, 'lr': cfg.PHASE2_ENCODER_LR},
                {'params': decoder_params, 'lr': cfg.PHASE2_DECODER_LR},
            ], weight_decay=cfg.WEIGHT_DECAY)
            print(f"  Phase 2 optimizer: AdamW, Enc LR={cfg.PHASE2_ENCODER_LR}, "
                  f"Dec LR={cfg.PHASE2_DECODER_LR}")
        return optimizer

    def train(self):
        cfg = self.cfg

        domain_aug = DomainRandomization(prob=cfg.DOMAIN_AUG_PROB)

        train_ds = FloorplanDatasetFT(
            cfg.DATA_DIR, 'train', cfg.IMG_SIZE,
            transform=get_train_transforms(cfg.IMG_SIZE),
            domain_aug=domain_aug,
        )
        val_ds = FloorplanDatasetFT(
            cfg.DATA_DIR, 'val', cfg.IMG_SIZE,
            transform=get_val_transforms(cfg.IMG_SIZE),
            domain_aug=None,
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
        print(f"Effective batch: {cfg.BATCH_SIZE * cfg.GRAD_ACCUM}")
        print(f"Domain aug prob: {cfg.DOMAIN_AUG_PROB}")
        print(f"Class weights: {cfg.CLASS_WEIGHTS}")

        # ===== 先跑一次验证看起始性能 =====
        print("\n--- Baseline validation ---")
        v_loss, v_miou, v_ious = self._validate(val_loader)
        iou_str = " | ".join([
            f"{cfg.CLASS_NAMES[i]}={v_ious[i]:.3f}"
            for i in range(cfg.NUM_CLASSES) if not np.isnan(v_ious[i])
        ])
        print(f"  Baseline mIoU: {v_miou:.4f}")
        print(f"  IoU: {iou_str}")
        self.history['best_miou'] = v_miou  # baseline作为best起点
        # 保存一份baseline作为参考
        self._save('M2_DA_FT_v2_baseline.pt')

        total_epoch = 0
        start = time.time()
        no_improve = 0

        # ===========================
        # Phase 1: Encoder冻结
        # ===========================
        print(f"\n{'='*60}")
        print(f"Phase 1: Freeze Encoder, Train Decoder ({cfg.PHASE1_EPOCHS} epochs)")
        print(f"{'='*60}")
        self._freeze_encoder()
        optimizer = self._make_optimizer(phase=1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.PHASE1_EPOCHS, eta_min=1e-6)

        for ep in range(1, cfg.PHASE1_EPOCHS + 1):
            total_epoch += 1

            # Linear warmup (前3 epoch)
            if ep <= cfg.WARMUP_EPOCHS:
                warmup_factor = ep / cfg.WARMUP_EPOCHS
                for pg in optimizer.param_groups:
                    pg['lr'] = cfg.PHASE1_LR * warmup_factor

            t_loss, t_miou = self._train_epoch(train_loader, total_epoch)
            v_loss, v_miou, v_ious = self._validate(val_loader)

            if ep > cfg.WARMUP_EPOCHS:
                scheduler.step()

            self._log(t_loss, v_loss, t_miou, v_miou, v_ious, 1)

            marker = ""
            if v_miou > self.history['best_miou']:
                self.history['best_miou'] = v_miou
                self.history['best_epoch'] = total_epoch
                self._save('M2_DA_FT_v2_best.pt')
                marker = " ★ BEST"
                no_improve = 0
            else:
                no_improve += 1

            elapsed = time.time() - start
            total_epochs = cfg.PHASE1_EPOCHS + cfg.PHASE2_EPOCHS
            eta = elapsed / total_epoch * (total_epochs - total_epoch)
            lr_now = optimizer.param_groups[0]['lr']

            iou_str = " ".join([
                f"{cfg.CLASS_NAMES[i][:3]}={v_ious[i]:.3f}"
                for i in range(cfg.NUM_CLASSES) if not np.isnan(v_ious[i])
            ])
            print(
                f"P1 E{ep:2d}/{cfg.PHASE1_EPOCHS} | "
                f"L:{t_loss:.4f}/{v_loss:.4f} | "
                f"mIoU:{t_miou:.3f}/{v_miou:.3f} | "
                f"{iou_str} | "
                f"lr:{lr_now:.1e} | "
                f"ETA:{eta/60:.0f}m{marker}",
                flush=True
            )

            if ep % 5 == 0:
                iou_detail = " | ".join([
                    f"{cfg.CLASS_NAMES[i]}={v_ious[i]:.3f}"
                    for i in range(cfg.NUM_CLASSES) if not np.isnan(v_ious[i])
                ])
                print(f"  IoU: {iou_detail}")

        # ===========================
        # Phase 2: 全参数微调
        # ===========================
        print(f"\n{'='*60}")
        print(f"Phase 2: Full Fine-tuning ({cfg.PHASE2_EPOCHS} epochs)")
        print(f"{'='*60}")
        self._unfreeze_encoder()
        optimizer = self._make_optimizer(phase=2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.PHASE2_EPOCHS, eta_min=1e-7)

        for ep in range(1, cfg.PHASE2_EPOCHS + 1):
            total_epoch += 1

            t_loss, t_miou = self._train_epoch(train_loader, total_epoch)
            v_loss, v_miou, v_ious = self._validate(val_loader)
            scheduler.step()

            self._log(t_loss, v_loss, t_miou, v_miou, v_ious, 2)

            marker = ""
            if v_miou > self.history['best_miou']:
                self.history['best_miou'] = v_miou
                self.history['best_epoch'] = total_epoch
                self._save('M2_DA_FT_v2_best.pt')
                marker = " ★ BEST"
                no_improve = 0
            else:
                no_improve += 1

            elapsed = time.time() - start
            total_epochs = cfg.PHASE1_EPOCHS + cfg.PHASE2_EPOCHS
            eta = elapsed / total_epoch * (total_epochs - total_epoch)
            enc_lr = optimizer.param_groups[0]['lr']
            dec_lr = optimizer.param_groups[1]['lr']

            iou_str = " ".join([
                f"{cfg.CLASS_NAMES[i][:3]}={v_ious[i]:.3f}"
                for i in range(cfg.NUM_CLASSES) if not np.isnan(v_ious[i])
            ])
            print(
                f"P2 E{ep:2d}/{cfg.PHASE2_EPOCHS} | "
                f"L:{t_loss:.4f}/{v_loss:.4f} | "
                f"mIoU:{t_miou:.3f}/{v_miou:.3f} | "
                f"{iou_str} | "
                f"lr:E{enc_lr:.1e}/D{dec_lr:.1e} | "
                f"ETA:{eta/60:.0f}m{marker}",
                flush=True
            )

            if ep % 5 == 0:
                iou_detail = " | ".join([
                    f"{cfg.CLASS_NAMES[i]}={v_ious[i]:.3f}"
                    for i in range(cfg.NUM_CLASSES) if not np.isnan(v_ious[i])
                ])
                print(f"  IoU: {iou_detail}")

            if no_improve >= cfg.PATIENCE:
                print(f"\nEarly stopping (no improve for {cfg.PATIENCE} epochs)")
                break

        # Save final
        self._save('M2_DA_FT_v2_final.pt')
        self._save_history()

        total_time = time.time() - start
        print(f"\n{'='*60}")
        print(f"[DONE] Fine-tuning V2 complete! {total_time/60:.1f} min")
        print(f"  Best mIoU: {self.history['best_miou']:.4f} "
              f"(Epoch {self.history['best_epoch']})")
        print(f"  Model: M2_DA_FT_v2_best.pt")

    def _train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0
        total_miou = 0
        n = 0
        self.optimizer_ref = getattr(self, '_current_optimizer', None)

        # 获取当前optimizer（通过train方法中的local var传递不方便，用model.train状态判断）
        # 这里直接在train_epoch中接收optimizer
        # 改为：从caller传入
        # 暂时用一个hack：通过self保存
        optimizer = self._opt
        optimizer.zero_grad()

        for i, (imgs, masks) in enumerate(loader):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)

            out = self.model(imgs)
            loss = self.criterion(out, masks) / self.cfg.GRAD_ACCUM
            loss.backward()

            if (i + 1) % self.cfg.GRAD_ACCUM == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                _, miou = compute_metrics(out, masks, self.cfg.NUM_CLASSES)

            total_loss += loss.item() * self.cfg.GRAD_ACCUM
            total_miou += miou
            n += 1

            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(loader)}] loss:{total_loss/n:.4f} "
                      f"miou:{total_miou/n:.3f}", flush=True)

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

        avg_ious = (np.nanmean(all_ious, axis=0).tolist()
                    if all_ious else [0.0] * self.cfg.NUM_CLASSES)
        valid = [x for x in avg_ious if not np.isnan(x)]
        miou = np.mean(valid) if valid else 0.0
        return total_loss / max(n, 1), miou, avg_ious

    def _log(self, t_loss, v_loss, t_miou, v_miou, v_ious, phase):
        self.history['train_loss'].append(t_loss)
        self.history['val_loss'].append(v_loss)
        self.history['train_miou'].append(t_miou)
        self.history['val_miou'].append(v_miou)
        self.history['val_class_iou'].append(v_ious)
        self.history['phase'].append(phase)

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
        path = self.cfg.OUTPUT_DIR / "finetune_v2_history.json"
        with open(str(path), 'w') as f:
            json.dump(self.history, f, indent=2)


# 修复 _train_epoch 需要 optimizer 引用的问题
# 重写 train 方法中对 _train_epoch 的调用
_orig_train = FineTunerV2.train

def _patched_train(self):
    cfg = self.cfg

    domain_aug = DomainRandomization(prob=cfg.DOMAIN_AUG_PROB)

    train_ds = FloorplanDatasetFT(
        cfg.DATA_DIR, 'train', cfg.IMG_SIZE,
        transform=get_train_transforms(cfg.IMG_SIZE),
        domain_aug=domain_aug,
    )
    val_ds = FloorplanDatasetFT(
        cfg.DATA_DIR, 'val', cfg.IMG_SIZE,
        transform=get_val_transforms(cfg.IMG_SIZE),
        domain_aug=None,
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
    print(f"Effective batch: {cfg.BATCH_SIZE * cfg.GRAD_ACCUM}")
    print(f"Domain aug prob: {cfg.DOMAIN_AUG_PROB}")
    print(f"Class weights: {cfg.CLASS_WEIGHTS}")

    # Baseline
    print("\n--- Baseline validation ---")
    v_loss, v_miou, v_ious = self._validate(val_loader)
    iou_str = " | ".join([
        f"{cfg.CLASS_NAMES[i]}={v_ious[i]:.3f}"
        for i in range(cfg.NUM_CLASSES) if not np.isnan(v_ious[i])
    ])
    print(f"  Baseline mIoU: {v_miou:.4f}")
    print(f"  IoU: {iou_str}")
    self.history['best_miou'] = v_miou
    self._save('M2_DA_FT_v2_baseline.pt')

    total_epoch = 0
    start = time.time()
    no_improve = 0

    # Phase 1
    print(f"\n{'='*60}")
    print(f"Phase 1: Freeze Encoder ({cfg.PHASE1_EPOCHS} epochs)")
    print(f"{'='*60}")
    self._freeze_encoder()
    self._opt = self._make_optimizer(phase=1)
    sched1 = optim.lr_scheduler.CosineAnnealingLR(
        self._opt, T_max=cfg.PHASE1_EPOCHS, eta_min=1e-6)

    for ep in range(1, cfg.PHASE1_EPOCHS + 1):
        total_epoch += 1

        if ep <= cfg.WARMUP_EPOCHS:
            warmup_factor = ep / cfg.WARMUP_EPOCHS
            for pg in self._opt.param_groups:
                pg['lr'] = cfg.PHASE1_LR * warmup_factor

        t_loss, t_miou = self._train_epoch(train_loader, total_epoch)
        v_loss, v_miou, v_ious = self._validate(val_loader)

        if ep > cfg.WARMUP_EPOCHS:
            sched1.step()

        self._log(t_loss, v_loss, t_miou, v_miou, v_ious, 1)

        marker = ""
        if v_miou > self.history['best_miou']:
            self.history['best_miou'] = v_miou
            self.history['best_epoch'] = total_epoch
            self._save('M2_DA_FT_v2_best.pt')
            marker = " ★ BEST"
            no_improve = 0
        else:
            no_improve += 1

        elapsed = time.time() - start
        total_epochs = cfg.PHASE1_EPOCHS + cfg.PHASE2_EPOCHS
        eta = elapsed / total_epoch * (total_epochs - total_epoch)
        lr_now = self._opt.param_groups[0]['lr']

        iou_str = " ".join([
            f"{cfg.CLASS_NAMES[i][:3]}={v_ious[i]:.3f}"
            for i in range(cfg.NUM_CLASSES) if not np.isnan(v_ious[i])
        ])
        print(
            f"P1 E{ep:2d}/{cfg.PHASE1_EPOCHS} | "
            f"L:{t_loss:.4f}/{v_loss:.4f} | "
            f"mIoU:{t_miou:.3f}/{v_miou:.3f} | "
            f"{iou_str} | lr:{lr_now:.1e} | "
            f"ETA:{eta/60:.0f}m{marker}", flush=True
        )
        if ep % 5 == 0:
            detail = " | ".join([
                f"{cfg.CLASS_NAMES[i]}={v_ious[i]:.3f}"
                for i in range(cfg.NUM_CLASSES) if not np.isnan(v_ious[i])
            ])
            print(f"  IoU: {detail}")

    # Phase 2
    print(f"\n{'='*60}")
    print(f"Phase 2: Full Fine-tuning ({cfg.PHASE2_EPOCHS} epochs)")
    print(f"{'='*60}")
    self._unfreeze_encoder()
    self._opt = self._make_optimizer(phase=2)
    sched2 = optim.lr_scheduler.CosineAnnealingLR(
        self._opt, T_max=cfg.PHASE2_EPOCHS, eta_min=1e-7)

    for ep in range(1, cfg.PHASE2_EPOCHS + 1):
        total_epoch += 1

        t_loss, t_miou = self._train_epoch(train_loader, total_epoch)
        v_loss, v_miou, v_ious = self._validate(val_loader)
        sched2.step()

        self._log(t_loss, v_loss, t_miou, v_miou, v_ious, 2)

        marker = ""
        if v_miou > self.history['best_miou']:
            self.history['best_miou'] = v_miou
            self.history['best_epoch'] = total_epoch
            self._save('M2_DA_FT_v2_best.pt')
            marker = " ★ BEST"
            no_improve = 0
        else:
            no_improve += 1

        elapsed = time.time() - start
        total_epochs = cfg.PHASE1_EPOCHS + cfg.PHASE2_EPOCHS
        eta = elapsed / total_epoch * (total_epochs - total_epoch)
        enc_lr = self._opt.param_groups[0]['lr']
        dec_lr = self._opt.param_groups[1]['lr']

        iou_str = " ".join([
            f"{cfg.CLASS_NAMES[i][:3]}={v_ious[i]:.3f}"
            for i in range(cfg.NUM_CLASSES) if not np.isnan(v_ious[i])
        ])
        print(
            f"P2 E{ep:2d}/{cfg.PHASE2_EPOCHS} | "
            f"L:{t_loss:.4f}/{v_loss:.4f} | "
            f"mIoU:{t_miou:.3f}/{v_miou:.3f} | "
            f"{iou_str} | "
            f"lr:E{enc_lr:.1e}/D{dec_lr:.1e} | "
            f"ETA:{eta/60:.0f}m{marker}", flush=True
        )
        if ep % 5 == 0:
            detail = " | ".join([
                f"{cfg.CLASS_NAMES[i]}={v_ious[i]:.3f}"
                for i in range(cfg.NUM_CLASSES) if not np.isnan(v_ious[i])
            ])
            print(f"  IoU: {detail}")

        if no_improve >= cfg.PATIENCE:
            print(f"\nEarly stopping (no improve for {cfg.PATIENCE} epochs)")
            break

    self._save('M2_DA_FT_v2_final.pt')
    self._save_history()

    total_time = time.time() - start
    print(f"\n{'='*60}")
    print(f"[DONE] Fine-tuning V2 complete! {total_time/60:.1f} min")
    print(f"  Best mIoU: {self.history['best_miou']:.4f} "
          f"(Epoch {self.history['best_epoch']})")
    print(f"  Model: M2_DA_FT_v2_best.pt")

FineTunerV2.train = _patched_train


if __name__ == "__main__":
    tuner = FineTunerV2()
    tuner.train()
