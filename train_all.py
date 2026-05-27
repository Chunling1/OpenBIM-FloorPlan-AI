"""
论文对比实验 - 统一训练脚本
支持三个模型:
  M1: LightUNet (baseline, 从零训练)
  M2: U-Net + ResNet34 (ImageNet pretrained)
  M3: DeepLabV3+ + EfficientNet-B4 (ImageNet pretrained)

特性:
  - 使用预计算的精确 mask (mask.npy)
  - 支持 checkpoint resume (中断后可继续)
  - AMP 混合精度
  - albumentations 数据增强

用法:
  python train_all.py --model m1
  python train_all.py --model m2
  python train_all.py --model m3
  python train_all.py --model m1 --resume   # 从 checkpoint 恢复
  python train_all.py --compare              # 对比评估
"""

import os
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"
import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

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

class BaseConfig:
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data" / "cubicasa5k"
    MODEL_DIR = BASE_DIR / "models"

    NUM_CLASSES = 4
    NUM_WORKERS = 0
    CLASS_NAMES = ["background", "wall", "window", "door"]
    CLASS_COLORS = [(0,0,0), (255,0,0), (0,0,255), (0,255,0)]
    CLASS_WEIGHTS = [0.5, 2.0, 3.0, 3.0]

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_EPOCHS = 100
    SAVE_EVERY = 5    # checkpoint 间隔（更频繁保存）
    PATIENCE = 25      # early stopping
    WARMUP_EPOCHS = 5  # LR warmup
    GRAD_CLIP = 0.5    # gradient clipping max norm
    COLLAPSE_THRESHOLD = 3  # epochs of collapse before recovery


class M1Config(BaseConfig):
    """LightUNet baseline"""
    MODEL_NAME = "M1_LightUNet"
    IMG_SIZE = 256
    BATCH_SIZE = 4
    GRAD_ACCUM = 2     # effective batch = 8
    LR = 3e-4
    WEIGHT_DECAY = 1e-4


class M2Config(BaseConfig):
    """U-Net + ResNet34"""
    MODEL_NAME = "M2_UNet_ResNet34_DA"
    IMG_SIZE = 512
    BATCH_SIZE = 2
    GRAD_ACCUM = 4
    LR = 3e-4
    ENCODER_LR = 3e-5
    WEIGHT_DECAY = 1e-4


class M3Config(BaseConfig):
    """DeepLabV3+ + EfficientNet-B4"""
    MODEL_NAME = "M3_DeepLabV3p_EffB4"
    IMG_SIZE = 512
    BATCH_SIZE = 2
    GRAD_ACCUM = 4
    LR = 3e-4
    ENCODER_LR = 2e-5
    WEIGHT_DECAY = 1e-4


CONFIGS = {'m1': M1Config, 'm2': M2Config, 'm3': M3Config}


# ============================
# LightUNet (M1)
# ============================

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class LightUNet(nn.Module):
    def __init__(self, in_c=3, num_classes=4, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for f in features:
            self.downs.append(DoubleConv(in_c, f))
            in_c = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.ups.append(DoubleConv(f*2, f))

        self.final = nn.Conv2d(features[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)
        return self.final(x)


# ============================
# 数据集 (使用预计算 mask.npy)
# ============================

def get_train_aug(sz):
    return A.Compose([
        A.RandomResizedCrop(size=(sz, sz), scale=(0.6, 1.0), ratio=(0.8, 1.25), p=0.5),
        A.Resize(sz, sz),
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
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.CoarseDropout(max_holes=6, max_height=24, max_width=24,
                        fill_value=0, mask_fill_value=0, p=0.15),
        # --- 域泛化增强（支持黑底CAD图） ---
        A.InvertImg(p=0.3),                          # 反色 → 模拟黑底CAD
        A.ToGray(p=0.2),                             # 灰度 → 模拟CAD线稿
        A.ColorJitter(brightness=0.4, contrast=0.4,
                      saturation=0.3, hue=0.1, p=0.3),  # 大幅色彩抖动
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_val_aug(sz):
    return A.Compose([
        A.Resize(sz, sz),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class FloorplanDatasetV3(Dataset):
    """使用预计算 mask.npy 的数据集"""
    def __init__(self, root_dir, split='train', img_size=512, transform=None):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.transform = transform

        split_file = self.root_dir / f"{split}.txt"
        all_samples = [l.strip() for l in split_file.read_text().strip().split('\n') if l.strip()]

        # 只保留有 mask.npy 的样本
        self.samples = []
        for s in all_samples:
            s = s.strip().strip('/')
            if (self.root_dir / s / "mask.npy").exists():
                self.samples.append(s)

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

            # 加载预计算 mask
            mask = np.load(str(sample_dir / "mask.npy"))

            # 如果 mask 和 img 尺寸不同，resize mask
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

            # 增强
            if self.transform:
                transformed = self.transform(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask']

            # to tensor
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img.transpose(2, 0, 1)).float()

            mask = np.clip(mask, 0, 3)
            mask = torch.from_numpy(mask.astype(np.int64)).long()
            return img, mask

        except Exception:
            img = torch.zeros(3, self.img_size, self.img_size)
            mask = torch.zeros(self.img_size, self.img_size, dtype=torch.long)
            return img, mask


# ============================
# 损失函数
# ============================

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, class_weights, gamma=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('weight', torch.FloatTensor(class_weights))

    def forward(self, pred, target):
        # Standard weighted CE (no Focal gamma - more stable)
        ce = nn.functional.cross_entropy(pred, target, weight=self.weight, reduction='mean')
        ce = torch.clamp(ce, max=50.0)  # prevent extreme values

        # Dice loss (with epsilon protection)
        pred_soft = torch.softmax(pred, dim=1)
        target_oh = nn.functional.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        inter = (pred_soft * target_oh).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
        dice = 1.0 - (2 * inter + 1e-6) / (union + 1e-6)

        total = ce + dice.mean()
        # Final NaN guard
        if torch.isnan(total) or torch.isinf(total):
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return total


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
# 训练器
# ============================

class Trainer:
    def __init__(self, model_key, resume=False):
        self.cfg = CONFIGS[model_key]()
        self.model_key = model_key
        self.device = torch.device(self.cfg.DEVICE)

        print("=" * 60)
        print(f"Model: {self.cfg.MODEL_NAME}")
        print("=" * 60)
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        os.makedirs(self.cfg.MODEL_DIR, exist_ok=True)

        # 创建模型
        self.model = self._create_model().to(self.device)
        params = sum(p.numel() for p in self.model.parameters())
        print(f"Params: {params:,} ({params/1e6:.1f}M)")

        # 损失
        self.criterion = CombinedLoss(
            self.cfg.NUM_CLASSES, self.cfg.CLASS_WEIGHTS
        ).to(self.device)

        # 优化器
        self.optimizer = self._create_optimizer()

        # AMP - disabled for training stability (FP32)
        self.scaler = GradScaler('cuda', init_scale=256, growth_interval=500)
        self.use_amp = False  # Disabled: FP16 causes gradient instability with small batch + high res

        # 历史
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_miou': [], 'val_miou': [],
            'val_class_iou': [],
            'best_miou': 0.0, 'best_epoch': 0,
            'model_name': self.cfg.MODEL_NAME,
        }
        self.start_epoch = 1

        # Resume
        if resume:
            self._load_checkpoint()

    def _create_model(self):
        if self.model_key == 'm1':
            return LightUNet(3, self.cfg.NUM_CLASSES)
        elif self.model_key == 'm2':
            return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                           in_channels=3, classes=self.cfg.NUM_CLASSES)
        elif self.model_key == 'm3':
            return smp.DeepLabV3Plus(encoder_name="efficientnet-b4",
                                     encoder_weights="imagenet",
                                     in_channels=3, classes=self.cfg.NUM_CLASSES)

    def _create_optimizer(self):
        if self.model_key == 'm1':
            return optim.AdamW(self.model.parameters(),
                              lr=self.cfg.LR, weight_decay=self.cfg.WEIGHT_DECAY)
        else:
            # 差分学习率
            encoder_params = list(self.model.encoder.parameters())
            encoder_ids = {id(p) for p in encoder_params}
            decoder_params = [p for p in self.model.parameters() if id(p) not in encoder_ids]
            return optim.AdamW([
                {'params': encoder_params, 'lr': self.cfg.ENCODER_LR},
                {'params': decoder_params, 'lr': self.cfg.LR},
            ], weight_decay=self.cfg.WEIGHT_DECAY)

    def _checkpoint_path(self):
        return self.cfg.MODEL_DIR / f"{self.cfg.MODEL_NAME}_checkpoint.pt"

    def _best_path(self):
        return self.cfg.MODEL_DIR / f"{self.cfg.MODEL_NAME}_best.pt"

    def _save_checkpoint(self, epoch, scheduler=None):
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'history': self.history,
            'config': {
                'num_classes': self.cfg.NUM_CLASSES,
                'img_size': self.cfg.IMG_SIZE,
                'class_names': self.cfg.CLASS_NAMES,
                'model_key': self.model_key,
            },
        }
        if scheduler is not None:
            save_dict['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(save_dict, str(self._checkpoint_path()))

    def _save_best(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_miou': self.history['best_miou'],
            'config': {
                'num_classes': self.cfg.NUM_CLASSES,
                'img_size': self.cfg.IMG_SIZE,
                'class_names': self.cfg.CLASS_NAMES,
                'model_key': self.model_key,
            },
        }, str(self._best_path()))

    def _load_checkpoint(self):
        ckpt_path = self._checkpoint_path()
        if ckpt_path.exists():
            self._ckpt_data = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
            self.model.load_state_dict(self._ckpt_data['model_state_dict'])
            self.optimizer.load_state_dict(self._ckpt_data['optimizer_state_dict'])
            self.scaler.load_state_dict(self._ckpt_data['scaler_state_dict'])
            self.history = self._ckpt_data['history']
            self.start_epoch = self._ckpt_data['epoch'] + 1
            print(f"[RESUME] from epoch {self._ckpt_data['epoch']}, best mIoU={self.history['best_miou']:.4f}")
        else:
            self._ckpt_data = None
            print("[RESUME] no checkpoint found, starting fresh")

    def train(self):
        cfg = self.cfg

        train_ds = FloorplanDatasetV3(cfg.DATA_DIR, 'train', cfg.IMG_SIZE,
                                      transform=get_train_aug(cfg.IMG_SIZE))
        val_ds = FloorplanDatasetV3(cfg.DATA_DIR, 'val', cfg.IMG_SIZE,
                                    transform=get_val_aug(cfg.IMG_SIZE))

        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                                  shuffle=True, num_workers=cfg.NUM_WORKERS,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE,
                                shuffle=False, num_workers=cfg.NUM_WORKERS,
                                pin_memory=True)

        # Warmup + CosineAnnealing scheduler
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=cfg.WARMUP_EPOCHS)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.NUM_EPOCHS - cfg.WARMUP_EPOCHS)
        scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[cfg.WARMUP_EPOCHS])

        # 如果 resume, 恢复 scheduler 状态
        if hasattr(self, '_ckpt_data') and self._ckpt_data and 'scheduler_state_dict' in self._ckpt_data:
            scheduler.load_state_dict(self._ckpt_data['scheduler_state_dict'])
            print(f"[RESUME] scheduler restored, LR={self.optimizer.param_groups[0]['lr']:.6f}")
        elif self.start_epoch > 1:
            # 兼容旧 checkpoint: 逐步跳 scheduler
            for _ in range(1, self.start_epoch):
                scheduler.step()

        print(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
        print(f"Val:   {len(val_ds)} samples")
        print(f"Effective batch: {cfg.BATCH_SIZE * cfg.GRAD_ACCUM}")

        start = time.time()
        no_improve = 0

        collapse_count = 0  # 连续崩溃计数
        prev_val_miou = 0.0

        for epoch in range(self.start_epoch, cfg.NUM_EPOCHS + 1):
            t_loss, t_miou = self._train_epoch(train_loader)
            v_loss, v_miou, v_ious = self._validate(val_loader)
            scheduler.step()

            self.history['train_loss'].append(t_loss)
            self.history['val_loss'].append(v_loss)
            self.history['train_miou'].append(t_miou)
            self.history['val_miou'].append(v_miou)
            self.history['val_class_iou'].append(v_ious)

            lr = self.optimizer.param_groups[0]['lr']
            marker = ""

            if v_miou > self.history['best_miou']:
                self.history['best_miou'] = v_miou
                self.history['best_epoch'] = epoch
                self._save_best()
                marker = " * BEST"
                no_improve = 0
                collapse_count = 0
            else:
                no_improve += 1

            # 崩溃检测：val_miou 大幅下降
            if epoch > 1 and v_miou < prev_val_miou * 0.5 and prev_val_miou > 0.3:
                collapse_count += 1
                print(f"  [WARN] Collapse detected ({collapse_count}/{cfg.COLLAPSE_THRESHOLD}): "
                      f"mIoU dropped {prev_val_miou:.3f} -> {v_miou:.3f}", flush=True)
                if collapse_count >= cfg.COLLAPSE_THRESHOLD:
                    # 从 best model 恢复
                    best_path = self._best_path()
                    if best_path.exists():
                        print(f"  [RECOVERY] Loading best model (mIoU={self.history['best_miou']:.4f})", flush=True)
                        ckpt = torch.load(str(best_path), map_location=self.device, weights_only=False)
                        self.model.load_state_dict(ckpt['model_state_dict'])
                        self.scaler = GradScaler('cuda', init_scale=256, growth_interval=500)
                        collapse_count = 0
            else:
                collapse_count = max(0, collapse_count - 1)
            prev_val_miou = v_miou

            if epoch % cfg.SAVE_EVERY == 0:
                self._save_checkpoint(epoch, scheduler=scheduler)

            elapsed = time.time() - start
            eta = elapsed / (epoch - self.start_epoch + 1) * (cfg.NUM_EPOCHS - epoch)

            print(
                f"E{epoch:3d}/{cfg.NUM_EPOCHS} | "
                f"L:{t_loss:.4f}/{v_loss:.4f} | "
                f"mIoU:{t_miou:.3f}/{v_miou:.3f} | "
                f"LR:{lr:.6f} | "
                f"ETA:{eta/60:.0f}m{marker}",
                flush=True
            )
            if epoch % 5 == 0:
                iou_str = " | ".join([f"{cfg.CLASS_NAMES[i]}={v_ious[i]:.3f}"
                                      for i in range(cfg.NUM_CLASSES) if not np.isnan(v_ious[i])])
                print(f"  IoU: {iou_str}", flush=True)

            if no_improve >= cfg.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        self._save_checkpoint(epoch, scheduler=scheduler)

        # 保存历史
        out_dir = cfg.BASE_DIR / "output_paper"
        os.makedirs(out_dir, exist_ok=True)
        with open(str(out_dir / f"{cfg.MODEL_NAME}_history.json"), 'w') as f:
            json.dump(self.history, f, indent=2)

        total = time.time() - start
        print(f"\n[DONE] {cfg.MODEL_NAME} complete! {total/60:.1f} min")
        print(f"  Best mIoU: {self.history['best_miou']:.4f} (Epoch {self.history['best_epoch']})")

    def _train_epoch(self, loader):
        self.model.train()
        total_loss, total_miou, n = 0, 0, 0
        nan_batches = 0
        self.optimizer.zero_grad()
        has_grads = False  # 追踪当前累积窗口是否有有效梯度

        for i, (imgs, masks) in enumerate(loader):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)

            if self.use_amp:
                with autocast('cuda'):
                    out = self.model(imgs)
                # Compute loss in FP32 for stability
                loss = self.criterion(out.float(), masks) / self.cfg.GRAD_ACCUM

                # NaN guard: skip batch if loss is NaN/inf
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_batches += 1
                    continue

                self.scaler.scale(loss).backward()
                has_grads = True
            else:
                out = self.model(imgs)
                loss = self.criterion(out, masks) / self.cfg.GRAD_ACCUM

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_batches += 1
                    continue

                loss.backward()
                has_grads = True

            if (i + 1) % self.cfg.GRAD_ACCUM == 0 or (i + 1) == len(loader):
                if has_grads:
                    if self.use_amp:
                        try:
                            self.scaler.unscale_(self.optimizer)
                            # Check for inf/nan gradients
                            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRAD_CLIP)
                            if torch.isfinite(torch.tensor(total_norm)):
                                self.scaler.step(self.optimizer)
                            self.scaler.update()
                        except (AssertionError, RuntimeError) as e:
                            # GradScaler internal state mismatch — reset scaler
                            print(f"  [WARN] AMP error: {e}, resetting scaler", flush=True)
                            self.scaler = GradScaler('cuda', init_scale=256, growth_interval=500)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRAD_CLIP)
                        self.optimizer.step()
                self.optimizer.zero_grad()
                has_grads = False

            with torch.no_grad():
                _, miou = compute_metrics(out, masks, self.cfg.NUM_CLASSES)
            loss_val = loss.item() * self.cfg.GRAD_ACCUM
            if not (np.isnan(loss_val) or np.isinf(loss_val)):
                total_loss += loss_val
                n += 1
            total_miou += miou

        if nan_batches > 0:
            print(f"  [WARN] {nan_batches} NaN batches skipped", flush=True)

        total_batches = n + nan_batches
        return total_loss / max(n, 1), total_miou / max(total_batches, 1)

    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()
        total_loss, n = 0, 0
        all_ious = []

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
            total_loss += loss.item()
            all_ious.append(ious)
            n += 1

        avg_ious = np.nanmean(all_ious, axis=0).tolist()
        valid = [x for x in avg_ious if not np.isnan(x)]
        miou = np.mean(valid) if valid else 0.0
        return total_loss / max(n, 1), miou, avg_ious


# ============================
# 对比评估
# ============================

def compare_models():
    cfg = BaseConfig()
    print("=" * 60)
    print("Model Comparison on Validation Set")
    print("=" * 60)

    results = {}
    for key, ConfigCls in CONFIGS.items():
        c = ConfigCls()
        best_path = c.MODEL_DIR / f"{c.MODEL_NAME}_best.pt"
        if not best_path.exists():
            print(f"[SKIP] {c.MODEL_NAME}: no best model found")
            continue

        ckpt = torch.load(str(best_path), map_location='cpu', weights_only=False)
        results[c.MODEL_NAME] = {
            'miou': ckpt.get('best_miou', 0),
            'params': 0,
        }

        hist_path = c.BASE_DIR / "output_paper" / f"{c.MODEL_NAME}_history.json"
        if hist_path.exists():
            with open(str(hist_path)) as f:
                hist = json.load(f)
            if hist.get('val_class_iou'):
                best_ep = hist.get('best_epoch', len(hist['val_class_iou']))
                idx = min(best_ep - 1, len(hist['val_class_iou']) - 1)
                results[c.MODEL_NAME]['class_iou'] = hist['val_class_iou'][idx]

    print(f"\n{'Model':<30} {'mIoU':>8} {'BG':>8} {'Wall':>8} {'Win':>8} {'Door':>8}")
    print("-" * 72)
    for name, r in results.items():
        ciou = r.get('class_iou', [0,0,0,0])
        print(f"{name:<30} {r['miou']:8.4f} {ciou[0]:8.3f} {ciou[1]:8.3f} {ciou[2]:8.3f} {ciou[3]:8.3f}")


# ============================
# 入口
# ============================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['m1', 'm2', 'm3'], default='m1')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--compare', action='store_true', help='Compare all models')
    args = parser.parse_args()

    if args.compare:
        compare_models()
    else:
        trainer = Trainer(args.model, resume=args.resume)
        trainer.train()
