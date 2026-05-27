"""
建筑平面图语义分割 - 训练脚本
U-Net + MobileNetV3 backbone
数据集: CubiCasa5K

目标类别:
  0 - background (背景)
  1 - wall (墙体)
  2 - window (窗户)  
  3 - door (门)

硬件要求: RTX 4050 (6GB VRAM) 足够
"""

import os
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"  
# Suppress libpng warnings by redirecting stderr for cv2
import ctypes
import sys

# Windows: suppress CRT stderr warnings from libpng 
if sys.platform == 'win32':
    try:
        kernel32 = ctypes.windll.kernel32
        # This doesn't fully suppress libpng but we'll handle it via PIL
    except:
        pass

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
import torchvision.transforms.functional as TF

warnings.filterwarnings('ignore')

# === 配置 ===
class Config:
    # 路径
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data" / "cubicasa5k"
    OUTPUT_DIR = BASE_DIR / "output"
    MODEL_DIR = BASE_DIR / "models"
    
    # 模型
    NUM_CLASSES = 4          # bg, wall, window, door
    IMG_SIZE = 256           # 减小尺寸节省内存
    ENCODER = "mobilenet_v3_small"  # 轻量级 backbone
    
    # 训练
    BATCH_SIZE = 2           # 保守内存配置
    NUM_EPOCHS = 80
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 0          # Windows 需要 0 避免 multiprocessing 问题
    
    # 数据增强
    AUG_PROBABILITY = 0.5
    
    # 类别权重（墙体最重要）
    CLASS_WEIGHTS = [0.5, 2.0, 3.0, 3.0]  # bg=0.5, wall=2, window=3, door=3
    
    # 设备
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 类别名
    CLASS_NAMES = ["background", "wall", "window", "door"]
    CLASS_COLORS = [
        (0, 0, 0),       # bg: 黑
        (255, 0, 0),     # wall: 红
        (0, 0, 255),     # window: 蓝
        (0, 255, 0),     # door: 绿
    ]


# ============================
# 数据集
# ============================

class FloorplanDataset(Dataset):
    """CubiCasa5K 平面图数据集"""
    
    # CubiCasa SVG 类别映射到我们的类别
    CATEGORY_MAP = {
        'Wall': 1,
        'Railing': 1,
        'Window': 2,
        'Door': 3,
    }
    
    def __init__(self, root_dir, split='train', img_size=512, augment=False):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.augment = augment
        
        # 读取分割文件
        split_file = self.root_dir / f"{split}.txt"
        if split_file.exists():
            self.samples = [
                line.strip() for line in split_file.read_text().strip().split('\n')
                if line.strip()
            ]
        else:
            # 如果没有分割文件，扫描目录
            all_dirs = sorted([
                d.name for d in self.root_dir.iterdir() 
                if d.is_dir() and (d / "F1_scaled.png").exists()
            ])
            
            # 自动分割 80/10/10
            n = len(all_dirs)
            if split == 'train':
                self.samples = all_dirs[:int(n * 0.8)]
            elif split == 'val':
                self.samples = all_dirs[int(n * 0.8):int(n * 0.9)]
            else:
                self.samples = all_dirs[int(n * 0.9):]
        
        print(f"[{split}] 加载了 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def _parse_svg_mask(self, svg_path, img_shape):
        """从 SVG 标注文件解析出分割 mask
        CubiCasa5K SVG 使用 viewBox 坐标系，需要缩放到图片像素坐标
        """
        img_h, img_w = img_shape[:2]
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        if not os.path.exists(svg_path):
            return mask
        
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            # SVG 命名空间
            ns = {'svg': 'http://www.w3.org/2000/svg'}
            
            # 获取 viewBox 用于坐标缩放
            viewbox = root.get('viewBox', '')
            if viewbox:
                parts = viewbox.split()
                svg_w = float(parts[2]) if len(parts) >= 3 else img_w
                svg_h = float(parts[3]) if len(parts) >= 4 else img_h
            else:
                svg_w = float(root.get('width', img_w))
                svg_h = float(root.get('height', img_h))
            
            scale_x = img_w / svg_w if svg_w > 0 else 1.0
            scale_y = img_h / svg_h if svg_h > 0 else 1.0
            
            for group in root.findall('.//svg:g', ns):
                class_name = group.get('class', '') or group.get('id', '')
                
                # 匹配类别
                mapped_class = 0
                for key, cls_id in self.CATEGORY_MAP.items():
                    if key.lower() in class_name.lower():
                        mapped_class = cls_id
                        break
                
                if mapped_class == 0:
                    continue
                
                # 解析此 group 及其子节点中的所有 polygon
                for polygon in group.findall('.//svg:polygon', ns) + group.findall('svg:polygon', ns):
                    points_str = polygon.get('points', '')
                    if not points_str:
                        continue
                    
                    try:
                        points = []
                        # 支持逗号和空格分隔的坐标
                        raw = points_str.strip().replace(',', ' ').split()
                        for i in range(0, len(raw) - 1, 2):
                            x = float(raw[i]) * scale_x
                            y = float(raw[i + 1]) * scale_y
                            points.append([int(x), int(y)])
                        
                        if len(points) >= 3:
                            pts = np.array(points, dtype=np.int32)
                            cv2.fillPoly(mask, [pts], mapped_class)
                    except (ValueError, IndexError):
                        continue
                
                # 解析直接子级的 polygon（非嵌套）
                for polygon in group.findall('svg:polygon', ns):
                    pass  # already handled above
                
                # 解析矩形
                for rect in group.findall('.//svg:rect', ns):
                    try:
                        x = float(rect.get('x', 0)) * scale_x
                        y = float(rect.get('y', 0)) * scale_y
                        w = float(rect.get('width', 0)) * scale_x
                        h = float(rect.get('height', 0)) * scale_y
                        if w > 0 and h > 0:
                            pts = np.array([
                                [int(x), int(y)], [int(x + w), int(y)],
                                [int(x + w), int(y + h)], [int(x), int(y + h)]
                            ], dtype=np.int32)
                            cv2.fillPoly(mask, [pts], mapped_class)
                    except (ValueError, TypeError):
                        continue
                
                # 解析 line 元素（一些墙体用 line 表示）
                for line in group.findall('.//svg:line', ns):
                    try:
                        x1 = float(line.get('x1', 0)) * scale_x
                        y1 = float(line.get('y1', 0)) * scale_y
                        x2 = float(line.get('x2', 0)) * scale_x
                        y2 = float(line.get('y2', 0)) * scale_y
                        thickness = max(2, int(3 * min(scale_x, scale_y)))
                        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), mapped_class, thickness)
                    except (ValueError, TypeError):
                        continue
        
        except ET.ParseError:
            pass
        
        return mask
    
    def _load_precomputed_mask(self, sample_dir):
        """尝试加载预计算的 mask（如果 CubiCasa5K 提供了 numpy 格式）"""
        mask_path = sample_dir / "F1_scaled_mask.npy"
        if mask_path.exists():
            return np.load(str(mask_path))
        
        # 尝试 PNG mask
        mask_path = sample_dir / "label.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            return mask
        
        return None

    def __getitem__(self, idx):
        # CubiCasa5K 路径格式: /high_quality_architectural/6044/
        rel_path = self.samples[idx].strip().strip('/')
        sample_path = self.root_dir / rel_path
        
        try:
            return self._load_sample(sample_path)
        except Exception:
            # 任何加载错误都返回空白
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            mask = np.zeros((self.img_size, self.img_size), dtype=np.int64)
            return self._to_tensor(img, mask)
    
    def _load_sample(self, sample_path):
        # 加载图片 - 使用 PIL 替代 cv2 避免内存问题
        img_path = sample_path / "F1_scaled.png"
        if not img_path.exists():
            for name in ["F1_original.png", "image.png", "floor.png"]:
                alt = sample_path / name
                if alt.exists():
                    img_path = alt
                    break
        
        # 用 PIL 加载并立即resize，避免大图占用过多内存
        pil_img = Image.open(str(img_path)).convert('RGB')
        pil_img = pil_img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = np.array(pil_img)
        pil_img.close()
        
        # 加载 mask - SVG 解析需要知道原图尺寸进行坐标映射
        # 但这里我们直接用 img_size 作为目标 mask 尺寸
        mask = self._load_precomputed_mask(sample_path)
        if mask is None:
            svg_path = sample_path / "model.svg"
            # 解析 SVG 时直接生成 img_size 大小的 mask
            mask = self._parse_svg_mask(str(svg_path), (self.img_size, self.img_size))
        
        # img 已经是正确尺寸了（PIL 已 resize）
        # mask 如果尺寸不对，resize 到正确尺寸
        if mask.shape[0] != self.img_size or mask.shape[1] != self.img_size:
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # 数据增强
        if self.augment:
            img, mask = self._augment(img, mask)
        
        # 确保 mask 值在合法范围内
        mask = np.clip(mask, 0, Config.NUM_CLASSES - 1)
        
        return self._to_tensor(img, mask)
    
    def _augment(self, img, mask):
        """数据增强"""
        if random.random() < 0.5:
            img = cv2.flip(img, 1)  # 水平翻转
            mask = cv2.flip(mask, 1)
        
        if random.random() < 0.5:
            img = cv2.flip(img, 0)  # 垂直翻转
            mask = cv2.flip(mask, 0)
        
        if random.random() < 0.3:
            k = random.choice([1, 2, 3])
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()
        
        if random.random() < 0.3:
            # 亮度/对比度调整
            alpha = random.uniform(0.8, 1.2)
            beta = random.randint(-20, 20)
            img = np.clip(img * alpha + beta, 0, 255).astype(np.uint8)
        
        return img, mask
    
    def _to_tensor(self, img, mask):
        """转换为 PyTorch tensor"""
        # img: [H, W, 3] -> [3, H, W], float32, normalized
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        
        # mask: [H, W] -> [H, W], int64
        mask = torch.from_numpy(mask.astype(np.int64)).long()
        
        return img, mask


# ============================
# 模型 (轻量级 U-Net)
# ============================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class LightUNet(nn.Module):
    """
    轻量级 U-Net
    适配 RTX 4050 6GB VRAM + 后续 CPU 推理
    参数量 ~1.5M
    """
    def __init__(self, in_channels=3, num_classes=4):
        super().__init__()
        
        # Encoder (下采样路径)
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        
        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)
        
        # Decoder (上采样路径)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        
        # 分类头
        self.head = nn.Conv2d(32, num_classes, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)          # [B, 32, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, 64, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [B, 128, H/4, W/4]
        e4 = self.enc4(self.pool(e3))  # [B, 256, H/8, W/8]
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # [B, 512, H/16, W/16]
        
        # Decoder + Skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.head(d1)


# ============================
# 损失函数
# ============================

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred_soft = torch.softmax(pred, dim=1)
        target_one_hot = nn.functional.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (pred_soft * target_one_hot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        super().__init__()
        if class_weights is not None:
            weight = torch.FloatTensor(class_weights)
        else:
            weight = None
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss(num_classes)
    
    def forward(self, pred, target):
        return self.ce(pred, target) + self.dice(pred, target)


# ============================
# 评估指标
# ============================

def compute_metrics(pred, target, num_classes):
    """计算每类 IoU 和总体 mIoU"""
    pred_cls = pred.argmax(dim=1)  # [B, H, W]
    
    ious = []
    for c in range(num_classes):
        pred_c = (pred_cls == c)
        target_c = (target == c)
        
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        
        if union > 0:
            ious.append((intersection / union).item())
        else:
            ious.append(float('nan'))
    
    # 过滤 nan
    valid_ious = [x for x in ious if not np.isnan(x)]
    miou = np.mean(valid_ious) if valid_ious else 0.0
    
    return ious, miou


# ============================
# 训练器
# ============================

class Trainer:
    def __init__(self, config=Config):
        self.cfg = config
        self.device = torch.device(config.DEVICE)
        
        print(f"[i] 设备: {self.device}")
        if self.device.type == 'cuda':
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 创建输出目录
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        
        # 模型
        self.model = LightUNet(
            in_channels=3,
            num_classes=config.NUM_CLASSES,
        ).to(self.device)
        
        params = sum(p.numel() for p in self.model.parameters())
        print(f"[i] 模型参数量: {params:,} ({params / 1e6:.1f}M)")
        
        # 损失函数
        self.criterion = CombinedLoss(
            config.NUM_CLASSES,
            class_weights=config.CLASS_WEIGHTS,
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY,
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.NUM_EPOCHS,
        )
        
        # 记录
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_miou': [], 'val_miou': [],
            'val_class_iou': [],
            'best_miou': 0.0,
            'best_epoch': 0,
        }
    
    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        # 数据加载
        train_dataset = FloorplanDataset(
            self.cfg.DATA_DIR, split='train',
            img_size=self.cfg.IMG_SIZE, augment=True,
        )
        val_dataset = FloorplanDataset(
            self.cfg.DATA_DIR, split='val',
            img_size=self.cfg.IMG_SIZE, augment=False,
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.cfg.BATCH_SIZE,
            shuffle=True, num_workers=self.cfg.NUM_WORKERS,
            pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.cfg.BATCH_SIZE,
            shuffle=False, num_workers=self.cfg.NUM_WORKERS,
            pin_memory=True,
        )
        
        print(f"[i] Train: {len(train_dataset)} 样本, {len(train_loader)} batch")
        print(f"[i] Val:   {len(val_dataset)} 样本, {len(val_loader)} batch")
        
        start_time = time.time()
        
        for epoch in range(1, self.cfg.NUM_EPOCHS + 1):
            # === 训练 ===
            train_loss, train_miou = self._train_epoch(train_loader, epoch)
            
            # === 验证 ===
            val_loss, val_miou, val_class_iou = self._validate(val_loader)
            
            # === 学习率调度 ===
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            
            # === 记录 ===
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_miou'].append(train_miou)
            self.history['val_miou'].append(val_miou)
            self.history['val_class_iou'].append(val_class_iou)
            
            # 保存最佳模型
            if val_miou > self.history['best_miou']:
                self.history['best_miou'] = val_miou
                self.history['best_epoch'] = epoch
                self._save_model('best_model.pt')
                marker = " * BEST"
            else:
                marker = ""
            
            # 每 10 epoch 保存 checkpoint
            if epoch % 10 == 0:
                self._save_model(f'checkpoint_epoch{epoch}.pt')
            
            # 打印
            elapsed = time.time() - start_time
            eta = elapsed / epoch * (self.cfg.NUM_EPOCHS - epoch)
            
            iou_str = " | ".join([
                f"{self.cfg.CLASS_NAMES[i]}={val_class_iou[i]:.3f}" 
                for i in range(self.cfg.NUM_CLASSES) 
                if not np.isnan(val_class_iou[i])
            ])
            
            print(
                f"Epoch {epoch:3d}/{self.cfg.NUM_EPOCHS} | "
                f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                f"mIoU: {train_miou:.3f}/{val_miou:.3f} | "
                f"LR: {lr:.6f} | "
                f"ETA: {eta/60:.0f}min{marker}"
            )
            if epoch % 5 == 0:
                print(f"  IoU: {iou_str}")
        
        # 最终保存
        self._save_model('final_model.pt')
        self._save_history()
        
        total_time = time.time() - start_time
        print(f"\n[DONE] Training complete! {total_time/60:.1f} min")
        print(f"    最佳 mIoU: {self.history['best_miou']:.4f} (Epoch {self.history['best_epoch']})")
        print(f"    模型保存至: {self.cfg.MODEL_DIR}")
    
    def _train_epoch(self, loader, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        total_miou = 0
        n = 0
        
        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # 前向
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # 反向
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 指标
            with torch.no_grad():
                _, miou = compute_metrics(outputs, masks, self.cfg.NUM_CLASSES)
            
            total_loss += loss.item()
            total_miou += miou
            n += 1
        
        return total_loss / max(n, 1), total_miou / max(n, 1)
    
    @torch.no_grad()
    def _validate(self, loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_ious = []
        n = 0
        
        for images, masks in loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            ious, _ = compute_metrics(outputs, masks, self.cfg.NUM_CLASSES)
            
            total_loss += loss.item()
            all_ious.append(ious)
            n += 1
        
        # 计算平均每类 IoU
        avg_ious = np.nanmean(all_ious, axis=0).tolist() if all_ious else [0.0] * self.cfg.NUM_CLASSES
        valid = [x for x in avg_ious if not np.isnan(x)]
        miou = np.mean(valid) if valid else 0.0
        
        return total_loss / max(n, 1), miou, avg_ious
    
    def _save_model(self, filename):
        """保存模型"""
        path = self.cfg.MODEL_DIR / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'num_classes': self.cfg.NUM_CLASSES,
                'img_size': self.cfg.IMG_SIZE,
                'class_names': self.cfg.CLASS_NAMES,
            },
            'best_miou': self.history['best_miou'],
        }, str(path))
    
    def _save_history(self):
        """保存训练历史"""
        path = self.cfg.OUTPUT_DIR / "training_history.json"
        with open(str(path), 'w') as f:
            json.dump(self.history, f, indent=2)


# ============================
# 推理 (训练完成后用这个)
# ============================

class FloorplanPredictor:
    """
    平面图分割推理器
    训练完成后导出到 BIMweb 用
    """
    
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        cfg = checkpoint['config']
        
        self.model = LightUNet(
            in_channels=3,
            num_classes=cfg['num_classes'],
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.img_size = cfg['img_size']
        self.class_names = cfg['class_names']
        
        print(f"[OK] 模型加载成功 (mIoU: {checkpoint.get('best_miou', 'N/A'):.4f})")
    
    @torch.no_grad()
    def predict(self, image_path):
        """
        输入: 图片路径
        输出: 分割 mask (H, W) 和各类概率 (C, H, W)
        """
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        
        # 预处理
        resized = cv2.resize(img, (self.img_size, self.img_size))
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
        tensor = tensor.to(self.device)
        
        # 推理
        output = self.model(tensor)
        probs = torch.softmax(output, dim=1)[0]  # [C, H, W]
        mask = probs.argmax(dim=0).cpu().numpy()   # [H, W]
        
        # 还原原始尺寸
        mask = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        return mask, probs.cpu().numpy()
    
    def predict_to_geometry(self, image_path, scale=1.0):
        """
        输入: 图片路径
        输出: 矢量化几何数据（可直接给 BIMweb API）
        """
        mask, _ = self.predict(image_path)
        
        result = {'walls': [], 'windows': [], 'doors': []}
        
        for cls_id, cls_name in [(1, 'walls'), (2, 'windows'), (3, 'doors')]:
            cls_mask = (mask == cls_id).astype(np.uint8) * 255
            
            # 形态学清理
            kernel = np.ones((3, 3), np.uint8)
            cls_mask = cv2.morphologyEx(cls_mask, cv2.MORPH_CLOSE, kernel)
            
            # 提取轮廓
            contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 50:  # 过滤噪点
                    continue
                
                # 简化轮廓
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                pts = [[float(p[0][0]) * scale, float(p[0][1]) * scale] for p in approx]
                result[cls_name].append({
                    'type': 'polyline',
                    'pts': pts,
                    'area': float(area) * scale * scale,
                })
        
        return result
    
    def visualize(self, image_path, output_path=None):
        """可视化分割结果"""
        mask, _ = self.predict(image_path)
        
        img = cv2.imread(str(image_path))
        overlay = np.zeros_like(img)
        
        colors = Config.CLASS_COLORS
        for cls_id in range(1, Config.NUM_CLASSES):
            overlay[mask == cls_id] = colors[cls_id]
        
        # 叠加
        result = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        
        if output_path:
            cv2.imwrite(str(output_path), result)
        
        return result


# ============================
# 入口
# ============================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='建筑平面图语义分割训练')
    parser.add_argument('--mode', choices=['train', 'predict', 'export'], default='train')
    parser.add_argument('--image', type=str, help='推理时的输入图片路径')
    parser.add_argument('--model', type=str, default='models/best_model.pt', help='模型路径')
    args = parser.parse_args()
    
    if args.mode == 'train':
        trainer = Trainer()
        trainer.train()
    
    elif args.mode == 'predict':
        if not args.image:
            print("请用 --image 指定输入图片")
            sys.exit(1)
        
        predictor = FloorplanPredictor(args.model)
        geom = predictor.predict_to_geometry(args.image)
        
        print(f"\n识别结果:")
        print(f"  墙体: {len(geom['walls'])} 个区域")
        print(f"  窗户: {len(geom['windows'])} 个区域")
        print(f"  门:   {len(geom['doors'])} 个区域")
        
        # 可视化
        out_path = Path(args.image).stem + "_result.png"
        predictor.visualize(args.image, out_path)
        print(f"  可视化: {out_path}")
    
    elif args.mode == 'export':
        # 导出为 ONNX (BIMweb 部署用)
        model = LightUNet(3, Config.NUM_CLASSES)
        ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        
        dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
        onnx_path = str(Config.MODEL_DIR / "floorplan_unet.onnx")
        
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=['image'], output_names=['mask'],
            dynamic_axes={'image': {0: 'batch'}, 'mask': {0: 'batch'}},
            opset_version=13,
        )
        print(f"[✓] ONNX 模型已导出: {onnx_path}")
