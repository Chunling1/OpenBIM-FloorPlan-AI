"""
Fine-tune脚本：在中国CAD标注数据上微调DA模型
加载Phase1训练的DA权重，冻结encoder浅层，小学习率微调

用法:
  python finetune_cad.py                          # 使用默认参数
  python finetune_cad.py --epochs 30 --lr 5e-5    # 自定义参数
  python finetune_cad.py --test                   # 在中国CAD测试图上评估
"""
import os
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"
import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
NUM_CLASSES = 4
CLASS_NAMES = ["background", "wall", "window", "door"]
CLASS_WEIGHTS = [0.5, 2.0, 3.0, 3.0]
COLORS_BGR = [(40,40,40), (60,76,231), (219,152,52), (113,204,46)]


# ========== 数据集 ==========

class ChineseCADDataset(Dataset):
    """中国CAD图纸数据集"""
    def __init__(self, img_dir, mask_dir, img_size=512, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.transform = transform

        # 匹配 image ↔ mask
        self.samples = []
        for mask_file in sorted(self.mask_dir.glob("*_mask.npy")):
            stem = mask_file.stem.replace("_mask", "")
            for ext in ['.jpg', '.png', '.jpeg']:
                img_file = self.img_dir / f"{stem}{ext}"
                if img_file.exists():
                    self.samples.append((img_file, mask_file))
                    break

        print(f"[CAD Dataset] {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = np.array(Image.open(str(img_path)).convert('RGB'))
        mask = np.load(str(mask_path))

        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = np.clip(mask, 0, 3)
        mask = torch.from_numpy(mask.astype(np.int64)).long()
        return img, mask


def get_finetune_aug(sz):
    """Fine-tune增强（包含域增强）"""
    return A.Compose([
        A.RandomResizedCrop(size=(sz, sz), scale=(0.7, 1.0), p=0.5),
        A.Resize(sz, sz),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.InvertImg(p=0.3),
        A.ToGray(p=0.2),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ========== 损失函数 ==========

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, class_weights):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('weight', torch.FloatTensor(class_weights))

    def forward(self, pred, target):
        ce = nn.functional.cross_entropy(pred, target, weight=self.weight, reduction='mean')
        ce = torch.clamp(ce, max=50.0)
        pred_soft = torch.softmax(pred, dim=1)
        target_oh = nn.functional.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        inter = (pred_soft * target_oh).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
        dice = 1.0 - (2 * inter + 1e-6) / (union + 1e-6)
        total = ce + dice.mean()
        if torch.isnan(total) or torch.isinf(total):
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return total


# ========== 训练器 ==========

class FineTuner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = 512

        # 模型
        self.model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                              in_channels=3, classes=NUM_CLASSES)

        # 加载DA权重
        da_path = BASE_DIR / "models" / "M2_UNet_ResNet34_DA_best.pt"
        orig_path = BASE_DIR / "models" / "M2_UNet_ResNet34_best.pt"
        weight_path = da_path if da_path.exists() else orig_path
        print(f"Loading weights: {weight_path.name}")
        ckpt = torch.load(str(weight_path), map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model = self.model.to(self.device)

        # 冻结encoder浅层
        frozen_count = 0
        for name, param in self.model.encoder.named_parameters():
            if any(k in name for k in ['conv1', 'bn1', 'layer1', 'layer2']):
                param.requires_grad = False
                frozen_count += 1
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Frozen: {frozen_count} params, Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M")

        # 优化器（极小学习率）
        encoder_params = [p for n, p in self.model.encoder.named_parameters() if p.requires_grad]
        encoder_ids = {id(p) for p in encoder_params}
        decoder_params = [p for p in self.model.parameters() 
                         if p.requires_grad and id(p) not in encoder_ids]
        self.optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': args.encoder_lr},
            {'params': decoder_params, 'lr': args.lr},
        ], weight_decay=1e-4)

        self.criterion = CombinedLoss(NUM_CLASSES, CLASS_WEIGHTS).to(self.device)
        self.best_miou = 0.0

    def train(self):
        cad_dir = Path(self.args.data_dir)
        img_dir = cad_dir / "images"
        mask_dir = cad_dir / "masks"

        if not img_dir.exists() or not mask_dir.exists():
            print(f"错误: 找不到 {img_dir} 或 {mask_dir}")
            print(f"请先运行: python auto_label.py --input {img_dir} --output {mask_dir}")
            return

        dataset = ChineseCADDataset(img_dir, mask_dir, self.img_size, get_finetune_aug(self.img_size))
        if len(dataset) == 0:
            print("没有找到匹配的 image-mask 对！")
            return

        # 80/20 split
        n_val = max(1, len(dataset) // 5)
        n_train = len(dataset) - n_val
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=0)

        print(f"\nTrain: {n_train}, Val: {n_val}")
        print(f"Epochs: {self.args.epochs}, LR: enc={self.args.encoder_lr} dec={self.args.lr}")

        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
        save_path = BASE_DIR / "models" / "M2_UNet_ResNet34_CAD_best.pt"

        for epoch in range(1, self.args.epochs + 1):
            # Train
            self.model.train()
            train_loss = 0
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                out = self.model(imgs)
                loss = self.criterion(out, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Val
            self.model.eval()
            val_loss = 0
            all_ious = []
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(self.device), masks.to(self.device)
                    out = self.model(imgs)
                    loss = self.criterion(out, masks)
                    val_loss += loss.item()
                    pred = out.argmax(dim=1)
                    ious = []
                    for c in range(NUM_CLASSES):
                        inter = ((pred == c) & (masks == c)).sum().float()
                        union = ((pred == c) | (masks == c)).sum().float()
                        ious.append((inter / union).item() if union > 0 else float('nan'))
                    all_ious.append(ious)
            val_loss /= len(val_loader)
            avg_ious = np.nanmean(all_ious, axis=0)
            miou = np.nanmean(avg_ious)

            marker = ""
            if miou > self.best_miou:
                self.best_miou = miou
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'best_miou': self.best_miou,
                }, str(save_path))
                marker = " * BEST"

            print(f"E{epoch:3d}/{self.args.epochs} | L:{train_loss:.4f}/{val_loss:.4f} "
                  f"| mIoU:{miou:.3f}{marker}")
            scheduler.step()

        print(f"\nDone! Best mIoU: {self.best_miou:.4f}")
        print(f"Model saved: {save_path}")


    def test(self):
        """在中国CAD测试图上评估"""
        vis_dir = BASE_DIR / "output_paper" / "visualizations"
        test_images = sorted(vis_dir.glob("test_china_cad_*"))

        if not test_images:
            print("没有找到中国CAD测试图！")
            return

        self.model.eval()
        aug = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"Testing on {len(test_images)} Chinese CAD images...\n")
        for img_path in test_images:
            img_rgb = np.array(Image.open(str(img_path)).convert('RGB'))
            processed = aug(image=img_rgb)
            tensor = torch.from_numpy(processed['image'].transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model(tensor).argmax(dim=1).squeeze().cpu().numpy()

            total = pred.size
            print(f"{img_path.name}: Wall={100*(pred==1).sum()/total:.1f}% "
                  f"Win={100*(pred==2).sum()/total:.1f}% Door={100*(pred==3).sum()/total:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/chinese_cad", help="中国CAD数据目录")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5, help="Decoder学习率")
    parser.add_argument("--encoder-lr", type=float, default=1e-5, help="Encoder学习率")
    parser.add_argument("--test", action="store_true", help="测试模式")
    args = parser.parse_args()

    ft = FineTuner(args)
    if args.test:
        ft.test()
    else:
        ft.train()


if __name__ == "__main__":
    main()
