import os
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"
import numpy as np
import cv2
from pathlib import Path
import torch
import segmentation_models_pytorch as smp
import albumentations as A

class FloorplanSegmenter:
    """
    BIM 户型图语义分割部署接口
    模型: M2_UNet_ResNet34_DA (Domain Adapted)
    """
    def __init__(self, model_path=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        if model_path is None:
            base_dir = Path(__file__).parent
            model_path = base_dir / "models" / "M2_DA_FT_v2_best.pt"
            
        self.img_size = 512
        self.num_classes = 4
        self.colors_bgr = [(40,40,40), (60,76,231), (219,152,52), (113,204,46)] # bg, wall, window, door
        
        self.model = self._load_model(model_path).to(self.device)
        self.aug = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def _load_model(self, model_path):
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                         in_channels=3, classes=self.num_classes)
        ckpt = torch.load(str(model_path), map_location='cpu', weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        model.eval()
        return model

    def remove_annotations(self, img_bgr):
        """预处理：移除标注线颜色"""
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        annotation_ranges = [
            ((50, 60, 60), (85, 255, 255)),    # 绿
            ((155, 60, 60), (175, 255, 255)),  # 品红
            ((130, 60, 60), (155, 255, 255)),  # 紫
            ((10, 80, 80), (35, 255, 255)),    # 橙黄
            ((0, 120, 100), (10, 255, 255)),   # 红
            ((170, 120, 100), (180, 255, 255)),
        ]
        
        combined_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        for lo, hi in annotation_ranges:
            mask = cv2.inRange(img_hsv, np.array(lo), np.array(hi))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        
        result = img_bgr.copy()
        result[combined_mask > 0] = 0
        return result, combined_mask

    def preprocess_dark_cad(self, img_bgr):
        """预处理：黑底白线转换增强"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        inverted = 255 - gray
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(inverted)
        _, binary = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
        result = cv2.addWeighted(enhanced, 0.4, binary, 0.6, 0)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    def postprocess_mask(self, pred, probs, img_shape, annotation_mask=None):
        """后处理：去噪、去假墙、门窗增强"""
        h, w = img_shape[:2]
        result = pred.copy()
        
        # 1. 边缘区域假墙过滤
        margin = 0.12
        edge_mask = np.zeros((h, w), dtype=bool)
        mh, mw = int(h * margin), int(w * margin)
        edge_mask[:mh, :] = True
        edge_mask[-mh:, :] = True
        edge_mask[:, :mw] = True
        edge_mask[:, -mw:] = True
        
        wall_mask = (result == 1).astype(np.uint8)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(wall_mask)
        
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            comp_mask = labels == i
            
            edge_ratio = comp_mask[edge_mask].sum() / max(comp_mask.sum(), 1)
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            compactness = area / max(bw * bh, 1)
            
            should_remove = False
            if edge_ratio > 0.7 and area < (h * w * 0.01):
                should_remove = True
            if aspect > 15 and compactness < 0.1:
                should_remove = True
            if annotation_mask is not None:
                annot_overlap = (comp_mask & (annotation_mask > 0)).sum() / max(comp_mask.sum(), 1)
                if annot_overlap > 0.3:
                    should_remove = True
                    
            if should_remove:
                result[comp_mask] = 0
        
        # 2. 门窗增强
        window_boost = (probs[2] > 0.15) & (result == 0)
        result[window_boost] = 2
        door_boost = (probs[3] > 0.12) & (result == 0)
        result[door_boost] = 3
        
        return result

    def predict(self, image_path_or_array, use_preprocessing=True):
        """
        端到端推理接口
        返回: dict包含 pred_mask, probs, overlay可以用于前端展示或SVG生成
        """
        if isinstance(image_path_or_array, (str, Path)):
            orig_bgr = cv2.imread(str(image_path_or_array))
        else:
            orig_bgr = image_path_or_array
            
        h_orig, w_orig = orig_bgr.shape[:2]
        
        if use_preprocessing:
            cleaned_bgr, annot_mask = self.remove_annotations(orig_bgr)
            model_input_img = self.preprocess_dark_cad(cleaned_bgr)  # 变为RGB白底
        else:
            model_input_img = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            annot_mask = None
            
        # 模型推理
        processed = self.aug(image=model_input_img)
        tensor = torch.from_numpy(processed['image'].transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
            
        probs_full = np.zeros((self.num_classes, h_orig, w_orig), dtype=np.float32)
        for c in range(self.num_classes):
            probs_full[c] = cv2.resize(probs[c], (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            
        pred_raw = probs_full.argmax(axis=0).astype(np.uint8)
        
        if use_preprocessing:
            pred_final = self.postprocess_mask(pred_raw, probs_full, orig_bgr.shape, annot_mask)
        else:
            pred_final = pred_raw
            
        return {
            "mask": pred_final,
            "probs": probs_full,
            "overlay": self._make_overlay(orig_bgr, pred_final)
        }
        
    def _make_overlay(self, img_bgr, pred_mask):
        seg_color = np.zeros_like(img_bgr)
        for cls_id, color in enumerate(self.colors_bgr):
            seg_color[pred_mask == cls_id] = color
        overlay = img_bgr.copy()
        fg = pred_mask > 0
        overlay[fg] = cv2.addWeighted(img_bgr, 0.4, seg_color, 0.6, 0)[fg]
        for cls_id in [1, 2, 3]:
            cls_mask = (pred_mask == cls_id).astype(np.uint8)
            contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, self.colors_bgr[cls_id], 2)
        return overlay

if __name__ == "__main__":
    # Test example
    segmenter = FloorplanSegmenter()
    print("[+] FloorplanSegmenter initialized")
