"""
BIM 户型图语义分割 - ONNX 部署版
服务器端只需: pip install onnxruntime opencv-python-headless numpy
无需 PyTorch / CUDA

模型: M2_UNet_ResNet34_DA (mIoU = 0.787)
"""

import os
import numpy as np
import cv2
from pathlib import Path

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("[WARN] onnxruntime not installed. Run: pip install onnxruntime")


class FloorplanSegmenterONNX:
    """
    ONNX 推理版本，服务器无需 PyTorch 即可部署。
    与 FloorplanSegmenter (PyTorch) 完全等价的推理管线。
    """

    CLASS_NAMES = ["background", "wall", "window", "door"]
    CLASS_COLORS_BGR = [(40, 40, 40), (60, 76, 231), (219, 152, 52), (113, 204, 46)]
    CLASS_COLORS_HEX = ["#282828", "#e74c3c", "#3498db", "#2ecc71"]

    def __init__(self, model_path=None):
        if not HAS_ONNX:
            raise ImportError("onnxruntime is required. Install: pip install onnxruntime")

        if model_path is None:
            model_path = str(Path(__file__).parent / "M2_DA_best.onnx")

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = 512
        self.num_classes = 4

        actual_provider = self.session.get_providers()[0]
        print(f"[+] FloorplanSegmenterONNX ready | provider={actual_provider}")

    # ============ 预处理 ============

    def remove_annotations(self, img_bgr):
        """移除CAD图中的标注线颜色"""
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        annotation_ranges = [
            ((50, 60, 60), (85, 255, 255)),
            ((155, 60, 60), (175, 255, 255)),
            ((130, 60, 60), (155, 255, 255)),
            ((10, 80, 80), (35, 255, 255)),
            ((0, 120, 100), (10, 255, 255)),
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
        """黑底白线转白底"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        inverted = 255 - gray
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(inverted)
        _, binary = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
        result = cv2.addWeighted(enhanced, 0.4, binary, 0.6, 0)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    # ============ 推理 ============

    def _run_inference(self, img_rgb):
        """ONNX推理核心"""
        h_orig, w_orig = img_rgb.shape[:2]

        # Resize + Normalize (ImageNet)
        resized = cv2.resize(img_rgb, (self.img_size, self.img_size)).astype(np.float32)
        resized = resized / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        resized = (resized - mean) / std

        # NCHW
        tensor = resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        # Run
        output = self.session.run(None, {self.input_name: tensor})[0]  # (1, 4, 512, 512)

        # Softmax
        output = output[0]  # (4, 512, 512)
        exp_out = np.exp(output - output.max(axis=0, keepdims=True))
        probs = exp_out / exp_out.sum(axis=0, keepdims=True)

        # Resize back
        probs_full = np.zeros((self.num_classes, h_orig, w_orig), dtype=np.float32)
        for c in range(self.num_classes):
            probs_full[c] = cv2.resize(probs[c], (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        pred = probs_full.argmax(axis=0).astype(np.uint8)
        return pred, probs_full

    # ============ 后处理 ============

    def postprocess_mask(self, pred, probs, img_shape, annotation_mask=None):
        """去噪、去假墙、门窗增强"""
        h, w = img_shape[:2]
        result = pred.copy()

        # 边缘假墙过滤
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

        # 门窗概率增强
        window_boost = (probs[2] > 0.15) & (result == 0)
        result[window_boost] = 2
        door_boost = (probs[3] > 0.12) & (result == 0)
        result[door_boost] = 3

        return result

    # ============ 端到端推理 ============

    def predict(self, image_input, use_preprocessing=True):
        """
        端到端推理
        image_input: 文件路径(str/Path) 或 BGR numpy array
        返回: dict{mask, overlay, stats, geometry}
        """
        if isinstance(image_input, (str, Path)):
            orig_bgr = cv2.imread(str(image_input))
        else:
            orig_bgr = image_input

        if orig_bgr is None:
            raise ValueError("Cannot read image")

        h_orig, w_orig = orig_bgr.shape[:2]

        if use_preprocessing:
            cleaned_bgr, annot_mask = self.remove_annotations(orig_bgr)
            model_input = self.preprocess_dark_cad(cleaned_bgr)
        else:
            model_input = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            annot_mask = None

        pred_raw, probs = self._run_inference(model_input)

        if use_preprocessing:
            pred_final = self.postprocess_mask(pred_raw, probs, orig_bgr.shape, annot_mask)
        else:
            pred_final = pred_raw

        overlay = self._make_overlay(orig_bgr, pred_final)
        stats = self._compute_stats(pred_final)
        geometry = self._extract_geometry(pred_final)

        return {
            "mask": pred_final,
            "overlay": overlay,
            "stats": stats,
            "geometry": geometry,
            "image_size": [w_orig, h_orig],
        }

    # ============ 可视化 ============

    def _make_overlay(self, img_bgr, pred_mask):
        seg_color = np.zeros_like(img_bgr)
        for cls_id, color in enumerate(self.CLASS_COLORS_BGR):
            seg_color[pred_mask == cls_id] = color
        overlay = img_bgr.copy()
        fg = pred_mask > 0
        overlay[fg] = cv2.addWeighted(img_bgr, 0.4, seg_color, 0.6, 0)[fg]
        for cls_id in [1, 2, 3]:
            cls_mask = (pred_mask == cls_id).astype(np.uint8)
            contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, self.CLASS_COLORS_BGR[cls_id], 2)
        return overlay

    def _compute_stats(self, mask):
        total = mask.size
        stats = {}
        for i, name in enumerate(self.CLASS_NAMES):
            count = int((mask == i).sum())
            stats[name] = {
                "pixels": count,
                "percentage": round(count / total * 100, 2),
                "color": self.CLASS_COLORS_HEX[i],
            }
        return stats

    def _extract_geometry(self, mask, scale=1.0):
        result = {"walls": [], "windows": [], "doors": []}
        class_map = {1: "walls", 2: "windows", 3: "doors"}
        for cls_id, key in class_map.items():
            cls_mask = (mask == cls_id).astype(np.uint8) * 255
            kernel = np.ones((3, 3), np.uint8)
            cls_mask = cv2.morphologyEx(cls_mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 50:
                    continue
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                pts = [[float(p[0][0]) * scale, float(p[0][1]) * scale] for p in approx]
                x, y, bw, bh = cv2.boundingRect(cnt)
                result[key].append({
                    "pts": pts,
                    "area": float(area) * scale * scale,
                    "bbox": [float(x)*scale, float(y)*scale, float(bw)*scale, float(bh)*scale],
                })
        return result


# ============ 与现有 web_server.py 集成的工厂函数 ============

_segmenter_instance = None

def get_segmenter(model_path=None):
    """单例模式获取分割器，避免重复加载模型"""
    global _segmenter_instance
    if _segmenter_instance is None:
        _segmenter_instance = FloorplanSegmenterONNX(model_path)
    return _segmenter_instance


if __name__ == "__main__":
    # Quick test
    seg = FloorplanSegmenterONNX()
    print("Model loaded. Ready for inference.")
