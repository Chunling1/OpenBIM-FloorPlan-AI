"""
论文级基准评测脚本 - benchmark_eval.py
在 CubiCasa5K 官方 test split (400张) 上评测所有模型
输出论文 Table 1 (方法对比) + Table 2 (消融实验) + LaTeX 格式

用法:
  python benchmark_eval.py              # 全量评测 (需 ~20min, GPU推荐)
  python benchmark_eval.py --quick      # 只跑 val 前100张快速确认
  python benchmark_eval.py --ablation   # 仅跑 M2-DA 消融实验
  python benchmark_eval.py --results    # 只打印已有结果 (不重新跑)
"""

import os, sys, json, time, warnings, argparse
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings('ignore')

import numpy as np
import cv2
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A

# ────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data" / "cubicasa5k"
MODEL_DIR = BASE_DIR / "models"
OUT_DIR   = BASE_DIR / "output_paper"
OUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["Background", "Wall", "Window", "Door"]
NUM_CLASSES = 4
IMG_SIZE_M1 = 256   # LightUNet 用 256
IMG_SIZE_MX = 512   # M2/M3 用 512

# ────────────────────────────────────────────
# LightUNet (M1) — 必须和训练时完全一致
# ────────────────────────────────────────────
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
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2, 2)
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
            x = down(x); skips.append(x); x = self.pool(x)
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


# ────────────────────────────────────────────
# 模型加载
# ────────────────────────────────────────────
def load_model(model_key: str, weight_path: Path, device):
    """model_key: 'm1' | 'm2' | 'm3'"""
    if model_key == 'm1':
        model = LightUNet(3, NUM_CLASSES)
    elif model_key == 'm2':
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                         in_channels=3, classes=NUM_CLASSES)
    elif model_key == 'm3':
        model = smp.DeepLabV3Plus(encoder_name="efficientnet-b4",
                                  encoder_weights=None,
                                  in_channels=3, classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    ckpt = torch.load(str(weight_path), map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    return model.eval().to(device)


# ────────────────────────────────────────────
# 预处理选项
# ────────────────────────────────────────────
def get_aug(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def remove_annotations(img_bgr):
    """移除彩色标注线（CubiCasa 图纸特有）"""
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ranges = [
        ((50, 60, 60), (85, 255, 255)),
        ((155, 60, 60), (175, 255, 255)),
        ((130, 60, 60), (155, 255, 255)),
        ((10, 80, 80), (35, 255, 255)),
        ((0, 120, 100), (10, 255, 255)),
        ((170, 120, 100), (180, 255, 255)),
    ]
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(img_hsv, np.array(lo), np.array(hi)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, k, iterations=1)
    result = img_bgr.copy(); result[mask > 0] = 0
    return result, mask

def postprocess_mask(pred, probs, img_shape, annot_mask=None):
    """边缘假墙过滤 + 门窗增强"""
    h, w = img_shape[:2]
    result = pred.copy()

    margin = 0.12
    mh, mw = int(h * margin), int(w * margin)
    edge = np.zeros((h, w), dtype=bool)
    edge[:mh, :] = edge[-mh:, :] = edge[:, :mw] = edge[:, -mw:] = True

    wall = (result == 1).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(wall)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        comp = labels == i
        edge_ratio = comp[edge].sum() / max(comp.sum(), 1)
        bw_s = stats[i, cv2.CC_STAT_WIDTH]; bh_s = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(bw_s, bh_s) / max(min(bw_s, bh_s), 1)
        compact = area / max(bw_s * bh_s, 1)
        rm = False
        if edge_ratio > 0.7 and area < (h * w * 0.01): rm = True
        if aspect > 15 and compact < 0.1: rm = True
        if annot_mask is not None:
            if (comp & (annot_mask > 0)).sum() / max(comp.sum(), 1) > 0.3: rm = True
        if rm: result[comp] = 0

    result[(probs[2] > 0.15) & (result == 0)] = 2
    result[(probs[3] > 0.12) & (result == 0)] = 3
    return result


# ────────────────────────────────────────────
# 单张图推理
# ────────────────────────────────────────────
def predict_one(model, img_rgb, img_size, device,
                use_preproc=False, use_postproc=False):
    aug = get_aug(img_size)
    h_orig, w_orig = img_rgb.shape[:2]
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    annot_mask = None
    if use_preproc:
        cleaned, annot_mask = remove_annotations(img_bgr)
        input_img = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
    else:
        input_img = img_rgb

    t = aug(image=input_img)
    tensor = torch.from_numpy(t['image'].transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        probs_512 = torch.softmax(out, dim=1).squeeze().cpu().numpy()

    # 还原到原始分辨率
    probs_full = np.stack([
        cv2.resize(probs_512[c], (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        for c in range(NUM_CLASSES)
    ])
    pred = probs_full.argmax(axis=0).astype(np.uint8)

    if use_postproc:
        pred = postprocess_mask(pred, probs_full, img_rgb.shape, annot_mask)

    return pred


# ────────────────────────────────────────────
# mIoU 计算
# ────────────────────────────────────────────
def compute_miou(pred, gt):
    ious = []
    for c in range(NUM_CLASSES):
        pc = pred == c; gc = gt == c
        inter = (pc & gc).sum(); union = (pc | gc).sum()
        ious.append(float(inter / union) if union > 0 else float('nan'))
    valid = [x for x in ious if not np.isnan(x)]
    return ious, float(np.mean(valid)) if valid else 0.0


# ────────────────────────────────────────────
# 评估整个 split
# ────────────────────────────────────────────
def eval_split(model, img_size, device, split='test',
               use_preproc=False, use_postproc=False,
               max_samples=None, verbose=True):

    split_file = DATA_DIR / f"{split}.txt"
    samples = [l.strip().strip('/') for l in split_file.read_text().strip().split('\n') if l.strip()]
    valid   = [s for s in samples if (DATA_DIR / s / "mask.npy").exists()]
    if max_samples:
        valid = valid[:max_samples]

    all_ious, all_m = [], []
    t0 = time.time()

    for i, rel in enumerate(valid):
        sd = DATA_DIR / rel
        pil = Image.open(str(sd / "F1_scaled.png")).convert('RGB')
        img = np.array(pil); pil.close()
        gt  = np.load(str(sd / "mask.npy"))
        if gt.shape[:2] != img.shape[:2]:
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
        gt = np.clip(gt, 0, 3)

        pred = predict_one(model, img, img_size, device, use_preproc, use_postproc)
        ious, m = compute_miou(pred, gt)
        all_ious.append(ious); all_m.append(m)

        if verbose and (i + 1) % 50 == 0:
            running = float(np.mean(all_m))
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(valid) - i - 1)
            print(f"    {i+1}/{len(valid)} | running mIoU={running:.4f} | ETA {eta:.0f}s")

    avg_ious = np.nanmean(all_ious, axis=0).tolist()
    avg_m    = float(np.mean(all_m))
    elapsed  = time.time() - t0
    return avg_m, avg_ious, len(valid), elapsed


# ────────────────────────────────────────────
# 打印 / LaTeX 输出
# ────────────────────────────────────────────
def fmt(v, bold=False):
    s = f"{v:.4f}" if not np.isnan(v) else "  —  "
    return f"\\textbf{{{s}}}" if bold else s

def print_table(title, rows):
    """rows: list of (name, split, miou, bg, wall, win, door)"""
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"{'Method':<30} {'Split':<6} {'mIoU':>8} {'BG':>8} {'Wall':>8} {'Win':>8} {'Door':>8}")
    print("-"*72)
    for r in rows:
        name, split, m, bg, wall, win, door = r
        print(f"{name:<30} {split:<6} {m:8.4f} {bg:8.4f} {wall:8.4f} {win:8.4f} {door:8.4f}")

def print_latex(title, rows, best_miou=None):
    print(f"\n% ─── LaTeX Table: {title} ───")
    print(r"\begin{tabular}{lccccccc}")
    print(r"\hline")
    print(r"Method & Split & mIoU & Background & Wall & Window & Door \\")
    print(r"\hline")
    for r in rows:
        name, split, m, bg, wall, win, door = r
        bold = (best_miou is not None and abs(m - best_miou) < 1e-5)
        vals = [fmt(x, bold and (x == m)) for x in [m, bg, wall, win, door]]
        print(f"{name} & {split} & {' & '.join(vals)} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")


# ────────────────────────────────────────────
# 主逻辑
# ────────────────────────────────────────────
def run_benchmark(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[device] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))

    split    = 'val' if args.quick else 'test'
    max_samp = 100  if args.quick else None

    # ── 已有结果文件判断 ──
    result_path = OUT_DIR / "benchmark_results.json"
    if args.results and result_path.exists():
        with open(result_path) as f:
            all_results = json.load(f)
        _report(all_results)
        return
    
    all_results = {}

    # ══════════════════════════════════════════════
    # TABLE 1: 主要方法对比
    # ══════════════════════════════════════════════
    print("\n\n" + "="*60)
    print("  TABLE 1 — 主要模型对比 (on CubiCasa5K)")
    print("="*60)

    models_cfg = [
        # (label,            model_key, weight_file,                          img_sz, preproc, postproc)
        ("M1-LightUNet",    'm1',      "M1_LightUNet_best.pt",                256,   False,   False),
        ("M2-UNet-R34",     'm2',      "M2_UNet_ResNet34_best.pt",            512,   False,   False),
        ("M3-DeepLabV3+",   'm3',      "M3_DeepLabV3p_EffB4_best.pt",         512,   False,   False),
        ("M2-DA (Ours)",    'm2',      "M2_UNet_ResNet34_DA_best.pt",         512,   False,   False),
        ("M2-DA+PP (Ours)", 'm2',      "M2_UNet_ResNet34_DA_best.pt",         512,   True,    True),
    ]

    if args.ablation:
        models_cfg = models_cfg[3:]  # 只跑 M2-DA 部分

    table1_rows = []
    for label, mkey, wfile, sz, preproc, postproc in models_cfg:
        wpath = MODEL_DIR / wfile
        if not wpath.exists():
            print(f"  [SKIP] {label}: {wfile} not found")
            continue

        print(f"\n── {label} {'(preproc)' if preproc else ''} {'(postproc)' if postproc else ''} ──")
        model = load_model(mkey, wpath, device)
        m, ious, n, elapsed = eval_split(model, sz, device, split, preproc, postproc, max_samp)
        del model; torch.cuda.empty_cache() if device.type=='cuda' else None

        row_key = label
        all_results[row_key] = {
            "split": split, "mIoU": m,
            "class_iou": {CLASS_NAMES[i]: ious[i] for i in range(NUM_CLASSES)},
            "n_samples": n, "elapsed_sec": round(elapsed, 1),
            "preproc": preproc, "postproc": postproc,
        }
        table1_rows.append((label, split, m, *ious))
        print(f"  mIoU={m:.4f}  BG={ious[0]:.4f}  Wall={ious[1]:.4f}  Win={ious[2]:.4f}  Door={ious[3]:.4f}  ({elapsed:.0f}s, {n} samples)")

    # ══════════════════════════════════════════════
    # TABLE 2: 消融实验 (M2-DA 各组件)
    # ══════════════════════════════════════════════
    if not args.ablation or args.ablation:  # always run ablation for M2-DA
        print("\n\n" + "="*60)
        print("  TABLE 2 — 消融实验 (M2-DA 组件贡献)")
        print("="*60)

        ablation_cfg = [
            # (label,                         preproc, postproc)
            ("M2-DA (no preproc, no postproc)", False,  False),
            ("M2-DA + Preproc only",            True,   False),
            ("M2-DA + Postproc only",           False,  True),
            ("M2-DA + Preproc + Postproc",      True,   True),
        ]

        wpath = MODEL_DIR / "M2_UNet_ResNet34_DA_best.pt"
        table2_rows = []

        if wpath.exists():
            for label, preproc, postproc in ablation_cfg:
                if label in all_results:
                    r = all_results[label]
                    ious = [r["class_iou"][c] for c in CLASS_NAMES]
                    row = (label, split, r["mIoU"], *ious)
                else:
                    print(f"\n  ablation: {label}")
                    model = load_model('m2', wpath, device)
                    m, ious, n, elapsed = eval_split(model, 512, device, split, preproc, postproc, max_samp)
                    del model; torch.cuda.empty_cache() if device.type=='cuda' else None
                    all_results[label] = {
                        "split": split, "mIoU": m,
                        "class_iou": {CLASS_NAMES[i]: ious[i] for i in range(NUM_CLASSES)},
                        "n_samples": n, "elapsed_sec": round(elapsed, 1),
                        "preproc": preproc, "postproc": postproc,
                    }
                    row = (label, split, m, *ious)
                    print(f"  mIoU={m:.4f}  Wall={ious[1]:.4f}  Win={ious[2]:.4f}  Door={ious[3]:.4f}")
                table2_rows.append(row)
        else:
            print("  [SKIP] M2_DA weight not found")
            table2_rows = []

    # ── Save all results ──
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {result_path}")

    _report(all_results, table1_rows if 'table1_rows' in dir() else [],
            table2_rows if 'table2_rows' in dir() else [])


def _report(all_results, table1_rows=None, table2_rows=None):
    """打印最终报告 + LaTeX"""

    # 从 all_results 还原 rows (--results 模式)
    if table1_rows is None:
        table1_rows = []
        for k, v in all_results.items():
            ious = [v["class_iou"].get(c, float('nan')) for c in CLASS_NAMES]
            table1_rows.append((k, v["split"], v["mIoU"], *ious))

    # ── 加入 published baseline ──
    published = [
        ("CubiCasa (Kalervo 2019)*", "test", 0.5750, float('nan'), 0.6500, 0.5200, 0.4800),
        ("Zeng et al. (ICCV 2019)*", "test", float('nan'), float('nan'), 0.7270, float('nan'), float('nan')),
    ]

    all_rows = published + table1_rows

    # 找最高 mIoU
    valid_miou = [r[2] for r in table1_rows if not np.isnan(r[2])]
    best_m = max(valid_miou) if valid_miou else None

    print_table("TABLE 1: Method Comparison on CubiCasa5K", all_rows)
    print("\n  * Published results from original papers (not re-evaluated)")

    if table2_rows:
        print_table("TABLE 2: Ablation Study (M2-DA Components)", table2_rows)

    # ── Paper-ready summary ──
    print("\n\n" + "="*60)
    print("  PAPER-READY SUMMARY")
    print("="*60)

    our_best = next((r for r in table1_rows if "PP" in r[0] or "M2-DA (Ours)" in r[0]), None)
    if our_best:
        _, _, m, bg, wall, win, door = our_best
        print(f"\n  Our best (M2-DA + Postproc):")
        print(f"    mIoU  = {m:.4f}  (+{m-0.5750:.4f} vs CubiCasa baseline)")
        print(f"    Wall  = {wall:.4f}")
        print(f"    Window= {win:.4f}")
        print(f"    Door  = {door:.4f}")
        print(f"    Background= {bg:.4f}")

    # ── LaTeX ──
    print_latex("Method Comparison", all_rows, best_m)
    if table2_rows:
        print_latex("Ablation Study", table2_rows)

    print("\n[Done] Results saved to output_paper/benchmark_results.json")


# ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick',    action='store_true', help='快速模式：仅 val set 前100张')
    parser.add_argument('--ablation', action='store_true', help='只跑 M2-DA 消融实验')
    parser.add_argument('--results',  action='store_true', help='仅打印已有结果，不重新推理')
    args = parser.parse_args()
    run_benchmark(args)
