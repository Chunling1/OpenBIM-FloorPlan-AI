"""
消融实验：量化各组件的贡献
1. DA增强 vs 原版模型在验证集上的对比
2. 各后处理步骤的贡献量化
3. 不同预处理策略对CAD图的影响
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "cubicasa5k"
OUTPUT_DIR = BASE_DIR / "output_paper" / "ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 512
NUM_CLASSES = 4
CLASS_NAMES = ["Background", "Wall", "Window", "Door"]

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


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
        output = model(tensor)
        pred = output.argmax(dim=1).squeeze().cpu().numpy()
        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
    return pred, probs


def compute_iou(pred, gt, num_classes):
    ious = []
    for c in range(num_classes):
        pc = (pred == c)
        gc = (gt == c)
        inter = (pc & gc).sum()
        union = (pc | gc).sum()
        if union > 0:
            ious.append(float(inter / union))
        else:
            ious.append(float('nan'))
    valid = [x for x in ious if not np.isnan(x)]
    miou = float(np.mean(valid)) if valid else 0.0
    return ious, miou


# ============ 消融实验1: DA模型 vs 原版模型 ============

def ablation_da_vs_orig():
    """在验证集上对比 M2-DA vs M2-Orig"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = {}
    model_paths = {
        'M2-Orig': BASE_DIR / "models" / "M2_UNet_ResNet34_best.pt",
        'M2-DA': BASE_DIR / "models" / "M2_UNet_ResNet34_DA_best.pt",
    }
    
    for name, path in model_paths.items():
        if path.exists():
            models[name] = load_model(path).to(device)
            print(f"  Loaded {name}")
    
    if len(models) < 2:
        print("  [SKIP] Need both models for comparison")
        return {}
    
    val_file = DATA_DIR / "val.txt"
    all_samples = [l.strip().strip('/') for l in val_file.read_text().strip().split('\n') if l.strip()]
    valid_samples = [s for s in all_samples if (DATA_DIR / s / "mask.npy").exists()]
    print(f"  Val samples: {len(valid_samples)}")
    
    results = {name: {'ious': [], 'mious': []} for name in models}
    
    for i, sample_rel in enumerate(valid_samples):
        sample_dir = DATA_DIR / sample_rel
        pil_img = Image.open(str(sample_dir / "F1_scaled.png")).convert('RGB')
        img_rgb = np.array(pil_img)
        pil_img.close()
        
        gt = np.load(str(sample_dir / "mask.npy"))
        if gt.shape[:2] != img_rgb.shape[:2]:
            gt = cv2.resize(gt, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt = np.clip(gt, 0, 3)
        
        for name, model in models.items():
            pred, _ = predict(model, img_rgb, device)
            pred_full = cv2.resize(pred.astype(np.uint8),
                                    (img_rgb.shape[1], img_rgb.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
            ious, miou = compute_iou(pred_full, gt, NUM_CLASSES)
            results[name]['ious'].append(ious)
            results[name]['mious'].append(miou)
        
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(valid_samples)}")
    
    # 汇总
    summary = {}
    for name in models:
        avg_ious = np.nanmean(results[name]['ious'], axis=0).tolist()
        avg_miou = float(np.mean(results[name]['mious']))
        summary[name] = {
            'mIoU': avg_miou,
            'class_iou': {CLASS_NAMES[i]: avg_ious[i] for i in range(NUM_CLASSES)},
        }
    
    del models
    torch.cuda.empty_cache()
    return summary


# ============ 消融实验2: 后处理各步骤贡献 ============

def ablation_postprocessing():
    """在CAD测试图上测试后处理各步骤的影响"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = BASE_DIR / "models" / "M2_UNet_ResNet34_DA_best.pt"
    model = load_model(model_path).to(device)
    
    cad_dir = BASE_DIR / "output_paper" / "visualizations"
    cad_images = sorted(cad_dir.glob("test_china_cad_*"))
    
    if not cad_images:
        # 也检查test_commercial
        cad_dir = BASE_DIR / "test_commercial"
        cad_images = sorted(cad_dir.glob("*.jpg")) + sorted(cad_dir.glob("*.png"))
    
    if not cad_images:
        print("  [SKIP] No test images found")
        return {}
    
    results = {}
    
    for img_path in cad_images:
        orig_bgr = cv2.imread(str(img_path))
        if orig_bgr is None:
            continue
        
        h, w = orig_bgr.shape[:2]
        
        # --- Step by step ---
        # A: 原图直接推理
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        pred_a, _ = predict(model, orig_rgb, device)
        pred_a = cv2.resize(pred_a.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # B: 去标注 + 推理  
        from test_commercial_buildings import remove_annotations, preprocess_dark_cad, postprocess_mask, predict_with_probs
        cleaned_bgr, annot_mask = remove_annotations(orig_bgr)
        cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
        pred_b, _ = predict(model, cleaned_rgb, device)
        pred_b = cv2.resize(pred_b.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # C: 去标注 + 白底预处理 + 推理
        preprocessed_rgb = preprocess_dark_cad(cleaned_bgr)
        pred_c, probs_c = predict_with_probs(model, preprocessed_rgb, device)
        
        # D: C + 后处理
        pred_d = postprocess_mask(pred_c, probs_c, orig_bgr.shape, annot_mask)
        
        def pixel_stats(pred):
            return {
                'wall_pct': float((pred == 1).sum() / pred.size * 100),
                'window_pct': float((pred == 2).sum() / pred.size * 100),
                'door_pct': float((pred == 3).sum() / pred.size * 100),
            }
        
        results[img_path.name] = {
            'A_raw': pixel_stats(pred_a),
            'B_deannot': pixel_stats(pred_b),
            'C_preprocess': pixel_stats(pred_c),
            'D_postprocess': pixel_stats(pred_d),
        }
    
    del model
    torch.cuda.empty_cache()
    return results


# ============ 消融实验3: 模型规模 vs 性能 ============

def ablation_model_scale():
    """从保存的history JSON中提取三模型的效率分析"""
    results = {}
    model_info = {
        'M1_LightUNet': {'params': '7.8M', 'resolution': '256x256', 'epoch_time_min': 4.1},
        'M2_UNet_ResNet34': {'params': '24.4M', 'resolution': '512x512', 'epoch_time_min': 6.8},
        'M2_UNet_ResNet34_DA': {'params': '24.4M', 'resolution': '512x512', 'epoch_time_min': 9.3},
        'M3_DeepLabV3p_EffB4': {'params': '18.6M', 'resolution': '512x512', 'epoch_time_min': 11.5},
    }
    
    for model_name, info in model_info.items():
        hist_path = BASE_DIR / "output_paper" / f"{model_name}_history.json"
        if not hist_path.exists():
            continue
        
        with open(str(hist_path)) as f:
            hist = json.load(f)
        
        best_ep = hist.get('best_epoch', len(hist['val_miou']))
        best_idx = min(best_ep - 1, len(hist['val_class_iou']) - 1)
        
        results[model_name] = {
            **info,
            'best_mIoU': hist['best_miou'],
            'best_epoch': best_ep,
            'total_epochs': len(hist['val_miou']),
            'class_iou': {CLASS_NAMES[i]: hist['val_class_iou'][best_idx][i] 
                          for i in range(NUM_CLASSES)},
            'convergence_epoch_70': next((e+1 for e, v in enumerate(hist['val_miou']) if v >= 0.70), None),
        }
    
    return results


# ============ 可视化 ============

def plot_ablation_results(da_results, pp_results, scale_results):
    """生成消融实验图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('消融实验结果', fontsize=18, fontweight='bold')
    
    # --- 1. DA vs Orig mIoU ---
    ax = axes[0, 0]
    if da_results:
        models = list(da_results.keys())
        mious = [da_results[m]['mIoU'] for m in models]
        colors = ['#e74c3c', '#3498db']
        bars = ax.bar(models, mious, color=colors[:len(models)], alpha=0.85, width=0.5)
        for bar, val in zip(bars, mious):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.set_ylabel('mIoU', fontsize=12)
        ax.set_title('域自适应 (DA) 消融: 验证集 mIoU', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0.7, 0.85)
    
    # --- 2. DA vs Orig per-class IoU ---
    ax = axes[0, 1]
    if da_results:
        x = np.arange(NUM_CLASSES)
        width = 0.3
        for i, (name, data) in enumerate(da_results.items()):
            vals = [data['class_iou'][c] for c in CLASS_NAMES]
            bars = ax.bar(x + i * width, vals, width, label=name, 
                         color=colors[i], alpha=0.8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylabel('IoU', fontsize=12)
        ax.set_title('域自适应消融: 各类别 IoU', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0.5, 1.0)
    
    # --- 3. 后处理步骤贡献 ---
    ax = axes[1, 0]
    if pp_results:
        steps = ['A_raw', 'B_deannot', 'C_preprocess', 'D_postprocess']
        step_labels = ['Raw', '+DeAnnot', '+WhiteBG', '+PostProc']
        step_colors = ['#95a5a6', '#e67e22', '#3498db', '#2ecc71']
        
        # 取所有图的平均
        avg_wall = [np.mean([pp_results[img][s]['wall_pct'] for img in pp_results]) for s in steps]
        avg_win = [np.mean([pp_results[img][s]['window_pct'] for img in pp_results]) for s in steps]
        avg_door = [np.mean([pp_results[img][s]['door_pct'] for img in pp_results]) for s in steps]
        
        x = np.arange(len(steps))
        width = 0.25
        ax.bar(x - width, avg_wall, width, label='Wall%', color='#e74c3c', alpha=0.8)
        ax.bar(x, avg_win, width, label='Window%', color='#3498db', alpha=0.8)
        ax.bar(x + width, avg_door, width, label='Door%', color='#2ecc71', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(step_labels, fontsize=10)
        ax.set_ylabel('Pixel Ratio (%)', fontsize=12)
        ax.set_title('预处理/后处理管线各步骤效果', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    # --- 4. 模型规模 vs 性能 ---
    ax = axes[1, 1]
    if scale_results:
        names = []
        mious = []
        epoch_times = []
        markers = ['o', 's', 'D', '^']
        colors_m = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
        
        for i, (name, data) in enumerate(scale_results.items()):
            short_name = name.replace('_', '\n')
            names.append(short_name)
            mious.append(data['best_mIoU'])
            epoch_times.append(data['epoch_time_min'])
            
            ax.scatter(data['epoch_time_min'], data['best_mIoU'], 
                      s=200, marker=markers[i], color=colors_m[i], 
                      label=f"{name} ({data['params']})", zorder=5, edgecolors='black')
            ax.annotate(f"{data['best_mIoU']:.4f}", 
                       (data['epoch_time_min'], data['best_mIoU']),
                       textcoords="offset points", xytext=(10, 5), fontsize=9, fontweight='bold')
        
        ax.set_xlabel('每Epoch训练时间 (min)', fontsize=12)
        ax.set_ylabel('Best mIoU', fontsize=12)
        ax.set_title('模型效率-性能权衡', fontsize=14, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "ablation_study.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n[PLOT] Saved: {save_path}")


# ============ Main ============

def main():
    print("=" * 70)
    print("消融实验 (Ablation Study)")
    print("=" * 70)
    
    # 1. DA vs Orig
    print("\n[1/3] DA模型 vs 原版模型 (验证集对比)...")
    t0 = time.time()
    da_results = ablation_da_vs_orig()
    print(f"  Done in {time.time()-t0:.0f}s")
    if da_results:
        print("\n  结果:")
        for name, data in da_results.items():
            cls_str = " | ".join([f"{c}: {data['class_iou'][c]:.4f}" for c in CLASS_NAMES])
            print(f"    {name}: mIoU={data['mIoU']:.4f} | {cls_str}")
    
    # 2. 后处理消融
    print("\n[2/3] 后处理管线消融...")
    pp_results = ablation_postprocessing()
    if pp_results:
        print("\n  结果:")
        for img, steps in pp_results.items():
            print(f"    {img}:")
            for step, stats in steps.items():
                print(f"      {step}: W:{stats['wall_pct']:.1f}% Wi:{stats['window_pct']:.1f}% D:{stats['door_pct']:.1f}%")
    
    # 3. 模型规模分析
    print("\n[3/3] 模型规模-性能分析...")
    scale_results = ablation_model_scale()
    if scale_results:
        print("\n  结果:")
        for name, data in scale_results.items():
            conv = data.get('convergence_epoch_70')
            conv_str = f"ep{conv}" if conv else "N/A"
            print(f"    {name}: mIoU={data['best_mIoU']:.4f} | {data['params']} | {data['epoch_time_min']}min/ep | 70%@{conv_str}")
    
    # 生成可视化
    print("\n生成消融实验可视化...")
    plot_ablation_results(da_results, pp_results, scale_results)
    
    # 保存JSON
    all_results = {
        'da_comparison': da_results,
        'postprocessing_ablation': pp_results,
        'model_scale': scale_results,
    }
    json_path = OUTPUT_DIR / "ablation_results.json"
    with open(str(json_path), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"[JSON] Saved: {json_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
