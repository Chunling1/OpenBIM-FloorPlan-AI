# -*- coding: utf-8 -*-
"""
Case Study: Full Pipeline Validation
Raster Floor Plan → Segmentation → IFC BIM → EnergyPlus IDF

Runs M2-DA model on test images, extracts geometry, generates IFC and IDF,
and collects all metrics for the paper's Case Study section.
"""
import os
import sys
import time
import json
import numpy as np
from pathlib import Path

# Setup paths
BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
os.chdir(str(BASE))

OUTPUT_DIR = BASE / "output_paper" / "case_study"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Step 1: Find test images
# ============================================================
def find_test_images():
    """Find available floor plan images for case study"""
    candidates = []
    
    # Check data/mask_verification for verification images
    verify_dir = BASE / "data" / "mask_verification"
    if verify_dir.exists():
        for f in verify_dir.glob("*.png"):
            candidates.append(("verify", f))
    
    # Check output directories for any existing floor plan images
    for d in ["output", "output_v2", "output_cad_test", "output_finetune"]:
        dd = BASE / d
        if dd.exists():
            for f in list(dd.glob("*.png"))[:3]:
                candidates.append((d, f))
    
    # Check for standalone test images
    for f in BASE.glob("F*_scaled_result.png"):
        candidates.append(("root", f))
    
    # Check test_commercial directory
    tc = BASE / "test_commercial"
    if tc.exists():
        for f in list(tc.glob("*.png")) + list(tc.glob("*.jpg")):
            candidates.append(("commercial", f))
    
    print(f"Found {len(candidates)} candidate images:")
    for tag, p in candidates:
        sz = p.stat().st_size / 1024
        print(f"  [{tag}] {p.name} ({sz:.0f} KB)")
    
    return candidates


# ============================================================
# Step 2: Run segmentation
# ============================================================
def run_segmentation(image_path, use_dark_preprocessing=False):
    """Run M2-DA segmentation on a single image"""
    import cv2
    from inference_api import FloorplanSegmenter
    
    segmenter = FloorplanSegmenter()
    
    t0 = time.time()
    result = segmenter.predict(str(image_path), use_preprocessing=use_dark_preprocessing)
    seg_time = time.time() - t0
    
    mask = result['mask']
    overlay = result['overlay']
    
    # Compute class statistics
    total_px = mask.size
    stats = {}
    class_names = ['background', 'wall', 'window', 'door']
    for cls_id, name in enumerate(class_names):
        count = (mask == cls_id).sum()
        stats[name] = {
            'pixels': int(count),
            'percent': round(count / total_px * 100, 2)
        }
    
    return {
        'mask': mask,
        'overlay': overlay,
        'seg_time': round(seg_time, 3),
        'image_shape': list(mask.shape),
        'class_stats': stats,
    }


# ============================================================
# Step 3: Geometry extraction + IFC + IDF
# ============================================================
def run_bim_pipeline(mask, case_name, scale=0.02):
    """Extract geometry, generate IFC and IDF"""
    from mask_to_ifc import extract_wall_segments, extract_openings, extract_rooms, generate_ifc
    from ifc_to_energyplus import generate_idf_from_geometry
    
    case_dir = OUTPUT_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    
    # Geometry extraction
    t0 = time.time()
    walls = extract_wall_segments(mask)
    windows = extract_openings(mask, 2, walls)
    doors = extract_openings(mask, 3, walls)
    rooms = extract_rooms(mask)
    geom_time = time.time() - t0
    
    # IFC generation
    t1 = time.time()
    ifc_path = str(case_dir / 'model.ifc')
    ifc_result = generate_ifc(walls, windows, doors, rooms,
                               scale=scale, floor_height=3.0, wall_height=2.8,
                               output_path=ifc_path)
    ifc_time = time.time() - t1
    
    # IDF generation
    t2 = time.time()
    idf_path = str(case_dir / 'model.idf')
    idf_result = generate_idf_from_geometry(walls, windows, doors, rooms,
                                             scale=scale, floor_height=3.0, wall_height=2.8,
                                             output_path=idf_path)
    idf_time = time.time() - t2
    
    # Compute physical dimensions
    total_wall_length_m = sum(w['length'] * scale for w in walls)
    total_floor_area_m2 = sum(r['area_px'] * scale * scale for r in rooms)
    avg_room_area_m2 = total_floor_area_m2 / len(rooms) if rooms else 0
    
    # Wall thickness stats
    wall_thicknesses = [w['thickness'] * scale for w in walls]
    
    return {
        'n_walls': len(walls),
        'n_windows': len(windows),
        'n_doors': len(doors),
        'n_rooms': len(rooms),
        'total_wall_length_m': round(total_wall_length_m, 2),
        'total_floor_area_m2': round(total_floor_area_m2, 2),
        'avg_room_area_m2': round(avg_room_area_m2, 2),
        'wall_thickness_range_m': [round(min(wall_thicknesses), 3), round(max(wall_thicknesses), 3)] if wall_thicknesses else [0, 0],
        'n_ifc_entities': ifc_result['n_entities'],
        'n_thermal_zones': idf_result['n_zones'],
        'geom_time': round(geom_time, 3),
        'ifc_time': round(ifc_time, 3),
        'idf_time': round(idf_time, 3),
        'total_bim_time': round(geom_time + ifc_time + idf_time, 3),
        'ifc_path': ifc_path,
        'idf_path': idf_path,
        'scale': scale,
    }


# ============================================================
# Step 4: Generate visualization
# ============================================================
def save_visualizations(image_path, seg_result, case_name):
    """Save segmentation visualization"""
    import cv2
    
    case_dir = OUTPUT_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    
    # Save overlay
    cv2.imwrite(str(case_dir / 'segmentation_overlay.png'), seg_result['overlay'])
    
    # Save mask as colored image
    mask = seg_result['mask']
    colors = [(40,40,40), (0,0,255), (255,165,0), (0,200,0)]  # bg, wall(red), window(orange), door(green)
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in enumerate(colors):
        vis[mask == cls_id] = color
    cv2.imwrite(str(case_dir / 'segmentation_mask.png'), vis)
    
    # Copy original
    orig = cv2.imread(str(image_path))
    if orig is not None:
        cv2.imwrite(str(case_dir / 'input_image.png'), orig)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("CASE STUDY: Full Pipeline Validation")
    print("Raster Floor Plan -> Segmentation -> IFC BIM -> EnergyPlus IDF")
    print("=" * 70)
    
    # Find images
    candidates = find_test_images()
    
    if not candidates:
        print("\n[!] No test images found. Generating synthetic test case...")
        # Create a realistic synthetic floor plan mask
        run_synthetic_case()
        return
    
    # Run pipeline on each image (limit to 5)
    all_results = []
    
    for i, (tag, img_path) in enumerate(candidates[:5]):
        case_name = f"case_{i+1}_{img_path.stem[:20]}"
        print(f"\n{'='*60}")
        print(f"Case {i+1}: {img_path.name} [{tag}]")
        print(f"{'='*60}")
        
        try:
            # Determine if dark preprocessing needed
            use_dark = (tag in ['cad_test', 'commercial'])
            
            # Stage 1-2: Segmentation
            print(f"\n[Stage 1-2] Running M2-DA segmentation...")
            seg_result = run_segmentation(img_path, use_dark_preprocessing=use_dark)
            print(f"  Segmentation time: {seg_result['seg_time']}s")
            print(f"  Image shape: {seg_result['image_shape']}")
            for cls, stats in seg_result['class_stats'].items():
                print(f"  {cls}: {stats['percent']:.1f}%")
            
            # Save visualizations
            save_visualizations(img_path, seg_result, case_name)
            
            # Stage 3-4: BIM + Energy
            print(f"\n[Stage 3-4] Running BIM pipeline...")
            bim_result = run_bim_pipeline(seg_result['mask'], case_name)
            print(f"  Walls: {bim_result['n_walls']}")
            print(f"  Windows: {bim_result['n_windows']}")
            print(f"  Doors: {bim_result['n_doors']}")
            print(f"  Rooms: {bim_result['n_rooms']}")
            print(f"  Floor area: {bim_result['total_floor_area_m2']} m2")
            print(f"  Thermal zones: {bim_result['n_thermal_zones']}")
            print(f"  BIM pipeline time: {bim_result['total_bim_time']}s")
            
            # Combine results
            case_result = {
                'case_name': case_name,
                'source': tag,
                'image_file': str(img_path.name),
                'segmentation': {
                    'time_s': seg_result['seg_time'],
                    'image_shape': seg_result['image_shape'],
                    'class_stats': seg_result['class_stats'],
                },
                'bim': bim_result,
                'total_pipeline_time_s': round(seg_result['seg_time'] + bim_result['total_bim_time'], 3),
            }
            all_results.append(case_result)
            
            # Save individual case summary
            case_dir = OUTPUT_DIR / case_name
            with open(str(case_dir / 'case_summary.json'), 'w', encoding='utf-8') as f:
                json.dump(case_result, f, indent=2, default=str)
            
            print(f"\n  TOTAL pipeline time: {case_result['total_pipeline_time_s']}s")
            
        except Exception as e:
            print(f"\n[ERROR] Case {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save overall summary
    summary = {
        'n_cases': len(all_results),
        'cases': all_results,
        'aggregate': {
            'avg_seg_time': round(np.mean([r['segmentation']['time_s'] for r in all_results]), 3),
            'avg_bim_time': round(np.mean([r['bim']['total_bim_time'] for r in all_results]), 3),
            'avg_total_time': round(np.mean([r['total_pipeline_time_s'] for r in all_results]), 3),
            'total_walls': sum(r['bim']['n_walls'] for r in all_results),
            'total_windows': sum(r['bim']['n_windows'] for r in all_results),
            'total_doors': sum(r['bim']['n_doors'] for r in all_results),
            'total_rooms': sum(r['bim']['n_rooms'] for r in all_results),
        }
    }
    
    with open(str(OUTPUT_DIR / 'case_study_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print final summary table
    print(f"\n{'='*70}")
    print("CASE STUDY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Case':<30} {'Seg(s)':<8} {'BIM(s)':<8} {'Total(s)':<9} {'Walls':<6} {'Win':<5} {'Door':<5} {'Rooms':<6} {'Area(m2)':<10}")
    print("-" * 97)
    for r in all_results:
        print(f"{r['case_name'][:30]:<30} "
              f"{r['segmentation']['time_s']:<8} "
              f"{r['bim']['total_bim_time']:<8} "
              f"{r['total_pipeline_time_s']:<9} "
              f"{r['bim']['n_walls']:<6} "
              f"{r['bim']['n_windows']:<5} "
              f"{r['bim']['n_doors']:<5} "
              f"{r['bim']['n_rooms']:<6} "
              f"{r['bim']['total_floor_area_m2']:<10}")
    print("-" * 97)
    agg = summary['aggregate']
    print(f"{'AVERAGE':<30} {agg['avg_seg_time']:<8} {agg['avg_bim_time']:<8} {agg['avg_total_time']:<9}")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


def run_synthetic_case():
    """Fallback: run on synthetic mask"""
    print("\n[Synthetic] Creating realistic floor plan mask...")
    mask = np.zeros((512, 512), dtype=np.uint8)
    # Outer walls
    mask[60:65, 60:450] = 1
    mask[390:395, 60:450] = 1
    mask[60:395, 60:65] = 1
    mask[60:395, 445:450] = 1
    # Interior walls
    mask[60:395, 245:250] = 1
    mask[220:225, 60:245] = 1
    mask[220:225, 250:450] = 1
    # Windows
    mask[60:65, 130:190] = 2
    mask[60:65, 310:380] = 2
    mask[390:395, 130:190] = 2
    mask[390:395, 310:380] = 2
    # Doors
    mask[260:290, 245:250] = 3
    mask[220:225, 130:160] = 3
    mask[220:225, 320:350] = 3
    mask[390:395, 250:280] = 3
    
    from mask_to_ifc import extract_wall_segments, extract_openings, extract_rooms, generate_ifc
    from ifc_to_energyplus import generate_idf_from_geometry
    
    case_dir = OUTPUT_DIR / "case_synthetic"
    case_dir.mkdir(parents=True, exist_ok=True)
    
    walls = extract_wall_segments(mask, min_area=50)
    windows = extract_openings(mask, 2, walls)
    doors = extract_openings(mask, 3, walls)
    rooms = extract_rooms(mask, min_area=200)
    
    ifc_result = generate_ifc(walls, windows, doors, rooms,
                               scale=0.02, output_path=str(case_dir / 'model.ifc'))
    idf_result = generate_idf_from_geometry(walls, windows, doors, rooms,
                                             scale=0.02, output_path=str(case_dir / 'model.idf'))
    
    print(f"  Walls: {len(walls)}, Windows: {len(windows)}, Doors: {len(doors)}, Rooms: {len(rooms)}")
    print(f"  Floor area: {idf_result['total_floor_area_m2']} m2")
    print(f"  IFC entities: {ifc_result['n_entities']}")
    print(f"  Thermal zones: {idf_result['n_zones']}")
    print(f"  Output: {case_dir}")


if __name__ == '__main__':
    main()
