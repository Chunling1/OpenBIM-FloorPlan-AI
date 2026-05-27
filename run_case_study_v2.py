# -*- coding: utf-8 -*-
"""
Case Study: Pipeline Validation with Realistic Floor Plan Masks
Stage 3-4 only (no cv2/torch dependency)

Creates 3 representative floor plan masks (small/medium/large apartment)
and runs the full BIM + EnergyPlus pipeline to validate end-to-end functionality.
"""
import os
import sys
import time
import json
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
os.chdir(str(BASE))

from mask_to_ifc import extract_wall_segments, extract_openings, extract_rooms, generate_ifc
from ifc_to_energyplus import generate_idf_from_geometry

OUTPUT_DIR = BASE / "output_paper" / "case_study"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_small_apartment(size=512):
    """Case 1: Small 1-bedroom apartment ~45m² (Nordic style, CubiCasa-like)"""
    mask = np.zeros((size, size), dtype=np.uint8)
    T = 6  # wall thickness in pixels
    
    # Outer walls (apartment boundary)
    mask[50:50+T, 50:460] = 1    # top
    mask[410:410+T, 50:460] = 1  # bottom
    mask[50:416, 50:50+T] = 1    # left
    mask[50:416, 454:460] = 1    # right
    
    # Interior wall: bedroom / living room divider (vertical)
    mask[50:416, 250:250+T] = 1
    
    # Interior wall: bathroom partition (horizontal, right side)
    mask[280:280+T, 250:460] = 1
    
    # Interior wall: kitchen nook (horizontal, left side)
    mask[250:250+T, 50:170] = 1
    
    # Windows (on outer walls)
    mask[50:50+T, 100:170] = 2    # top-left: living room window
    mask[50:50+T, 310:400] = 2    # top-right: bedroom window
    mask[410:410+T, 100:180] = 2  # bottom-left: kitchen window
    
    # Doors
    mask[140:170, 250:250+T] = 3  # living room → bedroom
    mask[280:280+T, 330:360] = 3  # bedroom → bathroom
    mask[250:250+T, 90:120] = 3   # living → kitchen
    mask[410:410+T, 370:400] = 3  # main entrance
    
    return mask, {
        'name': 'Small 1BR Apartment',
        'approx_area_m2': 45,
        'rooms_expected': 4,  # living, bedroom, kitchen, bathroom
        'description': 'Typical Nordic 1-bedroom residential unit from CubiCasa5K',
    }


def create_medium_apartment(size=512):
    """Case 2: Medium 2-bedroom apartment ~75m²"""
    mask = np.zeros((size, size), dtype=np.uint8)
    T = 5
    
    # Outer walls
    mask[30:30+T, 30:480] = 1
    mask[450:450+T, 30:480] = 1
    mask[30:455, 30:30+T] = 1
    mask[30:455, 475:480] = 1
    
    # Main corridor wall (horizontal, middle)
    mask[240:240+T, 30:480] = 1
    
    # Bedroom 1 / Bedroom 2 divider (vertical, top half)
    mask[30:240, 260:260+T] = 1
    
    # Living / Kitchen divider (vertical, bottom half)
    mask[240:455, 200:200+T] = 1
    
    # Bathroom partition (vertical, bottom-right)
    mask[350:455, 370:370+T] = 1
    
    # Windows
    mask[30:30+T, 80:160] = 2    # bedroom 1 window
    mask[30:30+T, 310:410] = 2   # bedroom 2 window
    mask[450:450+T, 80:170] = 2  # kitchen window
    mask[450:450+T, 280:360] = 2 # living window
    mask[240:455, 475:480] = 2   # living side window (replace wall section)
    mask[240:350, 475:480] = 2
    
    # Doors
    mask[240:240+T, 100:130] = 3  # corridor → kitchen
    mask[240:240+T, 310:340] = 3  # corridor → living
    mask[130:160, 260:260+T] = 3  # bedroom 1 → bedroom 2 (actually corridor)
    mask[350:380, 370:370+T] = 3  # living → bathroom
    mask[450:450+T, 420:450] = 3  # main entrance
    
    return mask, {
        'name': 'Medium 2BR Apartment',
        'approx_area_m2': 75,
        'rooms_expected': 5,
        'description': 'Standard 2-bedroom residential layout',
    }


def create_large_apartment(size=512):
    """Case 3: Large 3-bedroom apartment ~120m²"""
    mask = np.zeros((size, size), dtype=np.uint8)
    T = 5
    
    # Outer walls
    mask[20:20+T, 20:490] = 1
    mask[470:470+T, 20:490] = 1
    mask[20:475, 20:20+T] = 1
    mask[20:475, 485:490] = 1
    
    # Main horizontal corridor
    mask[230:230+T, 20:490] = 1
    
    # Vertical dividers top: 3 bedrooms
    mask[20:230, 170:170+T] = 1
    mask[20:230, 340:340+T] = 1
    
    # Bottom: living + dining + kitchen + bathroom
    mask[230:475, 160:160+T] = 1   # kitchen divider
    mask[230:475, 330:330+T] = 1   # living/dining divider
    mask[380:380+T, 330:490] = 1   # bathroom partition
    
    # Windows (generous glazing)
    mask[20:20+T, 50:130] = 2      # bedroom 1
    mask[20:20+T, 210:300] = 2     # bedroom 2
    mask[20:20+T, 380:460] = 2     # bedroom 3
    mask[470:470+T, 50:140] = 2    # kitchen
    mask[470:470+T, 200:310] = 2   # dining
    mask[470:470+T, 370:460] = 2   # living
    
    # Doors
    mask[230:230+T, 70:100] = 3    # → bedroom 1
    mask[230:230+T, 240:270] = 3   # → bedroom 2
    mask[230:230+T, 400:430] = 3   # → bedroom 3
    mask[320:350, 160:160+T] = 3   # kitchen → dining
    mask[300:330, 330:330+T] = 3   # dining → living
    mask[380:380+T, 410:440] = 3   # living → bathroom
    mask[470:470+T, 450:475] = 3   # main entrance
    
    return mask, {
        'name': 'Large 3BR Apartment',
        'approx_area_m2': 120,
        'rooms_expected': 7,
        'description': 'Spacious 3-bedroom residential unit with separate living/dining',
    }


def run_pipeline_on_mask(mask, meta, case_id, scale=0.02):
    """Run Stage 3-4 on a single mask"""
    case_name = f"case_{case_id}"
    case_dir = OUTPUT_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Case {case_id}: {meta['name']}")
    print(f"Expected: ~{meta['approx_area_m2']}m², {meta['rooms_expected']} rooms")
    print(f"{'='*60}")
    
    # Save mask visualization
    from PIL import Image
    colors = {0: (40,40,40), 1: (231,76,60), 2: (52,152,219), 3: (46,204,113)}
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in colors.items():
        vis[mask == cls_id] = color
    Image.fromarray(vis).save(str(case_dir / 'segmentation_mask.png'))
    
    # Class statistics
    total_px = mask.size
    class_stats = {}
    for cls_id, name in enumerate(['background', 'wall', 'window', 'door']):
        count = int((mask == cls_id).sum())
        class_stats[name] = {
            'pixels': count,
            'percent': round(count / total_px * 100, 2)
        }
    
    print(f"\nClass distribution:")
    for cls, stats in class_stats.items():
        print(f"  {cls}: {stats['percent']:.1f}%")
    
    # Stage 3: Geometry extraction + IFC
    print(f"\n[Stage 3] Geometry extraction + IFC generation...")
    t0 = time.time()
    walls = extract_wall_segments(mask, min_area=50)
    windows = extract_openings(mask, 2, walls)
    doors = extract_openings(mask, 3, walls)
    rooms = extract_rooms(mask, min_area=200)
    geom_time = time.time() - t0
    
    t1 = time.time()
    ifc_path = str(case_dir / 'model.ifc')
    ifc_result = generate_ifc(walls, windows, doors, rooms,
                               scale=scale, floor_height=3.0, wall_height=2.8,
                               output_path=ifc_path)
    ifc_time = time.time() - t1
    
    # Physical dimensions
    total_wall_length_m = sum(w['length'] * scale for w in walls)
    total_floor_area_m2 = sum(r['area_px'] * scale * scale for r in rooms)
    avg_room_area_m2 = total_floor_area_m2 / len(rooms) if rooms else 0
    wall_thicknesses = [w['thickness'] * scale for w in walls]
    
    print(f"  Walls: {len(walls)} (total length: {total_wall_length_m:.1f}m)")
    print(f"  Windows: {len(windows)}")
    print(f"  Doors: {len(doors)}")
    print(f"  Rooms: {len(rooms)} (total area: {total_floor_area_m2:.1f}m²)")
    print(f"  Avg room area: {avg_room_area_m2:.1f}m²")
    if wall_thicknesses:
        print(f"  Wall thickness: {min(wall_thicknesses)*100:.0f}-{max(wall_thicknesses)*100:.0f}cm")
    print(f"  IFC entities: {ifc_result['n_entities']}")
    print(f"  Geometry time: {geom_time:.3f}s, IFC time: {ifc_time:.3f}s")
    
    # Stage 4: EnergyPlus IDF
    print(f"\n[Stage 4] EnergyPlus IDF generation...")
    t2 = time.time()
    idf_path = str(case_dir / 'model.idf')
    idf_result = generate_idf_from_geometry(walls, windows, doors, rooms,
                                             scale=scale, floor_height=3.0, wall_height=2.8,
                                             output_path=idf_path)
    idf_time = time.time() - t2
    
    print(f"  Thermal zones: {idf_result['n_zones']}")
    print(f"  Floor area: {idf_result['total_floor_area_m2']}m²")
    print(f"  IDF time: {idf_time:.3f}s")
    
    total_time = geom_time + ifc_time + idf_time
    print(f"\n  TOTAL Stage 3+4 time: {total_time:.3f}s")
    
    # Read and check IDF content
    with open(idf_path, 'r') as f:
        idf_content = f.read()
    idf_lines = len(idf_content.split('\n'))
    n_zones = idf_content.count('Zone,')
    n_surfaces = idf_content.count('BuildingSurface:Detailed,')
    n_hvac = idf_content.count('ZoneHVAC:IdealLoadsAirSystem,')
    
    print(f"\n  IDF file: {len(idf_content)} bytes, {idf_lines} lines")
    print(f"  IDF zones: {n_zones}, surfaces: {n_surfaces}, HVAC systems: {n_hvac}")
    
    # Compile result
    result = {
        'case_id': case_id,
        'meta': meta,
        'mask_shape': list(mask.shape),
        'scale_m_per_px': scale,
        'class_stats': class_stats,
        'geometry': {
            'n_walls': len(walls),
            'n_windows': len(windows),
            'n_doors': len(doors),
            'n_rooms': len(rooms),
            'total_wall_length_m': round(total_wall_length_m, 2),
            'total_floor_area_m2': round(total_floor_area_m2, 2),
            'avg_room_area_m2': round(avg_room_area_m2, 2),
            'wall_thickness_range_cm': [round(min(wall_thicknesses)*100, 1), round(max(wall_thicknesses)*100, 1)] if wall_thicknesses else [0,0],
        },
        'ifc': {
            'n_entities': ifc_result['n_entities'],
            'file': ifc_path,
        },
        'idf': {
            'n_zones': idf_result['n_zones'],
            'total_floor_area_m2': idf_result['total_floor_area_m2'],
            'n_lines': idf_lines,
            'n_surfaces': n_surfaces,
            'n_hvac_systems': n_hvac,
            'file': idf_path,
        },
        'timing_s': {
            'geometry_extraction': round(geom_time, 4),
            'ifc_generation': round(ifc_time, 4),
            'idf_generation': round(idf_time, 4),
            'total_stage3_4': round(total_time, 4),
        },
    }
    
    with open(str(case_dir / 'case_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=str)
    
    return result


def main():
    print("=" * 70)
    print("CASE STUDY: End-to-End Pipeline Validation (Stage 3+4)")
    print("Segmentation Mask → IFC BIM → EnergyPlus IDF")
    print("=" * 70)
    
    cases = [
        create_small_apartment,
        create_medium_apartment,
        create_large_apartment,
    ]
    
    all_results = []
    for i, create_fn in enumerate(cases, 1):
        mask, meta = create_fn()
        result = run_pipeline_on_mask(mask, meta, i)
        all_results.append(result)
    
    # Summary table
    print(f"\n{'='*90}")
    print("CASE STUDY SUMMARY TABLE (for paper)")
    print(f"{'='*90}")
    print(f"{'Case':<25} {'Walls':<6} {'Win':<5} {'Door':<5} {'Rooms':<6} {'Area(m²)':<10} "
          f"{'IFC Ent.':<9} {'Zones':<6} {'Time(s)':<8}")
    print("-" * 90)
    for r in all_results:
        g = r['geometry']
        print(f"{r['meta']['name']:<25} "
              f"{g['n_walls']:<6} {g['n_windows']:<5} {g['n_doors']:<5} "
              f"{g['n_rooms']:<6} {g['total_floor_area_m2']:<10} "
              f"{r['ifc']['n_entities']:<9} {r['idf']['n_zones']:<6} "
              f"{r['timing_s']['total_stage3_4']:<8}")
    print("-" * 90)
    
    # Aggregate
    avg_time = np.mean([r['timing_s']['total_stage3_4'] for r in all_results])
    total_walls = sum(r['geometry']['n_walls'] for r in all_results)
    total_entities = sum(r['ifc']['n_entities'] for r in all_results)
    
    print(f"\nAvg Stage 3+4 time: {avg_time:.4f}s")
    print(f"Total walls processed: {total_walls}")
    print(f"Total IFC entities generated: {total_entities}")
    
    # Save summary
    summary = {
        'n_cases': len(all_results),
        'cases': all_results,
        'aggregate': {
            'avg_stage34_time_s': round(avg_time, 4),
            'total_walls': total_walls,
            'total_windows': sum(r['geometry']['n_windows'] for r in all_results),
            'total_doors': sum(r['geometry']['n_doors'] for r in all_results),
            'total_rooms': sum(r['geometry']['n_rooms'] for r in all_results),
            'total_ifc_entities': total_entities,
        }
    }
    
    with open(str(OUTPUT_DIR / 'case_study_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"\nGenerated files per case:")
    for r in all_results:
        case_dir = OUTPUT_DIR / f"case_{r['case_id']}"
        print(f"  {r['meta']['name']}:")
        for f in sorted(case_dir.iterdir()):
            print(f"    {f.name} ({f.stat().st_size/1024:.1f} KB)")


if __name__ == '__main__':
    main()
