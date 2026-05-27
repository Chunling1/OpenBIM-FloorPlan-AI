# -*- coding: utf-8 -*-
"""从 eplustbl.htm 提取5个案例的能耗数据"""
import re, json, os
os.environ['PYTHONIOENCODING'] = 'utf-8'

CASE_DIR = r'C:\Users\chunge\.gemini\antigravity\scratch\floorplan_segmentation\output_paper\case_study'
CASES = ['case_1_verify_0_333','case_2_verify_1_1654','case_3_verify_2_5559','case_4_verify_3_3015','case_5_verify_4_5748']

results = []
for case in CASES:
    htm_path = os.path.join(CASE_DIR, case, 'ep_output', 'eplustbl.htm')
    with open(htm_path, 'r', encoding='utf-8', errors='ignore') as f:
        htm = f.read()
    
    def find_vals_after(label):
        idx = htm.find(label)
        if idx < 0: return []
        chunk = htm[idx:idx+500]
        return [float(n) for n in re.findall(r'>\s*([0-9.]+)\s*<', chunk)]
    
    site = find_vals_after('Total Site Energy')
    area_vals = find_vals_after('Total Building Area')
    cond_area_vals = find_vals_after('Conditioned Building Area')
    
    # End Uses - 从 HTML 表中提取
    # 找 "End Uses" 表，提取 Heating/Cooling 行
    heating_gj = None
    cooling_gj = None
    eu_idx = htm.find('>End Uses<')
    if eu_idx > 0:
        eu_chunk = htm[eu_idx:eu_idx+8000]
        # 找 Heating 行
        h_idx = eu_chunk.find('>Heating<')
        if h_idx > 0:
            h_chunk = eu_chunk[h_idx:h_idx+500]
            h_vals = re.findall(r'>\s*([0-9.]+)\s*<', h_chunk)
            if h_vals:
                heating_gj = float(h_vals[0])
        c_idx = eu_chunk.find('>Cooling<')
        if c_idx > 0:
            c_chunk = eu_chunk[c_idx:c_idx+500]
            c_vals = re.findall(r'>\s*([0-9.]+)\s*<', c_chunk)
            if c_vals:
                cooling_gj = float(c_vals[0])
    
    r = {
        'case': case,
        'total_site_GJ': site[0] if site else None,
        'EUI_MJ_m2': site[1] if len(site)>1 else None,
        'EUI_kWh_m2': round(site[1]/3.6, 1) if len(site)>1 else None,
        'floor_area_m2': area_vals[0] if area_vals else None,
        'conditioned_area_m2': cond_area_vals[0] if cond_area_vals else None,
        'heating_GJ': heating_gj,
        'cooling_GJ': cooling_gj,
    }
    results.append(r)

# 打印
print("=" * 80)
print("EnergyPlus Annual Simulation Results (Chicago TMY3, ASHRAE Zone 5A)")
print("=" * 80)
for r in results:
    print(f"\n{r['case']}:")
    print(f"  Floor Area:     {r['floor_area_m2']} m2")
    print(f"  Total Energy:   {r['total_site_GJ']} GJ")
    print(f"  EUI:            {r['EUI_kWh_m2']} kWh/m2/yr ({r['EUI_MJ_m2']} MJ/m2)")
    print(f"  Heating:        {r['heating_GJ']} GJ")
    print(f"  Cooling:        {r['cooling_GJ']} GJ")

with open(os.path.join(CASE_DIR, 'ep_simulation_results.json'), 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to ep_simulation_results.json")
