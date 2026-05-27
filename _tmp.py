import json
with open(r'C:\Users\chunge\.gemini\antigravity\scratch\floorplan_segmentation\output_paper\case_study\case_study_summary.json') as f:
    cs = json.load(f)
for c in cs['cases']:
    area = c['bim']['total_floor_area_m2']
    zones = c['bim']['n_thermal_zones']
    walls = c['bim']['n_walls']
    wins = c['bim']['n_windows']
    doors = c['bim']['n_doors']
    name = c['case_name']
    print(f"{name}: area={area}m2, zones={zones}, walls={walls}, wins={wins}, doors={doors}")
