# -*- coding: utf-8 -*-
"""
Stage 4: IFC/几何信息 → EnergyPlus IDF 生成
从Stage 3提取的建筑几何直接生成EnergyPlus输入文件

无需安装EnergyPlus即可生成IDF，后续可手动运行仿真
"""
import numpy as np
from pathlib import Path
import json
import datetime

# ============================================================
# 从mask直接提取几何并生成IDF（跳过IFC中间步骤）
# ============================================================

def _default_constructions():
    """默认围护结构材料（GB50176中国居住建筑标准）"""
    return """
!-   ============  MATERIALS  ============

Material,
    Concrete200,             !- Name
    MediumRough,             !- Roughness
    0.200,                   !- Thickness {m}
    1.740,                   !- Conductivity {W/m-K}
    2400,                    !- Density {kg/m3}
    880,                     !- Specific Heat {J/kg-K}
    0.9,                     !- Thermal Absorptance
    0.6,                     !- Solar Absorptance
    0.6;                     !- Visible Absorptance

Material,
    Insulation50,            !- Name
    MediumSmooth,            !- Roughness
    0.050,                   !- Thickness {m}
    0.040,                   !- Conductivity {W/m-K}
    30,                      !- Density {kg/m3}
    1400,                    !- Specific Heat {J/kg-K}
    0.9,                     !- Thermal Absorptance
    0.5,                     !- Solar Absorptance
    0.5;                     !- Visible Absorptance

Material,
    Plaster20,               !- Name
    Smooth,                  !- Roughness
    0.020,                   !- Thickness {m}
    0.810,                   !- Conductivity {W/m-K}
    1600,                    !- Density {kg/m3}
    1050,                    !- Specific Heat {J/kg-K}
    0.9,                     !- Thermal Absorptance
    0.4,                     !- Solar Absorptance
    0.4;                     !- Visible Absorptance

Construction,
    ExteriorWall,            !- Name
    Plaster20,               !- Outside Layer
    Insulation50,            !- Layer 2
    Concrete200,             !- Layer 3
    Plaster20;               !- Layer 4

Construction,
    InteriorWall,            !- Name
    Plaster20,               !- Outside Layer
    Concrete200,             !- Layer 2
    Plaster20;               !- Layer 3

WindowMaterial:SimpleGlazingSystem,
    SimpleWindow,            !- Name
    2.700,                   !- U-Factor {W/m2-K}
    0.400,                   !- Solar Heat Gain Coefficient
    0.600;                   !- Visible Transmittance

Construction,
    WindowConst,             !- Name
    SimpleWindow;            !- Outside Layer

Construction,
    Floor,                   !- Name
    Concrete200;             !- Outside Layer

Construction,
    Roof,                    !- Name
    Plaster20,               !- Outside Layer
    Insulation50,            !- Layer 2
    Concrete200;             !- Layer 3
"""


def _schedules():
    """标准住宅运行时间表 (v23.2 compatible)"""
    return """
!-   ============  SCHEDULES  ============

ScheduleTypeLimits,
    Fraction,                !- Name
    0,                       !- Lower Limit Value
    1,                       !- Upper Limit Value
    CONTINUOUS;              !- Numeric Type

ScheduleTypeLimits,
    ActivityLevel,           !- Name
    0,                       !- Lower Limit Value
    500,                     !- Upper Limit Value
    CONTINUOUS;              !- Numeric Type

ScheduleTypeLimits,
    Temperature,             !- Name
    -100,                    !- Lower Limit Value
    200,                     !- Upper Limit Value
    CONTINUOUS;              !- Numeric Type

ScheduleTypeLimits,
    ControlType,             !- Name
    0,                       !- Lower Limit Value
    4,                       !- Upper Limit Value
    DISCRETE;                !- Numeric Type

Schedule:Compact,
    OccupancySch,            !- Name
    Fraction,                !- Schedule Type Limits Name
    Through: 12/31,          !- Field 1
    For: Weekdays,           !- Field 2
    Until: 7:00, 1.0,
    Until: 9:00, 0.5,
    Until: 17:00, 0.2,
    Until: 22:00, 0.8,
    Until: 24:00, 1.0,
    For: Weekends Holidays,
    Until: 9:00, 1.0,
    Until: 22:00, 0.8,
    Until: 24:00, 1.0;

Schedule:Compact,
    LightingSch,             !- Name
    Fraction,                !- Schedule Type Limits Name
    Through: 12/31,
    For: AllDays,
    Until: 7:00, 0.1,
    Until: 18:00, 0.3,
    Until: 23:00, 0.8,
    Until: 24:00, 0.1;

Schedule:Compact,
    EquipmentSch,            !- Name
    Fraction,                !- Schedule Type Limits Name
    Through: 12/31,
    For: AllDays,
    Until: 7:00, 0.3,
    Until: 9:00, 0.5,
    Until: 17:00, 0.3,
    Until: 22:00, 0.7,
    Until: 24:00, 0.3;

Schedule:Compact,
    ActivitySch,             !- Name
    ActivityLevel,           !- Schedule Type Limits Name
    Through: 12/31,
    For: AllDays,
    Until: 24:00, 120.0;

Schedule:Compact,
    HeatingSP,               !- Name
    Temperature,             !- Schedule Type Limits Name
    Through: 12/31,
    For: AllDays,
    Until: 24:00, 18.0;

Schedule:Compact,
    CoolingSP,               !- Name
    Temperature,             !- Schedule Type Limits Name
    Through: 12/31,
    For: AllDays,
    Until: 24:00, 26.0;

Schedule:Compact,
    AlwaysOn,                !- Name
    Fraction,                !- Schedule Type Limits Name
    Through: 12/31,
    For: AllDays,
    Until: 24:00, 1.0;

Schedule:Compact,
    DualSPControl,           !- Name
    ControlType,             !- Schedule Type Limits Name
    Through: 12/31,
    For: AllDays,
    Until: 24:00, 4;
"""


def generate_idf_from_geometry(walls, windows, doors, rooms,
                                scale=0.01, floor_height=3.0, wall_height=2.8,
                                epw_path=None, output_path='output.idf'):
    """
    从几何信息直接生成EnergyPlus IDF文件
    
    Args:
        walls, windows, doors, rooms: mask_to_ifc.py 提取的几何
        scale: 像素→米
        floor_height: 层高(m)
        wall_height: 墙高(m)
        epw_path: EPW气象文件路径（可选）
        output_path: 输出IDF路径
    """
    idf_lines = []
    
    def add(s):
        idf_lines.append(s)
    
    # === Header ===
    add("!- Generated by FloorPlan2BIM Pipeline")
    add(f"!- Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    add(f"!- Walls: {len(walls)}, Windows: {len(windows)}, Doors: {len(doors)}, Rooms: {len(rooms)}")
    add("")
    
    # === Version ===
    add("Version, 23.2;")
    add("")
    
    # === Simulation Parameters ===
    add("SimulationControl,")
    add("    No,                      !- Do Zone Sizing Calculation")
    add("    No,                      !- Do System Sizing Calculation")
    add("    No,                      !- Do Plant Sizing Calculation")
    add("    Yes,                     !- Run Simulation for Sizing Periods")
    add("    Yes;                     !- Run Simulation for Weather File Run Periods")
    add("")
    
    add("Building,")
    add("    AutoBIM Building,        !- Name")
    add("    0,                       !- North Axis {deg}")
    add("    City,                    !- Terrain")
    add("    0.04,                    !- Loads Convergence Tolerance Value")
    add("    0.4,                     !- Temperature Convergence Tolerance Value {deltaC}")
    add("    FullInteriorAndExterior, !- Solar Distribution")
    add("    25,                      !- Maximum Number of Warmup Days")
    add("    6;                       !- Minimum Number of Warmup Days")
    add("")
    
    add("Timestep, 4;")
    add("")
    
    add("RunPeriod,")
    add("    AnnualRun,               !- Name")
    add("    1,                       !- Begin Month")
    add("    1,                       !- Begin Day of Month")
    add("    ,                        !- Begin Year")
    add("    12,                      !- End Month")
    add("    31,                      !- End Day of Month")
    add("    ,                        !- End Year")
    add("    ,                        !- Day of Week for Start Day")
    add("    Yes,                     !- Use Weather File Holidays and Special Days")
    add("    Yes,                     !- Use Weather File Daylight Saving Period")
    add("    No,                      !- Apply Weekend Holiday Rule")
    add("    Yes,                     !- Use Weather File Rain Indicators")
    add("    Yes;                     !- Use Weather File Snow Indicators")
    add("")
    
    add("GlobalGeometryRules,")
    add("    UpperLeftCorner,         !- Starting Vertex Position")
    add("    Counterclockwise,        !- Vertex Entry Direction")
    add("    Relative;                !- Coordinate System")
    add("")
    
    # === Materials & Constructions ===
    add(_default_constructions())
    add(_schedules())
    
    # === Zones (每个房间一个热区) ===
    zone_names = []
    for r in rooms:
        zname = f"Room_{r['id']}"
        zone_names.append(zname)
        
        cx = r['center'][0] * scale
        cy = r['center'][1] * scale
        rw = r['width'] * scale
        rh = r['height'] * scale
        area = r['area_px'] * scale * scale
        
        add(f"Zone,")
        add(f"    {zname},                !- Name")
        add(f"    0,                       !- Direction of Relative North {{deg}}")
        add(f"    {cx:.3f},               !- X Origin {{m}}")
        add(f"    {cy:.3f},               !- Y Origin {{m}}")
        add(f"    0,                       !- Z Origin {{m}}")
        add(f"    1,                       !- Type")
        add(f"    1,                       !- Multiplier")
        add(f"    autocalculate,           !- Ceiling Height {{m}}")
        add(f"    autocalculate;           !- Volume {{m3}}")
        add("")
        
        # 地板
        x1, y1 = -rw/2, -rh/2
        x2, y2 = rw/2, rh/2
        
        add(f"BuildingSurface:Detailed,")
        add(f"    {zname}_Floor,           !- Name")
        add(f"    Floor,                   !- Surface Type")
        add(f"    Floor,                   !- Construction Name")
        add(f"    {zname},                 !- Zone Name")
        add(f"    ,                        !- Space Name")
        add(f"    Ground,                  !- Outside Boundary Condition")
        add(f"    ,                        !- Outside Boundary Condition Object")
        add(f"    NoSun,                   !- Sun Exposure")
        add(f"    NoWind,                  !- Wind Exposure")
        add(f"    ,                        !- View Factor to Ground")
        add(f"    4,                       !- Number of Vertices")
        add(f"    {x1:.3f},{y1:.3f},0,")
        add(f"    {x1:.3f},{y2:.3f},0,")
        add(f"    {x2:.3f},{y2:.3f},0,")
        add(f"    {x2:.3f},{y1:.3f},0;")
        add("")
        
        # 天花板
        add(f"BuildingSurface:Detailed,")
        add(f"    {zname}_Ceiling,         !- Name")
        add(f"    Ceiling,                 !- Surface Type")
        add(f"    Roof,                    !- Construction Name")
        add(f"    {zname},                 !- Zone Name")
        add(f"    ,                        !- Space Name")
        add(f"    Outdoors,                !- Outside Boundary Condition")
        add(f"    ,                        !- Outside Boundary Condition Object")
        add(f"    SunExposed,              !- Sun Exposure")
        add(f"    WindExposed,             !- Wind Exposure")
        add(f"    ,                        !- View Factor to Ground")
        add(f"    4,                       !- Number of Vertices")
        add(f"    {x1:.3f},{y1:.3f},{wall_height:.1f},")
        add(f"    {x2:.3f},{y1:.3f},{wall_height:.1f},")
        add(f"    {x2:.3f},{y2:.3f},{wall_height:.1f},")
        add(f"    {x1:.3f},{y2:.3f},{wall_height:.1f};")
        add("")
        
        # 4面墙
        for wi, (name, v) in enumerate([
            ('North', [(x1,y2,wall_height),(x2,y2,wall_height),(x2,y2,0),(x1,y2,0)]),
            ('East',  [(x2,y2,wall_height),(x2,y1,wall_height),(x2,y1,0),(x2,y2,0)]),
            ('South', [(x2,y1,wall_height),(x1,y1,wall_height),(x1,y1,0),(x2,y1,0)]),
            ('West',  [(x1,y1,wall_height),(x1,y2,wall_height),(x1,y2,0),(x1,y1,0)]),
        ]):
            add(f"BuildingSurface:Detailed,")
            add(f"    {zname}_Wall_{name},    !- Name")
            add(f"    Wall,                   !- Surface Type")
            add(f"    ExteriorWall,            !- Construction Name")
            add(f"    {zname},                 !- Zone Name")
            add(f"    ,                        !- Space Name")
            add(f"    Outdoors,                !- Outside Boundary Condition")
            add(f"    ,                        !- Outside Boundary Condition Object")
            add(f"    SunExposed,              !- Sun Exposure")
            add(f"    WindExposed,             !- Wind Exposure")
            add(f"    ,                        !- View Factor to Ground")
            add(f"    4,                       !- Number of Vertices")
            verts = ',\n    '.join(f"{p[0]:.3f},{p[1]:.3f},{p[2]:.1f}" for p in v)
            add(f"    {verts};")
            add("")
        
        # 内部负荷 (v23.2: People需要activity_level_schedule_name)
        add(f"People,")
        add(f"    {zname}_People,          !- Name")
        add(f"    {zname},                 !- Zone or ZoneList or Space or SpaceList Name")
        add(f"    OccupancySch,            !- Number of People Schedule Name")
        add(f"    Area/Person,             !- Number of People Calculation Method")
        add(f"    ,                        !- Zone Floor Area per Person {{m2/person}}")
        add(f"    ,                        !- People per Floor Area {{person/m2}}")
        add(f"    30,                      !- Floor Area per Person {{m2/person}}")
        add(f"    0.3,                     !- Fraction Radiant")
        add(f"    autocalculate,           !- Sensible Heat Fraction")
        add(f"    ActivitySch;             !- Activity Level Schedule Name")
        add("")
        
        add(f"Lights,")
        add(f"    {zname}_Lights,          !- Name")
        add(f"    {zname},                 !- Zone Name")
        add(f"    LightingSch,             !- Schedule Name")
        add(f"    Watts/Area,              !- Design Level Calculation Method")
        add(f"    ,                        !- Lighting Level {{W}}")
        add(f"    8.0,                     !- Watts per Zone Floor Area {{W/m2}}")
        add(f"    ,                        !- Watts per Person {{W/person}}")
        add(f"    0,                       !- Return Air Fraction")
        add(f"    0.4,                     !- Fraction Radiant")
        add(f"    0.2;                     !- Fraction Visible")
        add("")
        
        add(f"ElectricEquipment,")
        add(f"    {zname}_Equip,           !- Name")
        add(f"    {zname},                 !- Zone Name")
        add(f"    EquipmentSch,            !- Schedule Name")
        add(f"    Watts/Area,              !- Design Level Calculation Method")
        add(f"    ,                        !- Design Level {{W}}")
        add(f"    10.0,                    !- Watts per Zone Floor Area {{W/m2}}")
        add(f"    ,                        !- Watts per Person {{W/person}}")
        add(f"    0,                       !- Fraction Latent")
        add(f"    0.3,                     !- Fraction Radiant")
        add(f"    0;                       !- Fraction Lost")
        add("")
        
        # HVAC (v23.2: 需要显式EquipmentConnections + 节点名)
        add(f"ZoneHVAC:EquipmentConnections,")
        add(f"    {zname},                 !- Zone Name")
        add(f"    {zname}_EquipList,       !- Zone Conditioning Equipment List Name")
        add(f"    {zname}_SupplyInlet,     !- Zone Air Inlet Node or NodeList Name")
        add(f"    ,                        !- Zone Air Exhaust Node or NodeList Name")
        add(f"    {zname}_ZoneAirNode,     !- Zone Air Node Name")
        add(f"    {zname}_ReturnAirNode;   !- Zone Return Air Node or NodeList Name")
        add("")
        
        add(f"ZoneHVAC:EquipmentList,")
        add(f"    {zname}_EquipList,       !- Name")
        add(f"    SequentialLoad,          !- Load Distribution Scheme")
        add(f"    ZoneHVAC:IdealLoadsAirSystem,  !- Zone Equipment 1 Object Type")
        add(f"    {zname}_IdealHVAC,       !- Zone Equipment 1 Name")
        add(f"    1,                       !- Zone Equipment 1 Cooling Sequence")
        add(f"    1,                       !- Zone Equipment 1 Heating or No-Load Sequence")
        add(f"    ,                        !- Zone Equipment 1 Sequential Cooling Fraction Schedule Name")
        add(f"    ;                        !- Zone Equipment 1 Sequential Heating Fraction Schedule Name")
        add("")
        
        add(f"ZoneHVAC:IdealLoadsAirSystem,")
        add(f"    {zname}_IdealHVAC,       !- Name")
        add(f"    AlwaysOn,                !- Availability Schedule Name")
        add(f"    {zname}_SupplyInlet,     !- Zone Supply Air Node Name")
        add(f"    ,                        !- Zone Exhaust Air Node Name")
        add(f"    ,                        !- System Inlet Air Node Name")
        add(f"    50,                      !- Maximum Heating Supply Air Temperature {{C}}")
        add(f"    13,                      !- Minimum Cooling Supply Air Temperature {{C}}")
        add(f"    0.0156,                  !- Maximum Heating Supply Air Humidity Ratio {{kgWater/kgDryAir}}")
        add(f"    0.0077;                  !- Minimum Cooling Supply Air Humidity Ratio {{kgWater/kgDryAir}}")
        add("")
        
        add(f"ZoneControl:Thermostat,")
        add(f"    {zname}_Thermostat,      !- Name")
        add(f"    {zname},                 !- Zone Name")
        add(f"    DualSPControl,           !- Control Type Schedule Name")
        add(f"    ThermostatSetpoint:DualSetpoint,")
        add(f"    {zname}_DualSP;          !- Control Object Name")
        add("")
        
        add(f"ThermostatSetpoint:DualSetpoint,")
        add(f"    {zname}_DualSP,          !- Name")
        add(f"    HeatingSP,               !- Heating Setpoint Temperature Schedule Name")
        add(f"    CoolingSP;               !- Cooling Setpoint Temperature Schedule Name")
        add("")
    
    # === Output ===
    add("Output:Variable,*,Zone Ideal Loads Supply Air Total Heating Energy,Monthly;")
    add("Output:Variable,*,Zone Ideal Loads Supply Air Total Cooling Energy,Monthly;")
    add("Output:Variable,*,Zone Mean Air Temperature,Monthly;")
    add("OutputControl:Table:Style, HTML;")
    add("Output:Table:SummaryReports, AllSummary;")
    add("")
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(idf_lines))
    
    total_area = sum(r['area_px'] * scale * scale for r in rooms)
    
    return {
        'n_zones': len(rooms),
        'total_floor_area_m2': round(total_area, 1),
        'wall_height': wall_height,
        'output': output_path,
    }


# ============================================================
# 端到端Pipeline入口
# ============================================================

def run_full_pipeline(mask, output_dir='pipeline_output', scale=0.01,
                      floor_height=3.0, wall_height=2.8):
    """
    完整端到端：mask → IFC + IDF
    
    Returns:
        dict with all results and timing
    """
    import time
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from mask_to_ifc import extract_wall_segments, extract_openings, extract_rooms, generate_ifc
    
    results = {'scale': scale, 'floor_height': floor_height}
    
    # Stage 3: 几何提取 + IFC
    t0 = time.time()
    walls = extract_wall_segments(mask)
    windows = extract_openings(mask, 2, walls)
    doors = extract_openings(mask, 3, walls)
    rooms = extract_rooms(mask)
    results['geometry_time'] = round(time.time() - t0, 2)
    
    t1 = time.time()
    ifc_result = generate_ifc(walls, windows, doors, rooms,
                               scale=scale, floor_height=floor_height,
                               wall_height=wall_height,
                               output_path=str(output_dir / 'model.ifc'))
    results['ifc_time'] = round(time.time() - t1, 2)
    results['ifc'] = ifc_result
    
    # Stage 4: IDF生成
    t2 = time.time()
    idf_result = generate_idf_from_geometry(walls, windows, doors, rooms,
                                             scale=scale, floor_height=floor_height,
                                             wall_height=wall_height,
                                             output_path=str(output_dir / 'model.idf'))
    results['idf_time'] = round(time.time() - t2, 2)
    results['idf'] = idf_result
    
    results['total_time'] = round(time.time() - t0, 2)
    
    # 保存摘要
    with open(str(output_dir / 'pipeline_summary.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


if __name__ == '__main__':
    from mask_to_ifc import extract_wall_segments, extract_openings, extract_rooms
    
    base = Path(__file__).parent
    out_dir = base / "output_paper" / "pipeline_demo"
    
    # 合成测试mask
    print("=== Stage 3+4 Pipeline Demo ===")
    mask = np.zeros((512, 512), dtype=np.uint8)
    # 外墙
    mask[50:55, 50:450] = 1
    mask[350:355, 50:450] = 1
    mask[50:355, 50:55] = 1
    mask[50:355, 445:450] = 1
    # 内墙
    mask[50:355, 245:250] = 1
    mask[195:200, 250:450] = 1
    # 窗
    mask[50:55, 120:180] = 2
    mask[50:55, 300:370] = 2
    mask[350:355, 300:370] = 2
    # 门
    mask[220:250, 245:250] = 3
    mask[195:200, 330:360] = 3
    mask[350:355, 130:160] = 3
    
    result = run_full_pipeline(mask, str(out_dir), scale=0.02)
    
    print(f"\n=== Results ===")
    print(f"Geometry extraction: {result['geometry_time']}s")
    print(f"IFC generation:      {result['ifc_time']}s")
    print(f"IDF generation:      {result['idf_time']}s")
    print(f"Total:               {result['total_time']}s")
    print(f"Zones:               {result['idf']['n_zones']}")
    print(f"Floor area:          {result['idf']['total_floor_area_m2']} m2")
    print(f"Output:              {out_dir}")
