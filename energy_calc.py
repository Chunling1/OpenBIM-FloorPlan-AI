# -*- coding: utf-8 -*-
"""
建筑能耗模拟计算核心引擎
"""

import math

CLIMATE_DATA = {
    "harbin": {"city": "哈尔滨", "zone": "严寒 C 区", "hdd18": 4500, "cdd26": 80},
    "beijing": {"city": "北京", "zone": "寒冷 B 区", "hdd18": 2800, "cdd26": 180},
    "shanghai": {"city": "上海", "zone": "夏热冬冷 A 区", "hdd18": 1500, "cdd26": 320},
    "guangzhou": {"city": "广州", "zone": "夏热冬暖 A 区", "hdd18": 150, "cdd26": 750},
    "kunming": {"city": "昆明", "zone": "温和 A 区", "hdd18": 800, "cdd26": 20}
}

COP_HEATING = {
    "heat_pump_air": 3.2,
    "gas_boiler": 0.85,
    "electric": 0.98,
    "district_heating": 0.85
}

COP_COOLING = {
    "chiller": 4.5,
    "vrv": 3.8,
    "split": 3.2
}

def calculate_building_energy(params, geometry_data):
    """
    能耗核心计算方法
    params: dict, 前台表单传入的参数
    geometry_data: dict, AI 识别出的几何信息 (或回填的历史几何数据)
    """
    # 气候特征
    city_id = params.get("city_id", "beijing")
    climate = CLIMATE_DATA.get(city_id, CLIMATE_DATA["beijing"])
    hdd18 = climate["hdd18"]
    cdd26 = climate["cdd26"]
    
    # 几何基本量
    floor_area = float(params.get("floor_area_m2", 120))
    height = float(params.get("height", 3.2))
    floors = int(params.get("floors", 1))
    total_floor_area = floor_area * floors
    
    # AI几何提取的物理面积换算 (如果物理长度没传，根据默认公式估算)
    # 墙表面积 = 墙周长 * 高度
    wall_length = float(params.get("wall_length_m", 0))
    if wall_length <= 0:
        # 如果前台未上传图纸，估算周长
        wall_length = math.sqrt(floor_area) * 4
        
    window_length = float(params.get("window_length_m", 0))
    if window_length <= 0:
        window_length = wall_length * 0.25 # 估算窗长
        
    wall_area = wall_length * height * floors
    window_area = window_length * height * floors
    
    # 传热系数 U-value
    u_wall = float(params.get("u_wall", 0.55))
    u_win = float(params.get("u_win", 2.4))
    u_roof = float(params.get("u_roof", 0.38))
    u_floor = float(params.get("u_floor", 0.30))
    shgc = float(params.get("shgc", 0.42))
    
    # 系统能效 COP
    heating_system = params.get("heating_system", "heat_pump_air")
    cop_h = COP_HEATING.get(heating_system, 3.0)
    cooling_system = params.get("cooling_system", "vrv")
    cop_c = COP_COOLING.get(cooling_system, 3.5)
    
    # 设备与运行特征
    lpd = float(params.get("lpd", 9.0))
    epd = float(params.get("epd", 15.0))
    ach = float(params.get("ach", 1.0))
    dhw_liter = float(params.get("dhw_liter_pp", 5.0))
    op_hours = float(params.get("op_hours", 2500))
    
    # 1. 渗透/新风系数
    c_inf = 1.0 + 0.33 * ach
    
    # 2. 冬季供暖能耗 (kWh)
    # 热负荷 = (墙传热 + 窗传热 + 屋顶传热 + 地板传热) * HDD * 24 * 渗透系数
    q_envelope_h = (u_wall * wall_area + u_win * window_area + u_roof * floor_area + u_floor * floor_area) * 24 * hdd18 * c_inf / 1000.0
    heating_energy = q_envelope_h / cop_h
    
    # 3. 夏季制冷能耗 (kWh)
    # 冷负荷 = (包络结构热侵入 + 窗户日照得热 + 内部得热)
    # 太阳辐射累积能耗估计为 180 kWh/m²a
    q_envelope_c = (u_wall * wall_area + u_win * window_area + u_roof * floor_area + u_floor * floor_area) * 24 * cdd26 * c_inf / 1000.0
    q_solar = window_area * shgc * 180.0
    # 内部得热 (照明 + 设备 + 人体得热 50W/人，每 15平米 1人)
    people_density = total_floor_area / 15.0
    q_internal = ((lpd + epd) * total_floor_area + people_density * 50) * op_hours / 1000.0
    
    cooling_energy = (q_envelope_c + q_solar + q_internal * 0.4) / cop_c
    
    # 4. 室内照明能耗 (kWh)
    lighting_energy = lpd * total_floor_area * op_hours / 1000.0
    
    # 5. 机电插座能耗 (kWh)
    equipment_energy = epd * total_floor_area * op_hours / 1000.0
    
    # 6. 新风通风能耗 (风机功耗估算)
    # 新风风量 = 换气次数 * 建筑体积
    vent_volume = total_floor_area * height * ach
    # 假定风机功耗为 0.35 W/(m³/h)
    ventilation_energy = (vent_volume * 0.35) * op_hours / 1000.0
    
    # 7. 生活热水能耗 (DHW)
    # 热水人数每 20 平米 1 人
    dhw_people = max(1.0, total_floor_area / 20.0)
    # 热水温升 (60度 - 15度 = 45度)，热容 4.187 kJ/(kg.K)
    q_dhw = (365.0 * dhw_liter * dhw_people * 4.187 * 45.0) / 3600.0 # kWh
    dhw_energy = q_dhw / 0.90 # 假设电热水器/热泵平均制热效率 0.9
    
    # 总能耗汇总
    total_energy = heating_energy + cooling_energy + lighting_energy + equipment_energy + ventilation_energy + dhw_energy
    eui = total_energy / total_floor_area
    
    # 能效等级评定
    # 根据建筑性质和 EUI 分级
    btype = params.get("building_type", "office")
    rating = "C"
    # 建立分级阈值字典
    rating_thresholds = {
        "office": [35.0, 60.0, 95.0, 130.0],
        "commercial": [50.0, 90.0, 140.0, 190.0],
        "hotel": [60.0, 110.0, 160.0, 220.0],
        "residential": [20.0, 40.0, 70.0, 100.0],
        "school": [25.0, 45.0, 75.0, 110.0]
    }
    
    thresholds = rating_thresholds.get(btype, rating_thresholds["office"])
    if eui < thresholds[0]:
        rating = "A"
    elif eui < thresholds[1]:
        rating = "B"
    elif eui < thresholds[2]:
        rating = "C"
    elif eui < thresholds[3]:
        rating = "D"
    else:
        rating = "E"
        
    rating_labels = {
        "A": "超低能耗",
        "B": "低能耗",
        "C": "国家节能",
        "D": "一般能效",
        "E": "高能耗"
    }
    
    return {
        "summary": {
            "total_energy_kwh": round(total_energy, 1),
            "eui": round(eui, 2),
            "rating": rating,
            "rating_label": rating_labels[rating],
        },
        "breakdown_kwh": {
            "heating": round(heating_energy, 1),
            "cooling": round(cooling_energy, 1),
            "lighting": round(lighting_energy, 1),
            "equipment": round(equipment_energy, 1),
            "ventilation": round(ventilation_energy, 1),
            "dhw": round(dhw_energy, 1)
        },
        "breakdown_eui": {
            "heating": round(heating_energy / total_floor_area, 2),
            "cooling": round(cooling_energy / total_floor_area, 2),
            "lighting": round(lighting_energy / total_floor_area, 2),
            "equipment": round(equipment_energy / total_floor_area, 2),
            "ventilation": round(ventilation_energy / total_floor_area, 2),
            "dhw": round(dhw_energy / total_floor_area, 2)
        },
        "geometry_used": {
            "total_area_m2": round(total_floor_area, 1),
            "wall_area_m2": round(wall_area, 1),
            "window_area_m2": round(window_area, 1),
            "wwr": round(window_area / (wall_area + 1e-6), 2)
        },
        "climate": {
            "city": climate["city"],
            "zone": climate["zone"]
        }
    }
