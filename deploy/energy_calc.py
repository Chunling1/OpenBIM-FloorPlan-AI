"""
建筑能耗计算引擎 v2
基于 GB 50189-2015 简化稳态法 (度日数法)
支持详细的设备系统配置: 供暖/制冷/照明/通风/生活热水
"""

# ============ 气候区数据 ============
CLIMATE_DB = {
    "harbin":    {"name": "哈尔滨", "zone": "严寒A区", "hdd18": 5100, "cdd26": 50,  "t_avg": 4.2,  "rh": 62},
    "urumqi":    {"name": "乌鲁木齐","zone": "严寒B区", "hdd18": 4300, "cdd26": 80,  "t_avg": 7.0,  "rh": 55},
    "beijing":   {"name": "北京",   "zone": "寒冷A区", "hdd18": 2800, "cdd26": 180, "t_avg": 12.6, "rh": 57},
    "dalian":    {"name": "大连",   "zone": "寒冷B区", "hdd18": 2900, "cdd26": 50,  "t_avg": 10.9, "rh": 65},
    "xian":      {"name": "西安",   "zone": "寒冷B区", "hdd18": 2400, "cdd26": 200, "t_avg": 13.7, "rh": 65},
    "shanghai":  {"name": "上海",   "zone": "夏热冬冷", "hdd18": 1500, "cdd26": 350, "t_avg": 16.1, "rh": 75},
    "chongqing": {"name": "重庆",   "zone": "夏热冬冷", "hdd18": 1100, "cdd26": 450, "t_avg": 18.3, "rh": 80},
    "wuhan":     {"name": "武汉",   "zone": "夏热冬冷", "hdd18": 1500, "cdd26": 400, "t_avg": 16.6, "rh": 77},
    "changsha":  {"name": "长沙",   "zone": "夏热冬冷", "hdd18": 1400, "cdd26": 380, "t_avg": 17.2, "rh": 78},
    "nanjing":   {"name": "南京",   "zone": "夏热冬冷", "hdd18": 1600, "cdd26": 350, "t_avg": 15.6, "rh": 75},
    "guangzhou": {"name": "广州",   "zone": "夏热冬暖", "hdd18": 400,  "cdd26": 650, "t_avg": 22.0, "rh": 77},
    "shenzhen":  {"name": "深圳",   "zone": "夏热冬暖", "hdd18": 300,  "cdd26": 700, "t_avg": 22.8, "rh": 78},
    "haikou":    {"name": "海口",   "zone": "夏热冬暖", "hdd18": 100,  "cdd26": 900, "t_avg": 24.0, "rh": 85},
    "kunming":   {"name": "昆明",   "zone": "温和区",   "hdd18": 1200, "cdd26": 10,  "t_avg": 15.0, "rh": 68},
    "guiyang":   {"name": "贵阳",   "zone": "温和区",   "hdd18": 1300, "cdd26": 30,  "t_avg": 15.3, "rh": 77},
    "chengdu":   {"name": "成都",   "zone": "夏热冬冷", "hdd18": 1200, "cdd26": 250, "t_avg": 16.5, "rh": 82},
    "tianjin":   {"name": "天津",   "zone": "寒冷A区", "hdd18": 2700, "cdd26": 200, "t_avg": 12.9, "rh": 62},
    "jinan":     {"name": "济南",   "zone": "寒冷B区", "hdd18": 2300, "cdd26": 250, "t_avg": 14.2, "rh": 60},
    "zhengzhou": {"name": "郑州",   "zone": "寒冷B区", "hdd18": 2200, "cdd26": 250, "t_avg": 14.5, "rh": 63},
    "hangzhou":  {"name": "杭州",   "zone": "夏热冬冷", "hdd18": 1500, "cdd26": 350, "t_avg": 16.5, "rh": 76},
}

# ============ 建筑类型默认参数 ============
BUILDING_DEFAULTS = {
    "office": {
        "name": "办公建筑", "lpd": 9.0, "epd": 15.0, "occupancy": 0.1,
        "vent_rate": 30, "dhw_liter_pp": 5, "op_hours": 2500
    },
    "commercial": {
        "name": "商业建筑", "lpd": 12.0, "epd": 20.0, "occupancy": 0.15,
        "vent_rate": 20, "dhw_liter_pp": 3, "op_hours": 3500
    },
    "hotel": {
        "name": "酒店建筑", "lpd": 10.0, "epd": 10.0, "occupancy": 0.08,
        "vent_rate": 30, "dhw_liter_pp": 80, "op_hours": 8760
    },
    "hospital": {
        "name": "医院建筑", "lpd": 11.0, "epd": 25.0, "occupancy": 0.12,
        "vent_rate": 40, "dhw_liter_pp": 60, "op_hours": 8760
    },
    "school": {
        "name": "学校建筑", "lpd": 9.0, "epd": 10.0, "occupancy": 0.5,
        "vent_rate": 25, "dhw_liter_pp": 5, "op_hours": 2000
    },
    "residential": {
        "name": "居住建筑", "lpd": 6.0, "epd": 8.0, "occupancy": 0.04,
        "vent_rate": 30, "dhw_liter_pp": 50, "op_hours": 5000
    },
}

# ============ 设备效率参数 ============
HEATING_SYSTEMS = {
    "gas_boiler":    {"name": "燃气锅炉",     "efficiency": 0.89, "fuel": "gas"},
    "coal_boiler":   {"name": "燃煤锅炉",     "efficiency": 0.75, "fuel": "coal"},
    "heat_pump_air": {"name": "空气源热泵",   "efficiency": 3.2,  "fuel": "electric"},
    "heat_pump_geo": {"name": "地源热泵",     "efficiency": 4.5,  "fuel": "electric"},
    "district":      {"name": "集中供暖",     "efficiency": 0.80, "fuel": "district"},
    "electric":      {"name": "电加热",       "efficiency": 0.95, "fuel": "electric"},
    "none":          {"name": "无供暖",       "efficiency": 1.0,  "fuel": "none"},
}

COOLING_SYSTEMS = {
    "central_chiller": {"name": "中央冷水机组", "cop": 5.0},
    "vrv":             {"name": "VRV/VRF多联机","cop": 3.8},
    "split_ac":        {"name": "分体空调",     "cop": 3.2},
    "evaporative":     {"name": "蒸发冷却",     "cop": 8.0},
    "none":            {"name": "无制冷",       "cop": 1.0},
}


def calculate_energy(params):
    """
    综合能耗计算
    params: dict containing:
      - geometry: {floor_area_m2, wall_area_m2, window_area_m2, roof_area_m2, door_area_m2}
      - building: {height, floors, building_type, orientation}
      - envelope: {u_wall, u_window, u_roof, u_floor, shgc}
      - climate: {city_id}
      - heating: {system_type, t_set}
      - cooling: {system_type, t_set}
      - lighting: {lpd, control_factor}
      - ventilation: {ach, fan_power}
      - dhw: {occupants, daily_liter_pp}
      - equipment: {epd}
      - schedule: {op_hours}
    """
    # --- 解析参数 ---
    geo = params.get("geometry", {})
    bld = params.get("building", {})
    env = params.get("envelope", {})
    clim = params.get("climate", {})
    heat = params.get("heating", {})
    cool = params.get("cooling", {})
    light = params.get("lighting", {})
    vent = params.get("ventilation", {})
    dhw = params.get("dhw", {})
    equip = params.get("equipment", {})
    sched = params.get("schedule", {})

    # --- 几何 ---
    floors = int(bld.get("floors", 1))
    height = float(bld.get("height", 3.0))
    floor_area = float(geo.get("floor_area_m2", 500))
    total_area = floor_area * floors
    wall_area = float(geo.get("wall_area_m2", floor_area * 0.8))
    window_area = float(geo.get("window_area_m2", wall_area * 0.3))
    roof_area = float(geo.get("roof_area_m2", floor_area))
    door_area = float(geo.get("door_area_m2", 0))
    wwr = window_area / max(wall_area + window_area, 1)

    # --- 气候 ---
    city_id = clim.get("city_id", "beijing")
    city = CLIMATE_DB.get(city_id, CLIMATE_DB["beijing"])
    hdd = city["hdd18"]
    cdd = city["cdd26"]

    # --- 围护结构 ---
    u_wall = float(env.get("u_wall", 0.6))
    u_win = float(env.get("u_window", 2.5))
    u_roof = float(env.get("u_roof", 0.4))
    u_floor = float(env.get("u_floor", 0.3))
    shgc = float(env.get("shgc", 0.4))

    # --- 建筑类型默认值 ---
    btype = bld.get("building_type", "office")
    defaults = BUILDING_DEFAULTS.get(btype, BUILDING_DEFAULTS["office"])
    op_hours = float(sched.get("op_hours", defaults["op_hours"]))

    # === 1. 围护结构传热负荷 ===
    ua_wall = u_wall * wall_area
    ua_win = u_win * window_area
    ua_roof = u_roof * roof_area
    ua_floor = u_floor * floor_area
    ua_door = 3.0 * door_area
    total_ua = ua_wall + ua_win + ua_roof + ua_floor + ua_door

    # 供暖/制冷设定温度修正
    t_heat_set = float(heat.get("t_set", 18.0))
    t_cool_set = float(cool.get("t_set", 26.0))
    hdd_adj = max(0, hdd + (t_heat_set - 18.0) * 120)
    cdd_adj = max(0, cdd + (26.0 - t_cool_set) * 90)

    # Q = UA * DD * 24 / 1000 (kWh)
    heating_envelope = total_ua * hdd_adj * 24 / 1000
    cooling_envelope = total_ua * cdd_adj * 24 / 1000

    # 太阳得热 (简化)
    solar_gain = window_area * shgc * 150 * (cdd / max(cdd + hdd, 1))
    cooling_envelope += solar_gain
    heating_envelope = max(0, heating_envelope - solar_gain * 0.3)

    # === 2. 供暖能耗 ===
    h_sys = heat.get("system_type", "gas_boiler")
    h_info = HEATING_SYSTEMS.get(h_sys, HEATING_SYSTEMS["gas_boiler"])
    heating_energy = heating_envelope / h_info["efficiency"] if h_info["fuel"] != "none" else 0

    # === 3. 制冷能耗 ===
    c_sys = cool.get("system_type", "central_chiller")
    c_info = COOLING_SYSTEMS.get(c_sys, COOLING_SYSTEMS["central_chiller"])
    cooling_energy = cooling_envelope / c_info["cop"] if c_sys != "none" else 0

    # === 4. 照明能耗 ===
    lpd = float(light.get("lpd", defaults["lpd"]))
    ctrl = float(light.get("control_factor", 1.0))
    lighting_energy = lpd * total_area * op_hours * ctrl / 1000

    # === 5. 设备插座能耗 ===
    epd = float(equip.get("epd", defaults["epd"]))
    equipment_energy = epd * total_area * op_hours / 1000

    # === 6. 通风能耗 ===
    ach = float(vent.get("ach", 1.0))
    fan_power = float(vent.get("fan_power", 0.5))  # W/m³/h
    vol = total_area * height
    vent_flow = vol * ach
    ventilation_energy = vent_flow * fan_power * op_hours / 1000

    # === 7. 生活热水能耗 ===
    occupants = int(dhw.get("occupants", max(1, int(total_area * defaults["occupancy"]))))
    daily_liter = float(dhw.get("daily_liter_pp", defaults["dhw_liter_pp"]))
    dt_water = 35  # 温升 ΔT
    dhw_energy = occupants * daily_liter * 4.186 * dt_water * 365 / 3600 / 1000  # kWh
    dhw_eff = float(dhw.get("efficiency", 0.85))
    dhw_energy = dhw_energy / dhw_eff

    # === 汇总 ===
    total_energy = heating_energy + cooling_energy + lighting_energy + equipment_energy + ventilation_energy + dhw_energy
    eui = total_energy / total_area if total_area > 0 else 0

    # EUI 等级评定
    if eui < 50:
        rating, rating_label = "A", "超低能耗"
    elif eui < 80:
        rating, rating_label = "B", "低能耗"
    elif eui < 120:
        rating, rating_label = "C", "节能"
    elif eui < 180:
        rating, rating_label = "D", "一般"
    else:
        rating, rating_label = "E", "高能耗"

    breakdown = {
        "heating":     round(heating_energy, 1),
        "cooling":     round(cooling_energy, 1),
        "lighting":    round(lighting_energy, 1),
        "equipment":   round(equipment_energy, 1),
        "ventilation": round(ventilation_energy, 1),
        "dhw":         round(dhw_energy, 1),
    }
    breakdown_eui = {k: round(v / total_area, 2) if total_area > 0 else 0 for k, v in breakdown.items()}

    return {
        "success": True,
        "summary": {
            "total_energy_kwh": round(total_energy, 0),
            "eui": round(eui, 2),
            "rating": rating,
            "rating_label": rating_label,
            "total_floor_area_m2": round(total_area, 1),
        },
        "breakdown_kwh": breakdown,
        "breakdown_eui": breakdown_eui,
        "geometry_used": {
            "floor_area_m2": round(floor_area, 1),
            "total_area_m2": round(total_area, 1),
            "wall_area_m2": round(wall_area, 1),
            "window_area_m2": round(window_area, 1),
            "roof_area_m2": round(roof_area, 1),
            "wwr": round(wwr, 3),
        },
        "climate": {
            "city": city["name"],
            "zone": city["zone"],
            "hdd18": hdd,
            "cdd26": cdd,
        },
        "systems": {
            "heating": h_info["name"],
            "cooling": c_info["name"],
        },
    }


def get_climate_cities():
    """返回可选城市列表"""
    return [{"id": k, "name": v["name"], "zone": v["zone"]} for k, v in CLIMATE_DB.items()]


def get_building_types():
    """返回建筑类型列表"""
    return [{"id": k, "name": v["name"]} for k, v in BUILDING_DEFAULTS.items()]


def get_system_options():
    """返回设备系统选项"""
    return {
        "heating": [{"id": k, "name": v["name"]} for k, v in HEATING_SYSTEMS.items()],
        "cooling": [{"id": k, "name": v["name"]} for k, v in COOLING_SYSTEMS.items()],
    }
