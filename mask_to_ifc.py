# -*- coding: utf-8 -*-
"""
Stage 3: 分割Mask → IFC BIM 自动生成
从语义分割结果提取建筑构件几何，生成IFC 2x3文件

输入: segmentation mask (numpy array, H×W, values 0-3)
输出: IFC文件

类别: 0=Background, 1=Wall, 2=Window, 3=Door
"""
import numpy as np
from pathlib import Path
import uuid
import datetime
import json

# ============================================================
# 几何提取
# ============================================================

def extract_wall_segments(mask, min_area=100):
    """从wall mask提取墙段几何（连通区域 → 最小外接矩形）"""
    wall_mask = (mask == 1).astype(np.uint8)
    
    # 连通区域分析（纯numpy实现）
    from scipy import ndimage
    labeled, n_features = ndimage.label(wall_mask)
    
    walls = []
    for i in range(1, n_features + 1):
        component = (labeled == i)
        area = component.sum()
        if area < min_area:
            continue
        
        # 获取bounding box
        rows = np.where(component.any(axis=1))[0]
        cols = np.where(component.any(axis=0))[0]
        y_min, y_max = rows[0], rows[-1]
        x_min, x_max = cols[0], cols[-1]
        
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        # 判断水平/垂直
        if width >= height:
            orientation = 'horizontal'
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            length = width
            thickness = max(height, 3)
        else:
            orientation = 'vertical'
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            length = height
            thickness = max(width, 3)
        
        walls.append({
            'id': i,
            'bbox': (x_min, y_min, x_max, y_max),
            'center': (cx, cy),
            'length': length,
            'thickness': thickness,
            'orientation': orientation,
            'area_px': int(area),
        })
    
    return walls


def extract_openings(mask, class_id, walls, max_dist=30):
    """提取门/窗并关联宿主墙"""
    from scipy import ndimage
    
    opening_mask = (mask == class_id).astype(np.uint8)
    labeled, n_features = ndimage.label(opening_mask)
    
    openings = []
    for i in range(1, n_features + 1):
        component = (labeled == i)
        area = component.sum()
        if area < 20:
            continue
        
        rows = np.where(component.any(axis=1))[0]
        cols = np.where(component.any(axis=0))[0]
        y_min, y_max = rows[0], rows[-1]
        x_min, x_max = cols[0], cols[-1]
        
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        # 找最近的宿主墙
        host_wall = None
        min_d = float('inf')
        for w in walls:
            wcx, wcy = w['center']
            d = np.sqrt((cx - wcx)**2 + (cy - wcy)**2)
            if d < min_d and d < max_dist + w['length'] / 2:
                min_d = d
                host_wall = w['id']
        
        openings.append({
            'id': i,
            'type': 'window' if class_id == 2 else 'door',
            'bbox': (x_min, y_min, x_max, y_max),
            'center': (cx, cy),
            'width': max(width, height),
            'height': min(width, height),
            'host_wall_id': host_wall,
        })
    
    return openings


def extract_rooms(mask, min_area=500):
    """从背景区域提取房间多边形"""
    from scipy import ndimage
    
    # 房间 = 被墙围合的背景区域（排除最大的外部背景）
    bg_mask = (mask == 0).astype(np.uint8)
    labeled, n_features = ndimage.label(bg_mask)
    
    # 找到最大连通域（外部背景），排除它
    sizes = ndimage.sum(bg_mask, labeled, range(1, n_features + 1))
    if len(sizes) == 0:
        return []
    
    max_label = np.argmax(sizes) + 1
    
    rooms = []
    room_id = 0
    for i in range(1, n_features + 1):
        if i == max_label:
            continue
        component = (labeled == i)
        area = component.sum()
        if area < min_area:
            continue
        
        rows = np.where(component.any(axis=1))[0]
        cols = np.where(component.any(axis=0))[0]
        y_min, y_max = rows[0], rows[-1]
        x_min, x_max = cols[0], cols[-1]
        
        room_id += 1
        rooms.append({
            'id': room_id,
            'bbox': (x_min, y_min, x_max, y_max),
            'center': ((x_min+x_max)/2, (y_min+y_max)/2),
            'area_px': int(area),
            'width': x_max - x_min + 1,
            'height': y_max - y_min + 1,
        })
    
    return rooms


# ============================================================
# IFC 生成（纯STEP文件模板，不依赖ifcopenshell）
# ============================================================

def _guid():
    """生成IFC全局唯一ID（22字符base64）"""
    raw = uuid.uuid4().bytes
    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$'
    result = []
    for i in range(0, 16, 3):
        if i + 2 < 16:
            n = (raw[i] << 16) + (raw[i+1] << 8) + raw[i+2]
            for _ in range(4):
                result.append(chars[n % 64])
                n //= 64
        elif i + 1 < 16:
            n = (raw[i] << 8) + raw[i+1]
            for _ in range(3):
                result.append(chars[n % 64])
                n //= 64
        else:
            n = raw[i]
            for _ in range(2):
                result.append(chars[n % 64])
                n //= 64
    return ''.join(result[:22])


def generate_ifc(walls, windows, doors, rooms, 
                 scale=0.01, floor_height=3.0, wall_height=2.8,
                 output_path='output.ifc'):
    """
    生成IFC 2x3 STEP文件
    
    Args:
        walls: extract_wall_segments() 返回
        windows: extract_openings(mask, 2) 返回
        doors: extract_openings(mask, 3) 返回
        rooms: extract_rooms() 返回
        scale: 像素到米的换算（默认1px=0.01m=1cm）
        floor_height: 层高(m)
        wall_height: 墙高(m)
        output_path: 输出IFC文件路径
    """
    now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    
    lines = []
    entity_id = [0]  # mutable counter
    
    def next_id():
        entity_id[0] += 1
        return entity_id[0]
    
    def add(line):
        lines.append(line)
        return entity_id[0]
    
    # === HEADER ===
    lines.append("ISO-10303-21;")
    lines.append("HEADER;")
    lines.append(f"FILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');")
    lines.append(f"FILE_NAME('{Path(output_path).name}','{now}',('AutoBIM Pipeline'),(''),'',' ','');")
    lines.append("FILE_SCHEMA(('IFC2X3'));")
    lines.append("ENDSEC;")
    lines.append("DATA;")
    
    # === 基础实体 ===
    # Person/Org/App
    pid = next_id(); add(f"#{pid}=IFCPERSON($,$,'',$,$,$,$,$);")
    oid = next_id(); add(f"#{oid}=IFCORGANIZATION($,'AutoBIM',$,$,$);")
    poid = next_id(); add(f"#{poid}=IFCPERSONANDORGANIZATION(#{pid},#{oid},$);")
    appid = next_id(); add(f"#{appid}=IFCAPPLICATION(#{oid},'1.0','FloorPlan2BIM','FP2BIM');")
    ownerid = next_id(); add(f"#{ownerid}=IFCOWNERHISTORY(#{poid},#{appid},$,.NOCHANGE.,$,#{poid},#{appid},0);")
    
    # Units
    uid1 = next_id(); add(f"#{uid1}=IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.);")
    uid2 = next_id(); add(f"#{uid2}=IFCSIUNIT(*,.AREAUNIT.,$,.SQUARE_METRE.);")
    uid3 = next_id(); add(f"#{uid3}=IFCSIUNIT(*,.VOLUMEUNIT.,$,.CUBIC_METRE.);")
    uid4 = next_id(); add(f"#{uid4}=IFCSIUNIT(*,.PLANEANGLEUNIT.,$,.RADIAN.);")
    unitsid = next_id(); add(f"#{unitsid}=IFCUNITASSIGNMENT((#{uid1},#{uid2},#{uid3},#{uid4}));")
    
    # Geometric context
    orig3d = next_id(); add(f"#{orig3d}=IFCCARTESIANPOINT((0.,0.,0.));")
    dir_z = next_id(); add(f"#{dir_z}=IFCDIRECTION((0.,0.,1.));")
    dir_x = next_id(); add(f"#{dir_x}=IFCDIRECTION((1.,0.,0.));")
    axis3d = next_id(); add(f"#{axis3d}=IFCAXIS2PLACEMENT3D(#{orig3d},#{dir_z},#{dir_x});")
    ctx = next_id(); add(f"#{ctx}=IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-5,#{axis3d},$);")
    
    # Project
    projid = next_id(); add(f"#{projid}=IFCPROJECT('{_guid()}',#{ownerid},'AutoBIM Project',$,$,$,$,(#{ctx}),#{unitsid});")
    
    # Site
    site_place = next_id(); add(f"#{site_place}=IFCLOCALPLACEMENT($,#{axis3d});")
    siteid = next_id(); add(f"#{siteid}=IFCSITE('{_guid()}',#{ownerid},'Site',$,$,#{site_place},$,$,.ELEMENT.,$,$,$,$,$);")
    
    # Building
    bld_place = next_id(); add(f"#{bld_place}=IFCLOCALPLACEMENT(#{site_place},#{axis3d});")
    bldid = next_id(); add(f"#{bldid}=IFCBUILDING('{_guid()}',#{ownerid},'Building',$,$,#{bld_place},$,$,.ELEMENT.,$,$,$);")
    
    # Storey
    sty_place = next_id(); add(f"#{sty_place}=IFCLOCALPLACEMENT(#{bld_place},#{axis3d});")
    styid = next_id(); add(f"#{styid}=IFCBUILDINGSTOREY('{_guid()}',#{ownerid},'Ground Floor',$,$,#{sty_place},$,$,.ELEMENT.,0.);")
    
    # Spatial hierarchy
    rel1 = next_id(); add(f"#{rel1}=IFCRELAGGREGATES('{_guid()}',#{ownerid},$,$,#{projid},(#{siteid}));")
    rel2 = next_id(); add(f"#{rel2}=IFCRELAGGREGATES('{_guid()}',#{ownerid},$,$,#{siteid},(#{bldid}));")
    rel3 = next_id(); add(f"#{rel3}=IFCRELAGGREGATES('{_guid()}',#{ownerid},$,$,#{bldid},(#{styid}));")
    
    # === 墙体 ===
    wall_ids = []
    for w in walls:
        cx_m = w['center'][0] * scale
        cy_m = w['center'][1] * scale
        length_m = w['length'] * scale
        thick_m = max(w['thickness'] * scale, 0.1)  # 最小10cm
        
        # 墙体位置
        wp = next_id(); add(f"#{wp}=IFCCARTESIANPOINT(({cx_m:.4f},{cy_m:.4f},0.));")
        
        if w['orientation'] == 'horizontal':
            wd = next_id(); add(f"#{wd}=IFCDIRECTION((1.,0.,0.));")
        else:
            wd = next_id(); add(f"#{wd}=IFCDIRECTION((0.,1.,0.));")
        
        wa2d = next_id(); add(f"#{wa2d}=IFCAXIS2PLACEMENT3D(#{wp},$,#{wd});")
        wlp = next_id(); add(f"#{wlp}=IFCLOCALPLACEMENT(#{sty_place},#{wa2d});")
        
        # 挤出体（矩形截面 × 墙高）
        rp1 = next_id(); add(f"#{rp1}=IFCCARTESIANPOINT(({-length_m/2:.4f},{-thick_m/2:.4f}));")
        rp2 = next_id(); add(f"#{rp2}=IFCCARTESIANPOINT(({length_m/2:.4f},{-thick_m/2:.4f}));")
        rp3 = next_id(); add(f"#{rp3}=IFCCARTESIANPOINT(({length_m/2:.4f},{thick_m/2:.4f}));")
        rp4 = next_id(); add(f"#{rp4}=IFCCARTESIANPOINT(({-length_m/2:.4f},{thick_m/2:.4f}));")
        poly = next_id(); add(f"#{poly}=IFCPOLYLINE((#{rp1},#{rp2},#{rp3},#{rp4},#{rp1}));")
        profile = next_id(); add(f"#{profile}=IFCARBITRARYCLOSEDPROFILEDEF(.AREA.,$,#{poly});")
        
        ext_dir = next_id(); add(f"#{ext_dir}=IFCDIRECTION((0.,0.,1.));")
        ext = next_id(); add(f"#{ext}=IFCEXTRUDEDAREASOLID(#{profile},#{axis3d},#{ext_dir},{wall_height:.2f});")
        
        shape_rep = next_id(); add(f"#{shape_rep}=IFCSHAPEREPRESENTATION(#{ctx},'Body','SweptSolid',(#{ext}));")
        prod_shape = next_id(); add(f"#{prod_shape}=IFCPRODUCTDEFINITIONSHAPE($,$,(#{shape_rep}));")
        
        wid = next_id()
        add(f"#{wid}=IFCWALLSTANDARDCASE('{_guid()}',#{ownerid},'Wall_{w['id']}',$,$,#{wlp},#{prod_shape},$);")
        wall_ids.append(wid)
    
    # === 门窗 ===
    opening_ids = []
    for op in windows + doors:
        op_w = op['width'] * scale
        op_h = op['height'] * scale
        cx_m = op['center'][0] * scale
        cy_m = op['center'][1] * scale
        
        op_p = next_id(); add(f"#{op_p}=IFCCARTESIANPOINT(({cx_m:.4f},{cy_m:.4f},0.));")
        oa2d = next_id(); add(f"#{oa2d}=IFCAXIS2PLACEMENT3D(#{op_p},$,$);")
        olp = next_id(); add(f"#{olp}=IFCLOCALPLACEMENT(#{sty_place},#{oa2d});")
        
        sill_h = 0.9 if op['type'] == 'window' else 0.0
        op_height = 1.5 if op['type'] == 'window' else 2.1
        
        orp1 = next_id(); add(f"#{orp1}=IFCCARTESIANPOINT(({-op_w*scale/2:.4f},0.));")
        orp2 = next_id(); add(f"#{orp2}=IFCCARTESIANPOINT(({op_w*scale/2:.4f},0.));")
        orp3 = next_id(); add(f"#{orp3}=IFCCARTESIANPOINT(({op_w*scale/2:.4f},{op_height:.4f}));")
        orp4 = next_id(); add(f"#{orp4}=IFCCARTESIANPOINT(({-op_w*scale/2:.4f},{op_height:.4f}));")
        opoly = next_id(); add(f"#{opoly}=IFCPOLYLINE((#{orp1},#{orp2},#{orp3},#{orp4},#{orp1}));")
        oprofile = next_id(); add(f"#{oprofile}=IFCARBITRARYCLOSEDPROFILEDEF(.AREA.,$,#{opoly});")
        
        oext_dir = next_id(); add(f"#{oext_dir}=IFCDIRECTION((0.,0.,1.));")
        oext = next_id(); add(f"#{oext}=IFCEXTRUDEDAREASOLID(#{oprofile},#{axis3d},#{oext_dir},{op_height:.2f});")
        oshape = next_id(); add(f"#{oshape}=IFCSHAPEREPRESENTATION(#{ctx},'Body','SweptSolid',(#{oext}));")
        oprod = next_id(); add(f"#{oprod}=IFCPRODUCTDEFINITIONSHAPE($,$,(#{oshape}));")
        
        if op['type'] == 'window':
            eid = next_id()
            add(f"#{eid}=IFCWINDOW('{_guid()}',#{ownerid},'Window_{op['id']}',$,$,#{olp},#{oprod},$,{op_height:.2f},{op_w*scale:.2f});")
        else:
            eid = next_id()
            add(f"#{eid}=IFCDOOR('{_guid()}',#{ownerid},'Door_{op['id']}',$,$,#{olp},#{oprod},$,{op_height:.2f},{op_w*scale:.2f});")
        opening_ids.append(eid)
    
    # === 房间/空间 ===
    space_ids = []
    for r in rooms:
        cx_m = r['center'][0] * scale
        cy_m = r['center'][1] * scale
        rw = r['width'] * scale
        rh = r['height'] * scale
        area_m2 = r['area_px'] * scale * scale
        
        sp = next_id(); add(f"#{sp}=IFCCARTESIANPOINT(({cx_m:.4f},{cy_m:.4f},0.));")
        sa = next_id(); add(f"#{sa}=IFCAXIS2PLACEMENT3D(#{sp},$,$);")
        slp = next_id(); add(f"#{slp}=IFCLOCALPLACEMENT(#{sty_place},#{sa});")
        
        sid = next_id()
        add(f"#{sid}=IFCSPACE('{_guid()}',#{ownerid},'Room_{r['id']}','Area={area_m2:.1f}m2',$,#{slp},$,$,.ELEMENT.,.INTERNAL.,$);")
        space_ids.append(sid)
    
    # 空间包含关系
    all_elements = wall_ids + opening_ids
    if all_elements:
        elem_str = ','.join(f'#{e}' for e in all_elements)
        rel_contain = next_id()
        add(f"#{rel_contain}=IFCRELCONTAINEDINSPATIALSTRUCTURE('{_guid()}',#{ownerid},$,$,({elem_str}),#{styid});")
    
    if space_ids:
        space_str = ','.join(f'#{s}' for s in space_ids)
        rel_spaces = next_id()
        add(f"#{rel_spaces}=IFCRELAGGREGATES('{_guid()}',#{ownerid},$,$,#{styid},({space_str}));")
    
    # === FOOTER ===
    lines.append("ENDSEC;")
    lines.append("END-ISO-10303-21;")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return {
        'n_walls': len(walls),
        'n_windows': len(windows),
        'n_doors': len(doors),
        'n_rooms': len(rooms),
        'output': output_path,
        'n_entities': entity_id[0],
    }


# ============================================================
# 主入口
# ============================================================

def mask_to_ifc(mask, output_path='output.ifc', scale=0.01, 
                floor_height=3.0, wall_height=2.8):
    """
    端到端：分割mask → IFC文件
    
    Args:
        mask: np.ndarray (H,W), values 0-3
        output_path: 输出IFC路径
        scale: 像素到米 (默认1px=1cm)
        floor_height: 层高
        wall_height: 墙高
    
    Returns:
        dict 统计信息
    """
    print(f"[Stage 3] Extracting geometry from mask {mask.shape}...")
    
    walls = extract_wall_segments(mask)
    print(f"  Walls: {len(walls)}")
    
    windows = extract_openings(mask, 2, walls)
    print(f"  Windows: {len(windows)}")
    
    doors = extract_openings(mask, 3, walls)
    print(f"  Doors: {len(doors)}")
    
    rooms = extract_rooms(mask)
    print(f"  Rooms: {len(rooms)}")
    
    result = generate_ifc(walls, windows, doors, rooms,
                          scale=scale, floor_height=floor_height,
                          wall_height=wall_height, output_path=output_path)
    
    print(f"  IFC saved: {output_path} ({result['n_entities']} entities)")
    return result


if __name__ == '__main__':
    # 测试：加载一个已有的预测mask
    import sys
    base = Path(__file__).parent
    
    # 尝试从benchmark结果中加载
    test_mask_path = base / "output_paper" / "test_pred_sample.npy"
    if test_mask_path.exists():
        mask = np.load(str(test_mask_path))
    else:
        # 生成一个简单测试mask
        print("Generating synthetic test mask...")
        mask = np.zeros((512, 512), dtype=np.uint8)
        # 外墙
        mask[50:55, 50:450] = 1   # top
        mask[350:355, 50:450] = 1  # bottom
        mask[50:355, 50:55] = 1    # left
        mask[50:355, 445:450] = 1  # right
        # 内墙
        mask[50:355, 200:205] = 1
        # 窗
        mask[50:55, 100:150] = 2
        mask[50:55, 270:320] = 2
        # 门
        mask[200:230, 200:205] = 3
        mask[350:355, 120:150] = 3
    
    out = base / "output_paper" / "test_output.ifc"
    result = mask_to_ifc(mask, str(out), scale=0.01)
    print(f"\nResult: {json.dumps(result, indent=2)}")
