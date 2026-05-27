"""
预计算精确分割 Mask - 移植 CubiCasa5K 官方解析逻辑
生成 4 类 mask: 0=bg, 1=wall, 2=window, 3=door
保存为 mask.npy 到每个样本目录
"""
import os
import sys
import time
import numpy as np
from pathlib import Path
from xml.dom import minidom
from skimage.draw import polygon
import cv2
from PIL import Image

# 官方类别映射 (简化为4类)
ROOM_MAP = {
    "Wall": 1, "Railing": 1,
}

ICON_MAP = {
    "Window": 2, "Door": 3,
}


def get_XY(points):
    """解析 SVG points 字符串为 X,Y 坐标数组"""
    if not points:
        return np.array([]), np.array([])
    if points[-1] == "":
        points = points[:-1]
    if points[0] == "":
        points = points[1:]

    X, Y = [], []
    i = 0
    for a in points:
        if ',' in a:
            parts = a.split(',')
            X.append(float(parts[0]))
            Y.append(float(parts[1]))
        else:
            if i % 2:
                Y.append(float(a))
            else:
                X.append(float(a))
        i += 1
    return np.array(X), np.array(Y)


def get_polygon_from_element(e):
    """从 SVG group 元素提取多边形坐标并光栅化"""
    try:
        pol = next(p for p in e.childNodes if p.nodeName == "polygon")
        points = pol.getAttribute("points").split(' ')
        if points[-1] == '':
            points = points[:-1]

        X, Y = [], []
        for a in points:
            if ',' in a:
                y_val, x_val = a.split(',')  # 注意: CubiCasa SVG 中是 y,x 顺序
                X.append(float(x_val))
                Y.append(float(y_val))

        if len(X) < 3:
            return np.array([]), np.array([])

        rr, cc = polygon(np.array(X), np.array(Y))
        return rr, cc
    except (StopIteration, ValueError):
        return np.array([]), np.array([])


def get_wall_polygon(e, height, width):
    """精确解析墙体多边形"""
    try:
        pol = next(p for p in e.childNodes if p.nodeName == "polygon")
        points = pol.getAttribute("points").split(' ')
        if points[-1] == '':
            points = points[:-1]

        X, Y = [], []
        for a in points:
            if ',' in a:
                x_val, y_val = a.split(',')
                X.append(np.round(float(x_val)))
                Y.append(np.round(float(y_val)))

        X = np.clip(np.array(X), 0, width - 1)
        Y = np.clip(np.array(Y), 0, height - 1)

        if len(X) < 3:
            return np.array([]), np.array([])

        # 检查墙体是否太小
        if abs(max(X) - min(X)) < 4 or abs(max(Y) - min(Y)) < 4:
            return np.array([]), np.array([])

        rr, cc = polygon(Y, X)
        return rr, cc
    except (StopIteration, ValueError):
        return np.array([]), np.array([])


def get_opening_polygon(e, height, width):
    """解析门窗多边形"""
    try:
        pol = next(p for p in e.childNodes if p.nodeName == "polygon")
        points_str = pol.getAttribute("points").split(' ')
        if points_str[-1] == '':
            points_str = points_str[:-1]

        X, Y = [], []
        for a in points_str:
            if ',' in a:
                x_val, y_val = a.split(',')
                X.append(np.round(float(x_val)))
                Y.append(np.round(float(y_val)))

        X = np.clip(np.array(X), 0, width - 1)
        Y = np.clip(np.array(Y), 0, height - 1)

        if len(X) < 3:
            return np.array([]), np.array([])

        rr, cc = polygon(Y, X)
        return rr, cc
    except (StopIteration, ValueError):
        return np.array([]), np.array([])


def get_icon_polygon(e, height, width):
    """解析家具/设备图标的多边形"""
    try:
        # 获取变换矩阵
        transform = e.getAttribute("transform")
        if not transform or 'matrix' not in transform:
            return np.array([]), np.array([])

        parent_transform = None
        if (e.parentNode and
            e.parentNode.getAttribute("class") == "FixedFurnitureSet"):
            parent_transform = e.parentNode.getAttribute("transform")

        strings = transform.split(',')
        a = float(strings[0][7:])
        b = float(strings[1])
        c = float(strings[2])
        d = float(strings[3])
        ex = float(strings[-2])
        f = float(strings[-1][:-1])
        M = np.array([[a, c, ex], [b, d, f], [0, 0, 1]])

        if parent_transform and 'matrix' in parent_transform:
            ps = parent_transform.split(',')
            ap = float(ps[0][7:])
            bp = float(ps[1])
            cp = float(ps[2])
            dp = float(ps[3])
            ep = float(ps[-2])
            fp = float(ps[-1][:-1])
            M_p = np.array([[ap, cp, ep], [bp, dp, fp], [0, 0, 1]])
        else:
            M_p = None

        # 找 BoundaryPolygon
        X, Y = np.array([]), np.array([])
        try:
            bp_group = next(p for p in e.childNodes
                          if p.nodeName == 'g' and
                          p.getAttribute("class") == "BoundaryPolygon")
            for p in bp_group.childNodes:
                if p.nodeName == "polygon":
                    pts = p.getAttribute("points").split(' ')
                    X, Y = get_XY(pts)
                    break
            if len(X) == 0:
                # 从所有子元素收集角点
                x_all, y_all = [], []
                for p in bp_group.childNodes:
                    if p.nodeName == 'polygon':
                        tx, ty = get_XY(p.getAttribute("points").split(' '))
                        x_all.extend(tx)
                        y_all.extend(ty)
                    elif p.nodeName == 'rect':
                        rx = float(p.getAttribute('x') or 1)
                        ry = float(p.getAttribute('y') or 1)
                        rw = float(p.getAttribute('width'))
                        rh = float(p.getAttribute('height'))
                        x_all.extend([rx, rx+rw])
                        y_all.extend([ry, ry+rh])
                if x_all:
                    X = np.array([min(x_all), max(x_all), max(x_all), min(x_all)])
                    Y = np.array([min(y_all), min(y_all), max(y_all), max(y_all)])
        except StopIteration:
            # 没有 BoundaryPolygon, 从子 g 中收集
            x_all, y_all = [], []
            for g in e.childNodes:
                if g.nodeName == 'g':
                    for p in g.childNodes:
                        if p.nodeName == 'polygon':
                            tx, ty = get_XY(p.getAttribute("points").split(' '))
                            x_all.extend(tx)
                            y_all.extend(ty)
                        elif p.nodeName == 'rect':
                            rx = float(p.getAttribute('x') or 1)
                            ry = float(p.getAttribute('y') or 1)
                            rw = float(p.getAttribute('width'))
                            rh = float(p.getAttribute('height'))
                            x_all.extend([rx, rx+rw])
                            y_all.extend([ry, ry+rh])
            if x_all:
                X = np.array([min(x_all), max(x_all), max(x_all), min(x_all)])
                Y = np.array([min(y_all), min(y_all), max(y_all), max(y_all)])

        if len(X) < 3:
            return np.array([]), np.array([])

        # 应用变换矩阵
        for i in range(len(X)):
            v = np.array([[X[i]], [Y[i]], [1]])
            vv = np.matmul(M, v)
            if M_p is not None:
                vv = np.matmul(M_p, vv)
            X[i] = np.round(vv[0, 0])
            Y[i] = np.round(vv[1, 0])

        X = np.clip(X, 0, width - 1)
        Y = np.clip(Y, 0, height - 1)

        rr, cc = polygon(Y, X)
        return rr, cc

    except Exception:
        return np.array([]), np.array([])


def get_space_polygon(e, height, width):
    """解析房间空间多边形"""
    try:
        pol = next(p for p in e.childNodes if p.nodeName == "polygon")
        points_str = pol.getAttribute("points").split(' ')
        if points_str[-1] == '':
            points_str = points_str[:-1]

        X, Y = [], []
        for a in points_str:
            if ',' in a:
                y_val, x_val = a.split(',')
                X.append(float(x_val))
                Y.append(float(y_val))

        X = np.clip(np.array(X), 0, height - 1)
        Y = np.clip(np.array(Y), 0, width - 1)

        if len(X) < 3:
            return np.array([]), np.array([])

        rr, cc = polygon(X, Y)
        return rr, cc
    except (StopIteration, ValueError):
        return np.array([]), np.array([])


def parse_svg_official(svg_path, height, width):
    """
    使用官方逻辑解析 SVG 为 4 类分割 mask
    返回: mask (H, W), dtype=uint8, 值 0-3
    """
    # walls 通道: 存房间类型和墙体
    walls = np.zeros((height, width), dtype=np.uint8)
    # icons 通道: 存门窗
    icons = np.zeros((height, width), dtype=np.uint8)

    try:
        svg = minidom.parse(str(svg_path))
    except Exception:
        return walls  # 解析失败返回全零

    for e in svg.getElementsByTagName('g'):
        eid = e.getAttribute("id")
        eclass = e.getAttribute("class")

        # 墙体
        if eid == "Wall":
            rr, cc = get_wall_polygon(e, height, width)
            if len(rr) > 0:
                rr = np.clip(rr, 0, height - 1)
                cc = np.clip(cc, 0, width - 1)
                walls[rr, cc] = 1  # wall

        # 栏杆 (也归为墙体)
        if eid == "Railing":
            rr, cc = get_wall_polygon(e, height, width)
            if len(rr) > 0:
                rr = np.clip(rr, 0, height - 1)
                cc = np.clip(cc, 0, width - 1)
                walls[rr, cc] = 1  # wall

        # 窗户
        if eid == "Window":
            rr, cc = get_opening_polygon(e, height, width)
            if len(rr) > 0:
                rr = np.clip(rr, 0, height - 1)
                cc = np.clip(cc, 0, width - 1)
                icons[rr, cc] = 2  # window

        # 门
        if eid == "Door":
            rr, cc = get_opening_polygon(e, height, width)
            if len(rr) > 0:
                rr = np.clip(rr, 0, height - 1)
                cc = np.clip(cc, 0, width - 1)
                icons[rr, cc] = 3  # door

        # 房间空间 (标记为已占用，帮助 background 分割)
        if "Space " in eclass:
            rr, cc = get_space_polygon(e, height, width)
            if len(rr) > 0:
                rr = np.clip(rr, 0, height - 1)
                cc = np.clip(cc, 0, width - 1)
                # 房间内部暂标为一个中间值，后续合并
                walls[rr, cc] = np.where(walls[rr, cc] == 0, 0, walls[rr, cc])

    # 合并: icons 优先级 > walls
    # 最终 mask: 0=bg, 1=wall, 2=window, 3=door
    mask = walls.copy()
    mask[icons > 0] = icons[icons > 0]

    return mask


def precompute_all(data_dir, splits=('train', 'val', 'test')):
    """预计算所有样本的 mask"""
    data_dir = Path(data_dir)
    total = 0
    success = 0
    failed = 0
    skipped = 0

    start = time.time()

    for split in splits:
        split_file = data_dir / f"{split}.txt"
        if not split_file.exists():
            print(f"[SKIP] {split}.txt not found")
            continue

        samples = [l.strip() for l in split_file.read_text().strip().split('\n') if l.strip()]
        print(f"\n[{split}] Processing {len(samples)} samples...")

        for i, rel in enumerate(samples):
            total += 1
            rel = rel.strip().strip('/')
            sample_dir = data_dir / rel

            mask_path = sample_dir / "mask.npy"

            # 如果已经存在, 跳过
            if mask_path.exists():
                skipped += 1
                continue

            # 加载图片获取尺寸
            img_path = sample_dir / "F1_scaled.png"
            svg_path = sample_dir / "model.svg"

            if not img_path.exists() or not svg_path.exists():
                failed += 1
                continue

            try:
                pil_img = Image.open(str(img_path))
                w, h = pil_img.size
                pil_img.close()

                mask = parse_svg_official(svg_path, h, w)

                # 保存
                np.save(str(mask_path), mask)
                success += 1

            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  [ERR] {rel}: {e}")

            if (i + 1) % 200 == 0:
                elapsed = time.time() - start
                rate = (success + failed + skipped) / elapsed
                print(f"  {i+1}/{len(samples)} done ({success} ok, {failed} err, {skipped} skip) {rate:.1f}/s")

    elapsed = time.time() - start
    print(f"\n{'='*50}")
    print(f"Done in {elapsed:.0f}s")
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {failed}")
    print(f"  Total:   {total}")


def verify_masks(data_dir, n=5):
    """可视化验证几个 mask"""
    data_dir = Path(data_dir)
    split_file = data_dir / "val.txt"
    samples = [l.strip() for l in split_file.read_text().strip().split('\n') if l.strip()]

    colors = np.array([
        [0, 0, 0],       # bg
        [255, 0, 0],     # wall
        [0, 0, 255],     # window
        [0, 255, 0],     # door
    ], dtype=np.uint8)

    out_dir = data_dir.parent / "mask_verification"
    os.makedirs(out_dir, exist_ok=True)

    for i in range(min(n, len(samples))):
        rel = samples[i].strip().strip('/')
        sample_dir = data_dir / rel
        mask_path = sample_dir / "mask.npy"
        img_path = sample_dir / "F1_scaled.png"

        if not mask_path.exists():
            continue

        img = cv2.imread(str(img_path))
        mask = np.load(str(mask_path))

        # resize mask to match img
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        overlay = colors[mask_resized]
        result = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

        # 统计
        total_px = mask.size
        wall_pct = (mask == 1).sum() / total_px * 100
        win_pct = (mask == 2).sum() / total_px * 100
        door_pct = (mask == 3).sum() / total_px * 100

        out_path = out_dir / f"verify_{i}_{Path(rel).name}.png"
        cv2.imwrite(str(out_path), result)
        print(f"Sample {i}: wall={wall_pct:.1f}% win={win_pct:.1f}% door={door_pct:.1f}% -> {out_path.name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['compute', 'verify'], default='compute')
    parser.add_argument('--data', default='data/cubicasa5k')
    args = parser.parse_args()

    if args.mode == 'compute':
        precompute_all(args.data)
    elif args.mode == 'verify':
        verify_masks(args.data)
