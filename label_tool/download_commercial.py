"""
商业建筑 CAD 平面图下载器
从 Bing 图片搜索批量下载办公楼、酒店、商场、医院、学校等商业建筑的 CAD 平面图
下载到标注工具的原图目录，文件名以 comm_ 开头以区分居住建筑
"""
import os
import re
import urllib.request
import urllib.parse
import json
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

OUTPUT_DIR = Path(r"D:\我的坚果云\工作\博士\bim\标注\原图")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 商业建筑关键词 — 中英文混合，覆盖多种商业建筑类型
QUERIES = [
    # 办公楼
    "办公楼 CAD平面图",
    "办公建筑 施工图 平面图",
    "office building floor plan CAD",
    "office building architectural floor plan drawing",
    "写字楼 标准层平面图 CAD",
    # 商场
    "商场 CAD平面图",
    "shopping mall floor plan CAD drawing",
    "商业综合体 平面图 CAD",
    "retail building floor plan architecture",
    # 酒店
    "酒店 CAD平面图",
    "hotel floor plan CAD architectural",
    "酒店标准层 建筑平面图",
    "hotel building plan layout drawing",
    # 医院
    "医院 建筑平面图 CAD",
    "hospital floor plan CAD drawing",
    "医院门诊楼 平面图",
    # 学校
    "学校 教学楼 CAD平面图",
    "school building floor plan CAD",
    "教学楼 建筑施工图",
    # 图书馆/展览馆等公共建筑
    "图书馆 CAD平面图",
    "library floor plan CAD",
    "展览馆 建筑平面图",
    "museum floor plan architectural drawing",
    # 通用商业
    "commercial building floor plan CAD black background",
    "public building architectural plan CAD",
    "大型公共建筑 CAD 平面图",
    "工业厂房 CAD平面图",
    "factory floor plan CAD drawing",
]

MAX_RESULTS_PER_QUERY = 120
MAX_WORKERS = 16


def get_image_urls(query, max_results=120):
    """从 Bing 图片搜索抓取图片 URL"""
    urls = set()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
    }

    for offset in range(0, max_results, 35):
        try:
            encoded_query = urllib.parse.quote_plus(query)
            # 加 imagesize 过滤，要大图；加 filetype 过滤 png/jpg
            url = f"https://www.bing.com/images/search?q={encoded_query}&first={offset}&qft=+filterui:imagesize-large"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode('utf-8', errors='ignore')

            # 解析 iusc 元素
            matches = re.findall(r'class="iusc"[^>]*?m="([^"]+?)"', html)
            for m in matches:
                m_decoded = m.replace('&quot;', '"').replace('&#100;', 'd')
                try:
                    js = json.loads(m_decoded)
                    if 'murl' in js:
                        urls.add(js['murl'])
                except Exception:
                    murl_match = re.search(r'"murl"\s*:\s*"([^"]+?)"', m_decoded)
                    if murl_match:
                        urls.add(murl_match.group(1))

            time.sleep(0.8)
        except Exception as e:
            print(f"  搜索出错 offset={offset}: {e}")
            break

    return list(urls)


def download_one(url, filepath):
    """下载单张图片并验证"""
    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [
            ('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        ]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, str(filepath))

        # 基本验证：文件大于 5KB
        if filepath.stat().st_size < 5000:
            filepath.unlink()
            return False, "太小，可能是占位图"

        # 尝试用 cv2 验证
        try:
            import cv2
            import numpy as np
            img_data = np.fromfile(str(filepath), dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)
            if img is None:
                filepath.unlink()
                return False, "cv2 无法解码"
            h, w = img.shape[:2]
            if h < 200 or w < 200:
                filepath.unlink()
                return False, f"图太小 {w}x{h}"
        except ImportError:
            pass  # 没有 cv2 就跳过验证

        return True, "OK"
    except Exception as e:
        if filepath.exists():
            filepath.unlink()
        return False, str(e)


def main():
    print(f"输出目录: {OUTPUT_DIR}")

    # 统计现有的最大 comm_ 编号
    existing = list(OUTPUT_DIR.glob("comm_*"))
    if existing:
        nums = []
        for p in existing:
            m = re.match(r'comm_(\d+)', p.stem)
            if m:
                nums.append(int(m.group(1)))
        start_idx = max(nums) + 1 if nums else 1
    else:
        start_idx = 1

    print(f"起始编号: comm_{start_idx:04d}")

    # 收集所有 URL
    all_urls = []
    seen_urls = set()

    for i, query in enumerate(QUERIES):
        print(f"\n[{i+1}/{len(QUERIES)}] 搜索: {query}")
        urls = get_image_urls(query, max_results=MAX_RESULTS_PER_QUERY)
        new_urls = [u for u in urls if u not in seen_urls]
        seen_urls.update(new_urls)
        all_urls.extend(new_urls)
        print(f"  找到 {len(urls)} 个URL，{len(new_urls)} 个新URL，累计: {len(all_urls)}")

    print(f"\n总共收集到 {len(all_urls)} 个唯一 URL，开始下载...")

    # 去重：根据 URL hash
    idx = start_idx
    success = 0
    failed = 0

    for i, url in enumerate(all_urls):
        ext = '.png'
        url_lower = url.lower()
        if '.jpg' in url_lower or '.jpeg' in url_lower:
            ext = '.jpg'
        elif '.bmp' in url_lower:
            ext = '.bmp'

        filename = f"comm_{idx:04d}{ext}"
        filepath = OUTPUT_DIR / filename

        ok, msg = download_one(url, filepath)
        if ok:
            success += 1
            idx += 1
        else:
            failed += 1

        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(all_urls)}  成功: {success}  失败: {failed}")

    print(f"\n=== 下载完成 ===")
    print(f"成功: {success}")
    print(f"失败: {failed}")
    print(f"文件保存在: {OUTPUT_DIR}")
    print(f"文件名范围: comm_{start_idx:04d} ~ comm_{idx-1:04d}")


if __name__ == '__main__':
    main()
