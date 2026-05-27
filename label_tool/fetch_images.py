import os
import re
import urllib.request
import urllib.parse
import json
import time

def get_image_urls(query, max_results=200):
    urls = set()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36'
    }
    
    # We can try multiple pages/offsets
    for offset in range(0, max_results, 30):
        try:
            encoded_query = urllib.parse.quote_plus(query)
            url = f"https://www.bing.com/images/search?q={encoded_query}&first={offset}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode('utf-8', errors='ignore')
            
            # Find iusc elements containing metadata
            matches = re.findall(r'class="iusc"[^>]*?m="([^"]+?)"', html)
            if not matches:
                # Try escaping quote style
                matches = re.findall(r'class=&quot;iusc&quot;[^>]*?m=&quot;({[^}]+?})&quot;', html)
            
            for m in matches:
                # Decode HTML entities if present
                m_decoded = m.replace('&quot;', '"').replace('&#100;', 'd')
                try:
                    js = json.loads(m_decoded)
                    if 'murl' in js:
                        urls.add(js['murl'])
                except Exception:
                    # Try regex extraction from the attribute content
                    murl_match = re.search(r'"murl"\s*:\s*"([^"]+?)"', m_decoded)
                    if murl_match:
                        urls.add(murl_match.group(1))
            
            # Print progress
            print(f"Query: '{query}', Offset: {offset}, Total URLs found so far: {len(urls)}")
            time.sleep(1)
        except Exception as e:
            print(f"Error searching for query {query} at offset {offset}: {e}")
            break
            
    return list(urls)

def download_images(urls, output_dir, start_idx):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    idx = start_idx
    success_count = 0
    
    # Standard headers for download to avoid 403 Forbidden
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
    urllib.request.install_opener(opener)
    
    for url in urls:
        ext = '.jpg'
        if '.png' in url.lower():
            ext = '.png'
        elif '.jpeg' in url.lower():
            ext = '.jpg'
            
        filename = f"{idx}{ext}"
        filepath = os.path.join(output_dir, filename)
        
        print(f"Downloading {url} -> {filepath}...")
        try:
            # Check if file already exists (unlikely with index, but good practice)
            urllib.request.urlretrieve(url, filepath)
            
            # Validate image is readable and not corrupted
            # We can use PIL or cv2 to open it. Let's try importing PIL.
            from PIL import Image
            with Image.open(filepath) as img:
                img.verify()
            
            # If it's valid, increment index
            print(f"Success: {filepath} downloaded.")
            idx += 1
            success_count += 1
        except Exception as e:
            print(f"Failed to download/validate image: {e}")
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass
                    
    return success_count, idx

if __name__ == '__main__':
    queries = [
        "建筑平面图 CAD",
        "手绘 建筑平面图",
        "hand drawn architectural floor plan",
        "cad floor plan black background",
        "住宅建筑施工图 平面图"
    ]
    
    output_directory = r"D:\我的坚果云\工作\博士\bim\标注\原图"
    
    # We want to keep 1.png, so we start downloading from index 2
    current_idx = 2
    total_downloaded = 0
    
    for q in queries:
        print(f"Starting search for: {q}")
        image_urls = get_image_urls(q, max_results=150)
        print(f"Found {len(image_urls)} unique URLs for query: {q}")
        
        downloaded, next_idx = download_images(image_urls, output_directory, current_idx)
        total_downloaded += downloaded
        current_idx = next_idx
        
        print(f"Downloaded {downloaded} images for query: {q}. Current index is now: {current_idx}")
        
    print(f"All done! Total downloaded: {total_downloaded}")
