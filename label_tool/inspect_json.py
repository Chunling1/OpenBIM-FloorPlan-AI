import urllib.request
import json
import os

url = "https://huggingface.co/datasets/Voxel51/FloorPlanCAD/resolve/main/samples.json"
out_path = "samples.json"

if not os.path.exists(out_path):
    print("Downloading samples.json...")
    urllib.request.urlretrieve(url, out_path)
    print("Downloaded samples.json!")

with open(out_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    print(f"Data type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {data.keys()}")
        if "samples" in data:
            print(f"Number of samples: {len(data['samples'])}")
            print("First sample example:")
            print(json.dumps(data["samples"][0], indent=2)[:500])
    elif isinstance(data, list):
        print(f"Length of list: {len(data)}")
        print("First sample example:")
        print(json.dumps(data[0], indent=2)[:500])
