import os, sys, time
os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'
sys.path.insert(0, os.path.dirname(__file__))
from floorplan_onnx import FloorplanSegmenterONNX

BASE = os.path.dirname(os.path.dirname(__file__))
seg = FloorplanSegmenterONNX(os.path.join(BASE, 'models', 'M2_DA_best.onnx'))

test_img = os.path.join(BASE, 'output_paper', 'visualizations', 'test_china_cad_1.jpg')
t0 = time.time()
result = seg.predict(test_img, use_preprocessing=True)
elapsed = time.time() - t0

print(f"Elapsed: {elapsed:.3f}s")
print(f"Image size: {result['image_size']}")
for k, v in result['stats'].items():
    pct = v['percentage']
    print(f"  {k}: {pct}%")
geo = result['geometry']
print(f"Geometry: walls={len(geo['walls'])}, windows={len(geo['windows'])}, doors={len(geo['doors'])}")
