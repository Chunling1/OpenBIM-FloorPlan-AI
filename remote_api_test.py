import requests
import json
import time
import sys

BASE_URL = "http://localhost:5000"

# 1. Login
session = requests.Session()
res = session.post(f"{BASE_URL}/login", json={"username": "admin", "password": "chunge666"})
if res.status_code != 200 or res.json().get('status') != 'success':
    print(f"Login failed: {res.status_code} {res.text}")
    sys.exit(1)
print("✅ Login successful.")

# 2. Check AI status
res = session.get(f"{BASE_URL}/energy/ai_status")
if res.status_code != 200:
    print(f"AI status failed: {res.status_code}")
    sys.exit(1)
print(f"✅ AI Status: {res.json()}")

# 3. AI Recognize
print("⏳ Testing AI Recognize (Semantic Segmentation & Vectorization)...")
t0 = time.time()
with open("test_image.jpg", "rb") as f:
    files = {"raster_file": ("test_image.jpg", f, "image/jpeg")}
    data = {"report_number": "TEST_001", "preprocessing": "auto"}
    res = session.post(f"{BASE_URL}/energy/ai_recognize", data=data, files=files)

if res.status_code != 200:
    print(f"❌ AI Recognize failed: {res.status_code} {res.text}")
    sys.exit(1)

dataOut = res.json()
print(f"✅ AI Recognize completed in {time.time()-t0:.2f}s")
print(f"   Model: {dataOut.get('model')} (mIoU: {dataOut.get('mIoU')})")
print(f"   Image Size: {dataOut.get('image_size')}")
print(f"   Geometry Summary: {dataOut.get('geometry_summary')}")
print(f"   Pixel Lengths: {dataOut.get('pixel_lengths')}")

# 4. Energy Simulation based on AI results
print("\n⏳ Testing Energy Simulate (Simplified Model)...")
t0 = time.time()
sim_data = {
    "report_number": "TEST_001",
    "scale": 0.02, # 1px = 2cm
    "height": 3.0,
    "floors": 1,
    "u_wall": 0.6,
    "u_win": 2.5,
    "u_roof": 0.4,
    "u_floor": 0.3,
    "city": "Beijing"
}
res = session.post(f"{BASE_URL}/energy/ai_simulate", json=sim_data)

if res.status_code != 200:
    print(f"❌ AI Simulate failed: {res.status_code} {res.text}")
    sys.exit(1)

simOut = res.json()
print(f"✅ AI Simulate completed in {time.time()-t0:.2f}s")
print(f"   Geometry Calculated: Floor Area: {simOut['geometry']['floor_area_m2']}m2, Wall Area: {simOut['geometry']['wall_area_m2']}m2")
print(f"   WWR: {simOut['geometry']['wwr']}")
print(f"   Energy Loads (kWh): {json.dumps(simOut['loads'], indent=2)}")

print("\n🎉 Deployment Test Completed Successfully!")
