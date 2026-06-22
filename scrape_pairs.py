import requests, os
from PIL import Image
from io import BytesIO

MAPBOX_TOKEN = "YOUR_TOKEN_HERE"   # paste your token
ZOOM = 17
OUT_DIR = "datasets/scraped"
os.makedirs(OUT_DIR, exist_ok=True)

# Diverse cities not in original Maps dataset (which was mostly US-centric)
CITIES = [
    ("Mumbai", 19.0760, 72.8777), ("Delhi", 28.7041, 77.1025),
    ("Bangalore", 12.9716, 77.5946), ("London", 51.5074, -0.1278),
    ("Paris", 48.8566, 2.3522), ("Tokyo", 35.6762, 139.6503),
    ("Singapore", 1.3521, 103.8198), ("Cairo", 30.0444, 31.2357),
    ("Sydney", -33.8688, 151.2093), ("Berlin", 52.5200, 13.4050),
    ("Toronto", 43.6532, -79.3832), ("Sao Paulo", -23.5505, -46.6333),
    ("Bangkok", 13.7563, 100.5018), ("Istanbul", 41.0082, 28.9784),
    ("Seoul", 37.5665, 126.9780), ("Jakarta", -6.2088, 106.8456),
    ("Lagos", 6.5244, 3.3792), ("Mexico City", 19.4326, -99.1332),
    ("Moscow", 55.7558, 37.6173), ("Madrid", 40.4168, -3.7038),
]

PAIRS_PER_CITY = 100
STEP = 0.006   # ~600m spacing at zoom 17

def fetch(lat, lon, style):
    url = (f"https://api.mapbox.com/styles/v1/mapbox/{style}/static/"
           f"{lon},{lat},{ZOOM}/256x256@2x?access_token={MAPBOX_TOKEN}")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB").resize((256, 256))

idx = 0
for name, lat, lon in CITIES:
    grid = int(PAIRS_PER_CITY ** 0.5)
    for i in range(grid):
        for j in range(grid):
            sub_lat = lat + (i - grid/2) * STEP
            sub_lon = lon + (j - grid/2) * STEP
            try:
                sat = fetch(sub_lat, sub_lon, "satellite-v9")
                mp  = fetch(sub_lat, sub_lon, "streets-v12")
                pair = Image.new("RGB", (512, 256))
                pair.paste(sat, (0, 0))
                pair.paste(mp,  (256, 0))
                pair.save(f"{OUT_DIR}/{idx:06d}.jpg", quality=92)
                idx += 1
                if idx % 50 == 0:
                    print(f"{idx} pairs done ({name})")
            except Exception as e:
                print(f"skip {sub_lat:.4f},{sub_lon:.4f}: {e}")

print(f"DONE. Total new pairs: {idx}")
