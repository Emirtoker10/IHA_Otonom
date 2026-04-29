cat > ~/otonom_gorev.py << 'EOF'
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import asyncio
import math
import cv2
import numpy as np
import threading
from collections import deque

from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.laserscan_pb2 import LaserScan
from ultralytics import YOLO

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FOV_H = math.radians(60)
FOV_V = math.radians(45)

TARAMA_IRTIFA = 50
SERIT_ARALIGI = 20
ALAN_GENISLIK = 300
ALAN_UZUNLUK = 100
HOME_LAT = -35.363258
HOME_LON = 149.165207

latest_frame = [None]
tespit_piksel = [None]
islenen_koordinatlar = []
gorev_kuyrugu = asyncio.Queue()

lidar_buffer = deque(maxlen=5)

model = YOLO("yolov8n.pt")

node = Node()

def image_callback(msg):
    img = np.frombuffer(msg.data, dtype=np.uint8)
    img = img.reshape((msg.height, msg.width, 3))
    latest_frame[0] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def lidar_callback(msg):
    if not msg.ranges:
        return

    val = msg.ranges[0]

    if val is None or val == float('inf') or val <= 0:
        return

    lidar_buffer.append(val)

node.subscribe(Image, '/camera/image', image_callback)
node.subscribe(LaserScan, '/lidar/range', lidar_callback)

def get_altitude(fallback):
    if len(lidar_buffer) == 0:
        return fallback
    return sum(lidar_buffer) / len(lidar_buffer)

def pixel_to_gps(lat, lon, alt, heading, px, py):
    heading = math.radians(heading)

    norm_x = (px - IMAGE_WIDTH/2)/(IMAGE_WIDTH/2)
    norm_y = (py - IMAGE_HEIGHT/2)/(IMAGE_HEIGHT/2)

    dx = alt * math.tan(FOV_H/2) * norm_x
    dy = alt * math.tan(FOV_V/2) * norm_y

    dx_r = dx * math.cos(heading) - dy * math.sin(heading)
    dy_r = dx * math.sin(heading) + dy * math.cos(heading)

    R = 6371000
    new_lat = lat + math.degrees(dy_r / R)
    new_lon = lon + math.degrees(dx_r / (R * math.cos(math.radians(lat))))

    return new_lat, new_lon

def koordinat_islendi_mi(lat, lon, esik=25):
    for pl, plo in islenen_koordinatlar:
        dlat = abs(lat-pl)*111000
        dlon = abs(lon-plo)*111000*math.cos(math.radians(lat))
        if (dlat**2 + dlon**2)**0.5 < esik:
            return True
    return False

def yolo_thread():
    while True:
        if latest_frame[0] is not None:
            frame = latest_frame[0].copy()
            results = model(frame, conf=0.3, verbose=False)

            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        cx = (x1+x2)//2
                        cy = (y1+y2)//2
                        tespit_piksel[0] = (cx, cy)

            cv2.imshow("YOLO", frame)
            cv2.waitKey(1)

async def iha_loop(iha):
    while True:

        if tespit_piksel[0] is not None:
            px, py = tespit_piksel[0]
            tespit_piksel[0] = None

            async for pos in iha.telemetry.position():
                lat = pos.latitude_deg
                lon = pos.longitude_deg
                alt = pos.relative_altitude_m
                break

            async for h in iha.telemetry.heading():
                heading = h.heading_deg
                break

            alt = get_altitude(alt)

            tlat, tlon = pixel_to_gps(lat, lon, alt, heading, px, py)

            if koordinat_islendi_mi(tlat, tlon):
                continue

            islenen_koordinatlar.append((tlat, tlon))

            print(f"[IHA] Hedef bulundu: {tlat:.6f}, {tlon:.6f}")

            await gorev_kuyrugu.put((tlat, tlon))

        await asyncio.sleep(0.5)

async def drone_loop(drone):

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[DRONE] Baglandi")
            break

    async for health in drone.telemetry.health():
        if health.is_global_position_ok:
            break

    while True:
        lat, lon = await gorev_kuyrugu.get()

        print(f"[DRONE] Gorev alindi: {lat}, {lon}")

        await drone_gorev(drone, lat, lon)

async def drone_gorev(drone, lat, lon):

    await drone.action.arm()

    try:
        await drone.action.takeoff()
        await asyncio.sleep(8)
    except:
        pass

    await drone.action.goto_location(lat, lon, 15, 0)

    print("[DRONE] Gidiyor...")
    await asyncio.sleep(15)

    print("[DRONE] Yuk birakiliyor")
    await asyncio.sleep(3)

    await drone.action.return_to_launch()
    print("[DRONE] RTL")

async def main():

    iha = System()
    drone = System()

    await iha.connect(system_address="udpin://0.0.0.0:14550")
    await drone.connect(system_address="udpin://0.0.0.0:14560")

    threading.Thread(target=yolo_thread, daemon=True).start()

    await asyncio.gather(
        iha_loop(iha),
        drone_loop(drone)
    )

if __name__ == "__main__":
    asyncio.run(main())
EOF

echo "Dosya hazir: python3 ~/otonom_gorev.py ile calistir"