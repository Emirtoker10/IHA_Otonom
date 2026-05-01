cat > ~/otonom_pro.py << 'EOF'
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import asyncio
import math
import cv2
import numpy as np
import threading
from collections import deque
from mavsdk import System
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.laserscan_pb2 import LaserScan
from ultralytics import YOLO

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FOV_H = math.radians(60)
FOV_V = math.radians(45)

COOLDOWN = 5
MIN_TARGET_DISTANCE = 20

latest_frame = [None]
tespit_piksel = [None]

target_queue = asyncio.PriorityQueue()
islenen_koordinatlar = []
last_detection_time = 0

drone_busy = False

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

def mesafe(lat1, lon1, lat2, lon2):
    dlat = abs(lat1 - lat2) * 111000
    dlon = abs(lon1 - lon2) * 111000 * math.cos(math.radians(lat1))
    return math.sqrt(dlat**2 + dlon**2)

def pixel_to_gps(lat, lon, alt, heading, px, py):
    heading = math.radians(heading)
    norm_x = (px - IMAGE_WIDTH/2)/(IMAGE_WIDTH/2)
    norm_y = (py - IMAGE_HEIGHT/2)/(IMAGE_HEIGHT/2)

    dx = alt * math.tan(FOV_H/2) * norm_x
    dy = alt * math.tan(FOV_V/2) * norm_y

    dx_r = dx * math.cos(heading) - dy * math.sin(heading)
    dy_r = dx * math.sin(heading) + dy * math.cos(heading)

    R = 6371000
    return (
        lat + math.degrees(dy_r / R),
        lon + math.degrees(dx_r / (R * math.cos(math.radians(lat))))
    )

def hedef_var_mi(lat, lon):
    for pl, plo in islenen_koordinatlar:
        if mesafe(lat, lon, pl, plo) < MIN_TARGET_DISTANCE:
            return True
    return False

def yolo_thread():
    global last_detection_time

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        if latest_frame[0] is not None:
            frame = latest_frame[0]
            results = model(frame, conf=0.3, verbose=False)

            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:
                        now = loop.time()

                        if now - last_detection_time < COOLDOWN:
                            continue

                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        cx = (x1+x2)//2
                        cy = (y1+y2)//2

                        tespit_piksel[0] = (cx, cy)
                        last_detection_time = now

            cv2.imshow("YOLO", frame)
            cv2.waitKey(1)

async def iha_loop(iha):
    while True:
        if tespit_piksel[0]:
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

            if hedef_var_mi(tlat, tlon):
                continue

            print(f"[IHA] Yeni hedef: {tlat:.6f}")

            islenen_koordinatlar.append((tlat, tlon))
            await target_queue.put((0, (tlat, tlon)))

        await asyncio.sleep(0.3)

async def goto_control(drone, lat, lon):
    while True:
        async for pos in drone.telemetry.position():
            dist = mesafe(pos.latitude_deg, pos.longitude_deg, lat, lon)
            if dist < 3:
                return
            break
        await asyncio.sleep(1)

async def drone_loop(drone):
    global drone_busy

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[DRONE] Baglandi")
            break

    async for health in drone.telemetry.health():
        if health.is_global_position_ok:
            break

    while True:
        _, (lat, lon) = await target_queue.get()

        if drone_busy:
            continue

        drone_busy = True

        print(f"[DRONE] Gidiyor: {lat:.6f}")

        await drone.action.arm()
        await drone.action.takeoff()
        await asyncio.sleep(5)

        await drone.action.goto_location(lat, lon, 15, 0)
        await goto_control(drone, lat, lon)

        print("[DRONE] Hedefte")
        await asyncio.sleep(3)

        await drone.action.return_to_launch()
        print("[DRONE] RTL")

        drone_busy = False

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

echo "Hazır! Çalıştır: python3 ~/otonom_pro.py"