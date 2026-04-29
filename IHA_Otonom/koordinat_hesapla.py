import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import asyncio
import math
from collections import deque

from mavsdk import System
from gz.transport13 import Node
from gz.msgs10.laserscan_pb2 import LaserScan

# ============================================================
# KAMERA PARAMETRELERİ
# ============================================================
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FOV_H = math.radians(60)
FOV_V = math.radians(45)

# ============================================================
# LiDAR BUFFER (SMOOTHING)
# ============================================================
lidar_buffer = deque(maxlen=5)

def lidar_callback(msg):
    if not msg.ranges:
        return

    value = msg.ranges[0]

    # Geçersiz veri filtreleme
    if (
        value is None or
        value == float('inf') or
        math.isnan(value) or
        value <= 0
    ):
        return

    lidar_buffer.append(value)


def get_lidar_altitude(fallback_altitude):
    if len(lidar_buffer) == 0:
        return fallback_altitude

    # Ortalama al (smooth)
    return sum(lidar_buffer) / len(lidar_buffer)


# ============================================================
# PIXEL → GPS
# ============================================================
def pixel_to_gps(iha_lat, iha_lon, altitude, heading_deg, pixel_x, pixel_y):
    heading = math.radians(heading_deg)

    norm_x = (pixel_x - IMAGE_WIDTH / 2) / (IMAGE_WIDTH / 2)
    norm_y = (pixel_y - IMAGE_HEIGHT / 2) / (IMAGE_HEIGHT / 2)

    dx = altitude * math.tan(FOV_H / 2) * norm_x
    dy = altitude * math.tan(FOV_V / 2) * norm_y

    dx_r = dx * math.cos(heading) - dy * math.sin(heading)
    dy_r = dx * math.sin(heading) + dy * math.cos(heading)

    R = 6371000
    target_lat = iha_lat + math.degrees(dy_r / R)
    target_lon = iha_lon + math.degrees(dx_r / (R * math.cos(math.radians(iha_lat))))

    return target_lat, target_lon


# ============================================================
# TELEMETRY
# ============================================================
async def get_telemetry_once(iha):
    async for pos in iha.telemetry.position():
        lat = pos.latitude_deg
        lon = pos.longitude_deg
        rel_alt = pos.relative_altitude_m
        break

    async for head in iha.telemetry.heading():
        heading = head.heading_deg
        break

    return lat, lon, rel_alt, heading


# ============================================================
# ANA FONKSİYON
# ============================================================
async def run():
    # LiDAR dinle
    node = Node()
    node.subscribe(LaserScan, '/lidar/range', lidar_callback)

    # İHA bağlantı
    iha = System()
    await iha.connect(system_address="udpin://0.0.0.0:14550")

    print("[IHA] Baglaniliyor...")
    async for state in iha.core.connection_state():
        if state.is_connected:
            print("[IHA] Baglandi!")
            break

    # GPS kontrol
    print("[IHA] GPS bekleniyor...")
    async for health in iha.telemetry.health():
        if health.is_global_position_ok:
            print("[IHA] GPS OK!")
            break

    # Telemetry al
    iha_lat, iha_lon, rel_alt, iha_heading = await get_telemetry_once(iha)

    # LiDAR verisi bekle
    await asyncio.sleep(1)

    # İrtifa seçimi (gelişmiş)
    altitude = get_lidar_altitude(rel_alt)

    if altitude == rel_alt:
        print(f"[GPS] Irtifa: {altitude:.2f}m")
    else:
        print(f"[LiDAR] Irtifa: {altitude:.2f}m")

    print(f"[IHA] Konum: {iha_lat:.6f}, {iha_lon:.6f}")
    print(f"[IHA] Heading: {iha_heading:.1f}")

    # TEST (YOLO yerine)
    pixel_x, pixel_y = 599, 70

    target_lat, target_lon = pixel_to_gps(
        iha_lat, iha_lon, altitude, iha_heading, pixel_x, pixel_y
    )

    print("\n[SONUC] Kazazede Koordinati:")
    print(f"  Lat: {target_lat:.6f}")
    print(f"  Lon: {target_lon:.6f}")

    return target_lat, target_lon


# ============================================================
# ÇALIŞTIR
# ============================================================
if __name__ == "__main__":
    asyncio.run(run())