import asyncio
from mavsdk import System
import math

async def hedefe_ulasildi_mi(drone, target_lat, target_lon, threshold=2):
    async for pos in drone.telemetry.position():
        dlat = abs(pos.latitude_deg - target_lat) * 111000
        dlon = abs(pos.longitude_deg - target_lon) * 111000 * math.cos(math.radians(target_lat))
        mesafe = (dlat**2 + dlon**2) ** 0.5

        if mesafe < threshold:
            return True

async def drone_gorev(drone, lat, lon, alt=15):
    print(f"[DRONE] Yeni görev: {lat:.6f}, {lon:.6f}")

    await drone.action.arm()
    await drone.action.set_takeoff_altitude(alt)

    try:
        await drone.action.takeoff()
        await asyncio.sleep(8)
    except:
        pass

    await drone.action.goto_location(lat, lon, alt, 0)

    print("[DRONE] Hedefe gidiliyor...")

    await hedefe_ulasildi_mi(drone, lat, lon)

    print("[DRONE] Hedefe ulaşıldı!")

    await asyncio.sleep(3)

    print("[DRONE] Yük bırakılıyor")
    await asyncio.sleep(2)

    print("[DRONE] RTL")
    await drone.action.return_to_launch()
    await asyncio.sleep(10)