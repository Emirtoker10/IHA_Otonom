"""
coordinate_calculator.py
Sensör füzyonu ile piksel koordinatlarını gerçek dünya koordinatlarına çevirir.

Algoritma:
    1. Piksel merkezinden kamera açısını hesapla (fx, fy, cx, cy)
    2. LiDAR mesafesi + gimbal açısı → zemine olan yatay uzaklıkları bul
    3. İHA'nın GPS + yaw bilgisiyle kazazede konumunu hesapla

Dışa aktarılan:
    CoordinateCalculator
    CasualtyLocation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from config_loader import get

logger = logging.getLogger(__name__)


@dataclass
class CasualtyLocation:
    """Bir kazazedenin gerçek dünya koordinatları."""
    track_id: int
    latitude: float
    longitude: float
    altitude_m: float
    distance_m: float           # İHA'ya olan yatay mesafe
    bearing_deg: float          # İHA'dan kazazedeye yer yönü (kuzey=0)
    confidence: float


class CoordinateCalculator:
    """
    Kamera kalibrasyon + LiDAR + GPS + yaw → kazazede GPS koordinatı.

    Kullanım:
        calc = CoordinateCalculator()
        loc  = calc.compute(
                   track_id=5,
                   pixel_center=(640, 420),
                   lidar_distance_m=12.3,
                   uav_lat=39.9, uav_lon=32.8, uav_alt=15.0,
                   uav_yaw_deg=90.0,
                   gimbal_pitch_deg=-45.0,
                   confidence=0.82)
    """

    def __init__(self):
        cal = get("camera_calibration")
        cam = get("camera")

        self.fx: float = cal["fx"]
        self.fy: float = cal["fy"]
        self.cx: float = cal["cx"]        # optik merkez x (piksel)
        self.cy: float = cal["cy"]        # optik merkez y (piksel)
        self.img_w: int = cam["width"]
        self.img_h: int = cam["height"]

        logger.debug(f"CoordinateCalculator hazır: fx={self.fx}, fy={self.fy}, "
                     f"cx={self.cx}, cy={self.cy}")

    # ------------------------------------------------------------------
    def compute(
        self,
        track_id: int,
        pixel_center: Tuple[int, int],
        lidar_distance_m: float,
        uav_lat: float,
        uav_lon: float,
        uav_alt: float,
        uav_yaw_deg: float,
        gimbal_pitch_deg: float = -90.0,   # varsayılan: düz aşağı
        confidence: float = 0.0,
    ) -> Optional[CasualtyLocation]:
        """
        Parametre açıklamaları:
            pixel_center      — tespitin piksel merkezi (cx, cy)
            lidar_distance_m  — İHA'nın zeminden yüksekliği (LiDAR)
            uav_yaw_deg       — İHA'nın manyetik yaw açısı (kuzey=0, saat yönü +)
            gimbal_pitch_deg  — Gimbal açısı; -90 = düz aşağı, -45 = 45° öne
        """
        if lidar_distance_m <= 0:
            logger.warning("Geçersiz LiDAR mesafesi, hesaplama atlanıyor.")
            return None

        px, py = pixel_center

        # --- Adım 1: Piksel → normalize kamera koordinatı ---
        # (dx, dy): kameraya göre birim vektör bileşeni
        dx_cam = (px - self.cx) / self.fx     # sağ pozitif
        dy_cam = (py - self.cy) / self.fy     # aşağı pozitif

        # --- Adım 2: Gimbal pitch dönüşümü ---
        # Gimbal düz aşağı baktığında (-90°) x-offset = dx_cam * yükseklik
        pitch_rad = math.radians(gimbal_pitch_deg)

        # Kameradan yere olan mesafe (yükseklik / cos(pitch))
        slant_range_m = lidar_distance_m / math.cos(abs(pitch_rad) if pitch_rad != 0 else 1e-6)
        slant_range_m = max(slant_range_m, lidar_distance_m)

        # Zemine yatay uzaklıklar (metre)
        ground_x_m = dx_cam * slant_range_m   # sağa
        ground_y_m = dy_cam * slant_range_m   # ileriye (kamera yönü)

        # --- Adım 3: Kamera yönünü İHA yaw ile dünya koordinatlarına çevir ---
        yaw_rad = math.radians(uav_yaw_deg)

        # Kamera eksenleri → North/East eksenlerine döndür
        north_offset_m = (ground_y_m * math.cos(yaw_rad)
                          - ground_x_m * math.sin(yaw_rad))
        east_offset_m  = (ground_y_m * math.sin(yaw_rad)
                          + ground_x_m * math.cos(yaw_rad))

        # --- Adım 4: GPS offset → enlem/boylam ---
        d_lat, d_lon = self._meters_to_latlon(
            north_offset_m, east_offset_m, uav_lat
        )

        target_lat = uav_lat + d_lat
        target_lon = uav_lon + d_lon

        # Yatay mesafe ve yer yönü
        horiz_dist = math.hypot(north_offset_m, east_offset_m)
        bearing = math.degrees(math.atan2(east_offset_m, north_offset_m)) % 360

        loc = CasualtyLocation(
            track_id=track_id,
            latitude=round(target_lat, 8),
            longitude=round(target_lon, 8),
            altitude_m=round(uav_alt - lidar_distance_m, 2),  # kazazede irtifası
            distance_m=round(horiz_dist, 2),
            bearing_deg=round(bearing, 1),
            confidence=confidence,
        )

        logger.info(
            f"[Track {track_id}] Konum hesaplandı: "
            f"lat={loc.latitude:.7f}, lon={loc.longitude:.7f}, "
            f"dist={loc.distance_m}m, bearing={loc.bearing_deg}°"
        )
        return loc

    # ------------------------------------------------------------------
    @staticmethod
    def _meters_to_latlon(
        north_m: float, east_m: float, ref_lat: float
    ) -> Tuple[float, float]:
        """
        Kuzey/Doğu metre ofsetini enlem/boylam farkına çevirir.
        Küçük mesafeler için düzlemsel yaklaşım (< 1 km hatasız).
        """
        # 1 derece enlem ≈ 111 320 m (sabit)
        d_lat = north_m / 111_320.0
        # 1 derece boylam ≈ 111 320 * cos(lat) m
        d_lon = east_m / (111_320.0 * math.cos(math.radians(ref_lat)))
        return d_lat, d_lon
