"""
mavlink_controller.py
MAVLink / ArduPilot (Pixhawk) uçuş kontrolü.

Desteklenen komutlar:
    - Telemetri okuma (GPS, irtifa, yaw, hız)
    - Hover modu (GUIDED + sabit konum)
    - İniş noktasına dönüş (RTL)
    - Drone ARM/DEARM sinyali (GPIO üzerinden)

Dışa aktarılan:
    MAVLinkController
    UAVTelemetry
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from config_loader import get

logger = logging.getLogger(__name__)

# MAVLink uçuş modları (ArduCopter / ArduPlane)
MODE_GUIDED = "GUIDED"
MODE_AUTO   = "AUTO"
MODE_RTL    = "RTL"
MODE_LOITER = "LOITER"


@dataclass
class UAVTelemetry:
    """İHA'dan gelen anlık telemetri snapshot'ı."""
    lat: float = 0.0
    lon: float = 0.0
    alt_m: float = 0.0              # yerden yükseklik (relative)
    yaw_deg: float = 0.0            # manyetik yaw (kuzey=0)
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    ground_speed_ms: float = 0.0
    heading_deg: float = 0.0
    battery_pct: float = 100.0
    armed: bool = False
    mode: str = "UNKNOWN"
    gps_fix: int = 0                # 0=no fix, 3=3D fix
    timestamp: float = field(default_factory=time.time)


class MAVLinkController:
    """
    MAVLink bağlantısı, telemetri döngüsü ve komut gönderimi.

    Kullanım:
        ctrl = MAVLinkController()
        ctrl.connect()
        telem = ctrl.telemetry          # UAVTelemetry
        ctrl.set_hover()
        ctrl.return_to_launch()
        ctrl.disconnect()
    """

    def __init__(self):
        cfg = get("mavlink")
        self.conn_str: str  = cfg["connection_string"]
        self.baud: int      = cfg["baud"]
        self.timeout: float = cfg["timeout_s"]
        self.sysid: int     = cfg.get("system_id", 1)

        self._vehicle = None
        self._telem = UAVTelemetry()
        self._lock = threading.Lock()
        self._telem_thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Bağlantı
    # ------------------------------------------------------------------
    def connect(self):
        """Pixhawk'a bağlan ve telemetri döngüsünü başlat."""
        from pymavlink import mavutil
        logger.info(f"MAVLink bağlantısı kuruluyor: {self.conn_str}@{self.baud}")
        self._vehicle = mavutil.mavlink_connection(
            self.conn_str,
            baud=self.baud,
            source_system=255,
        )
        self._vehicle.wait_heartbeat(timeout=self.timeout)
        logger.info(
            f"Heartbeat alındı — sistem: {self._vehicle.target_system}, "
            f"bileşen: {self._vehicle.target_component}"
        )
        self._running = True
        self._telem_thread = threading.Thread(
            target=self._telemetry_loop, daemon=True, name="MAVTelemetry"
        )
        self._telem_thread.start()

    def disconnect(self):
        self._running = False
        if self._telem_thread:
            self._telem_thread.join(timeout=3.0)
        if self._vehicle:
            self._vehicle.close()
        logger.info("MAVLink bağlantısı kapatıldı.")

    # ------------------------------------------------------------------
    # Telemetri (read-only snapshot)
    # ------------------------------------------------------------------
    @property
    def telemetry(self) -> UAVTelemetry:
        with self._lock:
            return UAVTelemetry(**self._telem.__dict__)

    def _telemetry_loop(self):
        """Arka planda telemetri mesajlarını dinler."""
        from pymavlink import mavutil
        while self._running:
            try:
                msg = self._vehicle.recv_match(blocking=True, timeout=0.5)
                if msg is None:
                    continue
                mtype = msg.get_type()

                with self._lock:
                    if mtype == "GLOBAL_POSITION_INT":
                        self._telem.lat = msg.lat / 1e7
                        self._telem.lon = msg.lon / 1e7
                        self._telem.alt_m = msg.relative_alt / 1000.0
                        self._telem.heading_deg = msg.hdg / 100.0

                    elif mtype == "ATTITUDE":
                        self._telem.yaw_deg   = math.degrees(msg.yaw) % 360
                        self._telem.pitch_deg = math.degrees(msg.pitch)
                        self._telem.roll_deg  = math.degrees(msg.roll)

                    elif mtype == "VFR_HUD":
                        self._telem.ground_speed_ms = msg.groundspeed

                    elif mtype == "HEARTBEAT":
                        self._telem.armed = bool(
                            msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                        )
                        self._telem.mode = mavutil.mode_string_v10(msg)

                    elif mtype == "SYS_STATUS":
                        if msg.battery_remaining >= 0:
                            self._telem.battery_pct = float(msg.battery_remaining)

                    elif mtype == "GPS_RAW_INT":
                        self._telem.gps_fix = msg.fix_type

                    self._telem.timestamp = time.time()

            except Exception as e:
                logger.debug(f"Telemetri döngüsü hatası: {e}")
                time.sleep(0.05)

    # ------------------------------------------------------------------
    # Uçuş komutları
    # ------------------------------------------------------------------
    def set_mode(self, mode: str):
        """Uçuş modunu değiştirir (GUIDED, LOITER, RTL, AUTO...)."""
        from pymavlink import mavutil
        mode_id = self._vehicle.mode_mapping().get(mode)
        if mode_id is None:
            raise ValueError(f"Bilinmeyen mod: {mode}")
        self._vehicle.mav.set_mode_send(
            self._vehicle.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        logger.info(f"Mod değiştirildi: {mode}")

    def arm(self, force: bool = False):
        """İHA motorlarını ARM eder."""
        param2 = 21196 if force else 0
        self._vehicle.mav.command_long_send(
            self._vehicle.target_system,
            self._vehicle.target_component,
            400,   # MAV_CMD_COMPONENT_ARM_DISARM
            0, 1, param2, 0, 0, 0, 0, 0,
        )
        logger.info("ARM komutu gönderildi.")

    def disarm(self):
        """İHA motorlarını DEARM eder."""
        self._vehicle.mav.command_long_send(
            self._vehicle.target_system,
            self._vehicle.target_component,
            400,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
        logger.info("DEARM komutu gönderildi.")

    def set_hover(self):
        """
        LOITER moduna geçerek mevcut konumda asılı kalır.
        GPS ve irtifa kilitlenir.
        """
        self.set_mode(MODE_LOITER)
        logger.info("Hover modu aktif (LOITER).")

    def resume_mission(self):
        """AUTO moduna dönerek görev planına devam eder."""
        self.set_mode(MODE_AUTO)
        logger.info("Görev devam ediyor (AUTO).")

    def return_to_launch(self):
        """RTL moduna geçerek kalkış noktasına döner."""
        self.set_mode(MODE_RTL)
        logger.info("RTL moduna girildi.")

    def send_waypoint(self, lat: float, lon: float, alt_m: float):
        """
        GUIDED modda anlık hedef waypoint gönderir.
        (Görev planı dışında tek seferlik konum komutu)
        """
        from pymavlink import mavutil
        self.set_mode(MODE_GUIDED)
        self._vehicle.mav.send(
            mavutil.mavlink.MAVLink_set_position_target_global_int_message(
                0,
                self._vehicle.target_system,
                self._vehicle.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                0b0000111111111000,  # yalnızca konum maskesi
                int(lat * 1e7),
                int(lon * 1e7),
                alt_m,
                0, 0, 0,
                0, 0, 0,
                0, 0,
            )
        )
        logger.info(f"Waypoint gönderildi: lat={lat:.7f}, lon={lon:.7f}, alt={alt_m}m")

    def wait_for_altitude(self, target_alt: float, tolerance: float = 1.0, timeout: float = 30.0):
        """Belirtilen irtifaya ulaşılana kadar bekler."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if abs(self.telemetry.alt_m - target_alt) <= tolerance:
                return True
            time.sleep(0.3)
        logger.warning(f"İrtifa zaman aşımı: hedef={target_alt}m")
        return False


import math   # _telemetry_loop içinde kullanılıyor
