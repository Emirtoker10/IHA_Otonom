"""
lidar_reader.py
LiDAR'dan (TFmini / TFmini-S uyumlu) mesafe okur.
Serial protokol: 9 byte paket, başlangıç 0x59 0x59

Dışa aktarılan:
    LidarReader  — thread-safe, async okuma.
"""

import logging
import struct
import threading
import time
from typing import Optional

from config_loader import get

logger = logging.getLogger(__name__)

# TFmini paket yapısı: [0x59, 0x59, dist_L, dist_H, str_L, str_H, reserved, reserved, checksum]
_PACKET_LEN = 9
_HEADER = b"\x59\x59"


class LidarReader:
    """
    Thread-safe TFmini / TFmini-S okuyucu.
    read() her zaman en son geçerli mesafeyi döner.
    """

    def __init__(self):
        cfg = get("lidar")
        self.port: str    = cfg["port"]
        self.baud: int    = cfg["baud"]
        self.max_m: float = cfg["max_range_m"]
        self.min_m: float = cfg["min_range_m"]

        self._distance_m: Optional[float] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._serial = None

    # ------------------------------------------------------------------
    def start(self):
        if self._running:
            return
        import serial
        try:
            self._serial = serial.Serial(self.port, self.baud, timeout=1.0)
            self._running = True
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="LidarThread"
            )
            self._thread.start()
            logger.info(f"LiDAR başlatıldı: {self.port}@{self.baud}")
        except Exception as e:
            logger.error(f"LiDAR başlatılamadı: {e}")
            raise

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial and self._serial.is_open:
            self._serial.close()
        logger.info("LiDAR durduruldu.")

    def read(self) -> Optional[float]:
        """Metre cinsinden anlık mesafe. Veri yoksa None."""
        with self._lock:
            return self._distance_m

    # ------------------------------------------------------------------
    def _loop(self):
        buf = bytearray()
        while self._running:
            try:
                raw = self._serial.read(64)
                if not raw:
                    continue
                buf.extend(raw)

                # Paket arama
                while len(buf) >= _PACKET_LEN:
                    idx = buf.find(_HEADER)
                    if idx == -1:
                        buf.clear()
                        break
                    if idx > 0:
                        del buf[:idx]
                    if len(buf) < _PACKET_LEN:
                        break

                    packet = buf[:_PACKET_LEN]

                    # Checksum doğrulama
                    checksum = sum(packet[:8]) & 0xFF
                    if checksum != packet[8]:
                        del buf[:1]  # kaydır, yeniden dene
                        continue

                    dist_cm = struct.unpack_from("<H", packet, 2)[0]
                    dist_m  = dist_cm / 100.0

                    if self.min_m <= dist_m <= self.max_m:
                        with self._lock:
                            self._distance_m = dist_m

                    del buf[:_PACKET_LEN]

            except Exception as e:
                logger.warning(f"LiDAR okuma hatası: {e}")
                time.sleep(0.05)
