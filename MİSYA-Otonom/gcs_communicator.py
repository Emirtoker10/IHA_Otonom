"""
gcs_communicator.py
Yer Kontrol İstasyonu (GCS) ile haberleşme.

- Kazazede koordinatlarını JSON formatında UDP/TCP ile GCS'e gönderir.
- GCS'den onay (ACK) bekler.
- Bağlantı kesilirse otomatik yeniden bağlanma yapar.

Dışa aktarılan:
    GCSCommunicator
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import asdict
from typing import List, Optional

from config_loader import get
from coordinate_calculator import CasualtyLocation

logger = logging.getLogger(__name__)


class GCSCommunicator:
    """
    GCS ile JSON tabanlı UDP haberleşmesi.

    Mesaj formatı (GCS'e gönderilen):
    {
        "type": "CASUALTY",
        "track_id": 3,
        "latitude": 39.9123456,
        "longitude": 32.8654321,
        "altitude_m": 1200.5,
        "distance_m": 14.2,
        "bearing_deg": 135.0,
        "confidence": 0.87,
        "timestamp": 1718000000.0
    }

    GCS yanıtı (ACK):
    { "type": "ACK", "track_id": 3, "status": "ACCEPTED" }
    """

    def __init__(self):
        cfg = get("gcs")
        self.host: str     = cfg["host"]
        self.port: int     = cfg["port"]
        self.protocol: str = cfg.get("protocol", "udp").lower()
        self.retry_s: float = cfg.get("retry_interval_s", 2.0)

        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._sent_count = 0
        self._ack_count  = 0
        self._casualty_log: List[dict] = []   # yerel log

    # ------------------------------------------------------------------
    def connect(self):
        """Socket açar."""
        try:
            if self.protocol == "udp":
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._sock.settimeout(2.0)
                logger.info(f"GCS UDP hazır → {self.host}:{self.port}")
            else:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.settimeout(5.0)
                self._sock.connect((self.host, self.port))
                logger.info(f"GCS TCP bağlandı → {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"GCS bağlantı hatası: {e}")
            self._sock = None

    def disconnect(self):
        if self._sock:
            self._sock.close()
            self._sock = None
        logger.info("GCS bağlantısı kapatıldı.")

    # ------------------------------------------------------------------
    def send_casualty(self, loc: CasualtyLocation) -> bool:
        """
        Kazazede konumunu GCS'e gönderir.
        Returns True → başarılı / ACK alındı.
        """
        payload = {
            "type": "CASUALTY",
            **asdict(loc),
            "timestamp": time.time(),
        }
        return self._send_with_retry(payload, loc.track_id)

    def send_status(self, status: str, extra: dict = None):
        """Genel durum mesajı gönderir (örn. 'HOVERING', 'SCANNING')."""
        payload = {"type": "STATUS", "status": status, "timestamp": time.time()}
        if extra:
            payload.update(extra)
        self._send_raw(json.dumps(payload).encode())

    # ------------------------------------------------------------------
    def _send_with_retry(self, payload: dict, track_id: int, retries: int = 3) -> bool:
        data = json.dumps(payload).encode("utf-8")
        for attempt in range(1, retries + 1):
            try:
                ack = self._send_raw(data)
                if ack:
                    self._ack_count += 1
                    self._casualty_log.append(payload)
                    logger.info(
                        f"[Track {track_id}] GCS'e iletildi (deneme {attempt}). "
                        f"ACK: {ack}"
                    )
                    return True
                else:
                    logger.warning(f"[Track {track_id}] ACK yok, yeniden deneniyor ({attempt}/{retries})")
                    time.sleep(self.retry_s)
            except Exception as e:
                logger.error(f"Gönderim hatası: {e} (deneme {attempt}/{retries})")
                time.sleep(self.retry_s)
                self._reconnect()
        return False

    def _send_raw(self, data: bytes) -> Optional[dict]:
        """Ham veri gönderir, varsa JSON yanıtı parse eder."""
        if self._sock is None:
            self._reconnect()
        if self._sock is None:
            return None
        with self._lock:
            try:
                if self.protocol == "udp":
                    self._sock.sendto(data, (self.host, self.port))
                    self._sent_count += 1
                    try:
                        raw, _ = self._sock.recvfrom(1024)
                        return json.loads(raw.decode("utf-8"))
                    except socket.timeout:
                        return None
                else:
                    self._sock.sendall(data + b"\n")
                    self._sent_count += 1
                    try:
                        raw = self._sock.recv(1024)
                        return json.loads(raw.decode("utf-8"))
                    except socket.timeout:
                        return None
            except Exception as e:
                logger.error(f"_send_raw hatası: {e}")
                self._sock = None
                return None

    def _reconnect(self):
        logger.info("GCS yeniden bağlanıyor...")
        self.disconnect()
        time.sleep(self.retry_s)
        self.connect()

    # ------------------------------------------------------------------
    @property
    def stats(self) -> dict:
        return {
            "sent": self._sent_count,
            "acked": self._ack_count,
            "logged_casualties": len(self._casualty_log),
        }

    def get_casualty_log(self) -> List[dict]:
        """Gönderilen tüm kazazede kayıtlarını döner."""
        return list(self._casualty_log)
