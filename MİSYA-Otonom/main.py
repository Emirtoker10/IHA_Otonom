"""
main.py
VTOL SAR Sistemi — Ana Görev Orkestratörü

Modül bağımlılıkları:
    config_loader.py        ← merkezi YAML config
    logger_setup.py         ← loglama
    camera_stream.py        ← CSI kamera
    detection.py            ← YOLOv8 + ByteTrack + N-frame
    lidar_reader.py         ← TFmini LiDAR
    coordinate_calculator.py← piksel → GPS
    mavlink_controller.py   ← Pixhawk / ArduPilot
    drone_release.py        ← GPIO ARM + ayrılma
    gcs_communicator.py     ← UDP/TCP GCS
    state_machine.py        ← görev durum makinesi

Çalıştırma:
    python main.py
    python main.py --mock      # GPIO & LiDAR simülasyonu
    python main.py --no-gcs    # GCS'siz test
"""

import argparse
import signal
import sys
import time
import logging

# ── Loglama önce kurulur ──────────────────────────────────────────────
from logger_setup import setup_logging
setup_logging()
logger = logging.getLogger("main")

# ── Modüller ──────────────────────────────────────────────────────────
from config_loader import get
from camera_stream import CameraStream
from detection import DetectionEngine, ConfirmedTarget
from lidar_reader import LidarReader
from coordinate_calculator import CoordinateCalculator
from mavlink_controller import MAVLinkController
from drone_release import DroneRelease
from gcs_communicator import GCSCommunicator
from state_machine import StateMachine, State


# ═════════════════════════════════════════════════════════════════════
class SARMission:
    """
    Ana görev sınıfı.
    Tüm modülleri başlatır, durum makinesini yürütür.
    """

    def __init__(self, mock: bool = False, no_gcs: bool = False):
        self.mock   = mock
        self.no_gcs = no_gcs

        # Modül instance'ları
        self.camera  = CameraStream()
        self.engine  = DetectionEngine()
        self.lidar   = LidarReader() if not mock else _MockLidar()
        self.calc    = CoordinateCalculator()
        self.mav     = MAVLinkController()
        self.release = DroneRelease(mock=mock)
        self.gcs     = GCSCommunicator()
        self.sm      = StateMachine(initial=State.SCANNING)

        # Config
        flight_cfg = get("flight")
        self.hover_duration: float = flight_cfg["hover_duration_s"]

        self._running = False
        self._active_target: ConfirmedTarget | None = None

        # Sinyal yakalayıcı (Ctrl+C)
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ------------------------------------------------------------------
    def start(self):
        logger.info("══════ VTOL SAR Görevi Başlıyor ══════")
        try:
            self._init_hardware()
            self._running = True
            self._mission_loop()
        except Exception as e:
            logger.critical(f"Kritik hata: {e}", exc_info=True)
            self.sm.abort(str(e))
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Başlatma / Kapatma
    # ------------------------------------------------------------------
    def _init_hardware(self):
        logger.info("Donanım başlatılıyor...")

        self.camera.start()
        logger.info("✓ Kamera")

        self.lidar.start()
        logger.info("✓ LiDAR")

        self.engine.load()
        logger.info("✓ YOLO modeli")

        self.mav.connect()
        logger.info("✓ MAVLink / Pixhawk")

        self.release.setup()
        logger.info("✓ Drone ayrılma mekanizması")

        if not self.no_gcs:
            self.gcs.connect()
            logger.info("✓ GCS bağlantısı")
        else:
            logger.warning("GCS devre dışı (--no-gcs)")

        logger.info("Tüm donanım hazır.")

    def _shutdown(self):
        logger.info("Sistem kapatılıyor...")
        self._running = False
        self.camera.stop()
        self.lidar.stop()
        self.mav.disconnect()
        self.gcs.disconnect()
        self.release.teardown()

        stats = self.gcs.stats
        logger.info(
            f"Görev özeti — Gönderilen: {stats['sent']}, "
            f"ACK: {stats['acked']}, "
            f"Kazazede: {stats['logged_casualties']}"
        )
        logger.info("Sistem kapatıldı.")

    # ------------------------------------------------------------------
    # Ana döngü
    # ------------------------------------------------------------------
    def _mission_loop(self):
        logger.info("Görev döngüsü başladı.")

        while self._running:
            state = self.sm.current

            # ── SCANNING ──────────────────────────────────────────────
            if state == State.SCANNING:
                self._step_scanning()

            # ── DECELERATING ──────────────────────────────────────────
            elif state == State.DECELERATING:
                self._step_decelerating()

            # ── HOVERING ──────────────────────────────────────────────
            elif state == State.HOVERING:
                self._step_hovering()

            # ── COMPUTING ─────────────────────────────────────────────
            elif state == State.COMPUTING:
                self._step_computing()

            # ── TRANSMITTING ──────────────────────────────────────────
            elif state == State.TRANSMITTING:
                self._step_transmitting()

            # ── RELEASING ─────────────────────────────────────────────
            elif state == State.RELEASING:
                self._step_releasing()

            # ── RESUMING ──────────────────────────────────────────────
            elif state == State.RESUMING:
                self._step_resuming()

            # ── RTL / ABORT ───────────────────────────────────────────
            elif state in (State.RTL, State.ABORT):
                break

            time.sleep(0.01)  # CPU throttle

    # ------------------------------------------------------------------
    # Durum adımları
    # ------------------------------------------------------------------
    def _step_scanning(self):
        """Frame oku, YOLO çalıştır, onaylı hedef varsa geçiş yap."""
        frame = self.camera.read()
        if frame is None:
            return

        detections, confirmed = self.engine.process(frame)

        if confirmed:
            # En yüksek güvenli hedefi seç
            target = max(confirmed, key=lambda t: t.confidence)
            self._active_target = target
            logger.info(
                f"Kazazede onaylandı! Track={target.track_id}, "
                f"conf={target.confidence:.2f}, frame={target.frame_count}"
            )
            self.sm.transition(State.DECELERATING, f"Track {target.track_id}")

    def _step_decelerating(self):
        """Hızı düşür, hover konumuna getir."""
        if self.sm.state_duration() < 0.1:
            logger.info("İHA yavaşlıyor...")
            # Pixhawk'a hız azaltma komutu (LOITER öncesi)
            # Gerçek uygulamada waypoint hızı azaltılır.
            # Burada LOITER komutunu kısa gecikme ile gönderiyoruz.
            time.sleep(1.5)      # simüle yavaşlama
            self.sm.transition(State.HOVERING, "Hover konumuna girildi")

    def _step_hovering(self):
        """Hover moduna geç, drone ARM et, sensörlerin oturmasını bekle."""
        if self.sm.state_duration() < 0.1:
            # Hover komutu
            self.mav.set_hover()

            # Drone ARM
            self.release.arm_drone()

            # GCS'e durum bildir
            if not self.no_gcs:
                telem = self.mav.telemetry
                self.gcs.send_status("HOVERING", {
                    "lat": telem.lat, "lon": telem.lon, "alt": telem.alt_m
                })

        if self.sm.state_duration() >= self.hover_duration:
            self.sm.transition(State.COMPUTING, "Hover tamamlandı")

    def _step_computing(self):
        """Sensör füzyonu ile kazazede koordinatını hesapla."""
        target = self._active_target
        if target is None:
            self.sm.transition(State.HOVERING, "Hedef kayboldu")
            return

        telem   = self.mav.telemetry
        lidar_m = self.lidar.read()

        if lidar_m is None:
            logger.warning("LiDAR verisi yok, yeniden hover...")
            self.sm.transition(State.HOVERING, "LiDAR verisi eksik")
            return

        loc = self.calc.compute(
            track_id=target.track_id,
            pixel_center=target.center_px,
            lidar_distance_m=lidar_m,
            uav_lat=telem.lat,
            uav_lon=telem.lon,
            uav_alt=telem.alt_m,
            uav_yaw_deg=telem.yaw_deg,
            gimbal_pitch_deg=-90.0,
            confidence=target.confidence,
        )

        if loc is None:
            self.sm.transition(State.HOVERING, "Koordinat hesaplanamadı")
            return

        self._current_location = loc
        self.sm.transition(State.TRANSMITTING, f"Konum: {loc.latitude:.6f},{loc.longitude:.6f}")

    def _step_transmitting(self):
        """Koordinatı GCS'e ilet."""
        loc = self._current_location

        if not self.no_gcs:
            ok = self.gcs.send_casualty(loc)
            if not ok:
                logger.error("GCS iletimi başarısız!")
                # Yine de devam et (offline mod)
        else:
            logger.info(f"[no-gcs] Koordinat: {loc.latitude:.7f}, {loc.longitude:.7f}")

        # Engine'e teslim edildi olarak işaretle
        self.engine.mark_delivered(loc.track_id)
        self.sm.transition(State.RELEASING, "GCS iletimi tamamlandı")

    def _step_releasing(self):
        """Drone'u serbest bırak."""
        if self.sm.state_duration() < 0.1:
            self.release.release()
            time.sleep(1.0)      # ayrılma mekanizması tamamlanma süresi
            self.sm.transition(State.RESUMING, "Drone ayrıldı")

    def _step_resuming(self):
        """Göreve devam et (AUTO modu)."""
        if self.sm.state_duration() < 0.1:
            self.mav.resume_mission()
            self._active_target = None
            self._current_location = None
            logger.info("Tarama devam ediyor...")
            self.sm.transition(State.SCANNING, "Görev devam")

    # ------------------------------------------------------------------
    def _handle_signal(self, signum, frame):
        logger.warning(f"Sinyal alındı ({signum}), RTL başlatılıyor...")
        self._running = False
        self.mav.return_to_launch()
        self.sm.transition(State.RTL, "Manuel durdurma")


# ═════════════════════════════════════════════════════════════════════
# Mock LiDAR (--mock için)
# ═════════════════════════════════════════════════════════════════════
class _MockLidar:
    def start(self):  logger.info("[MOCK] LiDAR başlatıldı.")
    def stop(self):   pass
    def read(self):   return 12.5   # sabit 12.5 m mesafe simülasyonu


# ═════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════
def _parse_args():
    p = argparse.ArgumentParser(description="VTOL SAR Otonom Görev Sistemi")
    p.add_argument("--mock",   action="store_true", help="GPIO ve LiDAR simülasyonu")
    p.add_argument("--no-gcs", action="store_true", help="GCS olmadan çalış")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    mission = SARMission(mock=args.mock, no_gcs=args.no_gcs)
    mission.start()
