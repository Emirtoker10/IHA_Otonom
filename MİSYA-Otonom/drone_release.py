"""
drone_release.py
Yardım drone'unun ARM ve ayrılma mekanizması.
GPIO üzerinden servo/röle kontrolü (Raspberry Pi BCM pinleri).

Dışa aktarılan:
    DroneRelease
"""

import logging
import time
from config_loader import get

logger = logging.getLogger(__name__)


class DroneRelease:
    """
    Raspberry Pi GPIO üzerinden:
      - Drone motorlarını ARM eder (PWM sinyali veya dijital pin)
      - Mekanik ayrılma mekanizmasını tetikler (servo/röle)

    Donanım varsayımı:
        release_pin  → SG90 / DS3218 servo sinyal hattı
        arm_gpio_pin → drone FC'nin ARM girişi (3.3V lojik)

    Gerçek donanım yoksa MOCK modda çalışır (test için).
    """

    def __init__(self, mock: bool = False):
        cfg = get("drone")
        self.release_pin: int     = cfg["release_pin"]
        self.arm_pin: int         = cfg["arm_gpio_pin"]
        self.release_pulse: int   = cfg["release_pulse_ms"]   # servo: serbest
        self.lock_pulse: int      = cfg["lock_pulse_ms"]      # servo: kilitli
        self.mock = mock

        self._pwm = None
        self._gpio = None
        self._released = False
        self._armed    = False

    # ------------------------------------------------------------------
    def setup(self):
        """GPIO pinlerini hazırla."""
        if self.mock:
            logger.warning("DroneRelease MOCK modda — GPIO kullanılmıyor.")
            return

        try:
            import RPi.GPIO as GPIO
            self._gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            # Servo pini → PWM çıkış
            GPIO.setup(self.release_pin, GPIO.OUT)
            self._pwm = GPIO.PWM(self.release_pin, 50)  # 50 Hz servo frekansı
            self._pwm.start(self._ms_to_duty(self.lock_pulse))  # kilitli konumdan başla

            # ARM pini → dijital çıkış, başlangıçta LOW
            GPIO.setup(self.arm_pin, GPIO.OUT, initial=GPIO.LOW)

            logger.info(f"GPIO hazır — release_pin={self.release_pin}, arm_pin={self.arm_pin}")
        except ImportError:
            logger.error("RPi.GPIO bulunamadı. Mock moda geçiliyor.")
            self.mock = True

    def teardown(self):
        """GPIO'yu temizle."""
        if self.mock:
            return
        try:
            if self._pwm:
                self._pwm.stop()
            if self._gpio:
                self._gpio.cleanup()
            logger.info("GPIO temizlendi.")
        except Exception as e:
            logger.warning(f"GPIO teardown hatası: {e}")

    # ------------------------------------------------------------------
    def arm_drone(self):
        """Yardım drone'u motorlarını ARM eder."""
        if self._armed:
            logger.debug("Drone zaten ARM durumunda.")
            return

        logger.info("Drone ARM ediliyor...")
        if not self.mock:
            self._gpio.output(self.arm_pin, self._gpio.HIGH)
            time.sleep(0.5)  # FC ARM sinyali algılama süresi
        else:
            logger.info("[MOCK] Drone ARM sinyali gönderildi.")

        self._armed = True
        logger.info("Drone ARM tamamlandı.")

    def disarm_drone(self):
        """Yardım drone'u motorlarını DEARM eder."""
        if not self._armed:
            return
        if not self.mock:
            self._gpio.output(self.arm_pin, self._gpio.LOW)
        else:
            logger.info("[MOCK] Drone DEARM sinyali gönderildi.")
        self._armed = False
        logger.info("Drone DEARM edildi.")

    def release(self):
        """
        Ayrılma mekanizmasını tetikler.
        Servo kilitli konumdan serbest konuma hareket eder.
        """
        if self._released:
            logger.warning("Drone zaten ayrılmış durumda.")
            return

        logger.info("Ayrılma mekanizması aktive ediliyor...")
        if not self.mock:
            self._pwm.ChangeDutyCycle(self._ms_to_duty(self.release_pulse))
            time.sleep(0.8)  # servo hareket süresi
        else:
            logger.info("[MOCK] Servo → serbest konum.")

        self._released = True
        logger.info("Drone İHA'dan ayrıldı.")

    def lock(self):
        """Ayrılma mekanizmasını kilitli konuma geri alır (yeni drone için)."""
        if not self.mock:
            self._pwm.ChangeDutyCycle(self._ms_to_duty(self.lock_pulse))
            time.sleep(0.8)
        else:
            logger.info("[MOCK] Servo → kilitli konum.")
        self._released = False
        logger.info("Ayrılma mekanizması kilitlendi.")

    # ------------------------------------------------------------------
    @property
    def is_released(self) -> bool:
        return self._released

    @property
    def is_armed(self) -> bool:
        return self._armed

    @staticmethod
    def _ms_to_duty(pulse_ms: int) -> float:
        """Servo pulse (ms) → PWM duty cycle (%) @ 50 Hz."""
        # 50 Hz → periyot = 20 ms
        return (pulse_ms / 20.0) * 100.0
