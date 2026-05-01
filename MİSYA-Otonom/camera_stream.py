"""
camera_stream.py
CSI kamera akışını yönetir.
Birincil: Picamera2 (libcamera)
Geri dönüş: OpenCV VideoCapture (v4l2)

Kullanım:
    from camera_stream import CameraStream
    cam = CameraStream()
    cam.start()
    frame = cam.read()
    cam.stop()
"""

import logging
import threading
import time
import numpy as np

from config_loader import get

logger = logging.getLogger(__name__)


class CameraStream:
    """
    Thread-safe CSI kamera okuyucu.
    read() her zaman en son frame'i döner; eski frame'leri biriktirmez.
    """

    def __init__(self):
        cfg = get("camera")
        self.width: int  = cfg["width"]
        self.height: int = cfg["height"]
        self.fps: int    = cfg["fps"]
        self.use_libcamera: bool = cfg.get("use_libcamera", True)
        self.index: int  = cfg.get("index", 0)

        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._capture = None  # OpenCV cap veya Picamera2 instance

    # ------------------------------------------------------------------
    def start(self):
        if self._running:
            return
        self._running = True
        if self.use_libcamera:
            self._init_picamera2()
        else:
            self._init_opencv()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="CameraThread")
        self._thread.start()
        logger.info(f"Kamera başlatıldı ({self.width}x{self.height}@{self.fps}fps, "
                    f"libcamera={self.use_libcamera})")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self._release()
        logger.info("Kamera durduruldu.")

    def read(self) -> np.ndarray | None:
        """Son frame'i döner (BGR, uint8). Henüz frame yoksa None."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_opened(self) -> bool:
        return self._running and self._frame is not None

    # ------------------------------------------------------------------
    def _init_picamera2(self):
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            config = cam.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls={"FrameRate": float(self.fps)},
            )
            cam.configure(config)
            cam.start()
            self._capture = cam
            self._backend = "picamera2"
        except Exception as e:
            logger.warning(f"Picamera2 başlatılamadı ({e}), OpenCV'ye geçiliyor.")
            self.use_libcamera = False
            self._init_opencv()

    def _init_opencv(self):
        import cv2
        cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS,          self.fps)
        if not cap.isOpened():
            raise RuntimeError(f"Kamera açılamadı: /dev/video{self.index}")
        self._capture = cap
        self._backend = "opencv"

    def _loop(self):
        import cv2
        while self._running:
            try:
                if self._backend == "picamera2":
                    frame_rgb = self._capture.capture_array()          # RGB
                    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) # BGR
                else:
                    ret, frame = self._capture.read()
                    if not ret:
                        logger.warning("Kamera frame okunamadı, yeniden deneniyor...")
                        time.sleep(0.05)
                        continue

                with self._lock:
                    self._frame = frame

            except Exception as e:
                logger.error(f"Kamera döngüsü hatası: {e}")
                time.sleep(0.1)

    def _release(self):
        if self._capture is None:
            return
        try:
            if self._backend == "picamera2":
                self._capture.stop()
                self._capture.close()
            else:
                self._capture.release()
        except Exception as e:
            logger.warning(f"Kamera kapatma hatası: {e}")
        self._capture = None
