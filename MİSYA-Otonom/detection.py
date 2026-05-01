"""
detection.py
YOLOv8 nesne tespiti + ByteTrack takibi + N-frame doğrulama.

Dışa aktarılan:
    DetectionEngine  — tek instance, main tarafından kullanılır.
    Detection        — tek bir tespiti temsil eden dataclass.
    ConfirmedTarget  — N-frame eşiğini aşmış onaylı hedef.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config_loader import get

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Veri yapıları
# ─────────────────────────────────────────────

@dataclass
class Detection:
    """Tek bir YOLO tespiti."""
    track_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]      # x1, y1, x2, y2  (piksel)
    center_px: Tuple[int, int]           # (cx, cy)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConfirmedTarget:
    """N-frame doğrulama eşiğini aşmış kazazede."""
    track_id: int
    center_px: Tuple[int, int]
    confidence: float
    frame_count: int
    first_seen: float
    last_seen: float


# ─────────────────────────────────────────────
# Yardımcı: track geçmişi
# ─────────────────────────────────────────────

class _TrackHistory:
    """Bir track_id için ardışık frame sayacı."""

    def __init__(self, confirm_frames: int, max_lost: int):
        self.confirm_frames = confirm_frames
        self.max_lost = max_lost
        self._counters: Dict[int, dict] = {}

    def update(self, detections: List[Detection]) -> List[ConfirmedTarget]:
        """
        Tespit listesini al, sayaçları güncelle.
        Döner: Bu frame'de eşiği yeni aşan/devam eden ConfirmedTarget listesi.
        """
        seen_ids = {d.track_id for d in detections}

        # Kayıp frame sayacını artır
        for tid in list(self._counters.keys()):
            if tid not in seen_ids:
                self._counters[tid]["lost"] += 1
                if self._counters[tid]["lost"] > self.max_lost:
                    del self._counters[tid]

        # Mevcut tespitleri işle
        confirmed: List[ConfirmedTarget] = []
        for det in detections:
            tid = det.track_id
            if tid not in self._counters:
                self._counters[tid] = {
                    "count": 0,
                    "lost": 0,
                    "first_seen": det.timestamp,
                }
            entry = self._counters[tid]
            entry["count"] += 1
            entry["lost"] = 0

            if entry["count"] >= self.confirm_frames:
                confirmed.append(
                    ConfirmedTarget(
                        track_id=tid,
                        center_px=det.center_px,
                        confidence=det.confidence,
                        frame_count=entry["count"],
                        first_seen=entry["first_seen"],
                        last_seen=det.timestamp,
                    )
                )

        return confirmed

    def reset(self, track_id: int):
        self._counters.pop(track_id, None)


# ─────────────────────────────────────────────
# Ana motor
# ─────────────────────────────────────────────

class DetectionEngine:
    """
    YOLOv8 + ByteTrack + N-frame doğrulama motoru.

    Kullanım:
        engine = DetectionEngine()
        engine.load()
        detections, confirmed = engine.process(frame)
    """

    def __init__(self):
        yolo_cfg = get("yolo")
        det_cfg   = get("detection")

        self.model_path: str   = yolo_cfg["model_path"]
        self.conf_thresh: float = yolo_cfg["confidence_threshold"]
        self.iou_thresh: float  = yolo_cfg["iou_threshold"]
        self.target_class: str  = yolo_cfg["target_class"]
        self.device: str        = yolo_cfg.get("device", "cpu")

        self._confirm_frames: int = det_cfg["confirm_frames"]
        self._max_lost: int       = det_cfg["max_lost_frames"]
        self._min_age: int        = det_cfg["min_track_age"]

        self._model = None
        self._class_names: Dict[int, str] = {}
        self._history = _TrackHistory(self._confirm_frames, self._max_lost)

        # Daha önce GCS'e gönderilen track id'leri (tekrar gönderme önlemi)
        self._delivered_ids: set = set()

    # ------------------------------------------------------------------
    def load(self):
        """Modeli yükle (main başlangıcında bir kez çağrılır)."""
        from ultralytics import YOLO
        logger.info(f"YOLO modeli yükleniyor: {self.model_path} (device={self.device})")
        self._model = YOLO(self.model_path)
        self._model.to(self.device)
        self._class_names = self._model.names
        logger.info(f"Model yüklendi. Sınıflar: {list(self._class_names.values())[:10]}")

    # ------------------------------------------------------------------
    def process(
        self, frame: np.ndarray
    ) -> Tuple[List[Detection], List[ConfirmedTarget]]:
        """
        Tek bir frame'i işle.

        Returns:
            detections  — Bu frame'deki tüm geçerli tespitler
            confirmed   — N-frame eşiğini aşmış, henüz teslim edilmemiş hedefler
        """
        if self._model is None:
            raise RuntimeError("DetectionEngine.load() çağrılmadı.")

        results = self._model.track(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            persist=True,           # ByteTrack hafızası korunur
            tracker="bytetrack.yaml",
            verbose=False,
            device=self.device,
        )

        detections: List[Detection] = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                cls_name = self._class_names.get(cls_id, "unknown")
                if cls_name != self.target_class:
                    continue

                conf = float(box.conf[0])
                if conf < self.conf_thresh:
                    continue

                track_id = int(box.id[0]) if box.id is not None else -1
                if track_id < 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                detections.append(
                    Detection(
                        track_id=track_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        center_px=(cx, cy),
                    )
                )

        confirmed_all = self._history.update(detections)

        # Daha önce teslim edilmişleri filtrele
        confirmed_new = [
            t for t in confirmed_all if t.track_id not in self._delivered_ids
        ]

        return detections, confirmed_new

    def mark_delivered(self, track_id: int):
        """Bir hedefin koordinat iletiminin tamamlandığını işaretle."""
        self._delivered_ids.add(track_id)
        self._history.reset(track_id)

    def get_annotated_frame(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Görüntü üzerine bounding box ve bilgi çizer (opsiyonel / debug).
        """
        import cv2
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{det.track_id} {det.confidence:.2f}"
            cv2.putText(out, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return out
