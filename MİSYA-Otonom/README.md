# VTOL SAR — Otonom Kazazede Tespit ve Yardım Sistemi

## Proje Yapısı

```
vtol_sar/
├── main.py                   # Ana orkestratör — buradan çalıştır
├── state_machine.py          # Görev durum makinesi (SCANNING → RTL)
├── config_loader.py          # YAML config okuyucu (önbellekli)
├── logger_setup.py           # Merkezi loglama (console + dosya)
│
├── camera_stream.py          # CSI kamera (Picamera2 / OpenCV fallback)
├── detection.py              # YOLOv8 + ByteTrack + N-frame doğrulama
├── lidar_reader.py           # TFmini LiDAR seri okuyucu
├── coordinate_calculator.py  # Sensör füzyonu → GPS koordinatı
├── mavlink_controller.py     # Pixhawk MAVLink arayüzü
├── drone_release.py          # GPIO ARM + ayrılma servo kontrolü
├── gcs_communicator.py       # UDP/TCP GCS haberleşmesi
│
├── config/
│   └── config.yaml           # Tüm parametreler burada
├── models/
│   └── yolov8s.pt            # YOLO ağırlık dosyası (buraya koy)
└── logs/
    └── vtol_sar.log          # Otomatik oluşturulur
```

## Kurulum

```bash
# 1. Bağımlılıkları yükle
pip install -r requirements.txt --break-system-packages

# 2. YOLO modelini indir
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
mv yolov8s.pt models/

# 3. config/config.yaml dosyasını kendi donanımına göre düzenle
#    - mavlink.connection_string  → Pixhawk seri portu
#    - gcs.host                   → GCS IP adresi
#    - lidar.port                 → LiDAR seri portu
#    - camera_calibration.*       → fx, fy, cx, cy değerleri
```

## Çalıştırma

```bash
# Normal görev (tüm donanım bağlı)
python main.py

# GPIO & LiDAR simülasyonu (Pi olmayan bilgisayarda test)
python main.py --mock

# GCS'siz test
python main.py --mock --no-gcs
```

## Görev Akışı

```
SCANNING → tespit + N-frame doğrulama
         ↓
DECELERATING → yavaşlama
         ↓
HOVERING → LOITER modu + drone ARM + sensör bekleme
         ↓
COMPUTING → LiDAR + GPS + kamera kalibrasyonu → GPS koordinatı
         ↓
TRANSMITTING → GCS'e UDP/TCP JSON gönderimi
         ↓
RELEASING → GPIO servo → drone ayrılma
         ↓
RESUMING → AUTO moda dön, tarama devam
         ↓
SCANNING → (döngü)
```

## Modül İçeri Aktarma Örneği

```python
# Sadece deteksiyon motorunu test etmek için:
from detection import DetectionEngine
import cv2

engine = DetectionEngine()
engine.load()

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
detections, confirmed = engine.process(frame)
print(confirmed)
```

## Kamera Kalibrasyonu

`config/config.yaml` içindeki `camera_calibration` bölümündeki
`fx`, `fy`, `cx`, `cy` değerlerini kendi kamerana göre gir.
Pi Camera v2 için varsayılan değerler yaklaşık olarak doğrudur.
Gerçek kalibrasyon için OpenCV'nin chessboard kalibrasyonunu kullan.

## Donanım Bağlantıları

| Bileşen         | Bağlantı                      |
|-----------------|-------------------------------|
| Pixhawk UART    | `/dev/ttyAMA0` (Pi UART)      |
| LiDAR TFmini   | `/dev/ttyUSB0` (USB-serial)   |
| Servo (release) | GPIO 17 (BCM)                 |
| Drone ARM pin   | GPIO 27 (BCM)                 |
| CSI Kamera      | Pi Camera konektörü           |
