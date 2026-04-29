# 🚁 Teknofest İHA Otonom Arama-Kurtarma Sistemi

Dağlık alanlarda kaybolan bireylerin tespiti ve ilk yardım paketinin ulaştırılması için geliştirilmiş iki araçlı otonom İHA sistemi.

---

## 📌 Sistem Mimarisi

```
VTOL İHA (Sabit Kanat)
  → Geniş alan grid taraması
  → YOLOv8 ile kazazede tespiti
  → TF03-180 LiDAR ile irtifa ölçümü
  → GPS + LiDAR + Kamera FOV → Koordinat hesaplama
  → Koordinatı Drone'a MAVLink ile iletme

Multikopter Drone
  → Koordinatı al
  → Otonom uçuş (GUIDED mod)
  → Hover + Servo ile yük bırakma
  → RTL (Eve dönüş)
```

---

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Görev |
|-----------|-------|
| **ArduPilot SITL** | Uçuş kontrol simülasyonu |
| **Gazebo Harmonic** | 3D fizik simülasyonu |
| **MAVSDK Python** | İHA/Drone kontrolü |
| **YOLOv8x** | Gerçek zamanlı kişi tespiti |
| **TF03-180 LiDAR** | Yerden yükseklik ölçümü |
| **MAVLink** | İki araç arası haberleşme |
| **OpenCV** | Görüntü işleme |

---

## 🔧 Donanım

### VTOL İHA
- Pixhawk Orange Cube+ (Uçuş Kontrol)
- Raspberry Pi 5 + AI Hat+ 26T (Görev Bilgisayarı)
- Here 3 GNSS (RTK, 2.5cm hassasiyet)
- Raspberry Pi HQ Kamera (8mm, Global Shutter)
- TF03-180 LiDAR Mesafe Ölçer

### Multikopter Drone
- Pixhawk 2.4.8 (Uçuş Kontrol)
- Neo-M8N GNSS
- TF03-180 LiDAR
- MG90S Servo Motor (Yük Bırakma)

---

## 💻 Kurulum

### Gereksinimler
```bash
# ArduPilot SITL
git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot && git submodule update --init --recursive

# Gazebo Harmonic
sudo apt install gz-harmonic

# ardupilot_gazebo plugin
git clone https://github.com/ArduPilot/ardupilot_gazebo.git

# Python bağımlılıkları
pip install mavsdk ultralytics opencv-python numpy
```

### Ortam Değişkenleri
```bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo/models:$HOME/SITL_Models/Gazebo/models:$GZ_SIM_RESOURCE_PATH
```

---

## 🚀 Simülasyon Başlatma

### 1. Gazebo
```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia gz sim -v4 -r ~/SITL_Models/Gazebo/worlds/alti_transition_runway.sdf
```

### 2. ArduCopter SITL (Drone)
```bash
cd ~/ardupilot/ArduCopter
sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --console --map -I 1 --sysid 2
```

### 3. ArduPlane SITL (VTOL İHA)
```bash
cd ~/ardupilot/ArduPlane
sim_vehicle.py -v ArduPlane \
  --model JSON \
  --add-param-file=$HOME/SITL_Models/Gazebo/config/alti_transition_quad.param \
  --console --map -I 0 --sysid 1
```
### Her iki terminalde de : 
param set ARMING_SKIPCHK 1

### 4. VTOL İHA Kalkışı (MAVProxy Console)
```
mode GUIDED
arm throttle force
takeoff 50
```

### İnsan eklemek 
```
gz service -s /world/runway/create \
  --reqtype gz.msgs.EntityFactory \
  --reptype gz.msgs.Boolean \
  --timeout 1000 \
  --req 'sdf_filename: "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Standing%20person" pose: {position: {x: 50, y: 50, z: 0}}'
```
### 5. Otonom Görev Scripti
```bash
python3 otonom_gorev.py
```

### Simülasyonu Kapatmak
```bash
pkill -9 -f gz && pkill -9 -f arduplane && pkill -9 -f arducopter
```

---

## 📁 Dosya Yapısı

```
├── otonom_gorev.py        # Ana otonom görev scripti
├── koordinat_hesapla.py   # Piksel → GPS koordinat dönüşümü
├── drone_gonder.py        # Drone kontrolü
├── yolo_gazebo.py         # YOLO görüntü işleme
└── README.md
```

---

## 🎯 Görev Akışı

```
1. VTOL İHA kalkış (50m)
2. Gazebo kamerasından görüntü al
3. YOLOv8 ile kişi tespit et
4. TF03 LiDAR'dan irtifa al
5. Piksel + GPS + LiDAR → Koordinat hesapla
6. Drone'a koordinat gönder (MAVLink)
7. Drone hedefe git (GUIDED mod)
8. Hover + Yük bırak (Servo)
9. Drone RTL
10. VTOL taramaya devam
```

---

## 📐 Koordinat Hesaplama Algoritması

```python
# İHA'nın GPS konumu + LiDAR irtifası + Kamera FOV
# → Kazazedenin piksel konumu → GPS koordinatı

norm_x = (pixel_x - IMAGE_WIDTH/2) / (IMAGE_WIDTH/2)
norm_y = (pixel_y - IMAGE_HEIGHT/2) / (IMAGE_HEIGHT/2)

dx = altitude * tan(FOV_H/2) * norm_x
dy = altitude * tan(FOV_V/2) * norm_y

target_lat = iha_lat + degrees(dy_rotated / R)
target_lon = iha_lon + degrees(dx_rotated / (R * cos(radians(iha_lat))))
```

---

## ⚠️ Port Yapılandırması

| Araç | Instance | Gazebo Port | MAVProxy Port |
|------|----------|-------------|---------------|
| VTOL İHA | -I 0, sysid 1 | 9002 | 14550 |
| Drone | -I 1, sysid 2 | 9012 | 14560 |

---

## 📊 Kamera Parametreleri

| Parametre | Değer |
|-----------|-------|
| Lens | 8mm telephoto |
| Çözünürlük | 1920x1080 |
| FOV (Yatay) | 60° |
| Shutter | Global (1/2500s) |
| GSD @ 50m | 2.05 cm/px |
| Motion Blur | 0.448 px |

---

## 👥 Teknofest 2026

Bu proje Teknofest İHA yarışması kapsamında geliştirilmiştir.
