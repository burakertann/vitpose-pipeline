# 🕺 ViTPose Pipeline

Bu proje, **Hugging Face tabanlı bir ViTPose modelini** kullanarak videolardan insan pozu (2D keypoint) tespiti yapmanızı sağlar. Algılanan pozlar hem görselleştirilir hem de `.csv` formatında kaydedilir. 

## 🚀 Özellikler
- Hugging Face'den direkt model ve detektör desteği
- COCO keypoint formatında poz tahmini
- Her kare için CSV çıktısı
- Video çıktısı kaydı (isteğe bağlı)
- Kolayca genişletilebilir modüler yapı

## 🛠️ Kurulum

Python 3.9+ ve `pip` yüklü bir ortamda çalışır. İlk olarak sanal ortam oluşturmanızı öneririz:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## 📁 Dosya Yapısı

```
vitpose_pipeline/
│
├── config.py               # Tüm ayarlar burada
├── main.py                 # Ana çalıştırılabilir dosya
├── pipeline.py             # Pipeline akışı
├── estimator.py            # Hugging Face tabanlı poz tahminci
├── detectors/
│   └── hf_detector.py      # Hugging Face object detection
├── tests/
│   └── pipeline_test.py    # Pipeline testi
│   └── estimator_test.py   # Estimator testi
└── outputs/                # Video ve CSV çıktıları
```

## ⚙️ Kullanım

Projeyi çalıştırmak için:

```bash
python main.py
```

Ayarları `config.py` içinden değiştirebilirsin:
- `video_path`: Girdi videosu
- `model_id`: Kullanılan ViTPose modeli (HF'den)
- `save_video`: Görselleştirilmiş video çıktısı
- `save_csv`: CSV olarak keypoint kaydı

## 🧠 Desteklenen Anahtar Noktalar

Model, COCO formatına göre 17 keypoint verir:
```
nose, left_eye, right_eye, left_ear, right_ear,
left_shoulder, right_shoulder, left_elbow, right_elbow,
left_wrist, right_wrist, left_hip, right_hip,
left_knee, right_knee, left_ankle, right_ankle
```

## 📋 Örnek CSV Çıktısı

```csv
frame,person_id,keypoint_name,x,y
0,0,nose,123.4,245.1
0,0,left_eye,120.0,240.2
...
```

## 📦 Model & Detektör

- Pose Estimator: `usyd-community/vitpose-base-simple`
- Detector: `PekingU/rtdetr_r50vd_coco_o365`

## 📄 Lisans

[Apache 2.0](LICENSE)