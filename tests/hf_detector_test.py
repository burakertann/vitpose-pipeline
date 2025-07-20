import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from detectors.hf_detector import HFDetector
from config import CONFIG

def test_hf_detector():
    print("\n✅ HFDetector testi başlatıldı...\n")

    # 1. Config'ten ayarları al
    det_cfg = CONFIG["detector"]
    model_id = det_cfg["model_id"]
    print("Model = ",model_id)
    threshold = det_cfg["threshold"]
    print("threshold = ",threshold)
    device = det_cfg["device"]
    print("Device = ",device)

    # 2. Detector sınıfını başlat
    detector = HFDetector(model_id=model_id, threshold=threshold, device=device)

    # 3. Test için örnek bir görsel oku (kendi örnek görselini koyabilirsin)
    image_path = "/Users/burakertan/Desktop/vitpose_pipeline/tests/hf_detect_2.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Test görseli bulunamadı: {image_path}")
        return

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 4. Detection çalıştır
    bboxes = detector.detect(image_rgb)

    # 5. Sonuçları yazdır + doğrula
    print(f"🔍 Toplam {len(bboxes)} kişi bulundu.")
    for i, box in enumerate(bboxes):
        assert isinstance(box, list), "❌ Bbox tipi list değil"
        assert len(box) == 4, "❌ Bbox uzunluğu 4 değil"
        assert all(isinstance(coord, float) for coord in box), "❌ Koordinatlar float değil"
        print(f"✅ Kişi {i+1}: [x1={box[0]:.1f}, y1={box[1]:.1f}, x2={box[2]:.1f}, y2={box[3]:.1f}]")

    print("\n✅ HFDetector testi başarıyla tamamlandı.\n")
    
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

    cv2.imshow("Detection Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    test_hf_detector()
