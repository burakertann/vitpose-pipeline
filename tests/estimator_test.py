import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from config import CONFIG
from estimator import HFPoseEstimator
from detectors.hf_detector import HFDetector
from typing import List
import matplotlib.pyplot as plt


def draw_keypoints(image: np.ndarray, keypoints_list: List[np.ndarray]) -> np.ndarray:
    """
    Her kişi için keypoint'leri (eklemleri) ve iskeleti çizer.
    """
    # COCO formatına göre iskelet bağlantıları
    skeleton = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (0, 1), (0, 2), (1, 3), (2, 4)
    ]

    # Her bir kişi için döngü
    for keypoints in keypoints_list:
        # 1. İskelet hatlarını çiz
        for i, j in skeleton:
            start_point = keypoints[i]
            end_point = keypoints[j]

            # Koordinatları integer'a çevir
            x1, y1 = int(start_point[0]), int(start_point[1])
            x2, y2 = int(end_point[0]), int(end_point[1])

            # Çizgiyi çiz
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Yeşil çizgi

        # 2. Eklem noktalarını (keypoint'leri) çiz
        for x, y in keypoints:
            cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1) # Kırmızı nokta

    return image


def test_hf_pose_estimator():
    print("\n✅ HFPoseEstimator testi başlatıldı...\n")

    # Config'ten ayarları al
    det_cfg = CONFIG["detector"]
    pose_cfg = CONFIG["pose_estimator"]

    # Detector ve Pose modelini başlat
    detector = HFDetector(model_id=det_cfg["model_id"], threshold=det_cfg["threshold"], device=det_cfg["device"])
    pose_estimator = HFPoseEstimator(model_id=pose_cfg["model_id"], device=pose_cfg["device"])

    # Test görüntüsü
    image_path = "/Users/burakertan/Desktop/vitpose_pipeline/tests/hf_detect_2.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Test görseli bulunamadı: {image_path}")
        return

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Bbox tespiti
    bboxes = detector.detect(image_rgb)
    print(f"🔍 {len(bboxes)} kişi bulundu.")

    if len(bboxes) == 0:
        print("⚠️ Bbox bulunamadı, test iptal.")
        return

    # Pose tahmini
    keypoints_list = pose_estimator.predict(image_rgb, bboxes)

    # Format kontrolü
    for idx, keypoints in enumerate(keypoints_list):
        assert isinstance(keypoints, np.ndarray), "❌ Keypoint tipi ndarray değil"
        assert keypoints.shape == (17, 2), f"❌ Keypoint shape {keypoints.shape}, (17, 3) bekleniyordu"
        assert np.all(np.isfinite(keypoints)), "❌ Keypoint içinde NaN/inf var"
        print(f"✅ Kişi {idx+1} keypoint formatı doğru")

    # Görsel çizim ve gösterim
    output_image = draw_keypoints(image_rgb.copy(), keypoints_list)

    plt.imshow(output_image)
    plt.axis("off")
    plt.title("Pose Estimation")
    plt.show()

    print("\n✅ HFPoseEstimator testi başarıyla tamamlandı.\n")


if __name__ == "__main__":
    test_hf_pose_estimator()
