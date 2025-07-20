import torch
import numpy as np
from typing import List
from PIL import Image
from transformers import VitPoseImageProcessor, VitPoseForPoseEstimation


class HFPoseEstimator:
    def __init__(self, model_id: str, device: str = "cpu"):
        self.device = device

        # Hugging Face pose processor ve modeli yükleniyor
        self.processor = VitPoseImageProcessor.from_pretrained(model_id)
        self.model = VitPoseForPoseEstimation.from_pretrained(model_id).to(device)
        self.model.eval()
    def predict(self, image: np.ndarray, bboxes: List[List[float]]) -> List[np.ndarray]:
        """
        Her bbox için 2D keypoint tahmini yapar.
        Girdi:
            image: RGB np.ndarray
            bboxes: [[x1, y1, x2, y2], ...]
        Çıktı:
            List of np.ndarray, each of shape (17, 3) — (x, y, confidence)
        """
        # NumPy → PIL
        pil_image = Image.fromarray(image)

        # (x1, y1, x2, y2) → (x, y, w, h) formatına çevirmemiz gerekiyor
        boxes_xywh = []
        for box in bboxes:
            x1, y1, x2, y2 = box
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])

        # Processor ile input hazırla
        inputs = self.processor(
            pil_image,
            boxes=[boxes_xywh],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Tahmin sonuçlarını al
        results = self.processor.post_process_pose_estimation(outputs, boxes=[boxes_xywh])
        
        results_for_single_image = results[0]

        keypoints_list = []
        # Şimdi kişileri içeren doğru liste üzerinde döngü kur
        for person_result in results_for_single_image:
            # 'keypoints' anahtarından gelen tensör zaten (1, 17, 3) şeklinde olmalı.
            keypoints_tensor = person_result['keypoints']
            # Sadece numpy'a çevirip baştaki boyutu kaldırmamız yeterli.
            keypoints_np = keypoints_tensor.cpu().numpy().squeeze()
            
            keypoints_list.append(keypoints_np)

        return keypoints_list
