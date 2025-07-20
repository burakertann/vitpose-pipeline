import torch
import numpy as np
from typing import List
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection


class HFDetector:
    def __init__(self, model_id: str, threshold: float = 0.5, device: str = "cpu"):
        self.device = device
        self.threshold = threshold

        # Hugging Face processor ve model yükleniyor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = RTDetrForObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()
    def detect(self, image: np.ndarray) -> List[List[float]]:
        """
        RGB formatındaki görüntüdeki 'person' sınıfındaki bbox'ları döndürür.
        Girdi: RGB np.ndarray
        Çıktı: [[x1, y1, x2, y2], ...]
        """
        # NumPy → PIL formatına çeviriyoruz
        pil_image = Image.fromarray(image)

        # Image processor ile tensöre çevir
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Sonuçları post-process ediyoruz
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=[pil_image.size[::-1]],  # (height, width)
            threshold=self.threshold
        )[0]

        bboxes = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if label.item() == 0:  # COCO'da 0 → person sınıfı
                x1, y1, x2, y2 = box.tolist()
                bboxes.append([x1, y1, x2, y2])

        return bboxes
