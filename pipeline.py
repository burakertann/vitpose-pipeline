import os
import cv2
import csv
import time
import numpy as np
from config import CONFIG
from detectors.hf_detector import HFDetector
from estimator import HFPoseEstimator
from tests.estimator_test import draw_keypoints  # Aynı çizim fonksiyonunu kullanacağız

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def run_pipeline():
    print("\n🚀 Pipeline başlatılıyor...\n")

    # Config’leri al
    det_cfg = CONFIG["detector"]
    pose_cfg = CONFIG["pose_estimator"]
    video_path = CONFIG["input"]["video_path"]
    output_dir = CONFIG["output"]["output_dir"]
    save_video = CONFIG["output"]["save_video"]
    save_csv = CONFIG["output"]["save_csv"]

    os.makedirs(output_dir, exist_ok=True)

    # Detector ve Pose Estimator başlat
    detector = HFDetector(model_id=det_cfg["model_id"], threshold=det_cfg["threshold"], device=det_cfg["device"])
    pose_estimator = HFPoseEstimator(model_id=pose_cfg["model_id"], device=pose_cfg["device"])

    # Video aç
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Video açılamadı: {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Video kaydı (isteğe bağlı)
    if save_video:
        out_path = os.path.join(output_dir, "output_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # CSV için
    if save_csv:
        csv_path = os.path.join(output_dir, "output_keypoints.csv")
        csv_data = [["frame", "person_id", "keypoint_name", "x", "y"]]


    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Bbox bul
        bboxes = detector.detect(rgb)
        if len(bboxes) > 0:
            keypoints_list = pose_estimator.predict(rgb, bboxes)

            # Çizim
            frame = draw_keypoints(frame, keypoints_list)

            if save_csv:
                for pid, keypoints in enumerate(keypoints_list):
                    for kp_idx, (x, y) in enumerate(keypoints):
                        kp_name = COCO_KEYPOINT_NAMES[kp_idx]
                        csv_data.append([frame_id, pid, kp_name, float(x), float(y)])

        if save_video:
            out_writer.write(frame)

        frame_id += 1
        if frame_id % 10 == 0:
            print(f"▶️ {frame_id} frame işlendi...")

    # Kapat
    cap.release()
    if save_video:
        out_writer.release()

    if save_csv:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)

    print("\n✅ Pipeline başarıyla tamamlandı.\n")
