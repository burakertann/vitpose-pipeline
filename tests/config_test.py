import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CONFIG

def test_config():
    print("\n✅ CONFIG test başlatıldı...\n")

    det_cfg = CONFIG["detector"]
    pose_cfg = CONFIG["pose_estimator"]
    input_cfg = CONFIG["input"]
    output_cfg = CONFIG["output"]

    print("🔍 Detector Model:", det_cfg["model_id"])
    print("Detectorun kullandığı Device: ",det_cfg["device"])
    print("🧍 Pose Model:", pose_cfg["model_id"])
    print("Pose Modelin kullandığı Device: ",pose_cfg["device"])
    print("🎥 Video Path:", input_cfg["video_path"])
    print("💾 Save Video?", output_cfg["save_video"])
    print("📄 Save CSV?", output_cfg["save_csv"])
    print("📁 Output Dir:", output_cfg["output_dir"])

    print("\n✅ CONFIG test tamamlandı.\n")

if __name__ == "__main__":
    test_config()