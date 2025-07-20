import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import csv
from config import CONFIG
from pipeline import run_pipeline


def test_pipeline_output():
    print("\n🧪 Pipeline testi başlatıldı...\n")

    # Pipeline'ı çalıştır
    run_pipeline()

    output_dir = CONFIG["output"]["output_dir"]
    save_csv = CONFIG["output"]["save_csv"]
    save_video = CONFIG["output"]["save_video"]

    # CSV testi
    if save_csv:
        csv_path = os.path.join(output_dir, "output_keypoints.csv")
        assert os.path.exists(csv_path), "❌ CSV dosyası oluşturulmadı."
        with open(csv_path, newline="") as f:
            reader = list(csv.reader(f))
            assert len(reader) > 1, "❌ CSV boş görünüyor."
            header = reader[0]
            expected_header = ["frame", "person_id", "keypoint_name", "x", "y"]
            assert header == expected_header, f"❌ CSV başlıkları hatalı: {header}"
        print("✅ CSV çıktısı doğru.")

    # Video testi
    if save_video:
        video_path = os.path.join(output_dir, "output_video.mp4")
        assert os.path.exists(video_path), "❌ Video dosyası oluşturulmadı."
        assert os.path.getsize(video_path) > 0, "❌ Video dosyası boş."
        print("✅ Video çıktısı doğru.")

    print("\n✅ Pipeline testi başarıyla tamamlandı.\n")


if __name__ == "__main__":
    test_pipeline_output()
