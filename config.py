CONFIG = {
    "detector": {
        "type": "hf",
        "model_id": "PekingU/rtdetr_r50vd_coco_o365",
        "threshold": 0.3,
        "device": "cpu"
    },
    "pose_estimator": {
        "type": "hf",
        "model_id": "usyd-community/vitpose-base-simple",
        "device": "cpu"
    },
    "input": {
        "video_path": "/Users/burakertan/Desktop/vitpose_pipeline/video.mp4",
    },
    "output": {
        "save_video": True,
        "save_csv": True,
        "output_dir": "outputs/"
    }
}

"""
Object detection için en çok indirilen 20 Model:
--- Bulunan Modeller ---
PekingU/rtdetr_r101vd_coco_o365
PekingU/rtdetr_v2_r18vd
PekingU/rtdetr_r50vd_coco_o365
PekingU/rtdetr_r50vd
PaddlePaddle/RT-DETR-L_wired_table_cell_det
PaddlePaddle/RT-DETR-L_wireless_table_cell_det
PekingU/rtdetr_v2_r50vd
PekingU/rtdetr_r18vd
PekingU/rtdetr_v2_r101vd
PekingU/rtdetr_r101vd
PekingU/rtdetr_r18vd_coco_o365
Yifeng-Liu/rt-detr-finetuned-for-satellite-image-roofs-detection
PekingU/rtdetr_v2_r34vd
jadechoghari/RT-DETRv2
PaddlePaddle/RT-DETR-H_layout_3cls
PekingU/rtdetr_r34vd
amirivojdan/rtdetr-v2-r50-cppe5-finetune-2
goodcasper/see_ai_rt-detr_r18_only_bbox_da
goodcasper/see_ai_rt-detr_r18_only_bbox
goodcasper/see_ai_rt-detr_r50_4090_only_bbox_da
"""

"""

Pose Estimation için en çok indirilen 20 Model:

--- Sadece ViTPose++ Modelleri ---
Not: '++' genellikle model adlarında 'plus' olarak yazılır.

Model ID: usyd-community/vitpose-plus-huge                   | İndirme: 24,446
Model ID: usyd-community/vitpose-plus-base                   | İndirme: 12,631
Model ID: usyd-community/vitpose-plus-large                  | İndirme: 4,645
Model ID: usyd-community/vitpose-plus-small                  | İndirme: 3,179
Model ID: onnx-community/vitpose-plus-small-ONNX             | İndirme: 18
Model ID: danelcsb/vitpose-plus-base                         | İndirme: 2
Model ID: LPX55/vitpose-plus-base-ONNX                       | İndirme: 1

"""