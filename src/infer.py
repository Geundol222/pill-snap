from ultralytics import YOLO
from . import utils


def validation_score():
    model = YOLO('v12_runs/train/weights/best.pt')

    metrics = model.val(
        data='../configs/yolo_data.yaml',
        imgsz=960,
        project="v12_runs",
        seed=42,
        exist_ok=True
    )

    ap_75_95 = metrics.box.all_ap[:, 5:].mean()
    print(f'mAP@[0.75:0.95] : {ap_75_95:.2f}')

def test_loop():
    model = YOLO('v12_runs/train/weights/best.pt')

    # 보정된 이미지로 예측
    model.predict(
        source="./data/test_images",
        imgsz=960,
        conf=0.25,
        save=True,
        save_txt=True,
        save_conf=True,
        project="v12_runs",
        seed=42,
        exist_ok=True
    )

    utils.save_csv()