import os
import cv2
import numpy as np
from ultralytics import YOLO
from . import utils
from pathlib import Path
import shutil


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


def enhance_pipeline(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12, 12))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def test_loop():
    model = YOLO('v12_runs/train/weights/best.pt')

    test_dir = Path("./data/test_images")
    temp_dir = Path("./data/test_images_clahe")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # CLAHE 보정 후 임시 폴더에 저장
    for img_path in test_dir.glob("*"):
        img_bgr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        img_bgr = enhance_pipeline(img_bgr)
        ext = img_path.suffix
        out_path = temp_dir / img_path.name
        result, buf = cv2.imencode(ext, img_bgr)
        if result:
            buf.tofile(str(out_path))

    # 보정된 이미지로 예측
    model.predict(
        source=str(temp_dir),
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

    # predict 후 임시 폴더 삭제
    if temp_dir.exists():
        shutil.rmtree(temp_dir)