import os


import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO

def validation_score():
    model = YOLO('../../runs/detect/train/weights/best.pt')

    metrics = model.val(data='../../configs/yolo_data.yaml', imgsz=640)

    ap_75_95 = metrics.box.all_ap[:, 5:].mean()
    print("mAP@[0.75:0.95] :", ap_75_95)

def test_loop():
    model = YOLO('../../runs/detect/train/weights/best.pt')

    model.predict(
        source="../../data/test_images",
        imgsz=640,
        conf=0.25,
        save=True,
        save_txt=True
    )

def main():
    """
    Main 실행함수

    Description:
        Yolo의 경우 augmentation이나 여러 전처리가 전부 라이브러리 안에 내장되어있고 학습시 알아서 전처리를 해주기때문에 편합니다.
        또한 train함수를 제작할 필요 없이 model.train으로 파라미터만 잘 설정해 주면 되겠습니다.

        더 높은 학습 확률을 위해 파라미터에서 augmentation 옵션을 바꿀 수 있습니다.

        - 예시
        model.train(
            data='../../configs/yolo_data.yaml',
            epochs=100,     # 에폭 수
            imgsz=640,      # 이미지 resize(정사각형)
            batch=16,       # 배치 사이즈(DataLoader의 그것)
            degrees=10,     # RandomRoation
            scale=0.5,      # RandomScale
            fliplr=0.5,     # RandomHorizontalFlip
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,      # hsv 3종: ColorJitter와 비슷
            mosaic=1.0,     # 4장의 이미지를 랜덤하게 합쳐서 한장으로 만듬
            mixup=0.2       # 두장의 이미지를 섞어서 새로운 이미지를 만듬 -> 작은 데이터 셋에서 과적합 방지
        )
    """

    model = YOLO('yolov8s.pt')

    model.train(
        data='../../configs/yolo_data.yaml',
        epochs=5,
        imgsz=640,
        batch=16,
        device=0
    )

if __name__ == "__main__":
    rcParams['font.family'] = 'Malgun Gothic'
    rcParams['axes.unicode_minus'] = False

    main()
    validation_score()
    test_loop()