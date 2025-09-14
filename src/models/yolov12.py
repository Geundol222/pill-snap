import os


import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO

def validation_score():
    model = YOLO('../../runs/detect/train/weights/best.pt')

    metrics = model.val(
        data='../../configs/yolo_data.yaml',
        imgsz=640,
        project="v12_runs",
        exist_ok=True
    )

    ap_75_95 = metrics.box.all_ap[:, 5:].mean()
    print(f'mAP@[0.75:0.95] : {ap_75_95:.2f}')

def test_loop():
    model = YOLO('../../runs/detect/train/weights/best.pt')

    model.predict(
        source="../../data/test_images",
        imgsz=640,
        conf=0.25,
        save=True,
        save_txt=True,
        project="v12_runs",
        exist_ok=True
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
            epochs=100,         # 에폭 수
            imgsz=640,          # 이미지 resize(정사각형)
            batch=16,           # 배치 사이즈(DataLoader의 그것)
            degrees=10,         # RandomRoation
            scale=0.5,          # RandomScale
            fliplr=0.5,         # RandomHorizontalFlip
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,          # hsv 3종: ColorJitter와 비슷
            mosaic=1.0,         # 4장의 이미지를 랜덤하게 합쳐서 한장으로 만듬
            mixup=0.2,          # 두장의 이미지를 섞어서 새로운 이미지를 만듬 -> 작은 데이터 셋에서 과적합 방지
            project="v12_runs", # yolo model은 학습을 돌릴경우 디렉토리 에 runs폴더를 만들게 되는데, 버전마다 다르게 생성하기 위해 이름을 커스텀하여 폴더를 생성할 수 있습니다.
            name="v12_train",   # 해당 작업이 어떤 작업인지를 명시해서 폴더링할 수도 있습니다. 이 기능은 사용자의 선택이지만, 기본적으로 train, val, predict의 폴더가 작업마다 생성되므로 꼭 필요한 파라미터는 아닙니다.
            exist_ok=True       # 이미 폴더가 있어도 덮어쓰기 허용
        )

    ==========================================================================================================================================================================================================================
    LHS:
        가정:
            - 에폭수 증가 실험
            - 약들이 많이 기울어져 있는 경우가 있고, 약의 크기는 비슷하고 글자만 다른 약들이 존재하기때문에 Augmentation의 RandomAffine을 좀 강하게 걸어주고 실험 [degree(회전범위), trainslate(이미지 이동), scale(확대 축소), shear(기울임)]
            - 약들의 글자가 너무 흐릿하거나 알아볼 수 없는경우가 있음, hsv를 조절해서 약의 이미지를 조금 뚜렷하게 해보는 실험 [hsv_h(hue), hsv_s(saturation), hsv_v(value + constrast 혼합)]
            - mixup을 사용하면 글씨 디테일을 망가뜨릴 수가 있기 때문에 데이터의 양을 고려해서 0.1~0.2 혹은 0.0으로 실험
            - bbox의 누락이 문제가 되는 경우도 있으므로 cutmix를 진행할지, EDA의 결정을 따를지 고민
            - 이미지의 resize 크기가 작아 약의 글씨 디테일을 살리지 못할수 있으므로, 이미지 resize 크기 키우기

        results:
            - [No 전처리, No EDA] 5epochs => Validataion mAP@[0.75:0.95]: 0.7997302769703274(약 0.8)
    ==========================================================================================================================================================================================================================
    """

    model = YOLO('yolo12s.pt')

    model.train(
        data='../../configs/yolo_data.yaml',
        epochs=5,
        imgsz=640,
        batch=16,
        device=0,
        project="v12_runs",
        exist_ok=True
    )

if __name__ == "__main__":
    rcParams['font.family'] = 'Malgun Gothic'
    rcParams['axes.unicode_minus'] = False

    main()
    validation_score()
    test_loop()