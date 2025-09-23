import os
import json
import yaml


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from matplotlib import rcParams
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from ultralytics import YOLO

def validation_score():
    model = YOLO('v12_test_runs/train/weights/best.pt')

    metrics = model.val(
        data='../../configs/yolo_data_test.yaml',
        imgsz=960,
        project="v12_test_runs",
        seed=42,
        exist_ok=True
    )

    ap_75_95 = metrics.box.all_ap[:, 5:].mean()
    print(f'mAP@[0.75:0.95] : {ap_75_95:.2f}')


def test_loop():
    model = YOLO('v12_test_runs/train/weights/best.pt')

    model.predict(
        source="../../data/test_images",
        imgsz=960,
        conf=0.25,
        save=True,
        save_txt=True,
        save_conf=True,
        project="v12_test_runs",
        seed=42,
        exist_ok=True
    )

    save_csv()


def save_csv():
    pred_dir = "v12_test_runs/predict/labels"
    img_dir = "../../data/test_images"
    out_csv = "../../submissions/submission_test.csv"

    rows = []
    annotation_id = 1

    # 🔑 매핑 불러오기
    class_map = load_class_map()

    for label_file in os.listdir(pred_dir):
        if not label_file.endswith(".txt"):
            continue

        image_id = os.path.splitext(label_file)[0]   # "1"
        img_path = os.path.join(img_dir, image_id + ".png")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, image_id + ".jpg")

        with Image.open(img_path) as im:
            W, H = im.size

        with open(os.path.join(pred_dir, label_file)) as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 6:
                    cls, x, y, w, h, conf = map(float, parts)
                else:
                    continue  # conf 없는 라인 무시

                x_min = (x - w/2) * W
                y_min = (y - h/2) * H
                box_w = w * W
                box_h = h * H

                rows.append([
                    annotation_id,
                    int(image_id),
                    class_map[int(cls)],   # 🔑 여기서 변환
                    int(x_min),
                    int(y_min),
                    int(box_w),
                    int(box_h),
                    round(conf, 2)
                ])
                annotation_id += 1

    df = pd.DataFrame(rows, columns=[
        "annotation_id", "image_id", "category_id",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"
    ])
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


def load_class_map():
    anno_dir = "../../data/train_annotations_test"
    name2id = {}

    # 하위 모든 폴더 순회
    for root, dirs, files in os.walk(anno_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                if "categories" in data:
                    for cat in data["categories"]:
                        name2id[cat["name"].strip()] = cat["id"]

    # yolo_data.yaml 불러오기
    with open("../../configs/yolo_data_test.yaml", "r", encoding="utf-8") as f:
        yolo_cfg = yaml.safe_load(f)
    names = [n.strip() for n in yolo_cfg["names"]]

    # YOLO 인덱스 → 실제 category_id 매핑
    class_map = {}
    for i, name in enumerate(names):
        if name not in name2id:
            raise KeyError(f"⚠️ '{name}' 가 JSON categories에 없음")
        class_map[i] = name2id[name]

    return class_map


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
            - 약들이 많이 기울어져 있는 경우가 있고, 약의 크기는 비슷하고 글자만 다른 약들이 존재하기때문에 Augmentation의 RandomAffine을 좀 강하게 걸어주고 실험 [degrees(회전범위), trainslate(이미지 이동), scale(확대 축소), shear(기울임)]
            - 약들의 글자가 너무 흐릿하거나 알아볼 수 없는경우가 있음, hsv를 조절해서 약의 이미지를 조금 뚜렷하게 해보는 실험 [hsv_h(hue), hsv_s(saturation), hsv_v(value + constrast 혼합)]
            - mixup을 사용하면 글씨 디테일을 망가뜨릴 수가 있기 때문에 데이터의 양을 고려해서 0.1~0.2 혹은 0.0으로 실험
            - bbox의 누락이 문제가 되는 경우도 있으므로 cutmix를 진행할지, cutout을 진행할지, EDA의 결정을 따를지 고민
            - 이미지의 resize 크기가 작아 약의 글씨 디테일을 살리지 못할수 있으므로, 이미지 resize 크기 키우기 [imgsz]

        results:
            - [No 전처리, No EDA] 5epochs => Validataion mAP@[0.75:0.95]: 0.7997302769703274(약 0.8)
            - [No 전처리, No EDA] 30epochs => Validataion mAP@[0.75:0.95]: 0.87

            RandomAffine:
                - [degrees=120, translate=0.8, scale=0.8, shear=15] 30epochs => Validataion mAP@[0.75:0.95]: 0.73
                - [degrees=120, shear=15] 30epochs => Validataion mAP@[0.75:0.95]: 0.80
                - [degrees=60, shear=15] 30epochs => Validataion mAP@[0.75:0.95]: 0.81
                - [degrees=45, shear=15] 30epochs => Validataion mAP@[0.75:0.95]: 0.83
                - [degrees=30, shear=15] 30epochs => Validataion mAP@[0.75:0.95]: 0.79
                => 각도를 바꾸는게 Yolo 기본설정보다 좋지 않은 선택인거 같으므로 각도 변경은 잠시 보류

            Mixup:
                - [mixup=0.0] 30epochs => Validataion mAP@[0.75:0.95]: 0.86
                - [mixup=0.1] 30epochs => Validataion mAP@[0.75:0.95]: 0.87
                - [mixup=0.2] 30epochs => Validataion mAP@[0.75:0.95]: 0.86
                => mixup의 변경은 Yolo기본설정과 별 차이가 없음

            HSV:
                - [hsv_h=0.015, hsv_s=0.5, hsv_v=0.4] 30epochs => Validataion mAP@[0.75:0.95]: 0.87

            Image Size:
                - [imgsz=960, batch=4, nbs=16] 30epochs => Validataion mAP@[0.75:0.95]: 0.86
                - [imgsz=960, batch=4, nbs=16, hsv_h=0.1, hsv_s=0.5, hsv_v=0.3] 30epochs => Validataion mAP@[0.75:0.95]: 0.86

            결론:
                - ultralytics이 제공하는 auto_augment옵션이 개인이 진행하는 augmentation 보다 뛰어나다는 결론에 도달하여, [AutoAugment, RandAugment, AugMix] 이 세가지의 옵션을 실험해보는 것을 다음사람에게 인계하기로함
    ==========================================================================================================================================================================================================================
    """

    model = YOLO('yolo12s.pt')

    epochs = 50
    close_mosaic_ratio = int(epochs * 0.3)

    model.train(
        data='../../configs/yolo_data_test.yaml',
        epochs=epochs,
        close_mosaic=close_mosaic_ratio,
        imgsz=960,  # 640
        batch=4,    # 16
        nbs=16,     # nbs: 배치는 4인데 16의 효과를 주기위해 epoch마다 backward하는게 아니라 4번마다 한번 한다
        auto_augment='randaugment', # 'randaugment'는 기본값 'autoaugment'또는 'augmix'
        device=0,
        project="v12_test_runs",
        seed=42,
        exist_ok=True
    )

if __name__ == "__main__":
    rcParams['font.family'] = 'Malgun Gothic'
    rcParams['axes.unicode_minus'] = False

    main()
    validation_score()
    test_loop()