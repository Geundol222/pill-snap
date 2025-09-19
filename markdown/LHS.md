# 실험 및 공부내용

## EDA 실험
가정:
    - 데이터 라벨 자체가 없는 경우가 있음(annotation 파일자체가 없어서 bbox가 누락됨) pseudo-labeling
    - pseudo-labeling으로 채운 데이터의 bbox가 여전히 틀어짐이 있으므로, 그냥 iou가 아닌 ciou 지표를 비교하여 bbox보정, 0.976으로 점수가 상승한 모델로 변경

result:
    pseudo-labeling:
        - bbox를 채우기 위해 데이터를 살펴봤더니 이미지가 70, 75, 90도의 각도로 찍혔다고 생각됨
        - 모든 이미지가 세개의 각도로 찍혔다고 가정하고, 이미지와 대응되는 annotation이 누락이 되었다면, 가장 차이가 크지 않은 각도의 이미지를 기반으로 json파일을 복사
        - 이미지 구도에 따라 bbox가 전부 다르기 때문에, Baseline으로 돌려놨던 YOLOv12모델을 이용하여 bbox를 예측하고 iou를 비교하여 bbox를 보정
        - 보정 이후 학습에서 (0.965 -> 0.976)으로 점수 상승

    ciou:
        - 거의 모든 bbox가 약을 감싸는 형태로 보정된 것을 확인
        - 보정 이후 학습에서 (0.976 -> 0.983)으로 점수 상승

## 모델 실험
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