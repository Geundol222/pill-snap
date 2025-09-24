# 🧪 실험 및 공부 내용

## ⚙️ 실험 환경
- **Baseline 모델**: YOLOv12s (Ultralytics)
- **데이터셋**: 알약 이미지 데이터 (pseudo-labeling + 보정 적용)
- **평가지표**: Validation mAP@[0.75:0.95]
- **하드웨어**: RTX 3060 Ti (8GB VRAM)
- **프레임워크**: Ultralytics YOLO (v12), Python 3.12.10, CUDA 12.8

## 🔍 EDA 실험

### 📌 실험 내용
- 일부 데이터는 **라벨(Annotation) 자체가 없음** → pseudo-labeling 필요  
- pseudo-labeling으로 채운 bbox는 여전히 오차 존재 → **IoU 대신 CIoU 지표**로 비교 및 보정  
- 보정된 bbox를 이용하면 모델 성능 상승 가능  

### 📊 결과
**Pseudo-labeling**
- 이미지가 약 **3가지 각도(70°, 75°, 90°)** 로 촬영된 것으로 추정 및 가정
    - 이렇게 추정 및 가정한 이유는, 이미지에 annotation이 없는 것은 모델 학습에 영향을 주게 되지만,
    annotation이 있는데 이미지가 없는 것은 애초에 이미지가 없는 것이므로 학습에 영향을 주지 않을 것이라 판단했음.
    - 따라서, 모든 이미지가 3개의 각도에서 찍은 것이라고 가정하고 pseudo-labeling을 진행.
- 해당 이미지의 annotation이 누락된 경우, 가장 유사한 각도의 json 복사로 대체  
- Baseline YOLOv12 모델을 활용해 bbox 예측 후 IoU로 비교·보정  
- **점수: 0.965 → 0.976 (+0.011)**

**CIoU 적용**
- iou만으로는 정교한 bbox 보정을 기대할 수 없어서 ciou를 적용하여 bbox를 보정함
- 대부분의 bbox가 알약을 정확히 감싸는 형태로 보정됨  
- CIoU: 겹치는 영역뿐 아니라 **중심점 거리 + 너비·높이 비율**까지 고려  
- **점수: 0.976 → 0.983 (+0.007)**

---

## ⚙️ 모델 실험

### 📌 실험 내용
- **Epoch 수 증가** → 성능 상승 여부 확인  
- 알약의 회전/기울임이 심하므로 **RandomAffine 강하게 적용**  
- 흐릿한 글자를 살리기 위해 **HSV 조절(h, s, v)**  
- **Mixup**은 글자 디테일 손상 우려 → 0.0 ~ 0.2로 제한  
- bbox 누락 문제 → **CutMix, CutOut, 혹은 EDA 방식**과 비교 고민  
- 작은 글씨 디테일 보존을 위해 **imgsz 확대** 실험  
- 랜덤 서치 적용 실험

### 📊 결과

#### Epoch
- **5 epochs** (no EDA/전처리): mAP@[0.75:0.95] = **0.80**  
- **30 epochs** (no EDA/전처리): mAP@[0.75:0.95] = **0.87**

#### RandomAffine
| Params | Epochs | mAP@[0.75:0.95] |
|--------|--------|-----------------|
| degrees=120, translate=0.8, scale=0.8, shear=15 | 30 | 0.73 |
| degrees=120, shear=15 | 30 | 0.80 |
| degrees=60, shear=15 | 30 | 0.81 |
| degrees=45, shear=15 | 30 | 0.83 |
| degrees=30, shear=15 | 30 | 0.79 |

➡️ YOLO 기본 설정보다 성능 저하 → **각도 변경은 보류**

#### Mixup
| mixup | Epochs | mAP@[0.75:0.95] |
|-------|--------|-----------------|
| 0.0   | 30     | 0.86 |
| 0.1   | 30     | 0.87 |
| 0.2   | 30     | 0.86 |

➡️ YOLO 기본 설정과 큰 차이 없음

#### HSV
- hsv_h=0.015, hsv_s=0.5, hsv_v=0.4 → mAP@[0.75:0.95] = **0.87**

#### Image Size
| imgsz | batch | nbs | HSV 옵션 | Epochs | mAP@[0.75:0.95] |
|-------|-------|-----|----------|--------|-----------------|
| 960   | 4     | 16  | -        | 30     | 0.86 |
| 960   | 4     | 16  | h=0.1, s=0.5, v=0.3 | 30 | 0.86 |

#### 랜덤서치 적용
- 실험 Parameter

| Parameters   | Cases                                |
|--------------|--------------------------------------|
| mixup        | [0.0, 0.1, 0.2]                      |
| mosaic       | [0.0, 0.5, 1.0]                      |
| cutmix       | [0.0, 0.1, 0.3]                      |
| copy_paste   | [0.0, 0.1, 0.2]                      |
| augment      | ['autoaugment', 'augmix', 'randaugment'] |
| nbs          | [8, 16, 32]                          |
| lr0          | [0.001, 0.003, 0.005]                |



- 랜덤서치 결과

| Rank | Trial ID  | Best Epoch | mAP@[0.50:0.95] |
|------|-----------|------------|-----------------|
| 0    | trial_10  | 20         | 0.92590         |
| 1    | trial_15  | 20         | 0.92482         |
| 2    | trial_4   | 20         | 0.92428         |
| 3    | trial_12  | 20         | 0.92413         |
| 4    | trial_14  | 20         | 0.92345         |
| 5    | trial_3   | 20         | 0.92345         |
| 6    | trial_13  | 20         | 0.92320         |
| 7    | trial_5   | 20         | 0.92297         |
| 8    | trial_8   | 20         | 0.92094         |
| 9    | trial_2   | 20         | 0.92061         |
| 10   | trial_7   | 18         | 0.91901         |
| 11   | trial_9   | 20         | 0.91900         |
| 12   | trial_1   | 19         | 0.91833         |
| 13   | trial_6   | 19         | 0.91790         |
| 14   | trial_11  | 19         | 0.91781         |

- 15trial을 진행했지만, 노이즈 수준의 차이를 보임
- 15trial은 나올 수 있는 조합수에 비해 너무 적기 때문에 좋은 결과를 얻지 못한 것으로 생각됨
- 하지만 프로젝트의 기간이 정해져있고, 로컬컴퓨터에서 돌리는 한계상 15trial 만으로도 13시간 이상의 시간이 소모되었고 더 좋은 결과를 얻기 위해 trial을 늘리는 것은 무리라고 판단, 랜덤서치 실험은 종료
---
## 공부

- **AutoAugment**
    - 주요 augmentation: ***[shear, translate, rotate, auto_contrast, equalize, solarize, posterize, contrast, color, brightness, sharpness, invert, cutout, samplepairing]***
    - 각 연산은 probability와 magnitude를 가짐
    - RL 기반 탐색으로 가장 좋은 조합을 찾아 자동으로 적용됨
    - 위의 augmentation이 전부 진행되는 것이 아니며, 데이터 셋에 따라 **성능이 좋은 조합**을 찾아서 적용되는 것
- **RandAugment**
    - 주요 augmentation: autoaugment와 유사
    - autoaugment와의 차이점은 N개의 연산을 랜덤으로 선택하고 모든 연산에 대해 동일한 강도를 적용하여 augmentation 진행
    - autoaugment의 경우 가장 좋은 조합을 찾기 위해 탐색과정이 필요하여 시간이 오래걸리지만, RandAugment 이 부분을 개선하여 매번 랜덤한 연산을 진행
- **AugMix**
    - 주요 augmentation: 위의 두 방법론과 유사
    - 여러 augmentation chain을 무작위로 생성
    - chain들을 합성(Dirichlet 분포 기반 가중치) 후 원본 이미지와 섞음
    - 원본/증강본이 비슷한 embedding을 갖도록 학습
- **📘 용어 정리**
    - RL(Reinforcement Learning, 강화학습) 기반 탐색:
        - 강화학습 기반 탐색
        - 에이전트가 성능이 괜찮아 보이는 augmentation 조합을 제안 => 모델 학습 후 reward를 통해 성능점수를 주는 방법을 진행
        - 에이전트는 성능이 좋은 정책을 자주 뽑도록 학습함
    - Dirichlet 분포 기반 가중치:
        - 여러 값이 동시에 확률(합=1)이 되도록 만드는 분포
        - 예: chain이 3개가 존재한다면 [0.2, 0.5, 0.3]과 같이 합이 1이되는 비율을 랜덤하게 선정
        - 각 chain이 섞이는 비율이 매번 달라지며, 데이터의 다양성이 증가되는 효과
    - 원본/증강본이 비슷한 embedding을 갖도록 학습:
        - 원본 이미지와 증강본은 같은 클래스이므로, 모델이 출력하는 embedding이 비슷해지도록 추가적인 손실함수(Consistency Loss)사용
     
## 📚 참고자료
- [Ultralytics](https://docs.ultralytics.com/ko/modes/train/#introduction)
- [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501)
- [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719)
- [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781)

---

## 📝 결론
1. EDA 결과
- pseudo-labeling + CIoU 보정이 가장 큰 효과를 보였음: mAP@[0.75:0.95] 0.965 → 0.983 (+0.018)
- bbox 품질 자체를 높이는 것이 augmentation보다 효과적이라는 점 확인

2. 모델 실험 결과
- Epoch 증가(5 → 30) → 성능은 0.80 → 0.87로 확실히 상승
- imgsz를 키우는 것은 해상도 측면에서 조금 더 디테일한 부분을 파악할 수 있으므로 효과가 있는 것으로 생각됨
- RandomAffine은 각도 변화가 심하면 성능 저하
- Augmentations → 성능 개선 거의 없음 (baseline 수준 유지)
- 랜덤서치 → 조합 수(3^7=2,187) 대비 15 trial은 턱없이 부족 → 유의미한 개선 X

3. 종합
- **데이터 보정(Annotation 품질 관리)**이 가장 중요한 성능 향상 요인
- Augmentation은 효과가 제한적임 → 불필요하게 강한 증강은 오히려 성능 저하
- 향후에는 데이터 품질 개선 + 모델 구조 변경(Backbone, Fine-tuning 전략) 쪽이 더 유망할 것으로 판단
