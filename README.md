# 💊 경구약제 이미지 객체 검출 프로젝트

<p align="right">
  <a href="#-개요">개요</a> •
  <a href="#-팀-구성-및-역할">팀 구성</a> •
  <a href="#-설치-방법">설치</a> •
  <a href="#-프로젝트-구조">구조</a> •
  <a href="#-프로젝트-진행-과정">진행 과정</a> •
  <a href="#-실험-내용-및-결과">결과</a> •
  <a href="#-결론">결론</a> •
  <a href="#-참고자료">참고</a>
</p>

---

## 프로젝트 보고서
- [5팀_프로젝트 보고서.pdf](https://github.com/user-attachments/files/22525383/5._.pdf)

## 팀 프로젝트 Notion링크
- [링크](https://coal-sheet-752.notion.site/_AI-4-5-2770d71ee9698043b590c63f18ba22ea)

## 개인 협업일지 링크
- [김진욱](https://coal-sheet-752.notion.site/2770d71ee96980d6a8a9dde19e062d32?v=2770d71ee9698064a0e7000cc1b47e24&source=copy_link)
- [박병현](https://famous-gorilla-33d.notion.site/AI-_-_-269c7c1a009280dfb556e494268ea975?source=copy_link)
- [오형주](https://rose-laugh-280.notion.site/AI-09-09-09-24-2778de3ce62b80079a87e7926bbc98c5?source=copy_link)
- [이현석](https://bubbly-psychology-181.notion.site/Codeit-2252dfb1ef688054a879c45c276e8d85?source=copy_link)
- [진수경](https://puzzled-salto-827.notion.site/2696a4a5ec8380adb0bfd72fec737b86?v=2696a4a5ec8380fe9893000cccc037c7)
- [함건희](https://nostalgic-apricot-f75.notion.site/277fd289d4ef809880e8eef10d388fd3?v=277fd289d4ef8168aa96000c6c160de3&source=copy_link)

## 개인 실험 폴더
- [김진욱](./Personal/KJW/)
- [박병현](./Personal/PBH/)
- [오형주](./Personal/OHJ/)
- [이현석](./Personal/LHS/)
- [진수경](./Personal/JSG/)
- [함건희](./Personal/HKH/)

---

## 📌 개요
- **주제**: 알약 이미지 객체 검출(Object Detection)  
- **목표**: 정확한 약제 식별 및 **YOLOv8 vs YOLOv12** 성능 비교
- **프로젝트 기간**: 2025.09.09 - 202509.25(16일)
- **데이터(원본)**: [Ai Hub 경구약제 이미지 데이터 中 일부 (가공데이터는 private)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=576)

---

## 👥 팀 구성 및 역할
- **팀 인원**: 6명 — 김진욱 · 박병현 · 이현석 · 오형주 · 진수경 · 함건희  
- **역할 구조**:
  - YOLO 모델링 및 하이퍼파라미터 튜닝 (**YOLOv8, YOLOv12**)
  - EDA & 전처리 (라벨 검증, 데이터 증강)
- **운영 방식**:
  - 역할 **로테이션**을 통해 모델링과 데이터 작업을 모두 경험
  - 각 rotation 별로 **branch 전략** 수립: 작업 이력/발전 과정 추적

| 이름   | 역할         |
|--------|--------------|
| 이현석 | YOLOv12      |
| 함건희 | YOLOv8       |
| 김진욱 | EDA & 전처리 |
| 박병현 | EDA & 전처리 |
| 오형주 | EDA & 전처리 |
| 진수경 | EDA & 전처리 |

---

## 🔧 설치 방법

1. **프로젝트 다운로드**
```bash
git clone https://github.com/Geundol222/pill-snap.git
```

2. **가상환경 생성 (권장)**
```bash
vscode의 경우 interpreter 설정을 통해 가상환경 설치가능

or

python -m venv [가상환경 이름]

# 활성화
# Windows:
[가상환경 이름]\Scripts\activate
# Mac/Linux:
source [가상환경 이름]/bin/activate
```

3. **라이브러리 설치**
```bash
vscode의 경우 interpreter 설정을 통해 requirements 설치가능

or

pip install -r requirements.txt
```

4. **앱 실행**
```bash
python main.py
```

---


## 🏗️ 프로젝트 구조
```
pill-snap/
├── 📄 main.py              # 메인 실행 파일
├── 📁 configs/             # 설정 파일
├── 📁 data/                # train_images, train_annotations, test_images
├── 📁 data_yolo/           # YOLO용 image, txt label 묶음
├── 📁 src/ 
├───── 📁 models/             # 모델 정의
├───── 📄 train.py            # 학습 엔트리
├───── 📄 infer.py            # 추론/제출 생성
├───── 📄 utils.py            # 공통 함수
├── 📁 Personal/              # 개인 실험 폴더
├── 📁 notebooks/ 
├───── 📄 01_eda.ipynb        # 시각화/라벨 점검 EDA
├── 📁 submissions/          # 정답(캐글 제출)
├── 📁 test_models/          # 테스트용 비교 모델
├── 📄 README.md     
├── 📄 requirements.txt      # 필요한 라이브러리 목록
```

---

## 📅 프로젝트 진행 과정

### 🔹 1주차 – 모델 선정
- **적용 모델**: Faster R-CNN, ResNet-SSD, VGG-SSD, RetinaNet, YOLOv8  
- **mAP 결과**:  
  - ⚠️ FasterRCNN: 오류로 측정 불가  
  - ⚠️ ResNetSSD: 오류로 측정 불가  
  - 📉 VGGSSD: 0.35  
  - 📉 RetinaNet: 0.33  
  - ✅ YOLOv8: 0.86  
👉 다른 모델의 경우 mAP함수의 문제나 오류 등의 이유로 점수가 낮은 경향이 있었음, 프로젝트 기간상 오류를 전부 해결하기 힘들 것이라고 판단, 팀회의를 통해 YOLOv8모델을 베이스라인으로 잡는 것으로 합의

### 🔹 2주차 – YOLOv8 + YOLOv12 실험
#### 🧪 EDA & 전처리:
- bbox의 겹칩이 있는 이미지들이 발견됨, 수가 많지 않으므로 수동으로 bbox를 채우는 작업진행
  - **실험결과**:
    - 수동으로 bbox를 채우고 나서 모델이 예측하지 못하거나 잘못 예측하는 bbox들이 줄어드는 효과 확인
    - 적절한 라벨링으로 mAP의 개선 확인

- 데이터 클래스의 불균형발견, 개선을 위해 여러 방면으로 개선사항 검토
  - **실험결과**:
    - 소수 클래스가 포함된 이미지 자체를 복제하여 진행했으나, 이미지 내에 다수 클래스 객체까지 함께 증강되어 불균형의 해소 효과가 희석됨
    - 소수 클래스의 객체만 잘라내어 별도로 학습 시켰으나, Context가 제거된 비현실적인 데이터가 모델 학습에 혼란을 주어 오히려 일반화 성능이 크게 하락함
    
- 알약의 각인 및 선명도가 조금 떨어지는 문제가 있는것으로 추정되어 선명도 개선 실험
  - **실험결과**:
    - 실제 선명도의 개선 효과는 있었으나 점수에서 유의미한 변화를 얻어내지 못함

- 이미지에 따라 어노테이션 파일이 존재하지 않는 경우 확인, 가장 가까운 어노테이션 파일을 기반으로 새로운 json을 생성하고 모델 prediction을 통해 bbox 보정
  - **실험결과**:
    - 어노테이션의 추가가 데이터 추가와 비슷한 효과를 내었고, 누락되었던 라벨이 생성되면서 모델의 일반화 성능이 강화됨 **kaggle score: 0.965 → 0.983**

- 이미지의 개수를 늘리는 데이터 추가 실험
  - **실험결과**:
    - 데이터 추가 이후 모델이 학습하는 이미지 자체가 증가하여 성능이 크게 상승함 **kaggle score: 0.983 → 0.990**
    - **단, 모델의 prediction 결과 비정상적으로 많은 양의 bbox를 생성하는 현상이 포착되어 해당 점수가 일반화 성능이 강화된 것인지 확언할 수 없음**

#### ⚙️ 하이퍼파라미터 튜닝:
- YOLO에서 지정할 수 있는 augmentation 파라미터의 조정실험 진행 [Affine, mixup, hsv(ColorJitter), imgsz(Image Size)]
  - **실험결과**:
    - 이미지의 resize 크기를 키우는 것은 이미지의 해상도 증가를 가져오기 때문에 각인이나 흐릿한 약의 판단에 도움이 되었음
    - 각 augmentation 실험은 유의미한 mAP 증가 효과를 볼 수 없었으며, 공식문서를 찾아본 결과 auto_augment라는 옵션을 통해 자동으로 augmentation이 적용되기 때문에 수동으로 augmentation을 지정하여 실험하는 것은 의미가 없다고 판단, auto_augment의 옵션에 대한 실험을 진행해보기로 함
    
- auto_augment 실험
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
  - **실험결과**:
    - 세 옵션에 대해 mAP와 리더보드에 **큰 점수차이 없음**, 따라서 가장 빠른 **RandAugment를 적용**

- 랜덤 서치를 이용하여 다양한 augmentation 실험
  - **실험 Parameter**:
  
    | Parameters   | Cases                                |
    |--------------|--------------------------------------|
    | mixup        | [0.0, 0.1, 0.2]                      |
    | mosaic       | [0.0, 0.5, 1.0]                      |
    | cutmix       | [0.0, 0.1, 0.3]                      |
    | copy_paste   | [0.0, 0.1, 0.2]                      |
    | augment      | ['autoaugment', 'augmix', 'randaugment'] |
    | nbs          | [8, 16, 32]                          |
    | lr0          | [0.001, 0.003, 0.005]                |

  - **실험결과**:
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

- YOLO 모델 사이즈[n, s, m, l, xl] 모델의 성능 변화 확인
  - **실험결과**:
    - Large모델과 XLarge모델의 경우 모델의 크기가 너무 방대하고 VRAM을 크게 사용하기 때문에 개인이 실험하기에는 너무 무거운 모델로 판단하여 폐기함
    - Nano모델의 경우 많이 사용하는 모델이지만, 현재 BaseLine으로 잡아놓은 Small모델보다 성능면에서는 떨어질 것이라고 생각되어 폐기함
    - Medium 모델의 경우 성능을 확인하였을 때 같은 조건에서 Small 모델보다 좋은 결과를 보임 **kaggle score: 0.990 → 0.993**
    - **단, Medium 모델의 경우도 비정상적인 행 개수가 나타나는 것은 동일하기 때문에 일반화 성능이 올라갔다고 판단하기 어려움**
      
---

## 📊 실험 내용 및 결과
### 📈 결과 정리
| 실험 내용           | Kaggle 점수 |
| --------------- | --------- |
| BaseLine        | **0.965** |
| bbox 겹침 해결   | **0.968** |
| Pseudo-Labeling | **0.983** |
| 데이터 추가 + Pseudo-Labeling | **0.990** |
| 데이터 추가 + Pseudo-Labeling + Medium Model | **0.993** |

## 📝 결론
### 결론
- Pseudo-Labeling과 데이터의 추가가 큰 효과를 발휘하여 좋은 결과를 얻어낼 수 있었음
- EDA 과정에서 차마 생각하지 못한 시선에서의 데이터 분석을 많이 배울 수 있었음
- 꽤 많은 EDA과정이 있었고 이상치를 확인하였으나, 유의미한 효과를 내지 못했음 경험을 키우고 데이터 보는 방법을 조금 더 키운다면, 더 좋은 모델링 결과를 얻을 수 있을 것으로 생각됨
- 모델의 중복 탐지 문제 해결이 큰 과제일 것으로 보임, 실제 중복행을 제거한 버전에서는 kaggle점수가 하락하는 경향을 보였음

---

## 📚 참고자료
- [Ultralytics](https://docs.ultralytics.com/ko/modes/train/#introduction)
- [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501)
- [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719)
- [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781)
