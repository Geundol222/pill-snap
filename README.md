# 💊 경구약제 이미지 객체 검출 프로젝트

## 📌 개요
- **주제**: 알약 이미지 객체 검출(Object Detection)
- **목표**: 정확한 약제 식별 및 YOLOv8 vs YOLOv12 성능 비교
- **데이터**: [Ai Hub 경구약제 이미지 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=576)
  
---

## 👥 팀 구성 및 역할
- **팀 인원**: 6명 [김진욱, 박병현, 이현석, 오형주, 진수경, 함건희]
- **역할 구조**:
  - YOLO 모델링 및 하이퍼파라미터 튜닝 (YOLOv8, YOLOv12)
  - EDA & 전처리 (라벨 검증, 데이터 증강)
- **운영 방식**: 역할 **로테이션**을 통해 모델링과 데이터 작업을 모두 경험
- **예시**:
  | 이름 | 역할 |
  |------|-----|
  | 이현석 | YOLOv12 |
  | 함건희 | YOLOv8 |
  | 김진욱 | EDA&전처리 |
  | 박병현 | EDA&전처리 |
  | 오형주 | EDA&전처리 |
  | 진수경 | EDA&전처리 |
  
---

## 🔧 설치 방법

1. **프로젝트 다운로드**
```bash
git clone <repository-url>
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
├── 📁 notebooks/ 
├───── 📄 YOLO_eda_[이니셜].ipynb        # 시각화/라벨 점검 EDA
├───── 📄 99_report.ipynb     # 발표용 그래프/표 정리
├── 📁 submissions/          # 정답(캐글 제출)
├── 📄 requirements.txt      # 필요한 라이브러리 목록
├── 📄 README.md           
```

---

## 📅 프로젝트 진행 과정

### 🔹 1주차 – 모델 선정
- **적용 모델**: FasterRCNN, ResNetSSD, VGGSSD, RetinaNet, YOLOv8  
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
- 🧪데이터 클래스의 불균형발견, 개선을 위해 여러 방면으로 개선사항 검토 **(실험중)**
- 🧪알약의 각인 및 선명도가 조금 떨어지는 문제가 있는것으로 추정되어 선명도 개선 실험 **(실험중)**
- 이미지에 따라 어노테이션 파일이 존재하지 않는 경우 확인, 가장 가까운 어노테이션 파일을 기반으로 새로운 json을 생성하고 모델 prediction을 통해 bbox 보정
- **그 외의 문제들 파악 및 진행중**
#### ⚙️ 하이퍼파라미터 튜닝:
- YOLO에서 지정할 수 있는 augmentation 파라미터의 조정실험 진행 [Affine, mixup, hsv(ColorJitter), imgsz(Image Size)]
   - **실험결과**:
       - 이미지의 resize 크기를 키우는 것은 이미지의 해상도 증가를 가져오기 때문에 각인이나 흐릿한 약의 판단에 도움이 되었음
       - 각 augmentation 실험은 유의미한 mAP 증가 효과를 볼 수 없었으며, 공식문서를 찾아본 결과 auto_augment라는 옵션을 통해 자동으로 augmentation이 적용되기 때문에 수동으로 augmentation을 지정하여 실험하는 것은 의미가 없다고 판단, auto_augment의 옵션에 대한 실험을 진행해보기로 함
    
- auto_augment 실험
  - **AutoAugment**
      - 주요 augmentation: ***[shear, translate, rotate, auto_contrast, equlize, solarize, posterize, contrast, color, brightness, sharpness, invert, cutout, samplepairing]***
      - 각 연산은 probability와 magnitude를 가짐
      - RL 기반 탐색으로 가장 좋은 조합을 찾아 자동으로 적용됨
      - 위의 augmentation이 전부 진행되는 것이 아니며, 데이터 셋에 따라 **성능이 좋은 조합**을 찾아서 적용되는 것
  - **RandAugment**
      - 주요 augmentation: autoaugment와 유사
      - autoaugment와의 차이점은 N개의 연산을 랜덤으로 선택하고 모든 연산에 대해 동일한 강도를 적용하여 augmentation 진행
      - autoaugment의 경우 가장 좋은 조합을 찾기 위해 탐색과정이 필요하여 시간이 오래걸리지만, randaugmentation은 이 부분을 개선하여 매번 랜덤한 연산을 진행
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
      - Didichel 분포 기반 가중치:
          - 여러 값이 동시에 확률(합=1)이 되도록 만드는 분포
          - 예: chain이 3개가 존재한다면 [0.2, 0.5, 0.3]과 같이 합이 1이되는 비율을 랜덤하게 선정
          - 각 chain이 섞이는 비율이 매번 달라지며, 데이터의 다양성이 증가되는 효과
      - 원본/증강본이 비슷한 embedding을 갖도록 학습:
          - 원본 이미지와 증강본은 같은 클래스이므로, 모델이 출력하는 embedding이 비슷해지도록 추가적인 손실함수(Consistency Loss)사용
  - **실험결과**:
      - 세 옵션에 대해 mAP와 리더보드에 **큰 점수차이 없음**, 따라서 가장 빠른 **RandAugment를 적용**
        
---

## 📊 실험 내용 및 결과
### 🔧 베이스라인 실험
- 사용모델: YOLOv12
- 전처리: [Image resize(960)]
- 결과(Kaggle Leaderboard): **0.965**
- 예시 prediction Image:
![517](https://github.com/user-attachments/assets/0f60cc09-e990-487d-8427-44724d0597d3)

### 🔧 BBOX 겹침 문제 해결 실험
- 사용모델: YOLOv12
- 전처리: [Image resize(960), bbox 겹침 문제 수동 라벨링]
- 결과(Kaggle Leaderboard): **0.968**
- 예시 prediction Image:
![517](https://github.com/user-attachments/assets/52600d3c-2b85-43e1-ad1e-9cb1aff7eec1)

### 🔧 어노테이션 누락 pseudo-labeling 실험
- 사용모델: YOLOv12
- 전처리: [Image resize(960), annotation pseudo-labeling]
- 결과(Kaggle Leaderboard): **0.976**
- 예시 prediction Image:
![518](https://github.com/user-attachments/assets/5fdeb297-0570-4489-8bed-791ae596dc2f)

### 결과 정리
| 실험 내용 | 점수(Kaggle) |
|------|-----|
| BaseLine | 0.965 |
| Bbox 겹침 해결 | 0.968 |
| Pseudo-Labeling | 0.976 |

## 📝 결론 및 향후계획
### 결론
- 현재 진행중인 EDA 및 전처리 부분에서 조금씩 성능 개선이 이루어지고 있음
- 현재는 단독 전처리만을 확인하였지만, 효과가 있는 전처리 데이터를 모으면 성능의 개선이 더 이루어질 것으로 기대

### 향후계획
- 현재 YOLOv8의 경우 카테고리id가 매치 되지 않는 등 문제가 발생하여 리더보드 점수가 찍히지 않고 있음
- 빠른 개선을 통해 원래의 목적이었던 v8과 v12의 성능 차이를 확인할 수 있게 진행예정
- EDA가 진행중에 있으므로 필요한 문제를 더 찾아 개선하고, 모델도 성능개선이 가능한 하이퍼파라미터 탐색 예정 
