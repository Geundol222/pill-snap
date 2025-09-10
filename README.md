# 💊 경구약제 이미지 객체 검출(Object Detection) 프로젝트


### 🔧 설치 방법

1. **프로젝트 다운로드**
```bash
git clone <repository-url>
```

2. **가상환경 생성 (권장)**
```bash
python -m venv [가상환경 이름]

# 활성화
# Windows:
[가상환경 이름]\Scripts\activate
# Mac/Linux:
source [가상환경 이름]/bin/activate
```

3. **라이브러리 설치**
```bash
pip install -r requirements.txt
```

4. **앱 실행**
```bash
streamlit run main.py
```

<br>

## 🏗️ 프로젝트 구조
```
pill-snap/
├── 📄 main.py              # 메인 실행 파일
├── 📁 configs/             # 설정 파일
├── 📁 src/ 
├───── 📁 models/             # 모델 정의(필요시)
├───── 📄 train.py            # 학습 엔트리
├───── 📄 infer.py            # 추론/제출 생성
├───── 📄 utils.py            # 공통 함수
├── 📁 notebooks/ 
├───── 📄 01_eda.ipynb        # 시각화/라벨 점검
├───── 📄 99_report.ipynb     # 발표용 그래프/표 정리
├── 📁 submissions/          # 정답
├── 📄 requirements.txt      # 필요한 라이브러리 목록
├── 📄 README.md           
```