## warm-cool-personal-color

성령 / 선미 딥러닝 영상처리 프로젝트
AI 기반 퍼스널 컬러 진단 (Warm vs Cool)

얼굴 이미지를 입력하면 웜톤 / 쿨톤을 분류하고,
dlib 얼굴 랜드마크로 스킨 톤을 분석한 뒤,
진단된 톤에 맞는 가상 메이크업 합성과 실제 화장품 추천을 제공하는 프로젝트입니다.

## 📁 폴더 구조

warm-cool-personal-color/
├── models/                     # 학습된 모델(.h5), dlib 랜드마크(.dat)
│   ├── personal_color_mobilenetv2_model.h5
│   └── shape_predictor_68_face_landmarks.dat
├── notebook/                  # 실험용 Jupyter Notebook
├── src/
│   ├── app_gradio.py          # Gradio UI (업로드 + 웹캠)
│   ├── face_utils.py          # 랜드마크 검출, 스킨 톤 분석, 가상 메이크업
│   ├── model.py               # 딥러닝 모델 로드 & 예측
│   └── old/                   # 통합 테스트 코드(백업)
└── README.md


## 🔧 사용한 모델 / 데이터

- Backbone: MobileNetV2 기반 warm / cool 이진 분류 모델 (ImageNet 사전학습 후 전이학습)
- Input size: 224 x 224 RGB
- 학습 데이터: warm / cool 폴더 구조로 정리된 얼굴 이미지 (용량 문제로 레포에는 포함하지 않음)
  

## 🧪 주요 기능 (Workflow)

1. 입력
이미지 업로드 또는 웹캠 캡처(0.3초 마다 프레임 처리)

2. 분류 모델 예측
MobileNetV2 기반 warm/cool 분류

3. 얼굴 분석
- dlib 68 랜드마크 검출
- Lab 색공간에서 밝기(L), 붉은기(a), 노란기(b) 계산
- 볼 중심 영역 평균 색상 분석

4. 가상 메이크업 합성
- 톤에 맞는 립·아이브로우 색상 Overlay

5. 추천 출력
- 웜톤/쿨톤 화장품 리스트
- 스킨 톤 수치 분석 결과 제공

  
## 🖥️ 로컬 실행 방법

0) 필수 모델 파일 준비
warm-cool-personal-color/
└── models/
    ├── personal_color_mobilenetv2_model.h5
    └── shape_predictor_68_face_landmarks.dat

1) 레포 클론
git clone https://github.com/SeongRyeongJeong/warm-cool-personal-color.git
cd warm-cool-personal-color

2) 가상환경 생성 (선택)
Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

macOS / Linux
python3 -m venv venv
source venv/bin/activate

3) 라이브러리 설치
권장 방식:
pip install --upgrade pip
pip install -r requirements.txt

4) 앱 실행
cd src
python app_gradio.py

출력되는 URL (예: http://127.0.0.1:7860) 을 브라우저로 열면
웹캠/업로드 기반 AI 퍼스널 컬러 진단이 실행됩니다

## ☁️ 클라우드에서 실행 (Colab)
# 1) 레포 클론
!git clone https://github.com/SeongRyeongJeong/warm-cool-personal-color.git
%cd warm-cool-personal-color

# 2) 라이브러리 설치
!pip install gradio opencv-python dlib numpy tensorflow

# 3) Gradio 앱 실행
%cd src
from app_gradio import demo
demo.launch(debug=True, share=True)
*share=True는 코랩 런타임이 꺼지면 링크가 사라집니다.
