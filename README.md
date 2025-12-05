# warm-cool-personal-color
성령/선미 딥러닝 영상처리 과제
# AI 퍼스널 컬러 진단 (Warm vs Cool)

얼굴 이미지를 입력하면 쿨톤 / 웜톤을 분류하고, dlib 랜드마크 기반으로 스킨 톤을 분석한 뒤 톤에 맞는 가상 메이크업과 실제 제품을 추천해 주는 프로젝트입니다.

## 폴더 구조
├── models/ # 학습된 딥러닝 모델 (.h5)
├── notebook/ # 실험/학습용 Jupyter Notebook
├── src/ # 실제 실행 코드 (모델 로드, 얼굴 분석, Gradio 앱)
│ ├── app_gradio.py # Gradio UI (업로드 + 웹캠)
│ ├── face_utils.py # dlib 랜드마크, 스킨 톤 분석, 가상 메이크업 함수
│ ├── model.py # 학습된 모델 로드 및 예측 함수
│ └── old/ # 코랩에서 사용한 통합 실험 코드 백업
└── README.md

## 사용한 모델 / 데이터

- Backbone: MobileNetV2 기반 warm / cool 이진 분류 모델 (ImageNet 사전학습 후 전이학습)
- Input size: 224 x 224 RGB
- 학습 데이터: warm / cool 폴더 구조로 정리된 얼굴 이미지 (용량 문제로 레포에는 포함하지 않음)

## 실행 방법 (로컬)
git clone https://github.com/SeongRyeongJeong/warm-cool-personal-color.git
cd warm-cool-personal-color
(선택) 가상환경 생성
python -m venv venv
venv\Scripts\activate # Windows

필수 라이브러리 설치
pip install --upgrade pip
pip install gradio opencv-python dlib numpy tensorflow

앱 실행
cd src
python app_gradio.py

브라우저에서 출력된 Gradio URL(예: http://127.0.0.1:7860)을 열면,  
이미지 업로드 또는 웹캠으로 퍼스널 컬러를 진단할 수 있습니다.

## 워크플로우

1. 입력: 사용자가 얼굴 이미지를 업로드하거나 웹캠으로 촬영
2. 모델: MobileNetV2 기반 분류 모델이 warm / cool 톤 예측
3. 얼굴 분석: dlib 랜드마크로 얼굴과 볼 영역을 검출하고 Lab 색공간에서 밝기/붉은기/노란기 계산
4. 가상 메이크업: 예측된 톤에 맞는 아이브로우/립 컬러를 얼굴에 합성
5. 추천 출력: 톤에 맞는 실제 화장품 리스트와 스킨 톤 분석 결과를 텍스트로 제공

## 실행 방법 (코랩)
# 1) 레포 클론
!git clone https://github.com/SeongRyeongJeong/warm-cool-personal-color.git
%cd warm-cool-personal-color

# 2) 라이브러리 설치
!pip install gradio opencv-python dlib numpy tensorflow

# 3) Gradio 앱 실행
%cd src
from app_gradio import demo
demo.launch(debug=True, share=True)

