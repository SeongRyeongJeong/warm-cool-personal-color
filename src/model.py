import cv2
import numpy as np
from tensorflow.keras.models import load_model

# GitHub models/ 폴더에 올려둘 weight 파일 이름
MODEL_PATH = "../models/warm_cool_model.h5"

# 학습할 때 사용한 클래스 순서에 맞게
CLASS_NAMES = ["cool", "warm"]

# 전역 모델 객체 로드
model = load_model(MODEL_PATH)

def predict_tone_from_rgb(image_rgb):
    """RGB 이미지(224x224 이상)를 받아 warm/cool 톤을 예측."""
    img_resized = cv2.resize(image_rgb, (224, 224))
    img_input = np.expand_dims(img_resized / 255.0, axis=0)
    pred = model.predict(img_input, verbose=0)[0]
    idx = np.argmax(pred)
    label = CLASS_NAMES[idx]
    return label, pred

