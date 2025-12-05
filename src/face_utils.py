import dlib
import cv2
import numpy as np
import os

# ==========================================
# 1. Dlib 랜드마크 모델 설정
# ==========================================
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# 파일 없으면 다운로드
if not os.path.exists(PREDICTOR_PATH):
    print("📥 랜드마크 파일 다운로드 중...")
    os.system(
        "wget -O shape_predictor_68_face_landmarks.dat "
        "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
    )

# detector와 predictor 전역 변수
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    print("✅ Predictor 로드 성공!")
except Exception as e:
    print(f"🚨 Predictor 로드 실패: {e}")

# ==========================================
# 2. 메이크업 & 분석 함수 정의
# ==========================================
def apply_lipstick_virtual(frame, landmarks, color_bgr, alpha=0.4):
    points = landmarks[48:68]
    hull = cv2.convexHull(points)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [hull], 0, 255, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    color_layer = np.zeros_like(frame, dtype=np.uint8)
    color_layer[:] = color_bgr

    frame_float = frame.astype(float)
    color_layer_float = color_layer.astype(float)
    mask_float = mask.astype(float) / 255.0
    mask_3ch = cv2.merge([mask_float, mask_float, mask_float])

    output = frame_float * (1.0 - alpha * mask_3ch) + color_layer_float * (alpha * mask_3ch)
    return output.astype(np.uint8)


def apply_eyebrow_tint(frame, landmarks, color_bgr, alpha=0.3):
    result = frame.astype(float)
    color_layer = np.zeros_like(frame, dtype=np.uint8)
    color_layer[:] = color_bgr
    color_layer = color_layer.astype(float)
    mask_total = np.zeros(frame.shape[:2], dtype=np.float32)

    for idx_range in [(17, 22), (22, 27)]:
        points = landmarks[idx_range[0]:idx_range[1]]
        hull = cv2.convexHull(points)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [hull], 0, 255, -1)
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        mask_float = mask.astype(float) / 255.0
        mask_total = np.maximum(mask_total, mask_float)

    mask_3ch = cv2.merge([mask_total, mask_total, mask_total])
    output = result * (1.0 - alpha * mask_3ch) + color_layer * (alpha * mask_3ch)
    return output.astype(np.uint8)


def analyze_skin_tone(image, landmarks):
    pt1 = landmarks[2]
    pt2 = landmarks[31]
    x_min, x_max = min(pt1[0], pt2[0]), max(pt1[0], pt2[0])
    y_min, y_max = min(pt1[1], pt2[1]), max(pt1[1], pt2[1])
    cheek_roi = image[y_min:y_max, x_min:x_max]

    if cheek_roi.size == 0:
        return "분석 실패"

    lab_roi = cv2.cvtColor(cheek_roi, cv2.COLOR_BGR2LAB)
    l_mean = np.mean(lab_roi[:, :, 0])
    a_mean = np.mean(lab_roi[:, :, 1])
    b_mean = np.mean(lab_roi[:, :, 2])

    return f"• 밝기(L): {l_mean:.1f}\n• 붉은기(a): {a_mean-128:.1f}\n• 노란기(b): {b_mean-128:.1f}"

