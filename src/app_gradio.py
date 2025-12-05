import cv2
import numpy as np
import gradio as gr

from model import model, CLASS_NAMES as class_names, predict_tone_from_rgb
from face_utils import (
    detector,
    predictor,
    apply_lipstick_virtual,
    apply_eyebrow_tint,
    analyze_skin_tone,
)

# ==========================================
# 3. 데이터 및 설정
# ==========================================
MAKEUP_PALETTES = {
    "cool": {"lip": (147, 112, 219), "eyebrow": (60, 60, 60)},  # BGR
    "warm": {"lip": (80, 90, 255), "eyebrow": (40, 70, 100)},
}

PRODUCT_DB = {
    "cool": [
        {"brand": "롬앤", "name": "쥬시 래스팅 틴트 #베어그레이프", "desc": "차분한 쿨톤 핑크"},
        {"brand": "페리페라", "name": "잉크 무드 글로이 #갓기천사", "desc": "여쿨라 추천"},
    ],
    "warm": [
        {"brand": "헤라", "name": "센슈얼 파우더 매트 #팜파스", "desc": "웜톤 국민템"},
        {"brand": "3CE", "name": "벨벳 립 틴트 #다포딜", "desc": "가을 웜톤 추천"},
    ],
}

# ==========================================
# 4. Gradio 메인 로직
# ==========================================
def process_oliveyoung_style(input_image):
    if input_image is None:
        return None, "사진을 넣어주세요", ""

    frame_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    # 1. 예측 (RGB 기준)
    label, _ = predict_tone_from_rgb(input_image)

    # 2. 메이크업 & 분석
    faces = detector(frame_bgr, 1)
    analysis = "얼굴 감지 실패"

    if faces:
        lm = predictor(frame_bgr, faces[0])
        pts = np.array([[p.x, p.y] for p in lm.parts()])

        analysis = analyze_skin_tone(frame_bgr, pts)

        tone_key = "cool" if "cool" in label else "warm"
        palette = MAKEUP_PALETTES[tone_key]

        frame_bgr = apply_eyebrow_tint(frame_bgr, pts, palette["eyebrow"])
        frame_bgr = apply_lipstick_virtual(frame_bgr, pts, palette["lip"], alpha=0.5)

    output_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 3. 결과 텍스트
    rec_text = ""
    recs = PRODUCT_DB.get("cool" if "cool" in label else "warm", [])
    for r in recs:
        rec_text += f"[{r['brand']}] {r['name']}\n"

    return output_rgb, f"당신은 {label.upper()}톤 입니다.\n{analysis}", rec_text


# ==========================================
# 5. 앱 실행 (업로드 + 웹캠)
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 💄 AI 퍼스널 컬러 진단 (Olive Young Ver.)")

    # 1) 업로드용 (정적 이미지)
    gr.Markdown("### 🖼 이미지 업로드 진단")
    with gr.Row():
        inp_upload = gr.Image(
            label="얼굴 사진 업로드",
            type="numpy",
            sources=["upload"],  # 업로드만
            streaming=False,
        )
        out_upload = gr.Image(label="메이크업 결과 (업로드)")
    with gr.Row():
        txt_res_upload = gr.Textbox(label="분석 결과 (업로드)")
        txt_rec_upload = gr.Textbox(label="추천 제품 (업로드)")

    inp_upload.change(
        fn=process_oliveyoung_style,
        inputs=inp_upload,
        outputs=[out_upload, txt_res_upload, txt_rec_upload],
    )

    # 2) 웹캠용 (실시간 스트리밍)
    gr.Markdown("### 🎥 실시간 웹캠 진단")
    with gr.Row():
        cam = gr.Image(
            label="웹캠",
            type="numpy",
            sources=["webcam"],
            streaming=True,
        )
        out_cam = gr.Image(label="메이크업 결과 (웹캠)")
    with gr.Row():
        txt_res_cam = gr.Textbox(label="분석 결과 (웹캠)")
        txt_rec_cam = gr.Textbox(label="추천 제품 (웹캠)")

    cam.stream(
        fn=process_oliveyoung_style,
        inputs=cam,
        outputs=[out_cam, txt_res_cam, txt_rec_cam],
        stream_every=0.3,
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=True)

