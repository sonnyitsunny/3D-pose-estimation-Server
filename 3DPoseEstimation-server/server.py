from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from io import BytesIO
from fastapi.responses import StreamingResponse

# FastAPI 애플리케이션 생성
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TensorFlow 모델 로드
saved_model_dir = './'  # 모델 경로
try:
    model = tf.saved_model.load(saved_model_dir)
    infer = model.signatures["serving_default"]
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    raise RuntimeError("TensorFlow 모델 로드 실패")

# 키포인트 사전 정의
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# 몸통 연결선 정의
LINES_BODY = [
    [10, 8], [8, 6], [6, 5], [5, 7], [7, 9],  # 팔 연결
    [6, 12], [12, 11], [11, 5],              # 상체 연결
    [12, 14], [14, 16], [11, 13], [13, 15]   # 다리 연결
]

@app.post("/process-frame/")
async def process_frame(file: UploadFile = File(...)):
    """
    클라이언트에서 전송한 라이브 스트림 데이터(단일 프레임)를 처리하여 스켈레톤을 그린 이미지를 반환합니다.
    """
    try:
        # 업로드된 파일 읽기
        content = await file.read()
        np_arr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("유효하지 않은 이미지 파일입니다.")

        # 프레임 전처리
        input_img = cv2.resize(frame, (192, 192))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = np.expand_dims(input_img, axis=0)
        input_img = tf.convert_to_tensor(input_img, dtype=tf.int32)

        # 모델 추론
        outputs = infer(input=input_img)
        keypoints = outputs['output_0'].numpy()

        # 스켈레톤 그리기
        for person in keypoints[0]:  # keypoints[0]은 다수의 사람의 키포인트로 가정
            landmarks = []
            for i in range(0, len(person), 3):
                point = person[i:i+3]
                if len(point) == 3:
                    y, x, confidence = point
                    if confidence > 0.3:  # 신뢰도가 30% 이상인 경우에만 처리
                        px = int(x * frame.shape[1])
                        py = int(y * frame.shape[0])
                        landmarks.append((px, py))
                        cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)
                    else:
                        landmarks.append(None)
                else:
                    landmarks.append(None)

            # 연결된 선 그리기
            for start_idx, end_idx in LINES_BODY:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    if landmarks[start_idx] and landmarks[end_idx]:
                        cv2.line(frame, landmarks[start_idx], landmarks[end_idx], (0, 255, 128), 2)

        # 결과 이미지를 JPEG로 인코딩
        _, jpeg_image = cv2.imencode('.jpg', frame)
        return StreamingResponse(BytesIO(jpeg_image.tobytes()), media_type="image/jpeg")

    except Exception as e:
        print(f"오류 발생: {e}")
        raise HTTPException(status_code=400, detail=f"프레임 처리 중 오류: {e}")
