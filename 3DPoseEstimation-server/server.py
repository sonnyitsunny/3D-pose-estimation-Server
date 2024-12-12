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

# 각도 계산 함수
def calculate_angle(point_a, point_b, point_c):
    """
    세 점을 이용해 각도를 계산합니다.
    point_a, point_b, point_c: 각 점의 (x, y) 좌표
    """
    if None in [point_a, point_b, point_c]:
        return None

    # 벡터 계산
    ba = np.array([point_a[0] - point_b[0], point_a[1] - point_b[1]])  # 벡터 BA
    bc = np.array([point_c[0] - point_b[0], point_c[1] - point_b[1]])  # 벡터 BC

    # 벡터 사이의 각도 계산
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

@app.post("/process-frame/")
async def process_frame(file: UploadFile = File(...)):
    """
    클라이언트에서 전송한 라이브 스트림 데이터(단일 프레임)를 처리하여
    스켈레톤을 그린 이미지를 반환하고 [5,7], [5,11]과 [6,8], [6,12]의 각도를 서버 로그에 출력합니다.
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

        # 스켈레톤 그리기 및 각도 계산
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

            # 왼쪽 팔 각도 계산
            left_shoulder = landmarks[KEYPOINT_DICT['left_shoulder']]
            left_elbow = landmarks[KEYPOINT_DICT['left_elbow']]
            left_hip = landmarks[KEYPOINT_DICT['left_hip']]
            if left_shoulder and left_elbow and left_hip:
                left_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
                print(f"[5,7]과 [5,11]의 각도 (왼팔): {left_angle:.2f}도")

            # 오른쪽 팔 각도 계산
            right_shoulder = landmarks[KEYPOINT_DICT['right_shoulder']]
            right_elbow = landmarks[KEYPOINT_DICT['right_elbow']]
            right_hip = landmarks[KEYPOINT_DICT['right_hip']]
            if right_shoulder and right_elbow and right_hip:
                right_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
                print(f"[6,8]과 [6,12]의 각도 (오른팔): {right_angle:.2f}도")





            # 왼쪽 다리 상태 기본값 설정
            left_status = "데이터 없음"
            left_angle_degrees = None  # 각도를 따로 저장하려면 초기화

            # 오른쪽 다리 상태 기본값 설정
            right_status = "데이터 없음"
            right_angle_degrees = None  # 각도를 따로 저장하려면 초기화

            # 왼쪽 다리 각도 계산 및 상태 판단
            left_hip = landmarks[KEYPOINT_DICT['left_hip']]
            left_knee = landmarks[KEYPOINT_DICT['left_knee']]
            left_ankle = landmarks[KEYPOINT_DICT['left_ankle']]

            if left_hip and left_knee and left_ankle:
                # 벡터 계산
                vector_1 = np.array([left_knee[0] - left_hip[0], left_knee[1] - left_hip[1]])  # [hip -> knee]
                vector_2 = np.array([left_ankle[0] - left_knee[0], left_ankle[1] - left_knee[1]])  # [knee -> ankle]

                # 내적과 벡터 크기를 이용한 각도 계산
                dot_product = np.dot(vector_1, vector_2)
                magnitude_1 = np.linalg.norm(vector_1)
                magnitude_2 = np.linalg.norm(vector_2)
                cosine_angle = dot_product / (magnitude_1 * magnitude_2)
                angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

                # 각도를 도(degree)로 변환
                left_angle_degrees = np.degrees(angle)

                # 외적을 이용한 방향성 확인
                cross_product = np.cross(vector_1, vector_2)
                direction = "내반슬(O자 다리)" if cross_product > 0 else "외반슬(X자 다리)"

                # 상태 판단
                if left_angle_degrees < 170:
                    left_status = direction  # 내반슬 또는 외반슬로 설정
                elif 170 <= left_angle_degrees <= 180:
                    left_status = "정상"

                print(f"왼쪽 다리 상태: {left_status} (각도: {left_angle_degrees:.2f}도)")

            # 오른쪽 다리 각도 계산 및 상태 판단 (부호 반전 적용)
            right_hip = landmarks[KEYPOINT_DICT['right_hip']]
            right_knee = landmarks[KEYPOINT_DICT['right_knee']]
            right_ankle = landmarks[KEYPOINT_DICT['right_ankle']]

            if right_hip and right_knee and right_ankle:
                # 벡터 계산
                vector_1 = np.array([right_knee[0] - right_hip[0], right_knee[1] - right_hip[1]])  # [hip -> knee]
                vector_2 = np.array([right_ankle[0] - right_knee[0], right_ankle[1] - right_knee[1]])  # [knee -> ankle]

                # 내적과 벡터 크기를 이용한 각도 계산
                dot_product = np.dot(vector_1, vector_2)
                magnitude_1 = np.linalg.norm(vector_1)
                magnitude_2 = np.linalg.norm(vector_2)
                cosine_angle = dot_product / (magnitude_1 * magnitude_2)
                angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

                # 각도를 도(degree)로 변환
                right_angle_degrees = np.degrees(angle)

                # 외적을 이용한 방향성 확인 (부호 반전)
                cross_product = np.cross(vector_1, vector_2)
                direction = "내반슬(O자 다리)" if cross_product < 0 else "외반슬(X자 다리)"

                # 상태 판단
                if right_angle_degrees < 170:
                    right_status = direction  # 내반슬 또는 외반슬로 설정
                elif 170 <= right_angle_degrees <= 180:
                    right_status = "정상"

                print(f"오른쪽 다리 상태: {right_status} (각도: {right_angle_degrees:.2f}도)")

            # 양쪽 다리 상태 비교
            if left_status == "정상" and right_status == "정상":
                print("양쪽 다리 상태: 정상")
            elif left_status == right_status:
                print(f"양쪽 다리 상태: {left_status} (대칭적)")
            else:
                print(f"왼쪽 다리 상태: {left_status}, 오른쪽 다리 상태: {right_status} (비대칭)")







            # 왼쪽 다리와 오른쪽 다리가 이루는 각도 계산
            if left_hip and right_hip and left_knee and right_knee:
                # 왼쪽 다리 벡터 (왼쪽 엉덩이 -> 왼쪽 무릎)
                vector_left_leg = np.array([left_knee[0] - left_hip[0], left_knee[1] - left_hip[1]])
                # 오른쪽 다리 벡터 (오른쪽 엉덩이 -> 오른쪽 무릎)
                vector_right_leg = np.array([right_knee[0] - right_hip[0], right_knee[1] - right_hip[1]])

                # 벡터 간 각도 계산
                cosine_angle = np.dot(vector_left_leg, vector_right_leg) / (np.linalg.norm(vector_left_leg) * np.linalg.norm(vector_right_leg))
                leg_angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

                print(f"두 다리가 이루는 각도: {leg_angle:.2f}도")
            


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
