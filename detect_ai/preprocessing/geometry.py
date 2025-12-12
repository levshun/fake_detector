import numpy as np
from scipy.spatial import distance
import math
import cv2

def extract_geometry_features(landmarks):
    # Базовые точки лица
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)

    left_ear = (landmarks.part(0).x, landmarks.part(0).y)
    right_ear = (landmarks.part(16).x, landmarks.part(16).y)

    nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
    chin_tip = (landmarks.part(8).x, landmarks.part(8).y)

    mouth_left = (landmarks.part(48).x, landmarks.part(48).y)
    mouth_right = (landmarks.part(54).x, landmarks.part(54).y)

    # Базовые размеры лица для нормализации
    face_width = distance.euclidean(left_ear, right_ear)
    face_height = abs(landmarks.part(8).y - landmarks.part(27).y)

    # Основные нормализованные расстояния
    eye_distance = distance.euclidean(left_eye, right_eye) / face_width
    ear_distance = distance.euclidean(left_ear, right_ear) / face_height
    nose_to_chin_distance = distance.euclidean(nose_tip, chin_tip) / face_height
    mouth_width = distance.euclidean(mouth_left, mouth_right) / face_width
    face_symmetry = abs(landmarks.part(16).x - 2 * landmarks.part(27).x + landmarks.part(0).x) / face_width

    aspect_ratio = face_width / face_height if face_height > 0 else 0
    forehead_to_nose_ratio = abs(landmarks.part(27).y - landmarks.part(30).y) / face_height if face_height > 0 else 0

    # Дополнительные признаки

    # 1. Смещение центра глаз относительно центра лица
    eye_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    face_center = (landmarks.part(27).x, landmarks.part(27).y)
    eye_center_offset = distance.euclidean(eye_center, face_center) / face_width

    # 2. Расстояние между бровями и глазами
    left_brow_y = np.mean([landmarks.part(i).y for i in range(17, 22)])
    left_eye_y = np.mean([landmarks.part(i).y for i in range(36, 42)])
    left_brow_to_eye = abs(left_brow_y - left_eye_y)

    right_brow_y = np.mean([landmarks.part(i).y for i in range(22, 27)])
    right_eye_y = np.mean([landmarks.part(i).y for i in range(42, 48)])
    right_brow_to_eye = abs(right_brow_y - right_eye_y)

    brow_to_eye_distance = ((left_brow_to_eye + right_brow_to_eye) / 2.0) / face_height

    # 3. Угол линии челюсти (jawline_angle)
    v_left = (landmarks.part(0).x - landmarks.part(8).x, landmarks.part(0).y - landmarks.part(8).y)
    v_right = (landmarks.part(16).x - landmarks.part(8).x, landmarks.part(16).y - landmarks.part(8).y)
    dot_product = v_left[0] * v_right[0] + v_left[1] * v_right[1]
    norm_left = math.sqrt(v_left[0]**2 + v_left[1]**2)
    norm_right = math.sqrt(v_right[0]**2 + v_right[1]**2)
    if norm_left * norm_right == 0:
        jawline_angle = 0
    else:
        angle = math.acos(dot_product / (norm_left * norm_right))
        jawline_angle = angle / math.pi  # Нормализация: угол в диапазоне [0,1]

    # 4. Отношение ширины носа к ширине лица (nose_width_ratio)
    nose_width = distance.euclidean((landmarks.part(31).x, landmarks.part(31).y),
                                    (landmarks.part(35).x, landmarks.part(35).y))
    nose_width_ratio = nose_width / face_width

    # 5. Круглость лица (face_circularity)
    jawline_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)])
    jawline_points = jawline_points.reshape((-1, 1, 2)).astype(np.int32)
    area = cv2.contourArea(jawline_points)
    perimeter = cv2.arcLength(jawline_points, True)
    if perimeter == 0:
        face_circularity = 0
    else:
        face_circularity = (4 * math.pi * area) / (perimeter ** 2)

    # 6. Угол наклона глаз (eye_slant_angle)
    # Для левого глаза: между точками 36 (внешний угол) и 39 (внутренний угол)
    left_dx = landmarks.part(39).x - landmarks.part(36).x
    left_dy = landmarks.part(39).y - landmarks.part(36).y
    left_angle = abs(math.atan2(left_dy, left_dx))

    # Для правого глаза: между точками 42 (внутренний угол) и 45 (внешний угол)
    right_dx = landmarks.part(45).x - landmarks.part(42).x
    right_dy = landmarks.part(45).y - landmarks.part(42).y
    right_angle = abs(math.atan2(right_dy, right_dx))

    # Средний угол наклона глаз, нормированный на π
    eye_slant_angle = (left_angle + right_angle) / (2.0 * math.pi)

    # Объединяем все признаки в один словарь
    return {
        "eye_distance": eye_distance,
        "ear_distance": ear_distance,
        "aspect_ratio": aspect_ratio,
        "nose_to_chin_distance": nose_to_chin_distance,
        "mouth_width": mouth_width,
        "forehead_to_nose_ratio": forehead_to_nose_ratio,
        "face_symmetry": face_symmetry,
        "eye_center_offset": eye_center_offset,
        "brow_to_eye_distance": brow_to_eye_distance,
        "jawline_angle": jawline_angle,
        "nose_width_ratio": nose_width_ratio,
        "face_circularity": face_circularity,
        "eye_slant_angle": eye_slant_angle
    }
