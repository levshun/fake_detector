import numpy as np
import cv2
import math
from scipy.spatial import distance

def extract_perceptual_features(image, gray, landmarks):
    # Нормировка по ширине лица (между точками 0 и 16)
    face_width = distance.euclidean(
        (landmarks.part(0).x, landmarks.part(0).y),
        (landmarks.part(16).x, landmarks.part(16).y)
    )

    # 1. Вычисление показателя глаз (eye aspect ratio) для каждого глаза
    # Левый глаз: EAR = (||p37-p41|| + ||p38-p40||) / (2 * ||p36-p39||)
    left_eye_aspect_ratio = (
                                    distance.euclidean((landmarks.part(37).x, landmarks.part(37).y),
                                                       (landmarks.part(41).x, landmarks.part(41).y)) +
                                    distance.euclidean((landmarks.part(38).x, landmarks.part(38).y),
                                                       (landmarks.part(40).x, landmarks.part(40).y))
                            ) / (2.0 * distance.euclidean((landmarks.part(36).x, landmarks.part(36).y),
                                                          (landmarks.part(39).x, landmarks.part(39).y)))
    left_eye_aspect_ratio /= face_width  # нормировка

    # Правый глаз: EAR = (||p43-p47|| + ||p44-p46||) / (2 * ||p42-p45||)
    right_eye_aspect_ratio = (
                                     distance.euclidean((landmarks.part(43).x, landmarks.part(43).y),
                                                        (landmarks.part(47).x, landmarks.part(47).y)) +
                                     distance.euclidean((landmarks.part(44).x, landmarks.part(44).y),
                                                        (landmarks.part(46).x, landmarks.part(46).y))
                             ) / (2.0 * distance.euclidean((landmarks.part(42).x, landmarks.part(42).y),
                                                           (landmarks.part(45).x, landmarks.part(45).y)))
    right_eye_aspect_ratio /= face_width  # нормировка

    # 2. Разница между показателями глаз
    eye_aspect_ratio_diff = abs(left_eye_aspect_ratio - right_eye_aspect_ratio)

    # 3. Площадь глаз относительно горизонтальной ширины
    # Левый глаз: точки 36-41
    left_eye_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(36, 42)])
    left_eye_area = cv2.contourArea(left_eye_points)
    left_eye_width = distance.euclidean((landmarks.part(36).x, landmarks.part(36).y),
                                        (landmarks.part(39).x, landmarks.part(39).y))
    left_eye_area_ratio = left_eye_area / (left_eye_width ** 2 + 1e-8)

    # Правый глаз: точки 42-47
    right_eye_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(42, 48)])
    right_eye_area = cv2.contourArea(right_eye_points)
    right_eye_width = distance.euclidean((landmarks.part(42).x, landmarks.part(42).y),
                                         (landmarks.part(45).x, landmarks.part(45).y))
    right_eye_area_ratio = right_eye_area / (right_eye_width ** 2 + 1e-8)

    # 4. Площадь губ (внешний контур: точки 48-59) относительно ширины рта
    lip_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(48, 60)])
    lip_area = cv2.contourArea(lip_points)
    mouth_width = distance.euclidean((landmarks.part(48).x, landmarks.part(48).y),
                                     (landmarks.part(54).x, landmarks.part(54).y))
    lip_area_ratio = lip_area / (mouth_width ** 2 + 1e-8)

    # 5. Smile index – отношение ширины рта к площади губ
    smile_index = mouth_width / (lip_area + 1e-8)

    # 6. Nasolabial angle – угол между вектором от носа (точка 30) к верхней губе (точка 51)
    # и вектором от верхней губы (точка 51) к подбородку (точка 8)
    v1 = (landmarks.part(51).x - landmarks.part(30).x,
          landmarks.part(51).y - landmarks.part(30).y)
    v2 = (landmarks.part(8).x - landmarks.part(51).x,
          landmarks.part(8).y - landmarks.part(51).y)
    dot_prod = v1[0] * v2[0] + v1[1] * v2[1]
    norm_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    norm_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if norm_v1 * norm_v2 == 0:
        nasolabial_angle = 0
    else:
        angle_rad = math.acos(dot_prod / (norm_v1 * norm_v2))
        nasolabial_angle = angle_rad / math.pi  # нормировка в диапазоне [0, 1]

    # 7. Eye roundness – округлость глаз: (4 * area) / (pi * (width)^2)
    left_eye_roundness = (4 * left_eye_area) / (math.pi * (left_eye_width ** 2) + 1e-8)
    right_eye_roundness = (4 * right_eye_area) / (math.pi * (right_eye_width ** 2) + 1e-8)
    eye_roundness = (left_eye_roundness + right_eye_roundness) / 2.0

    # 8. Lip thickness ratio – расстояние между верхней (точка 51) и нижней губой (точка 57) относительно ширины рта
    lip_thickness = distance.euclidean((landmarks.part(51).x, landmarks.part(51).y),
                                       (landmarks.part(57).x, landmarks.part(57).y))
    lip_thickness_ratio = lip_thickness / (mouth_width + 1e-8)

    # 9. Expression index – агрегированный индекс перцептивного выражения (среднее значение показателей глаз и губ)
    expression_index = (left_eye_aspect_ratio + right_eye_aspect_ratio + lip_thickness_ratio) / 3.0

    # 10. Дополнительно, можно вернуть усреднённое значение eye_aspect_ratio и lip_thickness_ratio (альтернативное lip_aspect_ratio)
    avg_eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0

    return {
        "left_eye_aspect_ratio": left_eye_aspect_ratio,
        "right_eye_aspect_ratio": right_eye_aspect_ratio,
        "eye_aspect_ratio_diff": eye_aspect_ratio_diff,
        "left_eye_area_ratio": left_eye_area_ratio,
        "right_eye_area_ratio": right_eye_area_ratio,
        "lip_area_ratio": lip_area_ratio,
        "smile_index": smile_index,
        "nasolabial_angle": nasolabial_angle,
        "eye_roundness": eye_roundness,
        "lip_thickness_ratio": lip_thickness_ratio,
        "expression_index": expression_index,
        "avg_eye_aspect_ratio": avg_eye_aspect_ratio
    }
