import cv2
import numpy as np

def extract_eye_reflection_features(image, gray, landmarks):
    # Индексы для левого и правого глаз (dlib 68 landmarks)
    left_eye_indices = list(range(36, 42))
    right_eye_indices = list(range(42, 48))

    # Создаем маски для каждого глаза
    mask_left = np.zeros_like(gray)
    mask_right = np.zeros_like(gray)

    left_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in left_eye_indices])
    right_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in right_eye_indices])

    cv2.fillPoly(mask_left, [left_eye_points], 255)
    cv2.fillPoly(mask_right, [right_eye_points], 255)

    # Извлекаем области глаз из серого изображения
    left_eye_region = cv2.bitwise_and(gray, gray, mask=mask_left)
    right_eye_region = cv2.bitwise_and(gray, gray, mask=mask_right)

    # Применяем сглаживание для уменьшения шума
    left_eye_region = cv2.GaussianBlur(left_eye_region, (3, 3), 0)
    right_eye_region = cv2.GaussianBlur(right_eye_region, (3, 3), 0)

    # Вспомогательная функция для вычисления отношения ярких пикселей при заданном пороге
    def compute_reflection_ratio(eye_region, threshold):
        reflections = np.sum(eye_region >= threshold)
        total_pixels = np.sum(eye_region > 0)  # считаем ненулевые пиксели
        return reflections / total_pixels if total_pixels > 0 else 0

    # Для левого глаза вычисляем динамические пороги
    left_nonzero = left_eye_region[left_eye_region > 0]
    if left_nonzero.size > 0:
        left_mean = np.mean(left_nonzero)
        left_std = np.std(left_nonzero)
    else:
        left_mean = left_std = 0

    # Для правого глаза вычисляем динамические пороги
    right_nonzero = right_eye_region[right_eye_region > 0]
    if right_nonzero.size > 0:
        right_mean = np.mean(right_nonzero)
        right_std = np.std(right_nonzero)
    else:
        right_mean = right_std = 0

    # Определяем три варианта порогов: низкий, динамический и высокий
    left_thresh_low = left_mean
    left_thresh_dynamic = left_mean + left_std
    left_thresh_high = left_mean + 2 * left_std

    right_thresh_low = right_mean
    right_thresh_dynamic = right_mean + right_std
    right_thresh_high = right_mean + 2 * right_std

    # Вычисляем коэффициенты отражений для каждого варианта порога
    left_ratio_low = compute_reflection_ratio(left_eye_region, left_thresh_low)
    left_ratio_dynamic = compute_reflection_ratio(left_eye_region, left_thresh_dynamic)
    left_ratio_high = compute_reflection_ratio(left_eye_region, left_thresh_high)

    right_ratio_low = compute_reflection_ratio(right_eye_region, right_thresh_low)
    right_ratio_dynamic = compute_reflection_ratio(right_eye_region, right_thresh_dynamic)
    right_ratio_high = compute_reflection_ratio(right_eye_region, right_thresh_high)

    # Усредняем результаты по обоим глазам
    reflection_ratio_low = (left_ratio_low + right_ratio_low) / 2.0
    reflection_ratio_dynamic = (left_ratio_dynamic + right_ratio_dynamic) / 2.0
    reflection_ratio_high = (left_ratio_high + right_ratio_high) / 2.0

    return {
        "eye_reflection_ratio_low": reflection_ratio_low,
        "eye_reflection_ratio_dynamic": reflection_ratio_dynamic,
        "eye_reflection_ratio_high": reflection_ratio_high
    }
