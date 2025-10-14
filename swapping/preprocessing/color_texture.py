import numpy as np
import cv2

def extract_color_texture_features(image, gray, landmarks):
    # Создаем маску для области лица (landmarks с 17 по 67)
    mask = np.zeros_like(gray)
    points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 68)])
    cv2.fillPoly(mask, [points], 255)

    # Извлечение области лица по маске для дальнейших вычислений
    masked_color = image[mask == 255]
    masked_gray = gray[mask == 255]

    # Существующие признаки
    skin_tone_std = np.std(masked_color)
    skin_color_variation = np.mean(np.abs(masked_color - np.median(masked_color)))
    noise_variation = np.std(cv2.Laplacian(gray, cv2.CV_64F))
    edge_sharpness = np.mean(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5))

    # Новые признаки на основе преобразования в градации серого (уже имеется gray) и HSV
    # 1. Среднее значение интенсивности (грейскейл)
    skin_color_mean = np.mean(masked_gray)

    # 2. Медианное значение интенсивности
    skin_color_median = np.median(masked_gray)

    # 3. Энтропия распределения интенсивностей
    hist, _ = np.histogram(masked_gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]  # исключаем нулевые вероятности
    skin_color_entropy = -np.sum(hist * np.log2(hist))

    # Преобразование в HSV для цветовых признаков
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masked_hsv = hsv[mask == 255]
    saturation = masked_hsv[:, 1]
    brightness = masked_hsv[:, 2]
    hue = masked_hsv[:, 0]

    # 4. Стандартное отклонение насыщенности
    saturation_std = np.std(saturation)

    # 5. Стандартное отклонение яркости
    brightness_std = np.std(brightness)

    # 6. Стандартное отклонение оттенка
    hue_std = np.std(hue)

    # 7. Контраст: (max - min) / (mean + epsilon)
    epsilon = 1e-6
    color_contrast = (np.max(masked_gray) - np.min(masked_gray)) / (np.mean(masked_gray) + epsilon)

    # 8. Плотность краёв в области лица
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum((edges > 0) & (mask == 255)) / (np.sum(mask == 255) + epsilon)

    return {
        "skin_tone_std": skin_tone_std,
        "skin_color_variation": skin_color_variation,
        "noise_variation": noise_variation,
        "edge_sharpness": edge_sharpness,
        "skin_color_mean": skin_color_mean,
        "skin_color_median": skin_color_median,
        "skin_color_entropy": skin_color_entropy,
        "saturation_std": saturation_std,
        "brightness_std": brightness_std,
        "hue_std": hue_std,
        "color_contrast": color_contrast,
        "edge_density": edge_density
    }
