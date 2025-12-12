import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_hair_structure_features(image, gray, landmarks):
    # Определяем область волос на основе точек бровей (landmarks 17-26)
    eyebrow_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(17, 27)])
    min_x = int(np.min(eyebrow_points[:, 0]))
    max_x = int(np.max(eyebrow_points[:, 0]))
    min_y = int(np.min(eyebrow_points[:, 1]))

    # Оценка высоты лица: от минимальной точки бровей до подбородка (landmarks.part(8))
    face_height = landmarks.part(8).y - min_y
    # Высота области волос — 50% от высоты лица (параметр можно настроить)
    hair_height = int(0.5 * face_height)

    # Определяем прямоугольную область для волос: сверху от бровей
    y1 = max(min_y - hair_height, 0)
    y2 = min_y
    x1 = min_x
    x2 = max_x

    hair_region = image[y1:y2, x1:x2]
    if hair_region.size == 0:
        return {
            "hair_texture_std": 0,
            "hair_edge_density": 0,
            "hair_color_mean": [0, 0, 0],
            "hair_color_std": [0, 0, 0],
            "hair_color_entropy": 0,
            "hair_gradient_mean": 0,
            "hair_gradient_std": 0,
            "hair_lbp_entropy": 0,
            "hair_sharpness": 0,
            "hair_density": 0,
            "hair_contrast": 0
        }

    # Признак 1. Стандартное отклонение текстуры (по всем каналам)
    hair_texture_std = np.std(hair_region)

    # Признак 2. Плотность краёв в области волос (детектор Canny)
    gray_hair = cv2.cvtColor(hair_region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_hair, 100, 200)
    hair_edge_density = np.sum(edges > 0) / (edges.size + 1e-8)

    # Признак 3. Среднее значение цветовых каналов (B, G, R)
    hair_color_mean = np.mean(hair_region, axis=(0, 1)).tolist()

    # Признак 4. Стандартное отклонение цветовых каналов
    hair_color_std = np.std(hair_region, axis=(0, 1)).tolist()

    # Признак 5. Энтропия распределения интенсивностей в области волос (градации серого)
    hist, _ = np.histogram(gray_hair, bins=256, range=(0, 256), density=True)
    hist_nonzero = hist[hist > 0]
    hair_color_entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero + 1e-8))

    # Признак 6. Градиентные характеристики
    grad_x = cv2.Sobel(gray_hair, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_hair, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    hair_gradient_mean = np.mean(grad_magnitude)
    hair_gradient_std = np.std(grad_magnitude)

    # Признак 7. Энтропия LBP (Local Binary Pattern)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_hair, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
    lbp_hist_nonzero = lbp_hist[lbp_hist > 0]
    hair_lbp_entropy = -np.sum(lbp_hist_nonzero * np.log2(lbp_hist_nonzero + 1e-8))

    # Признак 8. Резкость волос (вариация Лапласиана)
    hair_sharpness = cv2.Laplacian(gray_hair, cv2.CV_64F).var()

    # Признак 9. Плотность волос: отношение числа пикселей с яркостью > 50 к общему числу пикселей
    hair_dense_pixels = np.sum(gray_hair > 50)
    hair_density = hair_dense_pixels / (gray_hair.size + 1e-8)

    # Признак 10. Контрастность: (max - min) / (mean + epsilon)
    hair_contrast = (np.max(gray_hair) - np.min(gray_hair)) / (np.mean(gray_hair) + 1e-8)

    return {
        "hair_texture_std": hair_texture_std,
        "hair_edge_density": hair_edge_density,
        "hair_color_mean": hair_color_mean,
        "hair_color_std": hair_color_std,
        "hair_color_entropy": hair_color_entropy,
        "hair_gradient_mean": hair_gradient_mean,
        "hair_gradient_std": hair_gradient_std,
        "hair_lbp_entropy": hair_lbp_entropy,
        "hair_sharpness": hair_sharpness,
        "hair_density": hair_density,
        "hair_contrast": hair_contrast
    }
