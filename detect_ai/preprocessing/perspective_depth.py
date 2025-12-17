import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import math
from scipy.stats import skew, kurtosis

def load_midas_model():
    """
    Загружает предобученную модель MiDaS для оценки глубины.
    """
    model_type = "DPT_Large"  # Можно использовать "MiDaS_small" для меньшей модели
    model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    return model, transform, device

# Загружаем модель один раз (глобально)
_model, _transform, _device = load_midas_model()

def extract_perspective_depth_features(image, gray, landmarks):
    """
    Извлекает признаки перспективы и глубины на основе моноскопической оценки глубины.

    Args:
        avg_face_depth: Средняя нормализованная глубина в области лица.

        std_face_depth: Стандартное отклонение глубины в области лица.

        max_face_depth: Максимальная глубина в области лица.

        min_face_depth: Минимальная глубина в области лица.

        face_depth_range: Разница между максимальной и минимальной глубиной в области лица.

        depth_skewness: Асимметрия распределения глубин.

        depth_kurtosis: Куртозис распределения глубин.

        depth_std_ratio: Отношение стандартного отклонения глубины лица к стандартному отклонению всей карты.

        face_depth_gradient: Разница между средней глубиной верхней и нижней частей лица.

        boundary_depth_discontinuity: Разница между средней глубиной вокруг лица и внутри лица.

        mean_depth_gradient_orientation: Средняя ориентация градиента глубины в области лица.

        std_depth_gradient_orientation: Стандартное отклонение ориентации градиента глубины.

        depth_ratio: Отношение средней глубины лица к средней глубине всего изображения.

    """
    eps = 1e-8

    # Преобразуем изображение для модели: BGR -> RGB -> PIL Image -> numpy массив
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_pil = transforms.ToPILImage()(image_rgb)
    input_array = np.array(input_pil)  # Преобразование в numpy-массив

    # Применяем преобразование MiDaS
    sample = _transform(input_array)
    if isinstance(sample, dict):
        input_tensor = sample["image"].to(_device)
    else:
        input_tensor = sample.to(_device)

    # Добавляем измерение батча, если его нет (ожидаем форма [B, 3, H, W])
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        prediction = _model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Нормализуем карту глубины к диапазону [0, 1]
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min + eps)

    # Определяем область лица по landmarks (используем все 68 точек)
    face_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
    x_min, y_min = np.min(face_points, axis=0)
    x_max, y_max = np.max(face_points, axis=0)

    # Извлекаем область лица из нормализованной карты глубины
    face_depth = depth_norm[int(y_min):int(y_max), int(x_min):int(x_max)]
    if face_depth.size == 0:
        avg_face_depth = std_face_depth = max_face_depth = min_face_depth = face_depth_gradient = 0
        face_depth_range = depth_skewness = depth_kurtosis = depth_std_ratio = 0
        boundary_depth_discontinuity = mean_depth_gradient_orientation = std_depth_gradient_orientation = 0
    else:
        avg_face_depth = np.mean(face_depth)
        std_face_depth = np.std(face_depth)
        max_face_depth = np.max(face_depth)
        min_face_depth = np.min(face_depth)
        face_depth_range = max_face_depth - min_face_depth

        # Статистика распределения глубин в области лица
        depth_skewness = skew(face_depth.flatten())
        depth_kurtosis = kurtosis(face_depth.flatten())

        # Разбиваем область лица на верхнюю и нижнюю части для оценки градиента глубины
        mid_y = int((y_min + y_max) / 2)
        upper_face = depth_norm[int(y_min):mid_y, int(x_min):int(x_max)]
        lower_face = depth_norm[mid_y:int(y_max), int(x_min):int(x_max)]
        upper_mean = np.mean(upper_face) if upper_face.size > 0 else avg_face_depth
        lower_mean = np.mean(lower_face) if lower_face.size > 0 else avg_face_depth
        face_depth_gradient = np.abs(upper_mean - lower_mean)

        # Стандартное отклонение по всей карте глубины
        std_image_depth = np.std(depth_norm)
        depth_std_ratio = std_face_depth / (std_image_depth + eps)

        # Граница лица: расширяем bounding box на margin пикселей
        margin = 10
        x1_border = max(int(x_min) - margin, 0)
        y1_border = max(int(y_min) - margin, 0)
        x2_border = min(int(x_max) + margin, depth_norm.shape[1])
        y2_border = min(int(y_max) + margin, depth_norm.shape[0])
        # Область за пределами лица в расширенном box, но без самой области лица
        border_region = depth_norm[y1_border:int(y_min), x1_border:x2_border]
        if border_region.size == 0:
            avg_border_depth = avg_face_depth
        else:
            avg_border_depth = np.mean(border_region)
        boundary_depth_discontinuity = np.abs(avg_border_depth - avg_face_depth)

        # Градиент глубины в области лица: вычисляем ориентацию
        grad_y, grad_x = np.gradient(face_depth)
        gradient_orientation = np.arctan2(grad_y, grad_x)
        # Нормируем углы на диапазон [0, π]
        gradient_orientation = np.abs(gradient_orientation)
        mean_depth_gradient_orientation = np.mean(gradient_orientation) / math.pi
        std_depth_gradient_orientation = np.std(gradient_orientation) / math.pi

    # Средняя глубина всего изображения
    avg_image_depth = np.mean(depth_norm)
    depth_ratio = avg_face_depth / (avg_image_depth + eps)

    return {
        "avg_face_depth": avg_face_depth,
        "std_face_depth": std_face_depth,
        "max_face_depth": max_face_depth,
        "min_face_depth": min_face_depth,
        "face_depth_range": face_depth_range,
        "depth_skewness": depth_skewness,
        "depth_kurtosis": depth_kurtosis,
        "depth_std_ratio": depth_std_ratio,
        "face_depth_gradient": face_depth_gradient,
        "boundary_depth_discontinuity": boundary_depth_discontinuity,
        "mean_depth_gradient_orientation": mean_depth_gradient_orientation,
        "std_depth_gradient_orientation": std_depth_gradient_orientation,
        "depth_ratio": depth_ratio
    }
