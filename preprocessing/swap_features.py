import ast
import logging
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from skimage.feature import graycoprops, graycomatrix, local_binary_pattern
from skimage.util import img_as_ubyte

from preprocessing.face_feature_extractor_main import FaceFeatureExtractor

try:
    from swapping.exceptions import FeatureExtractionError
except Exception:  # pragma: no cover - fallback for standalone use
    class FeatureExtractionError(Exception):
        """Fallback exception when swapping package is unavailable."""


logger = logging.getLogger(__name__)

LBP_METHOD = "uniform"
LBP_RADIUS = 3
LBP_N_POINTS = 8 * LBP_RADIUS

mp_face_mesh = mp.solutions.face_mesh


def get_landmark_points(landmarks):
    return np.asarray([[lm.x, lm.y, lm.z] for lm in landmarks])


def get_geometric_distances(points):
    # TODO: заполнить реальными метриками; сейчас заглушка для совместимости.
    return [0.0] * 10


def extract_statistical_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "mean_intensity": np.mean(gray_image),
        "std_intensity": np.std(gray_image),
        "median_intensity": np.median(gray_image),
    }


def extract_glcm_features(image, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
    gray_image = img_as_ubyte(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    return {
        "contrast": graycoprops(glcm, "contrast").mean(),
        "energy": graycoprops(glcm, "energy").mean(),
        "homogeneity": graycoprops(glcm, "homogeneity").mean(),
        "correlation": graycoprops(glcm, "correlation").mean(),
    }


def calculate_features_one(image_path: str) -> tuple:
    start_total = time.perf_counter()

    logger.debug(f"Начало расчета признаков для изображения: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Не удалось прочитать изображение по пути: {image_path}")
        raise FeatureExtractionError(f"Не удалось прочитать изображение: {image_path}")

    data_dict = {}
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            logger.error(f"MediaPipe не нашел лицо на изображении: {image_path}")
            raise FeatureExtractionError(f"MediaPipe не нашел лицо на изображении: {image_path}")

        face_landmarks = results.multi_face_landmarks[0]
        points = get_landmark_points(face_landmarks.landmark)
        logger.debug(f"MediaPipe нашел {len(points)} ориентиров.")
        hull = ConvexHull(points)
        geometric_distances = get_geometric_distances(points)

        base_features_names = [
            "face_k",
            "l_eye_k",
            "r_eye_k",
            "mouth_k",
            "k1_l",
            "k1_r",
            "k2_l",
            "k2_r",
            "k3_l",
            "k3_r",
        ]
        for i, name in enumerate(base_features_names):
            data_dict[name] = [geometric_distances[i]]

        data_dict["volume"] = [hull.volume]
        data_dict["area"] = [hull.area]

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray_image, LBP_N_POINTS, LBP_RADIUS, LBP_METHOD)
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_N_POINTS + 3), range=(0, LBP_N_POINTS + 2))
        hist = hist.astype("float")
        hist /= hist.sum() + 1e-7
        for i, val in enumerate(hist):
            data_dict[f"hist_{i}"] = [val]

        stat_dict = extract_statistical_features(image)
        texture_dict = extract_glcm_features(image)
        for key, val in stat_dict.items():
            data_dict[key] = [val]
        for key, val in texture_dict.items():
            data_dict[key] = [val]

        for i, point in enumerate(points):
            data_dict[str(i)] = [list(point)]

    end_standard = time.perf_counter()

    feat_extr = FaceFeatureExtractor()
    try:
        custom_features = feat_extr.extract_features(image)
        if not custom_features:
            logger.error("FaceFeatureExtractor вернул пустой список признаков.")
            raise ValueError("FaceFeatureExtractor вернул пустой список признаков.")

        for key, val in custom_features[0].items():
            data_dict[key] = [val]

    except Exception as e:
        logger.error(f"Критическая ошибка при извлечении структурных признаков (FaceFeatureExtractor): {e}")
        raise FeatureExtractionError(f"Ошибка при извлечении структурных признаков лица: {e}")

    end_custom = time.perf_counter()

    df = pd.DataFrame.from_dict(data_dict)
    logger.info(f"Расчет признаков завершен. Итоговый DataFrame содержит {len(df.columns)} столбцов.")

    timings = {
        "std_features_time": end_standard - start_total,
        "custom_lib_time": end_custom - end_standard,
    }

    return df, timings


def flatten_landmarks(df_landmarks: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Начало преобразования DataFrame с ориентирами...")
    if df_landmarks.empty:
        return pd.DataFrame()

    sample_cell = df_landmarks.values[0][0]

    if isinstance(sample_cell, str):
        try:
            parsed_df = df_landmarks.apply(lambda col: col.apply(ast.literal_eval))
        except (ValueError, SyntaxError) as e:
            raise FeatureExtractionError(f"Ошибка парсинга строковых ориентиров: {e}")
    else:
        parsed_df = df_landmarks

    flat_coords = np.hstack(parsed_df.values[0])
    num_landmarks = len(parsed_df.columns)
    feature_names = [f"{i}{axis}" for i in range(num_landmarks) for axis in ["x", "y", "z"]]

    if len(flat_coords) != len(feature_names):
        raise ValueError(f"Несоответствие размеров: {len(flat_coords)} признаков и {len(feature_names)} имен.")

    result_df = pd.DataFrame([flat_coords], columns=feature_names)
    logger.debug("Преобразование DataFrame с ориентирами успешно завершено.")
    return result_df


__all__ = [
    "calculate_features_one",
    "extract_statistical_features",
    "extract_glcm_features",
    "flatten_landmarks",
    "FeatureExtractionError",
]
