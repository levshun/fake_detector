import os
from pathlib import Path

import cv2
import dlib

from preprocessing.geometry import extract_geometry_features
from preprocessing.color_texture import extract_color_texture_features
from preprocessing.frequency_artifacts import extract_frequency_artifacts_features
from preprocessing.perceptual_analysis import extract_perceptual_features
from preprocessing.eye_reflections import extract_eye_reflection_features
from preprocessing.hair_structure import extract_hair_structure_features
from preprocessing.perspective_depth import extract_perspective_depth_features

_DEFAULT_PREDICTOR = Path(__file__).resolve().parent.parent / "swapping" / "shape_predictor_68_face_landmarks.dat"


class FaceFeatureExtractor:
    """
    Извлекает совокупность признаков лица, опираясь на dlib landmarks.

    Принимает как путь к изображению, так и уже загруженный BGR numpy массив.
    """

    def __init__(self, predictor_path: str | os.PathLike | None = None):
        predictor = Path(predictor_path) if predictor_path else _DEFAULT_PREDICTOR
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(predictor))

    def _load_image(self, image_or_path):
        if isinstance(image_or_path, (str, os.PathLike)):
            return cv2.imread(str(image_or_path))
        return image_or_path

    def extract_features(self, image_or_path):
        image = self._load_image(image_or_path)
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) == 0:
            return None

        features_list = []
        for face in faces:
            landmarks = self.predictor(gray, face)

            # Извлечение признаков из разных модулей
            geometry_features = extract_geometry_features(landmarks)
            color_texture_features = extract_color_texture_features(image, gray, landmarks)
            frequency_artifacts_features = extract_frequency_artifacts_features(image, gray, landmarks)
            perceptual_features = extract_perceptual_features(image, gray, landmarks)
            eye_reflection_features = extract_eye_reflection_features(image, gray, landmarks)
            hair_structure_features = extract_hair_structure_features(image, gray, landmarks)
            perspective_depth_features = extract_perspective_depth_features(image, gray, landmarks)

            # Объединение всех признаков в один словарь
            features = {**geometry_features,
                        **color_texture_features,
                        **frequency_artifacts_features,
                        **perceptual_features,
                        **eye_reflection_features,
                        **hair_structure_features,
                        **perspective_depth_features}

            features_list.append(features)
        return features_list
