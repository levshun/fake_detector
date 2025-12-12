import os
from pathlib import Path

import cv2
import dlib

import detect_ai as dai

_DEFAULT_PREDICTOR = os.path.join('..', 'models', 'swapping', 'shape_predictor_68_face_landmarks.dat')


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
            geometry_features = dai.extract_geometry_features(landmarks)
            color_texture_features = dai.extract_color_texture_features(image, gray, landmarks)
            frequency_artifacts_features = dai.extract_frequency_artifacts_features(image, gray, landmarks)
            perceptual_features = dai.extract_perceptual_features(image, gray, landmarks)
            eye_reflection_features = dai.extract_eye_reflection_features(image, gray, landmarks)
            hair_structure_features = dai.extract_hair_structure_features(image, gray, landmarks)
            perspective_depth_features = dai.extract_perspective_depth_features(image, gray, landmarks)

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
