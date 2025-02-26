import cv2
import dlib

from geometry import extract_geometry_features
from color_texture import extract_color_texture_features
from frequency_artifacts import extract_frequency_artifacts_features
from perceptual_analysis import extract_perceptual_features
from eye_reflections import extract_eye_reflection_features
from hair_structure import extract_hair_structure_features
from perspective_depth import extract_perspective_depth_features

class FaceFeatureExtractor:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def extract_features(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print("Ошибка загрузки изображения:", image_path)
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) == 0:
            print("Лица не обнаружены на изображении:", image_path)
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
