import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib
from catboost import CatBoostClassifier, CatBoostError
import numpy as np
import pandas as pd
import logging
import cv2

from swapping.feature_calculator import calculate_features_one, flatten_landmarks
from swapping.exceptions import ModelLoadingError, FeatureExtractionError, SwappingError

logger = logging.getLogger(__name__)


class FaceCropper:
    def __init__(self, models_dir: str):
        model_path = os.path.join(models_dir, 'face_detection_yunet_2023mar.onnx')
        if not os.path.exists(model_path):
            logger.error(f"Файл модели детекции лиц не найден: {model_path}")
            raise ModelLoadingError(f"Отсутствует обязательный файл модели детекции лиц: {model_path}")

        try:
            self.face_detector = cv2.FaceDetectorYN.create(
                model=model_path,
                config='',
                input_size=(320, 320),
                score_threshold=0.5,
                nms_threshold=0.3,
                top_k=5000,
                backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
                target_id=cv2.dnn.DNN_TARGET_CPU
            )
        except Exception as e:
            logger.error(f"Ошибка инициализации FaceDetectorYN: {e}")
            raise ModelLoadingError(f"Ошибка инициализации FaceDetectorYN: {e}")

    def crop_face(self, image: Image.Image) -> Image.Image:
        if image is None:
            return None

        try:
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            h_orig, w_orig = frame.shape[:2]
            self.face_detector.setInputSize((w_orig, h_orig))

            _, faces = self.face_detector.detect(frame)

            if faces is None or len(faces) == 0:
                return None

            face = max(faces, key=lambda f: f[14])
            x, y, w, h = map(int, face[0:4])

            margin = int(0.2 * max(w, h))
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(w_orig - x, w + 2 * margin)
            h = min(h_orig - y, h + 2 * margin)

            max_dim = max(w, h)
            x_center = x + w // 2
            y_center = y + h // 2

            x1 = max(0, x_center - max_dim // 2)
            y1 = max(0, y_center - max_dim // 2)
            x2 = min(w_orig, x1 + max_dim)
            y2 = min(h_orig, y1 + max_dim)

            if x2 - x1 < max_dim:
                x1 = max(0, x2 - max_dim)
            if y2 - y1 < max_dim:
                y1 = max(0, y2 - max_dim)

            face_crop = frame[y1:y1 + max_dim, x1:x1 + max_dim]

            if face_crop.size == 0:
                return None

            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_crop = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_AREA)

            return Image.fromarray(face_crop)

        except Exception as e:
            logger.warning(f"Ошибка при обрезке лица: {e}")
            return None

    def get_fallback_crop(self, image: Image.Image) -> Image.Image:
        try:
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            size = min(h, w)
            x = (w - size) // 2
            y = (h - size) // 2
            cropped = img_array[y:y + size, x:x + size]
            cropped = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_AREA)
            return Image.fromarray(cropped)
        except Exception as e:
            logger.error(f"Ошибка при создании fallback кропа: {e}")
            return image.resize((256, 256))


class DeepfakePredictor:
    def __init__(self, models_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Используемое устройство: {self.device}")

        try:
            self.face_cropper = FaceCropper(models_dir)

            fb_dir = os.path.join(models_dir, "feature_based")
            nn_dir = os.path.join(models_dir, "efficientnet")

            if not os.path.exists(fb_dir):
                raise ModelLoadingError(f"Папка с feature-based моделями не найдена: {fb_dir}")

            self.rf_ef = joblib.load(os.path.join(fb_dir, "random_forest_ef.pkl"))
            self.rf_fl = joblib.load(os.path.join(fb_dir, "random_forest_fl.pkl"))
            self.cb_lbp = CatBoostClassifier().load_model(os.path.join(fb_dir, "catboost_lbp.cbm"))
            self.cb_tf_sf = CatBoostClassifier().load_model(os.path.join(fb_dir, "catboost_tf_sf.cbm"))
            self.meta_model = joblib.load(os.path.join(fb_dir, "meta_model_ensemble.pkl"))
            logging.info("Все модели на признаках успешно загружены.")

            self.effnet_models = {}
            if os.path.isdir(nn_dir):
                for filename in os.listdir(nn_dir):
                    if filename.endswith(".pth"):
                        tech_name = os.path.splitext(filename)[0].replace("effnet_", "")
                        logging.info(f"Загрузка модели EfficientNet для технологии '{tech_name}'...")
                        model_path = os.path.join(nn_dir, filename)
                        model = models.efficientnet_b4(weights=None)
                        model.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                                         nn.Linear(model.classifier[1].in_features, 2))
                        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                        model.eval()
                        self.effnet_models[tech_name] = model.to(self.device)
        except (FileNotFoundError, CatBoostError, OSError, RuntimeError) as e:
            # Ловим стандартные ошибки и оборачиваем их в ModelLoadingError для единообразия
            raise ModelLoadingError(f"Не удалось загрузить модель: {e}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _validate_features(self, df: pd.DataFrame, required_features: list, model_name: str):
        """Проверяет наличие необходимых колонок в DataFrame."""
        missing_features = [col for col in required_features if col not in df.columns]
        if missing_features:
            logger.error(f"Отсутствуют признаки для модели {model_name}: {missing_features}")
            raise FeatureExtractionError(
                f"Не удалось сформировать полный вектор признаков. Отсутствуют поля для {model_name}: {missing_features[:5]}..."
            )

    def _get_feature_based_probs(self, df: pd.DataFrame) -> dict:
        try:
            # RF EF
            self._validate_features(df, self.rf_ef.feature_names_in_, "RandomForest_EF")
            X_ef = df[self.rf_ef.feature_names_in_]
            prob_ef = self.rf_ef.predict_proba(X_ef)[0]

            # RF FL (Landmarks)
            num_landmark_cols = len(df.filter(regex=r'^\d+$', axis=1).columns)
            df_landmarks = df.loc[:, '0':str(num_landmark_cols - 1)]
            X_fl_expanded = flatten_landmarks(df_landmarks)

            self._validate_features(X_fl_expanded, self.rf_fl.feature_names_in_, "RandomForest_FL")
            X_fl = X_fl_expanded[self.rf_fl.feature_names_in_]
            prob_fl = self.rf_fl.predict_proba(X_fl)[0]

            # CatBoost LBP
            self._validate_features(df, self.cb_lbp.feature_names_, "CatBoost_LBP")
            X_lbp = df[self.cb_lbp.feature_names_]
            prob_lbp = self.cb_lbp.predict_proba(X_lbp)[0]

            # CatBoost TF SF
            self._validate_features(df, self.cb_tf_sf.feature_names_, "CatBoost_TF_SF")
            X_tf_sf = df[self.cb_tf_sf.feature_names_]
            prob_tf_sf = self.cb_tf_sf.predict_proba(X_tf_sf)[0]

            return {'ef': prob_ef, 'fl': prob_fl, 'lbp': prob_lbp, 'tf_sf': prob_tf_sf}

        except KeyError as e:
            # На случай, если что-то проскочило валидацию, но вызвало ошибку pandas
            raise FeatureExtractionError(f"Ошибка доступа к признакам при прогнозировании: {e}")
        except Exception as e:
            raise SwappingError(f"Ошибка при расчете вероятностей на основе признаков: {e}")

    def predict(self, image_path: str) -> dict:
        features_df = calculate_features_one(image_path)

        # Этот метод теперь гарантированно выбросит понятное исключение, если признаков не хватает
        fb_probs = self._get_feature_based_probs(features_df)

        effnet_predictions = {}
        effnet_probs_for_meta = np.array([0.5, 0.5])

        try:
            image = Image.open(image_path).convert('RGB')

            cropped_image = self.face_cropper.crop_face(image)
            if cropped_image is None:
                logger.info("Лицо не найдено, используется fallback кроп.")
                cropped_image = self.face_cropper.get_fallback_crop(image)

            image_tensor = self.transform(cropped_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if self.effnet_models:
                    main_effnet_name = next(iter(self.effnet_models))
                    output = self.effnet_models[main_effnet_name](image_tensor)
                    effnet_probs_for_meta = torch.softmax(output, dim=1)[0].cpu().numpy()

                    for tech_name, model in self.effnet_models.items():
                        if tech_name == main_effnet_name:
                            prob = effnet_probs_for_meta
                        else:
                            output = model(image_tensor)
                            prob = torch.softmax(output, dim=1)[0].cpu().numpy()
                        effnet_predictions[tech_name] = {'real_prob': float(prob[0]), 'fake_prob': float(prob[1])}
                else:
                    logging.warning("Модели EfficientNet не загружены.")
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения для EfficientNet: {e}")
            effnet_probs_for_meta = np.array([0.5, 0.5])

        try:
            meta_features = np.hstack([
                effnet_probs_for_meta, fb_probs['ef'], fb_probs['fl'],
                fb_probs['lbp'], fb_probs['tf_sf']
            ]).reshape(1, -1)

            final_prediction = self.meta_model.predict(meta_features)[0]
            final_probability = self.meta_model.predict_proba(meta_features)[0]

            result = {
                "final_decision": "Fake" if final_prediction == 1 else "Real",
                "final_probability": {'real_prob': float(final_probability[0]),
                                      'fake_prob': float(final_probability[1])},
                "base_models_prob": {
                    "ef_rf": {'real_prob': float(fb_probs['ef'][0]), 'fake_prob': float(fb_probs['ef'][1])},
                    "fl_rf": {'real_prob': float(fb_probs['fl'][0]), 'fake_prob': float(fb_probs['fl'][1])},
                    "lbp_cb": {'real_prob': float(fb_probs['lbp'][0]), 'fake_prob': float(fb_probs['lbp'][1])},
                    "tf_sf_cb": {'real_prob': float(fb_probs['tf_sf'][0]), 'fake_prob': float(fb_probs['tf_sf'][1])}
                },
                "specialized_effnet_prob": effnet_predictions
            }
            return result
        except Exception as e:
            logger.error(f"Ошибка в финальном мета-прогнозе: {e}")
            raise SwappingError(f"Ошибка при формировании финального решения: {e}")