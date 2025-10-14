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

from swapping.feature_calculator import calculate_features_one, flatten_landmarks
from swapping.exceptions import ModelLoadingError

logger = logging.getLogger(__name__)


class DeepfakePredictor:
    def __init__(self, models_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Используемое устройство: {self.device}")

        try:
            fb_dir = os.path.join(models_dir, "feature_based")
            nn_dir = os.path.join(models_dir, "efficientnet")

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
        except (FileNotFoundError, CatBoostError, OSError) as e:
            raise ModelLoadingError(f"Не удалось загрузить модель: {e}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_feature_based_probs(self, df: pd.DataFrame) -> dict:
        X_ef = df[self.rf_ef.feature_names_in_]
        prob_ef = self.rf_ef.predict_proba(X_ef)[0]

        num_landmark_cols = len(df.filter(regex=r'^\d+$', axis=1).columns)
        df_landmarks = df.loc[:, '0':str(num_landmark_cols - 1)]
        X_fl_expanded = flatten_landmarks(df_landmarks)
        X_fl = X_fl_expanded[self.rf_fl.feature_names_in_]
        prob_fl = self.rf_fl.predict_proba(X_fl)[0]

        X_lbp = df[self.cb_lbp.feature_names_]
        prob_lbp = self.cb_lbp.predict_proba(X_lbp)[0]

        X_tf_sf = df[self.cb_tf_sf.feature_names_]
        prob_tf_sf = self.cb_tf_sf.predict_proba(X_tf_sf)[0]

        return {'ef': prob_ef, 'fl': prob_fl, 'lbp': prob_lbp, 'tf_sf': prob_tf_sf}

    def predict(self, image_path: str) -> dict:
        features_df = calculate_features_one(image_path)
        fb_probs = self._get_feature_based_probs(features_df)

        effnet_predictions = {}
        effnet_probs_for_meta = np.array([0.5, 0.5])
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

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

        meta_features = np.hstack([
            effnet_probs_for_meta, fb_probs['ef'], fb_probs['fl'],
            fb_probs['lbp'], fb_probs['tf_sf']
        ]).reshape(1, -1)

        final_prediction = self.meta_model.predict(meta_features)[0]
        final_probability = self.meta_model.predict_proba(meta_features)[0]

        result = {
            "final_decision": "Fake" if final_prediction == 1 else "Real",
            "final_probability": {'real_prob': float(final_probability[0]), 'fake_prob': float(final_probability[1])},
            "base_models_prob": {
                "ef_rf": {'real_prob': float(fb_probs['ef'][0]), 'fake_prob': float(fb_probs['ef'][1])},
                "fl_rf": {'real_prob': float(fb_probs['fl'][0]), 'fake_prob': float(fb_probs['fl'][1])},
                "lbp_cb": {'real_prob': float(fb_probs['lbp'][0]), 'fake_prob': float(fb_probs['lbp'][1])},
                "tf_sf_cb": {'real_prob': float(fb_probs['tf_sf'][0]), 'fake_prob': float(fb_probs['tf_sf'][1])}
            },
            "specialized_effnet_prob": effnet_predictions
        }
        return result