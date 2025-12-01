import json
import os
import time
from copy import copy

import joblib
import numpy as np
from scipy import stats

from modifying.classifier import ModClassifier
from preprocessing.feature_extraction import (
    extract_face_landmarks,
    extract_hog_features,
    extract_lbp_features,
    load_img,
    resize_img,
)


class _TimingLogger:
    """Helper to print timings when env flag allows."""

    def __init__(self) -> None:
        flag = os.environ.get("MODIFYING_TIMING", "1").strip().lower()
        self.enabled = flag not in {"0", "false", "off"}

    def log(self, label: str, elapsed: float) -> None:
        if self.enabled:
            print(f"[ModifyingTiming] {label}: {elapsed:.3f}s")

    def measure(self, label: str):
        start = time.perf_counter()

        class _Context:
            def __enter__(self_nonlocal):
                return None

            def __exit__(self_nonlocal, exc_type, exc, tb):
                self.log(label, time.perf_counter() - start)
                return False

        return _Context()


class ModDetector:
    """
    Facial Image Modification Detector.

    Params:
        clf_model: ModClassifier object, classifier.
        feature_type: String, type of feature extraction to use ('lbp', 'hog', 'facial_landmarks' or None).
        feature_params: Dict, parameters for feature extraction.
        pca_model: sklearn PCA, model for principal component analysis.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._timings = _TimingLogger()
        init_start = time.perf_counter()
        self.clf_model = self._get_model()
        self.feature_type, self.feature_params = self._get_feature_type()
        self.pca_model = self._get_pca_model()
        self._timings.log(f"init:{os.path.basename(model_path)}", time.perf_counter() - init_start)

    def _get_model(self):
        if os.path.exists(f'{os.path.dirname(self.model_path)}/class.json'):
            class_indices_path = f'{os.path.dirname(self.model_path)}/class.json'
        else:
            class_indices_path = None
        return ModClassifier(model_path=self.model_path, class_indices_path=class_indices_path)

    def _get_feature_type(self):
        if 'lbp_' in self.model_path:
            return self._load_lbp_features()
        elif 'hog_' in self.model_path:
            return self._load_hog_features()
        elif 'face_landmarks_' in self.model_path:
            return self._load_face_landmarks_features()
        else:
            return None, None

    def _get_pca_model(self):
        if 'pca' in self.model_path:
            return joblib.load(f'{os.path.dirname(self.model_path)}/pca.pkl')
        else:
            return None

    def _load_lbp_features(self):
        with open(f'{os.path.dirname(self.model_path)}/lbp.json', 'r') as f:
            params = json.load(f)
        params = {key: val for key, val in params.items() if key in extract_lbp_features.__code__.co_varnames}
        return 'lbp', params

    def _load_hog_features(self):
        with open(f'{os.path.dirname(self.model_path)}/hog.json', 'r') as f:
            params = json.load(f)
        params = {key: val for key, val in params.items()
                  if (key in extract_hog_features.__code__.co_varnames) or (key == 'input_size')}
        if 'pixels_per_cell' in params:
            params['pixels_per_cell'] = tuple(params['pixels_per_cell'])
        if 'cells_per_block' in params:
            params['cells_per_block'] = tuple(params['cells_per_block'])
        return 'hog', params

    def _load_face_landmarks_features(self):
        with open(f'{os.path.dirname(self.model_path)}/face_landmarks.json', 'r') as f:
            params = json.load(f)
        params = {key: val for key, val in params.items() if key in extract_face_landmarks.__code__.co_varnames}
        return 'face_landmarks', params

    def _get_input_shape(self):
        return self.clf_model._get_model_input_shape()

    def prepare_data(self, img_path: str) -> np.ndarray | None:
        total_start = time.perf_counter()
        input_shape = self._get_input_shape()

        if self.feature_type == 'face_landmarks':
            with self._timings.measure("prepare:face_landmarks"):
                data = load_img(img_path, as_mediapipe_image=True)
                data = extract_face_landmarks(data, **self.feature_params)
            if data is None:
                return None

        elif self.feature_type == 'lbp':
            with self._timings.measure("prepare:lbp"):
                data = load_img(img_path, as_gray=True)
                if (self.feature_params.get('histogram') in [None, False]) and (data.shape != input_shape):
                    data = resize_img(data, input_shape)
                data = extract_lbp_features(data, **self.feature_params)
            if self.pca_model is not None:
                with self._timings.measure("prepare:pca"):
                    data = self.pca_model.transform(data.reshape(1, -1))

        elif self.feature_type == 'hog':
            with self._timings.measure("prepare:hog"):
                data = load_img(img_path, as_gray=True)
                if 'input_size' in self.feature_params:
                    if data.shape != self.feature_params['input_size']:
                        data = resize_img(data, self.feature_params['input_size'])
                    feature_params = {k: v for k, v in self.feature_params.items() if k != 'input_size'}
                else:
                    feature_params = copy(self.feature_params)

                data = extract_hog_features(data, **feature_params)
            if self.pca_model is not None:
                with self._timings.measure("prepare:pca"):
                    data = self.pca_model.transform(data.reshape(1, -1))[0]

        else:
            with self._timings.measure("prepare:plain"):
                data = load_img(img_path, as_gray=False)
                if data.shape != input_shape:
                    data = resize_img(data, input_shape[:-1])

        data = np.expand_dims(data, axis=0)
        self._timings.log("prepare:total", time.perf_counter() - total_start)
        return data

    def run(self, img_path: str) -> dict:
        """
        Detecting modifications in facial images.
        Args:
            img_path: String, path to image to be detected.
        Return:
            Dict, modification labels and their probabilities corresponding to input images.
        """
        total_start = time.perf_counter()
        data = self.prepare_data(img_path)
        inference_start = time.perf_counter()
        output = self.clf_model.predict(data, [img_path])[0]
        self._timings.log("run:predict", time.perf_counter() - inference_start)
        self._timings.log("run:total", time.perf_counter() - total_start)
        return output

    def set_label_encoding(self, encoding: bool):
        """
        Set label encoding for image data or delete.
        If encoding is set to true, read labels from 'class.json' in model path.
        Params:
            encoding: Boolean, whether to set label encoding or not.
        """
        if encoding:
            if os.path.exists(f'{os.path.dirname(self.model_path)}/class.json'):
                class_indices_path = f'{os.path.dirname(self.model_path)}/class.json'
            else:
                class_indices_path = None
            self.clf_model.load_class_indices(class_indices_path)
        else:
            self.clf_model.class_indices = None


class ModVotingDetector:
    """
     Facial Image Modification Voting Detector.

    Params:
        ensemble_models: Dict, dictionary in format {model_path: ModClassifier}.
        class_indices: Dict, class indices to label encoding in format {class_id: class_label}.
    """

    def __init__(self, model_paths: list, class_indices: dict | None = None):
        self.ensemble_models = {model_path: ModDetector(model_path) for model_path in model_paths}
        output_count = len(
            set([detector.clf_model._get_model_output_shape() for detector in self.ensemble_models.values()]))
        assert output_count == 1, 'All models in an ensemble must have the same output size.'
        self.class_indices = class_indices

    def run(self, img_path: str, voting='hard') -> dict:
        """
        Detecting modifications in facial image.
        Args:
            img_path: String, path to image to be detected.
            voting: String, type of voting ('hard' or 'soft').
        Return:
            Dict, modification labels and their probabilities corresponding to input images.
        """
        assert voting == 'hard' or voting == 'soft', "Error in <voting> type. Must be 'hard' or 'soft'."
        predictions = {}
        for model_name, detector in self.ensemble_models.items():
            data = detector.prepare_data(img_path)
            predictions[model_name] = detector.clf_model._get_predictions(data)
        if voting == 'hard':
            labels, probabilities = self._hard_voting(predictions)
        else:
            labels, probabilities = self._soft_voting(predictions)
        return self._get_output(labels, probabilities, [img_path])[0]

    @staticmethod
    def _hard_voting(predictions: dict) -> tuple:
        all_labels = np.array([pred[0] for pred in predictions.values()])
        labels, _ = stats.mode(all_labels)
        all_proba = np.array(
            [[np.max(p) if labels[i] == np.argmax(p) else 1 - np.max(p) for i, p in enumerate(pred[1])]
             for pred in predictions.values()])
        probabilities = np.mean(all_proba, axis=0)
        return labels, probabilities

    @staticmethod
    def _soft_voting(predictions: dict) -> tuple:
        length = len(predictions[list(predictions.keys())[0]][0])
        labels = []
        probabilities = []
        for i in range(length):
            i_proba = np.column_stack([pred[1][i] for pred in predictions.values()])
            avg_proba = np.mean(i_proba, axis=1)
            labels.append(np.argmax(avg_proba))
            probabilities.append(np.max(avg_proba))
        return labels, probabilities

    def _get_output(self, labels: np.ndarray, probabilities: np.ndarray, index: list | None = None) -> list:
        if index is None:
            index = [f'image_{i}' for i in range(len(labels))]
        if self.class_indices is not None:
            labels = [self.class_indices[label] for label in labels]
        return [{'img_name': name,
                 'label': labels[i],
                 'probability': probabilities[i].round(4)}
                for i, name in enumerate(index)]

    def set_label_encoding(self, encoding: bool):
        """
        Set label encoding for image data or delete.
        If encoding is set to true, read labels from 'class.json' in models path.
        Params:
            encoding: Boolean, whether to set label encoding or not.
        """
        if encoding:
            for model_name in self.ensemble_models.keys():
                if os.path.exists(f'{os.path.dirname(model_name.model_path)}/class.json'):
                    class_indices_path = f'{os.path.dirname(model_name.model_path)}/class.json'
                else:
                    class_indices_path = None
                model_name.clf_model.load_class_indices(class_indices_path)
            self.class_indices = self.ensemble_models[list(self.ensemble_models.keys())[0]].clf_model.class_indices
        else:
            for model_name in self.ensemble_models.keys():
                self.ensemble_models[model_name].set_label_encoding(False)
            self.class_indices = None


def start_model(model_path: str) -> ModDetector:
    """
    Load trained model from disk.
    Args:
        model_path: String or List, path to trained model or several models (for ensemble).
    Return:
        ModDetector, object for modification detection.
    """
    if os.path.isdir(model_path):
        if os.path.isfile(f'{model_path}/{model_path.split("/")[-1]}.keras'):
            model_path = f'{model_path}/{model_path.split("/")[-1]}.keras'
        elif os.path.isfile(f'{model_path}/{model_path.split("/")[-1]}.pkl'):
            model_path = f'{model_path}/{model_path.split("/")[-1]}.pkl'
        else:
            raise FileNotFoundError(f"File '{model_path.split('/')[-1]}.keras' or '{model_path.split('/')[-1]}.pkl' "
                                    f"not found in the specified directory '{model_path}'.")
    else:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"File '{model_path}' not found.")

    model = ModDetector(model_path)
    print(model.model_path)
    if model.feature_type is not None:
        print(f'Feature Extraction: {dict([("type", model.feature_type)]) | model.feature_params}.')
    else:
        print(f'Feature Extraction: deep features.')
    if model.pca_model is not None:
        print(f'PCA model: {model.pca_model.get_params()}')
    print(f'Classifier Parameters: {model.clf_model.summary()}')
    return model


def start_ensemble(from_file: str) -> ModVotingDetector:
    """
    Load ensemble model from JSON-file configuration.
    Args:
        from_file: String, path to configuration JSON-file.
    Return:
        ModDetector, object for modification detection.
    """
    ensemble_info = json.load(open(from_file, 'r'))
    class_indices = {int(k): v for k, v in ensemble_info['class_indices'].items()}
    model = ModVotingDetector(ensemble_info['model_paths'], class_indices=class_indices)
    for path, detector in model.ensemble_models.items():
        print(f"Parameters of model '{path}'")
        if detector.feature_type is not None:
            print(f'    Feature Extraction: {dict([("type", detector.feature_type)]) | detector.feature_params}.')
        else:
            print(f'    Feature Extraction: deep features.')
        if detector.pca_model is not None:
            print(f'    PCA model: {detector.pca_model.get_params()}')
        print(f'    Classifier Parameters: {detector.clf_model.summary()}')
        print()
    return model

