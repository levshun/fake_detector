from __future__ import annotations
import numpy as np
import joblib
import json
from keras import models
import tensorflow as tf
tf.function(reduce_retracing=True)


class ModClassifier:
    """
    Facial Image Modification Classifier.

    Params:
        _use_keras: Boolean, whether to use deep learning.
        model: Keras or SkLearn model to classification.
        class_indices: Dict, class indices to label encoding in format {class_id: class_label}.
    """

    def __init__(self, model_path: str, class_indices_path: str | None = None):
        self._use_keras = False
        self.model = self.load_model(model_path)
        self.class_indices = self.load_class_indices(class_indices_path)

    def load_model(self, path):
        """
        Load model.
        Args:
            path: String, path to model file.
        Return:
            Keras or SkLearn model.
        """
        if '.keras' in path:
            self._use_keras = True
            return models.load_model(path)
        else:
            return joblib.load(path)

    def load_class_indices(self, class_indices_path: str | None) -> dict:
        """
        Load class indices' dictionary.
        Args:
            class_indices_path: String, path to class indices.
        Return:
            Dictionary of class indices.
        """
        if class_indices_path is None:
            if self._get_model_output_shape() <= 2:
                return {0: 'no modification', 1: 'modification'}
            else:
                return {i: f'class_{i}' for i in range(self._get_model_output_shape())}
        else:
            with open(class_indices_path, 'r') as f:
                ci = json.load(f)
            ci = {int(k): ci[k] for k in ci}
            return ci

    def predict(self, data: np.array, index: list | None = None) -> list:
        """
        Predict class labels and their probabilities for given data.
        Args:
            data: Array, image data.
            index: List, index or name of image.
        Return:
            List of predicted class labels and probabilities.
        """
        if data is None:
            return self._get_output([0], [[0]], index)
        labels, probabilities = self._get_predictions(data)
        return self._get_output(labels, probabilities, index)

    def _get_predictions(self, data: np.array) -> tuple:
        if self._use_keras:
            return self._get_deep_predictions(data)
        else:
            return self._get_shallow_predictions(data)

    def _get_shallow_predictions(self, data: np.array) -> tuple:
        labels = self.model.predict(data)
        probabilities = self.model.predict_proba(data)
        return labels, probabilities

    def _get_deep_predictions(self, data: np.array) -> tuple:
        data = tf.convert_to_tensor(data)
        probabilities = self.model.predict(data, verbose=0)
        if self.model.output_shape[1] <= 2:
            labels = np.array([0 if p < 0.5 else 1 for p in probabilities])
            probabilities = np.array([[p, 1 - p] if p < 0.5 else [1 - p, p] for p in probabilities])
        else:
            labels = np.argmax(probabilities, axis=1)
        return labels, probabilities

    def _get_output(self, labels: np.ndarray, probabilities: np.ndarray, index: list | None = None) -> list:
        if index is None:
            index = [f'image_{i}' for i in range(len(labels))]
        if self.class_indices is not None:
            labels = [self.class_indices[label] for label in labels]
        probabilities = np.max(probabilities, axis=1)
        return [{'img_name': name,
                 'label': labels[i],
                 'probability': round(float(probabilities[i]), 5)}
                for i, name in enumerate(index)]

    def _get_model_params(self) -> dict:
        if self._use_keras:
            return self._get_deep_model_params()
        else:
            return self._get_shallow_model_params()

    def _get_shallow_model_params(self) -> dict:
        return self.model.get_params()

    def _get_deep_model_params(self) -> dict:
        config = {'num_layers': len(self.model.layers)}
        for i, layer in enumerate(self.model.layers):
            layer_config = layer.get_config()
            config[f'layer_{i}'] = {}
            if 'name' in layer_config.keys():
                config[f'layer_{i}']['name'] = layer_config['name']
            if 'units' in layer_config.keys():
                config[f'layer_{i}']['units'] = layer_config['units']
            if 'filters' in layer_config.keys():
                config[f'layer_{i}']['filters'] = layer_config['filters']
            if 'kernel_size' in layer_config.keys():
                config[f'layer_{i}']['kernel_size'] = layer_config['kernel_size']
            if 'strides' in layer_config.keys():
                config[f'layer_{i}']['strides'] = layer_config['strides']
            if 'padding' in layer_config.keys():
                config[f'layer_{i}']['padding'] = layer_config['padding']
            if 'axis' in layer_config.keys():
                config[f'layer_{i}']['axis'] = layer_config['axis']
            if 'momentum' in layer_config.keys():
                config[f'layer_{i}']['momentum'] = layer_config['momentum']
            if 'epsilon' in layer_config.keys():
                config[f'layer_{i}']['epsilon'] = layer_config['epsilon']
            if 'rate' in layer_config.keys():
                config[f'layer_{i}']['rate'] = layer_config['rate']
            if 'activation' in layer_config.keys():
                config[f'layer_{i}']['activation'] = layer_config['activation']
            if 'kernel_regularizer' in layer_config.keys():
                config[f'layer_{i}']['kernel_regularizer'] = layer_config['kernel_regularizer']
            if 'layers' in layer_config.keys():
                config[f'layer_{i}']['num_sublayers'] = len(layer_config['layers'])
        return config

    def _get_model_input_shape(self) -> tuple | set | None:
        if self._use_keras:
            return self._get_deep_model_input_shape()
        else:
            return self._get_shallow_model_input_shape()

    def _get_shallow_model_input_shape(self) -> tuple | set | None:
        try:
            return self.model.n_features_in_
        except:
            return None

    def _get_deep_model_input_shape(self) -> tuple | set:
        return self.model.input_shape[1:]

    def _get_model_output_shape(self) -> int:
        if self._use_keras:
            return self._get_deep_model_output_shape()
        else:
            return self._get_shallow_model_output_shape()

    def _get_shallow_model_output_shape(self) -> int:
        return len(self.model.classes_)

    def _get_deep_model_output_shape(self) -> int:
        return self.model.output_shape[1]

    def summary(self) -> dict:
        """
        Returns a summary of the model.
        """
        return {
            'model type': type(self.model).__name__,
            'input shape': self._get_model_input_shape(),
            'output shape': self._get_model_output_shape(),
            'class indices': self.class_indices,
            'model parameters': self._get_model_params()
        }

