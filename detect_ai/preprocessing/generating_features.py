import numpy as np


def _stats(values) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return np.array([arr.min(), arr.max(), arr.mean(), np.median(arr), arr.var()], dtype=float)


def build_features_from_scores(scores_model_vit, scores_model_conv) -> np.ndarray:
    """Агрегирует выходы моделей в единый вектор признаков."""
    vit = np.asarray(list(scores_model_vit), dtype=float)
    conv = np.asarray(list(scores_model_conv), dtype=float)
    return np.concatenate((_stats(vit), _stats(conv), _stats(vit + conv)))


__all__ = ["build_features_from_scores"]
