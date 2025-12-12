from detect_ai.modifying.detection import \
    start_model as mod_start_model, \
    start_ensemble as mod_start_ensemble
from detect_ai.modifying.ensemble import mod_ens_dict
from detect_ai.modifying.classifier import ModClassifier

from detect_ai.generating.GeneratedImageDetector import GeneratedImageDetector

from detect_ai.swapping.predictor import DeepfakePredictor
from detect_ai.swapping.exceptions import SwappingError
from detect_ai.swapping.exceptions import \
    ModelLoadingError, FeatureExtractionError, SwappingError, FeatureExtractionError
from detect_ai.swapping.logger_setup import setup_logger

from detect_ai.preprocessing.feature_extraction import \
    load_img, resize_img, \
    extract_lbp_features, extract_hog_features, \
    extract_face_landmarks
from detect_ai.preprocessing.swap_features import \
    calculate_features_one, flatten_landmarks
from detect_ai.preprocessing.face_feature_extractor_main import FaceFeatureExtractor
from detect_ai.preprocessing.geometry import extract_geometry_features
from detect_ai.preprocessing.color_texture import extract_color_texture_features
from detect_ai.preprocessing.frequency_artifacts import extract_frequency_artifacts_features
from detect_ai.preprocessing.perceptual_analysis import extract_perceptual_features
from detect_ai.preprocessing.eye_reflections import extract_eye_reflection_features
from detect_ai.preprocessing.hair_structure import extract_hair_structure_features
from detect_ai.preprocessing.perspective_depth import extract_perspective_depth_features
from detect_ai.preprocessing.generating_features import build_features_from_scores
from detect_ai.preprocessing.img_features import blurring, normalize_img

from detect_ai.reporting.report_generator import generate_pdf_report, generate_text_report