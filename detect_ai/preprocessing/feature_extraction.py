from skimage import filters
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage.feature import local_binary_pattern, hog
import skimage.util as util
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import numpy as np


def load_img(filename, as_gray=True, as_mediapipe_image=False) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        filename: String, name of the image.

        as_gray: Boolean, if True, the image is converted to grayscale (not available for Mediapipe image).

        as_mediapipe_image: Boolean, if True, the image load as Mediapipe image.

    Return:
        Array, the loaded image.
    """
    if as_mediapipe_image:
        return mp.Image.create_from_file(filename)
    return imread(filename, as_gray=as_gray)


def resize_img(img: np.ndarray, output_shape: tuple = (150, 150)) -> np.ndarray:
    """
    Resize the image.

    Args:
        img: Array, input image.

        output_shape: Tuple, resized image shape.

    Return:
        Array, resized image.
    """
    if not isinstance(output_shape, tuple):
        raise TypeError(f"Expected output_shape to be a <tuple>, but got <{type(output_shape).__name__}>.")
    return util.img_as_ubyte(resize(img, output_shape))


def extract_lbp_features(img: np.ndarray,
                         n_points: int = 15,
                         radius: int = 3,
                         method='uniform',
                         histogram: bool = False) -> np.ndarray:
    """
    Extract the local binary patterns (LBP) from an image.

    Args:
        img: Array, input image.

        n_points: Integer, number of circularly symmetric neighbor set points.

        radius: Integer, radius of circle.

        method: String, method to determine the pattern.

        histogram: Boolean, if true, LBP histogram returned.

    Return:
        Array, LBP features or their corresponding histograms (if histogram).
    """
    img = util.img_as_ubyte(img)
    lbp_img = local_binary_pattern(img, n_points, radius, method)
    if histogram:
        lbp_hist = np.histogram(lbp_img.ravel(), bins=n_points + 2, range=(0, n_points + 2))[0]
        return lbp_hist
    else:
        return lbp_img


def extract_hog_features(img,
                         orientations: int = 9,
                         pixels_per_cell: tuple = (8, 8),
                         cells_per_block: tuple = (3, 3),
                         block_norm: str = 'L2-Hys',
                         visualize: bool = False,
                         feature_vector: bool = True) -> np.ndarray:
    """
    Extract Histogram of Oriented Gradients (HOG) for a given image.

    Args:
        img: Array, input image.

        orientations: Integer, number of orientation bins.

        pixels_per_cell: Tuple, size (in pixels) of a cell.

        cells_per_block:  Tuple, number of cells in each block.

        block_norm: String, block normalization method.

        visualize: Boolean, if True, return an image of the HOG.

        feature_vector: Boolean, if True, return the data as a flat feature vector.

    Return:
        Array, HOG features.
    """
    return hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
               block_norm=block_norm, visualize=visualize, feature_vector=feature_vector)


def extract_face_landmarks(img: mp.Image,
                           model_asset_path='models/face_landmarker_v2_with_blendshapes.task'):
    """
    Extract face landmarks from an image.

    Args:
        img: Mediapipe image.

        model_asset_path: String, path to model assets.

    Return:
        Array, FaceLandmarker object in form of coordinates array.
    """
    if not isinstance(img, mp.Image):
        raise TypeError(f"Expected input image to be a <mediapipe.Image>, but got <{type(img).__name__}>.")
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    img_landmark = detector.detect(img)
    if not img_landmark.face_landmarks:
        return None
    output = np.array([[coords.x, coords.y, coords.z] for coords in img_landmark.face_landmarks[0]])
    return output
