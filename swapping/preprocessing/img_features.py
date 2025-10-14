from __future__ import annotations
import os
from matplotlib import pyplot as plt
from skimage import filters
from skimage.transform import resize
from skimage.io import imread
from skimage.feature import local_binary_pattern, hog
import numpy as np
from sklearn.model_selection import train_test_split
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import pandas as pd


class Images:
    """
    Class to handle images read from local disk.
    Attributes:
        X: array of images.
        y: array of image labels.
    """
    def __init__(self,
                 paths: str | list,
                 labels: int | list = 0,
                 as_gray: bool = True,
                 blurred_sigma: int | None = None,
                 resize_shape: tuple | None = None,
                 normalize: bool = True):
        """
        Initialize the Images class.
        :param paths: path(s) to images.
        :param labels: label(s) of images.
        :param as_gray: if true, convert images to grayscale.
        :param blurred_sigma: if given, apply blurred filter.
        :param resize_shape: if given, resize image.
        :param normalize: if true, normalize images.
        """
        self.X = None
        self.y = None
        self.load(paths, labels, as_gray)
        if blurred_sigma is not None:
            self.blurred(blurred_sigma)
        if resize_shape is not None:
            self.resize(resize_shape)
        if normalize:
            self.normalize()

    def load(self, paths, labels, as_gray):
        """
        Load images from local disk.
        :param paths: path(s) to images.
        :param labels: label(s) of images.
        :param as_gray: if true, convert images to grayscale.
        """
        self.X = []
        self.y = []
        if isinstance(paths, str):
            paths = [paths]
        if isinstance(labels, int):
            labels = [labels for _ in range(len(paths))]
        for i, path in enumerate(paths):
            for file in os.listdir(path):
                self.X.append(load_img(filename=f'{path}/{file}', as_gray=as_gray))
                self.y.append(labels[i])
        self.X = np.array(self.X)
        self.y = np.array(self.y)


    def blurred(self, sigma=1):
        """
        Blurred the images.
        :param sigma: standard deviation for Gaussian kernel.
        """
        for i in range(len(self.X)):
            self.X[i] = blurred_img(self.X[i], sigma)

    def resize(self, output_shape=(150, 150)):
        """
        Resize the images.
        :param output_shape: resized image shape
        """
        for i in range(len(self.X)):
            self.X[i] = resize_img(self.X[i], output_shape)

    def normalize(self):
        """
        Normalize the images.
        """
        for i in range(len(self.X)):
            self.X[i] = normalize_img(self.X[i])


class MPImages:
    """
    Class for Mediapipe image.
    Attributes:
        X: array of Mediapipe images.
        y: array of image labels.
        landmarks: face landmarks for images.
    """
    def __init__(self,
                 paths: str | list,
                 labels: int | list = 0):
        """
        Initialize the Images class.
        :param paths: path(s) to images.
        :param labels: label(s) of images.
        """
        self.X = None
        self.y = None
        self.landmarks = None
        self.load(paths, labels)

    def load(self, paths, labels):
        """
        Load images from local disk.
        :param paths: path(s) to images.
        :param labels: label(s) of images.
        """
        self.X = []
        self.y = []
        if isinstance(paths, str):
            paths = [paths]
        if isinstance(labels, int):
            labels = [labels for _ in range(len(paths))]
        for i, path in enumerate(paths):
            for file in os.listdir(path):
                self.X.append(load_img(filename=f'{path}/{file}', as_mediapipe_image=True))
                self.y.append(labels[i])
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def extract_face_landmarks(self):
        """
        Extract face landmarks from images.
        """
        self.landmarks = []
        img_to_delete = []
        for i, img in enumerate(self.X):
            img_landmarks = extract_face_landmarks(img)
            if not img_landmarks.face_landmarks:
                print(f'No face landmarks for image {i}')
                img_to_delete.append(i)
            else:
                self.landmarks.append(img_landmarks)
        self.X = [x for i, x in enumerate(self.X) if i not in img_to_delete]
        self.y = [y for i, y in enumerate(self.y) if i not in img_to_delete]

    def get_landmark_frames(self) -> np.ndarray | None:
        """
        Convert all face landmarks into numpy arrays.
        :return: array of landmarks as frames.
        """
        result = []
        if self.landmarks is None:
            return result
        for img_landmark in self.landmarks:
            values = np.array([[coords.x, coords.y, coords.z] for coords in img_landmark.face_landmarks[0]])
            result.append(values)
        result = np.array(result)
        return result

    def draw_landmarks(self, index=0):
        """
        Draw face landmarks.
        :param index: number of images to draw.
        """
        landmarks = self.landmarks[index]
        image_to_draw = self.X[index].numpy_view()
        face_landmarks_list = landmarks.face_landmarks
        annotated_image = np.copy(image_to_draw)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())
        plt.imshow(annotated_image)


class FeatureSamples:
    """
    Class to generate feature samples.
    Attributes:
        X_train: train image sample.
        X_test: test image sample.
        y_train: train label sample.
        y_test: test label sample.
    """
    def __init__(self, X, y, train_size=0.8):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=train_size)
        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)


class LBPFeatures:
    """
    Class to generate local binary pattern (LBP) features.
    Attributes:
        values: Local Binary Pattern
        histograms: Local Binary Pattern h
    """
    def __init__(self, data: np.ndarray,
                 n_points: int = 15,
                 radius: int = 3,
                 method='uniform'):
        """
        Initializes LBP features object.
        :param data: images.
        :param n_points: number of circularly symmetric neighbor set points.
        :param radius: radius of circle.
        :param method: method to determine the pattern.
        """
        self.values = []
        self.histograms = []
        for x in data:
            lbp, lbp_hist = extract_lbp_features(x, n_points, radius, method, histogram=True)
            self.values.append(lbp)
            self.histograms.append(lbp_hist)
        self.values = np.array(self.values)
        self.histograms = np.array(self.histograms)

    def show(self,
             index=0,
             type='image'):
        """
        Shows LBP features.
        :param index: number of image to show.
        :param type: show 'image' or 'histogram'.
        """
        if type == 'image':
            plt.imshow(self.values[index])
        if type == 'histogram':
            plt.grid()
            plt.bar(x=range(len(self.histograms[index])), height=self.histograms[index])


class HOGFeatures:
    """
    Class to handle HOG features extracted from images.
    Attributes:
        values: list of HOG features.
        images: list of images of the HOG.
    """
    def __init__(self,
                 data: np.ndarray,
                 orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3),
                 block_norm='L2-Hys',
                 feature_vector=True):
        """
        Initializes HOG features.
        :param data:
        :param orientations: number of orientation bins.
        :param pixels_per_cell: size (in pixels) of a cell.
        :param cells_per_block:  number of cells in each block.
        :param block_norm: block normalization method.
        :param feature_vector: if True, return the data as a flat feature vector.
        """
        self.values = []
        self.images = []
        for x in data:
            fd, hog_image = extract_hog_features(x, orientations, pixels_per_cell, cells_per_block,
                                                 block_norm, visualize=True, feature_vector=feature_vector)
            self.values.append(fd)
            self.images.append(hog_image)
        self.values = np.array(self.values)
        self.images = np.array(self.images)

    def show(self, index=0):
        """
        Show HOG image.
        :param index: number of images to show.
        """
        plt.imshow(self.images[index])


def load_img(filename, as_gray=True, as_mediapipe_image=False):
    """
    Load an image from disk.
    :param filename: name of the image.
    :param as_gray: if True, the image is converted to grayscale (not available for Mediapipe image).
    :param as_mediapipe_image: if True, the image load as Mediapipe image.
    :return: the loaded image.
    """
    if as_mediapipe_image:
        return mp.Image.create_from_file(filename)
    return imread(filename, as_gray=as_gray)


def blurred_img(img, sigma=1):
    """
    Blur the image.
    :param img: image.
    :param sigma: standard deviation for Gaussian kernel.
    :return: blurred image.
    """
    return filters.gaussian(img, sigma=sigma)


def resize_img(img, output_shape=(150, 150)):
    """
    Resize the image.
    :param img: image.
    :param output_shape: resized image shape.
    :return: resized image.
    """
    return resize(img, output_shape)


def normalize_img(img):
    """
    Normalize the image.
    :param img: image.
    :return: normalized image.
    """
    return img / 255.0


def extract_lbp_features(img: np.ndarray,
                         n_points: int = 15,
                         radius: int = 3,
                         method='uniform',
                         histogram: bool = False) -> np.ndarray | tuple:
    """
    Extract the local binary patterns (LBP) from an image.
    :param img: image.
    :param n_points: number of circularly symmetric neighbor set points.
    :param radius: radius of circle.
    :param method: method to determine the pattern.
    :param histogram: if true, LBP histogram returned.
    :return: LBP features and their corresponding histograms (if histogram).
    """
    lbp_img = local_binary_pattern(img, n_points, radius, method)
    if histogram:
        lbp_hist = np.histogram(lbp_img.ravel(), bins=n_points + 2, range=(0, n_points + 2))[0]
        return lbp_img, lbp_hist
    else:
        return lbp_img


def extract_hog_features(img,
                         orientations=9,
                         pixels_per_cell=(8, 8),
                         cells_per_block=(3, 3),
                         block_norm='L2-Hys',
                         visualize=False,
                         feature_vector=True):
    """
    Extract Histogram of Oriented Gradients (HOG) for a given image.
    :param img: image.
    :param orientations: number of orientation bins.
    :param pixels_per_cell: size (in pixels) of a cell.
    :param cells_per_block:  number of cells in each block.
    :param block_norm: block normalization method.
    :param visualize: if True, return an image of the HOG.
    :param feature_vector: if True, return the data as a flat feature vector.
    :return: HOG features.
    """
    return hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
               block_norm=block_norm, visualize=visualize, feature_vector=feature_vector)


def extract_face_landmarks(img: mp.Image,
                           model_asset_path='face_landmarker_v2_with_blendshapes.task'):
    """
    Extract face landmarks from an image.
    :param img: Mediapipe image.
    :param model_asset_path: path to model assets.
    :return: FaceLandmarker object
    """
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    detection_result = detector.detect(img)
    return detection_result



