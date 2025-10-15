import os
import unittest
import inspect
from preprocessing.feature_extraction import *
from modifying.detection import *


class TestModifying(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestModifying, self).__init__(*args, **kwargs)
        TestModifying.n = 1
        self.test_image = 'datasets_test/beauty_gan/modification/face_0000000005_XMY-074.png'

    def setUp(self):
        print('Test {} BEGIN'.format(TestModifying.n))

    def tearDown(self):
        print('Test {} END'.format(TestModifying.n))
        TestModifying.n += 1

    def test_load_img(self):
        print(inspect.stack()[0][3])
        non_existent_filename = "this_image_does_not_exist.jpg"
        self.assertRaises(FileNotFoundError, load_img, non_existent_filename)

    def test_resize_img(self):
        print(inspect.stack()[0][3])
        non_existent_filename = "this_image_does_not_exist.jpg"
        img = load_img(self.test_image)
        self.assertRaises(ValueError, resize_img, img, (-1, -1))
        self.assertRaises(TypeError, resize_img, img, 25)

    def test_extract_lbp_features(self):
        print(inspect.stack()[0][3])
        img = load_img(self.test_image)
        self.assertRaises(ValueError, extract_lbp_features, -1)
        self.assertRaises(TypeError, extract_lbp_features, img, 'n_points')

    def test_extract_hog_features(self):
        print(inspect.stack()[0][3])
        img = load_img(self.test_image)
        self.assertRaises(ValueError, extract_hog_features, img, -1)
        self.assertRaises(TypeError, extract_hog_features, img, 'n_points')

    def test_extract_face_landmarks(self):
        print(inspect.stack()[0][3])
        img = load_img(self.test_image, as_mediapipe_image=True)
        self.assertRaises(RuntimeError, extract_face_landmarks, img, 'unknown.task')
        img = load_img(self.test_image)
        self.assertRaises(TypeError, extract_face_landmarks, img, 'models/face_landmarker_v2_with_blendshapes.task')

    def test_start_model(self):
        print(inspect.stack()[0][3])
        self.assertRaises(FileNotFoundError, start_model, 'unknown_model.keras')
        self.assertRaises(FileNotFoundError, start_model, 'unknown_model')

    def test_start_ensemble(self):
        print(inspect.stack()[0][3])
        self.assertRaises(FileNotFoundError, start_ensemble, 'unknown_model.json')

    def test_run_ensemble(self):
        print(inspect.stack()[0][3])
        model = start_ensemble('models/ensembles/beautification_detection.json')
        self.assertRaises(AssertionError, model.run, self.test_image, 'unknown')





