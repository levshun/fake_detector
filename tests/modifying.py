import os
import unittest
import inspect
import detect_ai as dai


class TestModifying(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestModifying, self).__init__(*args, **kwargs)
        TestModifying.n = 1
        self.test_image = os.path.join(
            '..', 'datasets', 'modifying', 'beauty_gan', 'modification', 'face_0000000005_XMY-074.png'
        )

    def setUp(self):
        print('Test {} BEGIN'.format(TestModifying.n))

    def tearDown(self):
        print('Test {} END'.format(TestModifying.n))
        TestModifying.n += 1

    def test_load_img(self):
        print(inspect.stack()[0][3])
        non_existent_filename = "this_image_does_not_exist.jpg"
        self.assertRaises(FileNotFoundError, dai.load_img, non_existent_filename)

    def test_resize_img(self):
        print(inspect.stack()[0][3])
        non_existent_filename = "this_image_does_not_exist.jpg"
        img = dai.load_img(self.test_image)
        self.assertRaises(ValueError, dai.resize_img, img, (-1, -1))
        self.assertRaises(TypeError, dai.resize_img, img, 25)

    def test_extract_lbp_features(self):
        print(inspect.stack()[0][3])
        img = dai.load_img(self.test_image)
        self.assertRaises(ValueError, dai.extract_lbp_features, -1)
        self.assertRaises(TypeError, dai.extract_lbp_features, img, 'n_points')

    def test_extract_hog_features(self):
        print(inspect.stack()[0][3])
        img = dai.load_img(self.test_image)
        self.assertRaises(ValueError, dai.extract_hog_features, img, -1)
        self.assertRaises(TypeError, dai.extract_hog_features, img, 'n_points')

    def test_extract_face_landmarks(self):
        print(inspect.stack()[0][3])
        img = dai.load_img(self.test_image, as_mediapipe_image=True)
        self.assertRaises(RuntimeError, dai.extract_face_landmarks, img, 'unknown.task')
        img = dai.load_img(self.test_image)
        self.assertRaises(
            TypeError, dai.extract_face_landmarks, img,
            os.path.join('..', 'models', 'modifying', 'face_landmarker_v2_with_blendshapes.task')
        )

    def test_start_model(self):
        print(inspect.stack()[0][3])
        self.assertRaises(FileNotFoundError, dai.mod_start_model, 'unknown_model.keras')
        self.assertRaises(FileNotFoundError, dai.mod_start_model, 'unknown_model')

    def test_run_ensemble(self):
        print(inspect.stack()[0][3])
        model = dai.mod_start_ensemble(dai.mod_ens_dict)
        self.assertRaises(AssertionError, model.run, self.test_image, 'unknown')





