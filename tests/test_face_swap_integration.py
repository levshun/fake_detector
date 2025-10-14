import unittest
import os
from swapping.predictor import DeepfakePredictor
from swapping.exceptions import SwappingError

class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.models_directory = 'swapping/models'
        self.test_image_path = 'tests/face_swap_test_data/real.jpg'

    def test_full_pipeline_on_real_image(self):
        self.assertTrue(os.path.exists(self.models_directory), "Папка с моделями 'models' не найдена в корне проекта")
        self.assertTrue(os.path.exists(self.test_image_path), f"Тестовое изображение не найдено: {self.test_image_path}")

        try:
            predictor = DeepfakePredictor(models_dir=self.models_directory)
            result = predictor.predict(image_path=self.test_image_path)

            self.assertIsInstance(result, dict)
            self.assertIn("final_decision", result)
            self.assertIn("final_probability", result)
            self.assertIn("real_prob", result["final_probability"])
            self.assertIn("fake_prob", result["final_probability"])

            self.assertAlmostEqual(
                result["final_probability"]["real_prob"] + result["final_probability"]["fake_prob"],
                1.0,
                places=5
            )

        except SwappingError as e:
            self.fail(f"Интеграционный тест провалился с ошибкой модуля: {e}")
        except Exception as e:
            self.fail(f"Интеграционный тест провалился с непредвиденной ошибкой: {e}")