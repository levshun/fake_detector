import unittest
import os
import detect_ai as dai

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.models_directory = os.path.join('..', 'models', 'swapping')
        self.test_image_path = os.path.join('swapping_data', 'real.jpg')

    def test_full_pipeline_on_real_image(self):
        self.assertTrue(
            os.path.exists(self.models_directory),
            "Папка с моделями 'models' не найдена в корне проекта")
        self.assertTrue(
            os.path.exists(self.test_image_path),
            f"Тестовое изображение не найдено: {self.test_image_path}")

        try:
            predictor = dai.DeepfakePredictor(models_dir=self.models_directory)
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

        except dai.SwappingError as e:
            self.fail(f"Интеграционный тест провалился с ошибкой модуля: {e}")
        except Exception as e:
            self.fail(f"Интеграционный тест провалился с непредвиденной ошибкой: {e}")
