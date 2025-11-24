import unittest
import os
import time
import logging
from swapping.predictor import DeepfakePredictor
from swapping.exceptions import SwappingError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestPerformance(unittest.TestCase):

    def setUp(self):
        self.models_directory = 'swapping/models'
        self.test_data_directory = 'tests/dataset_test_fake_2'

    def test_batch_processing_performance(self):
        if not os.path.exists(self.test_data_directory):
            self.skipTest(f"Директория {self.test_data_directory} не найдена.")

        image_files = [
            os.path.join(self.test_data_directory, f)
            for f in os.listdir(self.test_data_directory)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]

        if not image_files:
            self.skipTest("Изображения для тестирования не найдены.")

        logger.info(f"Запуск теста производительности на {len(image_files)} изображениях.")

        start_load_time = time.perf_counter()
        try:
            predictor = DeepfakePredictor(models_dir=self.models_directory)
        except Exception as e:
            self.fail(f"Загрузка модели провалилась с непредвиденной ошибкой: {e}")
        end_load_time = time.perf_counter()

        load_duration = end_load_time - start_load_time

        prediction_times = []

        time_accum_std_features = 0.0
        time_accum_custom_lib = 0.0
        time_accum_models = 0.0
        valid_time_samples = 0

        results_stats = {
            "Real": 0,
            "Fake": 0,
            "Error": 0
        }

        sub_models_fake_probs = {}
        detailed_results = []

        start_total_execution = time.perf_counter()

        for image_path in image_files:
            start_img = time.perf_counter()
            file_name = os.path.basename(image_path)
            try:
                result = predictor.predict(image_path=image_path, debug=False)
                decision = result.get("final_decision", "Unknown")

                if decision == "Real":
                    results_stats["Real"] += 1
                elif decision == "Fake":
                    results_stats["Fake"] += 1
                else:
                    decision = "Unknown"

                if "performance_metrics" in result:
                    metrics = result["performance_metrics"]
                    time_accum_std_features += metrics.get("std_features_time", 0)
                    time_accum_custom_lib += metrics.get("custom_lib_time", 0)
                    time_accum_models += metrics.get("model_inference_time", 0)
                    valid_time_samples += 1

                if "base_models_prob" in result:
                    for m_name, probs in result["base_models_prob"].items():
                        if m_name not in sub_models_fake_probs:
                            sub_models_fake_probs[m_name] = []
                        sub_models_fake_probs[m_name].append(probs['fake_prob'])

                if "specialized_effnet_prob" in result:
                    for m_name, probs in result["specialized_effnet_prob"].items():
                        if m_name not in sub_models_fake_probs:
                            sub_models_fake_probs[m_name] = []
                        sub_models_fake_probs[m_name].append(probs['fake_prob'])

                if "final_probability" in result:
                    if "Meta_Model" not in sub_models_fake_probs:
                        sub_models_fake_probs["Meta_Model"] = []
                    sub_models_fake_probs["Meta_Model"].append(result["final_probability"]['fake_prob'])

                detailed_results.append((file_name, decision))
                logger.info(f"Обработано {file_name}: {decision}")

            except SwappingError as e:
                logger.warning(f"Пропуск {file_name} (SwappingError): {e}")
                results_stats["Error"] += 1
                detailed_results.append((file_name, "Error"))
            except Exception as e:
                self.fail(f"Непредвиденная ошибка при обработке {image_path}: {e}")
            finally:
                end_img = time.perf_counter()
                prediction_times.append(end_img - start_img)

        end_total_execution = time.perf_counter()
        total_duration = end_total_execution - start_total_execution

        avg_prediction_time = sum(prediction_times) / len(prediction_times) if prediction_times else 0.0

        avg_std_time = time_accum_std_features / valid_time_samples if valid_time_samples > 0 else 0.0
        avg_custom_time = time_accum_custom_lib / valid_time_samples if valid_time_samples > 0 else 0.0
        avg_model_time = time_accum_models / valid_time_samples if valid_time_samples > 0 else 0.0

        total_processed = len(image_files)
        real_percent = (results_stats["Real"] / total_processed) * 100 if total_processed > 0 else 0
        fake_percent = (results_stats["Fake"] / total_processed) * 100 if total_processed > 0 else 0
        error_percent = (results_stats["Error"] / total_processed) * 100 if total_processed > 0 else 0

        print("\n" + "=" * 60)
        print("ИТОГОВЫЙ ОТЧЕТ ПО ТЕСТИРОВАНИЮ")
        print("=" * 60)

        print(f"{'Имя файла':<40} | {'Результат':<10}")
        print("-" * 60)
        for name, res in detailed_results:
            print(f"{name:<40} | {res:<10}")

        print("-" * 60)
        print("СТАТИСТИКА ПО МОДЕЛЯМ (Средняя вероятность Fake)")
        print(f"{'Модель':<30} | {'Avg Fake Prob':<15}")
        print("-" * 60)

        sorted_models = sorted(sub_models_fake_probs.keys())
        for m_name in sorted_models:
            probs = sub_models_fake_probs[m_name]
            avg_prob = sum(probs) / len(probs) if probs else 0.0
            print(f"{m_name:<30} | {avg_prob:.4f}")

        print("-" * 60)
        print("СТАТИСТИКА ПРОИЗВОДИТЕЛЬНОСТИ (Средние значения)")
        print(f"1. Расчет стандартных признаков:    {avg_std_time:.4f} сек")
        print(f"2. Расчет preprocessing lib:        {avg_custom_time:.4f} сек")
        print(f"3. Работа ML моделей (инференс):    {avg_model_time:.4f} сек")
        print("-" * 60)
        print(f"Общее среднее время на фото:        {avg_prediction_time:.4f} сек")
        print(f"Время загрузки моделей:             {load_duration:.4f} сек")
        print(f"Общее время выполнения теста:       {total_duration:.4f} сек")
        print("-" * 60)
        print("СТАТИСТИКА РАСПРЕДЕЛЕНИЯ")
        print(f"Всего изображений: {total_processed}")
        print(f"Real:              {results_stats['Real']} ({real_percent:.2f}%)")
        print(f"Fake:              {results_stats['Fake']} ({fake_percent:.2f}%)")
        print(f"Необработанные:    {results_stats['Error']} ({error_percent:.2f}%)")
        print("=" * 60 + "\n")