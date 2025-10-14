import os
import csv
import time
from face_feature_extractor_main import FaceFeatureExtractor

def format_time(seconds: float) -> str:
    """
    Преобразует время в секундах в строку формата ЧЧ:ММ:СС.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# Путь к файлу предсказателя и папке с изображениями
predictor_path = "swapping/shape_predictor_68_face_landmarks.dat"
images_folder = "images"  # Корневая папка с изображениями (включая вложенные папки)
output_csv = "features_output.csv"  # Имя выходного CSV-файла

# Создаем экземпляр экстрактора признаков
extractor = FaceFeatureExtractor(predictor_path)

# Рекурсивно собираем список всех изображений в папке (включая вложенные папки)
image_files = []
for root, dirs, files in os.walk(images_folder):
    for file in files:
        if file.lower().endswith(('png', 'jpg', 'jpeg')):
            # Относительный путь к файлу относительно images_folder
            rel_dir = os.path.relpath(root, images_folder)
            rel_path = file if rel_dir == "." else os.path.join(rel_dir, file)
            image_files.append(rel_path)

total_files = len(image_files)
if total_files == 0:
    print("Изображения не найдены.")
    exit()

# Открываем CSV-файл для записи
with open(output_csv, mode='w', newline='') as csvfile:
    csv_writer = None  # Будет инициализирован при первой записи
    start_time = time.time()
    processed_files = 0
    last_printed_percent = -1  # для отслеживания изменения процента

    # Начальный вывод прогресса (0%)
    print(f"Progress: 0.00% (0/{total_files}) | Elapsed: 00:00:00 | Remaining: 00:00:00")

    # Обработка каждого изображения
    for rel_path in image_files:
        image_path = os.path.join(images_folder, rel_path)
        features = extractor.extract_features(image_path)

        if features:
            # Создаем заголовки CSV при первой записи
            for i, feature_set in enumerate(features):
                if csv_writer is None:
                    fieldnames = ["image_name", "relative_path", "face_index"] + list(feature_set.keys())
                    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    csv_writer.writeheader()
                # Формируем строку с метаданными и признаками
                row = {
                    "image_name": os.path.basename(rel_path),
                    "relative_path": rel_path,
                    "face_index": i + 1
                }
                row.update(feature_set)
                csv_writer.writerow(row)
        # Если признаки не извлечены, изображение пропускается

        processed_files += 1
        elapsed = time.time() - start_time
        percent = (processed_files / total_files) * 100
        int_percent = int(percent)
        remaining = (elapsed / processed_files) * (total_files - processed_files) if processed_files else 0

        # Формируем строки времени
        elapsed_str = format_time(elapsed)
        remaining_str = format_time(remaining)

        # Пока не достигли 1% — выводим прогресс для каждого файла,
        # далее — только при изменении целочисленного процента
        if percent < 1 or int_percent > last_printed_percent:
            print(
                f"Progress: {percent:.2f}% ({processed_files}/{total_files}) | "
                f"Elapsed: {elapsed_str} | Remaining: {remaining_str}"
            )
            last_printed_percent = int_percent

print(f"\nCSV файл сохранён: {output_csv}")
