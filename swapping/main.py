import json
import sys
import logging
from swapping.logger_setup import setup_logger
from swapping.predictor import DeepfakePredictor
from swapping.exceptions import SwappingError

if __name__ == '__main__':
    setup_logger()

    MODELS_DIRECTORY = 'swapping/models'

    if len(sys.argv) < 2:
        logging.error("Не указан путь к изображению. Пример: python -m swapping.main <path_to_image>")
        sys.exit(1)

    TEST_IMAGE_PATH = sys.argv[1]

    try:
        predictor = DeepfakePredictor(models_dir=MODELS_DIRECTORY)
        result = predictor.predict(image_path=TEST_IMAGE_PATH)

        logging.info("\n--- Результат проверки ---")
        print(json.dumps(result, indent=2))

    except SwappingError as e:
        logging.error(f"Произошла ошибка в модуле swapping: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Произошла непредвиденная критическая ошибка: {e}", exc_info=True)
        sys.exit(1)