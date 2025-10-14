import logging
import sys

def setup_logger():
    """Настраивает логирование в файл и в консоль."""
    log_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-5.5s] [%(module)-15.s] %(message)s'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("debug.log", mode='w', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logging.info("Логгер успешно настроен.")