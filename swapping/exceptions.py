class SwappingError(Exception):
    """Базовый класс для исключений в этом модуле."""
    pass

class ModelLoadingError(SwappingError):
    """Возникает при ошибке загрузки файла модели."""
    pass

class FeatureExtractionError(SwappingError):
    """Возникает, когда не удается извлечь признаки из изображения."""
    pass