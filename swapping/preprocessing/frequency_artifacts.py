import numpy as np
import cv2
from scipy.fftpack import dct
from scipy.stats import skew, kurtosis

def extract_frequency_artifacts_features(image, gray, landmarks):
    eps = 1e-8

    # 1. Частотный анализ с использованием DCT
    # Применяем 2D DCT с нормировкой для стабильных результатов
    dct_transformed = dct(dct(gray, axis=0, norm='ortho'), axis=1, norm='ortho')
    abs_dct = np.abs(dct_transformed)

    # Базовые метрики DCT:
    dct_std = np.std(dct_transformed)
    dct_energy = np.sum(abs_dct)
    prob_dct = abs_dct / (np.sum(abs_dct) + eps)
    dct_entropy = -np.sum(prob_dct * np.log2(prob_dct + eps))

    # Дополнительные DCT-метрики:
    dct_max = np.max(abs_dct)
    nonzero_dct = abs_dct[abs_dct > eps]
    dct_min = np.min(nonzero_dct) if nonzero_dct.size > 0 else 0
    dct_mean = np.mean(abs_dct)
    dct_median = np.median(abs_dct)
    dct_skewness = skew(abs_dct.ravel())
    dct_kurtosis = kurtosis(abs_dct.ravel())

    # 2. Разница яркости между левой и правой половинами изображения
    half_width = image.shape[1] // 2
    brightness_left = np.mean(image[:, :half_width])
    brightness_right = np.mean(image[:, half_width:])
    brightness_difference = np.abs(brightness_left - brightness_right)

    # 3. Несоответствие теней между определенными областями лица
    # Используем области вокруг кончика носа (landmarks.part(30)) и подбородка (landmarks.part(8))
    def get_region_mean(point, window=5):
        y, x = point.y, point.x
        y1 = max(y - window, 0)
        y2 = y + window
        x1 = max(x - window, 0)
        x2 = x + window
        region = image[y1:y2, x1:x2]
        return np.mean(region) if region.size > 0 else 0

    nose_mean = get_region_mean(landmarks.part(30))
    chin_mean = get_region_mean(landmarks.part(8))
    shadow_mismatch = np.abs(nose_mean - chin_mean)

    # 4. Дополнительный анализ с использованием FFT
    fft_transformed = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft_transformed)
    fft_magnitude = np.abs(fft_shifted)
    fft_total_energy = np.sum(fft_magnitude)

    # Определяем центральную область как низкочастотную:
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    # Центральный прямоугольник размером в 1/2 от высоты и ширины спектра
    r_size = rows // 4
    c_size = cols // 4
    low_freq_region = fft_magnitude[crow - r_size:crow + r_size, ccol - c_size:ccol + c_size]
    fft_low_freq_energy = np.sum(low_freq_region)
    fft_high_freq_energy = fft_total_energy - fft_low_freq_energy
    fft_energy_ratio = fft_low_freq_energy / (fft_total_energy + eps)

    # Энтропия FFT
    prob_fft = fft_magnitude / (fft_total_energy + eps)
    fft_entropy = -np.sum(prob_fft * np.log2(prob_fft + eps))

    return {
        # Исходные метрики
        "dct_std": dct_std,
        "dct_energy": dct_energy,
        "dct_entropy": dct_entropy,
        "brightness_difference": brightness_difference,
        "shadow_mismatch": shadow_mismatch,
        # Дополнительные DCT-метрики
        "dct_max": dct_max,
        "dct_min": dct_min,
        "dct_mean": dct_mean,
        "dct_median": dct_median,
        "dct_skewness": dct_skewness,
        "dct_kurtosis": dct_kurtosis,
        # Дополнительные FFT-метрики
        "fft_energy_ratio": fft_energy_ratio,
        "fft_entropy": fft_entropy,
        "fft_low_freq_energy": fft_low_freq_energy,
        "fft_high_freq_energy": fft_high_freq_energy
    }
