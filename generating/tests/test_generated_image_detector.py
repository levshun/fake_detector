from __future__ import annotations

import random
import time
import sys
from pathlib import Path

import numpy as np
import torch
import pytest

ROOT = Path(__file__).resolve().parent.parent  # one folder up from tests/
sys.path.insert(0, str(ROOT))  # make parent dir importable

from GeneratedImageDetector import GeneratedImageDetector


MODELS_DIR = ROOT / "models"


@pytest.fixture(scope="module")
def detector() -> GeneratedImageDetector:
    return GeneratedImageDetector(
        vit_model_path=MODELS_DIR / "eva_model.pth",
        convnext_model_path=MODELS_DIR / "convnext_model.pth",
        dt_model=MODELS_DIR / "final_decisiontree.pkl",
        device="cuda",
    )


def _run_prediction(detector: GeneratedImageDetector, image_path: Path) -> dict[str, float | int]:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    return detector.get_final_score(image_path)


@pytest.mark.parametrize(
    ("image_name", "expected_label", "expected_prob"),
    (
        ("good.png", 0, 0.0010679196924391287),
        ("bad.jpg", 1, 0.9993678887484198),
    ),
)
def test_generated_image_scores(
    detector: GeneratedImageDetector,
    image_name: str,
    expected_label: int,
    expected_prob: float,
) -> None:
    image_path = Path(__file__).resolve().parent / "test_images" / image_name
    result = _run_prediction(detector, image_path)

    assert result["is_fake"] == expected_label
    assert np.isclose(result["prob_of_fake"], expected_prob, rtol=1e-3, atol=1e-6)


PERF_THRESHOLD_SECONDS = 5.0
PERF_NUM_RUNS = 10


def test_generated_image_performance(detector: GeneratedImageDetector) -> None:
    """Simple performance test: mean runtime over 10 runs must be < 5 seconds."""
    image_path = Path(__file__).resolve().parent / "test_images" / "good.png"

    start = time.perf_counter()
    for _ in range(PERF_NUM_RUNS):
        _run_prediction(detector, image_path)
    elapsed = time.perf_counter() - start

    mean_time = elapsed / PERF_NUM_RUNS
    print(f"[Performance] Mean inference time over {PERF_NUM_RUNS} runs: {mean_time:.4f}s")

    assert mean_time < PERF_THRESHOLD_SECONDS, (
        f"Mean inference time too slow: {mean_time:.3f}s "
        f"(threshold: {PERF_THRESHOLD_SECONDS:.3f}s)"
    )
