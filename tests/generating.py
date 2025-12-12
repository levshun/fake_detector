import random
import time
import os

import numpy as np
import torch
import pytest

import detect_ai as dai


MODELS_DIR = os.path.join('..', 'models', 'generating')


@pytest.fixture(scope="module")
def detector() -> dai.GeneratedImageDetector:
    return dai.GeneratedImageDetector(
        vit_model_path=os.path.join(MODELS_DIR, "eva_model.pth"),
        convnext_model_path=os.path.join(MODELS_DIR, "convnext_model.pth"),
        dt_model=os.path.join(MODELS_DIR, "final_decisiontree.pkl"),
        # device="cuda",
    )


def _run_prediction(detector: dai.GeneratedImageDetector, image_path: str) -> dict[str, float | int]:
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
    detector: dai.GeneratedImageDetector,
    image_name: str,
    expected_label: int,
    expected_prob: float,
) -> None:
    image_path = os.path.join("generating_data", image_name)
    result = _run_prediction(detector, image_path)

    assert result["is_fake"] == expected_label
    assert np.isclose(result["prob_of_fake"], expected_prob, rtol=1e-3, atol=1e-6)


PERF_THRESHOLD_SECONDS = 5.0
PERF_NUM_RUNS = 10


def test_generated_image_performance(detector: dai.GeneratedImageDetector) -> None:
    """Simple performance test: mean runtime over 10 runs must be < 5 seconds."""
    image_path = os.path.join("generating_data", "good.png")

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
