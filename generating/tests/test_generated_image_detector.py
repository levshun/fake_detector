from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import pytest

from GeneratedImageDetector import GeneratedImageDetector


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT.parent / "models"
CLASSIFIER_PATH = ROOT.parent / "final_decisiontree.pkl"


@pytest.fixture(scope="module")
def detector() -> GeneratedImageDetector:
    return GeneratedImageDetector(
        vit_model_path=MODELS_DIR / "eva_jpgtest_4.pth",
        convnext_model_path=MODELS_DIR / "convnext_jpgtest_4.pth",
        dt_model=CLASSIFIER_PATH,
        device="cpu",
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
def test_generated_image_scores(detector: GeneratedImageDetector, image_name: str, expected_label: int, expected_prob: float) -> None:
    image_path = Path(__file__).resolve().parent / "test_images" / image_name
    result = _run_prediction(detector, image_path)

    assert result["is_fake"] == expected_label
    assert np.isclose(result["prob_of_fake"], expected_prob, rtol=1e-3, atol=1e-6)
