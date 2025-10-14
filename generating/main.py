"""CLI entry point for running the generated image detector."""
from __future__ import annotations

import argparse
from pathlib import Path

from GeneratedImageDetector import GeneratedImageDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the generated image detector.")
    parser.add_argument("image", type=Path, help="Path to the image to score")
    parser.add_argument("model_vit", type=Path, help="Path to the ViT torch model file")
    parser.add_argument("model_convnext", type=Path, help="Path to the ConvNeXt torch model file")
    parser.add_argument("classifier", type=Path, help="Path to the pickled decision tree classifier")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string, typically 'cuda' or 'cpu'. Defaults to cuda when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = GeneratedImageDetector(
        args.model_vit,
        args.model_convnext,
        args.classifier,
        device=args.device,
    )
    result = detector.get_final_score(args.image)
    print(f"Predicted label: {result['is_fake']}")
    print(f"Probability fake: {result['prob_of_fake']}")


if __name__ == "__main__":
    main()
