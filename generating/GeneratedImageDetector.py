from __future__ import annotations

import math
import os
import pickle
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from preprocessing.generating_features import build_features_from_scores

Image.MAX_IMAGE_PIXELS = None

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODELS_DIR = _REPO_ROOT / "models" / "generating"


class _TimingLogger:
    """Simple helper for printing timing information when enabled."""

    def __init__(self) -> None:
        flag = os.environ.get("GENERATING_TIMING", "1").strip().lower()
        self.enabled = flag not in {"0", "false", "off"}

    def log(self, label: str, elapsed: float) -> None:
        if self.enabled:
            print(f"[GeneratingTiming] {label}: {elapsed:.3f}s")

    def measure(self, label: str):
        start = time.perf_counter()

        class _Context:
            def __enter__(self_nonlocal):
                return None

            def __exit__(self_nonlocal, exc_type, exc, tb):
                self.log(label, time.perf_counter() - start)
                return False

        return _Context()


class GeneratedImageDetector:
    """Wrapper around the notebook pipeline functions."""

    def __init__(
        self,
        vit_model_path: str | Path = _DEFAULT_MODELS_DIR / "eva_model.pth",
        convnext_model_path: str | Path = _DEFAULT_MODELS_DIR / "convnext_model.pth",
        dt_model: str | Path = _DEFAULT_MODELS_DIR / "final_decisiontree.pkl",
        *,
        device: str | None = None,
    ) -> None:
        self._timings = _TimingLogger()
        init_start = time.perf_counter()

        with self._timings.measure("init:setup_transforms"):
            self._transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            self.random_crop = transforms.RandomCrop((256, 256))
            self.center_crop = transforms.CenterCrop((252, 252))
            self.random_crop_iterations = 4

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        map_location = self.device if self.device == "cpu" else "cpu"
        with self._timings.measure("init:load_vit"):
            self.model_vit = torch.load(
                vit_model_path, map_location=map_location, weights_only=False
            ).eval()
            if not hasattr(self.model_vit, "reg_token"):
                self.model_vit.reg_token = None
            if not hasattr(self.model_vit, "no_embed_class"):
                self.model_vit.no_embed_class = False
            if not hasattr(self.model_vit, "norm_pre"):
                self.model_vit.norm_pre = torch.nn.Identity()
            if not hasattr(self.model_vit, "attn_pool"):
                self.model_vit.attn_pool = None
            for module in self.model_vit.modules():
                if module.__class__.__name__ == "EvaAttention":
                    if not hasattr(module, "q_norm"):
                        module.q_norm = torch.nn.Identity()
                    if not hasattr(module, "k_norm"):
                        module.k_norm = torch.nn.Identity()
                    if not hasattr(module, "num_prefix_tokens"):
                        module.num_prefix_tokens = 1

        with self._timings.measure("init:load_convnext"):
            self.model_convnext = torch.load(
                convnext_model_path, map_location=map_location, weights_only=False
            ).eval()

        with self._timings.measure("init:move_models"):
            if self.device != "cpu":
                self.model_vit = self.model_vit.to(self.device)
                self.model_convnext = self.model_convnext.to(self.device)
            else:
                self.model_vit = self.model_vit.to("cpu")
                self.model_convnext = self.model_convnext.to("cpu")

        with self._timings.measure("init:load_classifier"):
            with open(dt_model, "rb") as file:
                self.clf = pickle.load(file)
            if not hasattr(self.clf, "monotonic_cst"):
                self.clf.monotonic_cst = None

        self._timings.log("init:total", time.perf_counter() - init_start)

    @staticmethod
    def read_img_file(path: str | Path) -> Image.Image:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def get_features(
        self,
        scores_model_vit: Iterable[float],
        scores_model_convnext: Iterable[float],
    ) -> np.ndarray:
        return build_features_from_scores(scores_model_vit, scores_model_convnext)

    def get_model_scores(self, img_path: str | Path) -> tuple[list[float], list[float]]:
        total_start = time.perf_counter()
        with self._timings.measure("run:read_and_resize"):
            img = self.read_img_file(img_path)
            width, height = img.size
            factor = max(1.0, 256 / min(width, height))
            if factor != 1.0:
                new_size = (math.ceil(width * factor), math.ceil(height * factor))
                img = img.resize(new_size, Image.LANCZOS)

        model_vit_scores: list[float] = []
        model_convnext_scores: list[float] = []

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if isinstance(self.device, str) and self.device.startswith("cuda")
            else nullcontext()
        )

        with self._timings.measure("run:model_forward"):
            with autocast_ctx:
                with torch.inference_mode():
                    for iteration in range(1, self.random_crop_iterations + 1):
                        iter_start = time.perf_counter()
                        crop_256 = self.random_crop(img)
                        crop_252 = self.center_crop(crop_256)

                        tensor_256 = self._transform(crop_256).unsqueeze(0).to(self.device)
                        tensor_252 = self._transform(crop_252).unsqueeze(0).to(self.device)

                        vit_output = self.model_vit.forward(tensor_252)
                        vit_score = torch.sigmoid(vit_output).cpu().numpy()
                        model_vit_scores.append(float(vit_score[0][0]))

                        conv_output = self.model_convnext.forward(tensor_256)
                        conv_score = torch.sigmoid(conv_output).cpu().numpy()
                        model_convnext_scores.append(float(conv_score[0][0]))
                        self._timings.log(f"run:crop_iteration_{iteration}", time.perf_counter() - iter_start)

        self._timings.log("run:get_model_scores", time.perf_counter() - total_start)
        return model_vit_scores, model_convnext_scores

    def get_final_score(self, img_path: str | Path) -> dict[str, float | int]:
        total_start = time.perf_counter()

        scores_vit, scores_conv = self.get_model_scores(img_path)

        feature_start = time.perf_counter()
        features = self.get_features(scores_vit, scores_conv).reshape(1, -1)
        self._timings.log("run:build_features", time.perf_counter() - feature_start)

        clf_start = time.perf_counter()
        pred = self.clf.predict_proba(features)
        row_sum = pred.sum(axis=1, keepdims=True)
        if not np.allclose(row_sum, 1.0):
            pred = np.divide(pred, row_sum, out=np.zeros_like(pred), where=row_sum != 0)
        self._timings.log("run:classifier", time.perf_counter() - clf_start)
        self._timings.log("run:total", time.perf_counter() - total_start)
        return {"prob_of_fake": float(pred[0][1]), "is_fake": int(np.argmax(pred))}
