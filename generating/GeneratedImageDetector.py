from __future__ import annotations

import math
import pickle
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None


class GeneratedImageDetector:
    """Wrapper around the notebook pipeline functions."""

    def __init__(
        self,
        vit_model_path: str | Path = "./models/convnext_model.pth",
        convnext_model_path: str | Path = "./models/convnext_model.pth",
        dt_model: str | Path = "./models/final_decisiontree.pkl",
        *,
        device: str | None = None,
    ) -> None:
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.random_crop = transforms.RandomCrop((256, 256))
        self.center_crop = transforms.CenterCrop((252, 252))
        self.random_crop_iterations = 5

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        map_location = self.device if self.device == "cpu" else "cpu"
        self.model_vit = torch.load(vit_model_path, map_location=map_location,weights_only=False).eval()
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
        self.model_convnext = torch.load(convnext_model_path, map_location=map_location,weights_only=False).eval()

        if self.device != "cpu":
            self.model_vit = self.model_vit.to(self.device)
            self.model_convnext = self.model_convnext.to(self.device)
        else:
            self.model_vit = self.model_vit.to("cpu")
            self.model_convnext = self.model_convnext.to("cpu")

        with open(dt_model, "rb") as file:
            self.clf = pickle.load(file)
        if not hasattr(self.clf, "monotonic_cst"):
            self.clf.monotonic_cst = None

    @staticmethod
    def read_img_file(path: str | Path) -> Image.Image:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    @staticmethod
    def _stats(values: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(values), dtype=float)
        return np.array([arr.min(), arr.max(), arr.mean(), np.median(arr), arr.var()], dtype=float)

    def get_features(
        self,
        scores_model_vit: Iterable[float],
        scores_model_convnext: Iterable[float],
    ) -> np.ndarray:
        vit = np.asarray(list(scores_model_vit), dtype=float)
        conv = np.asarray(list(scores_model_convnext), dtype=float)
        return np.concatenate((self._stats(vit), self._stats(conv), self._stats(vit + conv)))

    def get_model_scores(self, img_path: str | Path) -> tuple[list[float], list[float]]:
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

        with autocast_ctx:
            with torch.inference_mode():
                for _ in range(self.random_crop_iterations):
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

        return model_vit_scores, model_convnext_scores

    def get_final_score(self, img_path: str | Path) -> dict[str, float | int]:
        scores_vit, scores_conv = self.get_model_scores(img_path)
        features = self.get_features(scores_vit, scores_conv).reshape(1, -1)
        pred = self.clf.predict_proba(features)
        row_sum = pred.sum(axis=1, keepdims=True)
        if not np.allclose(row_sum, 1.0):
            pred = np.divide(pred, row_sum, out=np.zeros_like(pred), where=row_sum != 0)
        return {"prob_of_fake": float(pred[0][1]), "is_fake": int(np.argmax(pred))}