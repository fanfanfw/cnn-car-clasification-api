from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from app.config import get_settings


GEN_YEARS = {
    "gen2": "2008-2014",
    "gen3prefacelift": "2015-2017",
    "gen3facelift": "2018-2023",
}


def format_label(variant: str, gen: str) -> str:
    years = GEN_YEARS.get(gen, "")
    return f"{variant}_{gen}", (f"{variant} {gen} ({years})" if years else f"{variant} {gen}")


def build_model(model_name: str, num_classes: int, dropout: float) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(model.fc.in_features, num_classes))
        return model

    if model_name == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(in_features, num_classes))
        return model

    if model_name == "mobilenet":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1024, num_classes),
        )
        return model

    raise ValueError(f"Unknown model: {model_name}")


def get_transform(img_size: int) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


@dataclass(frozen=True)
class LoadedModel:
    model: nn.Module
    idx_to_class: Dict[int, str]
    transform: transforms.Compose


class HierarchicalCarClassifier:
    def __init__(self) -> None:
        self.device: torch.device = torch.device("cpu")
        self.variant: Optional[LoadedModel] = None
        self.gen_sa: Optional[LoadedModel] = None
        self.gen_sc: Optional[LoadedModel] = None
        self.gen_x: Optional[LoadedModel] = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device_name(self) -> str:
        return str(self.device)

    def load_models(
        self,
        *,
        variant_path: Optional[str] = None,
        gen_sa_path: Optional[str] = None,
        gen_sc_path: Optional[str] = None,
        gen_x_path: Optional[str] = None,
    ) -> None:
        settings = get_settings()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        variant_ckpt = Path(variant_path or settings.hier_variant_model_path)
        gen_sa_ckpt = Path(gen_sa_path or settings.hier_gen_sa_model_path)
        gen_sc_ckpt = Path(gen_sc_path or settings.hier_gen_sc_model_path)
        gen_x_ckpt = Path(gen_x_path or settings.hier_gen_x_model_path)

        self.variant = self._load_checkpoint_model(variant_ckpt)
        self.gen_sa = self._load_checkpoint_model(gen_sa_ckpt)
        self.gen_sc = self._load_checkpoint_model(gen_sc_ckpt)
        self.gen_x = self._load_checkpoint_model(gen_x_ckpt)

        self._loaded = True

    def _load_checkpoint_model(self, checkpoint_path: Path) -> LoadedModel:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        idx_to_class = checkpoint["idx_to_class"]
        train_config = checkpoint.get("train_config") or {}
        model_name = train_config.get("model_name", "resnet18")
        dropout = float(train_config.get("dropout", 0.3))
        img_size = int(train_config.get("img_size", 224))

        model = build_model(model_name, num_classes=len(idx_to_class), dropout=dropout)
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)
        model.eval()

        return LoadedModel(model=model, idx_to_class=idx_to_class, transform=get_transform(img_size))

    @torch.inference_mode()
    def _predict_probs(self, loaded: LoadedModel, image: Image.Image) -> Dict[str, float]:
        x = loaded.transform(image).unsqueeze(0).to(self.device)
        logits = loaded.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        out = {loaded.idx_to_class[i]: float(probs[i].item()) for i in range(len(loaded.idx_to_class))}
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    @staticmethod
    def _top1(probs: Dict[str, float]) -> Tuple[str, float]:
        (label, conf), *_rest = probs.items()
        return label, float(conf)

    def predict(self, image_bytes: bytes) -> Dict:
        if not self._loaded:
            raise RuntimeError("Hierarchical models not loaded. Call load_models() first.")
        if self.variant is None:
            raise RuntimeError("Variant model not loaded.")

        settings = get_settings()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        variant_probs = self._predict_probs(self.variant, image)
        variant, variant_conf = self._top1(variant_probs)

        gen_loaded: Optional[LoadedModel]
        if variant == "SA":
            gen_loaded = self.gen_sa
        elif variant == "SC":
            gen_loaded = self.gen_sc
        elif variant == "X":
            gen_loaded = self.gen_x
        else:
            gen_loaded = None

        gen_probs: Dict[str, float] = {}
        gen = "unknown"
        gen_conf = 0.0
        decision_overridden = False

        if gen_loaded is not None:
            gen_probs = self._predict_probs(gen_loaded, image)
            gen, gen_conf = self._top1(gen_probs)

            if (
                variant == "X"
                and gen == "gen3prefacelift"
                and settings.hier_x_prefacelift_min_prob > 0
                and gen_probs.get("gen3prefacelift", 0.0) < settings.hier_x_prefacelift_min_prob
            ):
                gen = "gen3facelift"
                gen_conf = float(gen_probs.get("gen3facelift", gen_conf))
                decision_overridden = True

        final_code, final_display = format_label(variant, gen)

        needs_review = False
        if variant_conf < settings.hier_variant_min_conf:
            needs_review = True
        if gen_loaded is None or gen_conf < settings.hier_gen_min_conf:
            needs_review = True

        return {
            "final_label": final_code,
            "final_display": final_display,
            "needs_review": needs_review,
            "policy_applied": {"x_prefacelift_threshold": settings.hier_x_prefacelift_min_prob, "decision_overridden": decision_overridden},
            "stage1": {"predicted": variant, "confidence": variant_conf, "probabilities": variant_probs},
            "stage2": {"predicted": gen, "confidence": gen_conf, "probabilities": gen_probs},
        }


hier_classifier = HierarchicalCarClassifier()

