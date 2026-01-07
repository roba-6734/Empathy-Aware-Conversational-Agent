"""Vision-based emotion/context inference module."""

from __future__ import annotations

import os
from typing import Optional

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency in constrained envs
    Image = None

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional dependency in constrained envs
    pipeline = None

from ..config import DEFAULT_VISION_MODEL, TARGET_EMOTIONS
from ..schemas import EmotionPrediction


class VisionEmotionContextModule:
    """Infer emotional cues from an image using a pretrained image classifier."""

    def __init__(self, model_name: str = DEFAULT_VISION_MODEL, load_model: bool = True):
        self.model_name = model_name
        self.classifier = None
        self._label_map = {
            "happy": "joy",
            "joy": "joy",
            "sad": "sadness",
            "sadness": "sadness",
            "angry": "anger",
            "anger": "anger",
            "neutral": "neutral",
            "fear": "frustration",
            "disgust": "frustration",
            "surprise": "frustration",
            "contempt": "frustration",
        }

        if load_model and pipeline is not None:
            try:
                self.classifier = pipeline(
                    task="image-classification",
                    model=self.model_name,
                    top_k=5,
                )
            except Exception:
                self.classifier = None

    def analyze(self, image_path: Optional[str]) -> Optional[EmotionPrediction]:
        if not image_path:
            return None

        if not os.path.exists(image_path):
            return EmotionPrediction(
                emotion="neutral",
                confidence=0.0,
                scores={e: 0.0 for e in TARGET_EMOTIONS},
                source="vision_error",
                details=f"Image not found: {image_path}",
            )

        if self.classifier is None or Image is None:
            return EmotionPrediction(
                emotion="neutral",
                confidence=0.35,
                scores={"neutral": 1.0},
                source="vision_fallback",
                details="Vision model unavailable; defaulting to neutral.",
            )

        try:
            image = Image.open(image_path).convert("RGB")
            predictions = self.classifier(image)
            aggregated = {e: 0.0 for e in TARGET_EMOTIONS}

            for item in predictions:
                label = str(item.get("label", "")).lower()
                score = float(item.get("score", 0.0))
                mapped = self._map_label(label)
                aggregated[mapped] += score

            total = sum(aggregated.values()) or 1.0
            normalized = {k: v / total for k, v in aggregated.items()}
            emotion = max(normalized, key=normalized.get)
            confidence = normalized[emotion]
            return EmotionPrediction(
                emotion=emotion,
                confidence=confidence,
                scores=normalized,
                source="vision_classifier",
                details=f"Model={self.model_name}",
            )
        except Exception as exc:
            return EmotionPrediction(
                emotion="neutral",
                confidence=0.0,
                scores={e: 0.0 for e in TARGET_EMOTIONS},
                source="vision_error",
                details=f"Vision inference failed: {exc}",
            )

    def _map_label(self, label: str) -> str:
        for raw, mapped in self._label_map.items():
            if raw in label:
                return mapped
        return "neutral"
