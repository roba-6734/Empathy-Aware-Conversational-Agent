"""Multimodal fusion logic for emotion state estimation."""

from __future__ import annotations

from typing import Dict

from ..config import TARGET_EMOTIONS
from ..schemas import EmotionPrediction, FusedEmotionState


class MultimodalFusionModule:
    """Combine text and vision emotion signals with transparent weighted scoring."""

    def __init__(self, text_weight: float = 0.7, vision_weight: float = 0.3):
        self.text_weight = text_weight
        self.vision_weight = vision_weight

    def fuse(
        self,
        text_emotion: EmotionPrediction,
        vision_emotion: EmotionPrediction | None,
    ) -> FusedEmotionState:
        text_scores = self._normalized_scores(text_emotion)

        if vision_emotion is None:
            return FusedEmotionState(
                emotion=text_emotion.emotion,
                confidence=text_emotion.confidence,
                fused_scores=text_scores,
                text_emotion=text_emotion,
                vision_emotion=None,
                conflict_detected=False,
                fusion_notes="Text-only mode (no image provided).",
            )

        vision_scores = self._normalized_scores(vision_emotion)
        conflict = (
            text_emotion.emotion != vision_emotion.emotion
            and text_emotion.confidence >= 0.45
            and vision_emotion.confidence >= 0.45
        )

        text_weight = self.text_weight
        vision_weight = self.vision_weight
        adaptation_note = "Using default fusion weights."

        # When signals disagree and one modality is much more confident,
        # rebalance weights for a transparent conflict-resolution rule.
        if conflict and (vision_emotion.confidence - text_emotion.confidence) >= 0.25:
            text_weight, vision_weight = 0.5, 0.5
            adaptation_note = "Conflict with stronger vision confidence; rebalanced to text=0.5, vision=0.5."
        elif conflict and (text_emotion.confidence - vision_emotion.confidence) >= 0.25:
            text_weight, vision_weight = 0.8, 0.2
            adaptation_note = "Conflict with stronger text confidence; rebalanced to text=0.8, vision=0.2."

        fused: Dict[str, float] = {}
        for emotion in TARGET_EMOTIONS:
            fused[emotion] = (
                text_weight * text_scores.get(emotion, 0.0)
                + vision_weight * vision_scores.get(emotion, 0.0)
            )

        total = sum(fused.values()) or 1.0
        fused = {k: v / total for k, v in fused.items()}
        top_emotion = max(fused, key=fused.get)
        top_conf = fused[top_emotion]

        note = (
            f"Conflict detected: text and vision suggest different emotions. {adaptation_note}"
            if conflict
            else "Text and vision are aligned or low-confidence disagreement."
        )

        return FusedEmotionState(
            emotion=top_emotion,
            confidence=top_conf,
            fused_scores=fused,
            text_emotion=text_emotion,
            vision_emotion=vision_emotion,
            conflict_detected=conflict,
            fusion_notes=note,
        )

    @staticmethod
    def _normalized_scores(prediction: EmotionPrediction) -> Dict[str, float]:
        if prediction.scores:
            # Ensure full support on the shared emotion space.
            scores = {emotion: float(prediction.scores.get(emotion, 0.0)) for emotion in TARGET_EMOTIONS}
            total = sum(scores.values()) or 1.0
            return {k: v / total for k, v in scores.items()}

        one_hot = {emotion: 0.0 for emotion in TARGET_EMOTIONS}
        one_hot[prediction.emotion] = prediction.confidence
        total = sum(one_hot.values()) or 1.0
        return {k: v / total for k, v in one_hot.items()}
