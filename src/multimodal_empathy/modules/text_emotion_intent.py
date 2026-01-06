"""Text emotion and intent inference module."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - environment-dependent dependency availability
    pipeline = None

from ..config import DEFAULT_TEXT_MODEL, TARGET_EMOTIONS
from ..schemas import EmotionPrediction, IntentPrediction, TextAnalysis


@dataclass
class _IntentRule:
    label: str
    keywords: List[str]


class TextEmotionIntentModule:
    """Predict emotion from text with a pretrained transformer and infer lightweight intent."""

    def __init__(self, model_name: str = DEFAULT_TEXT_MODEL, load_model: bool = True):
        self.model_name = model_name
        self.classifier = None
        self._label_map = {
            "joy": "joy",
            "sadness": "sadness",
            "anger": "anger",
            "neutral": "neutral",
            "disgust": "frustration",
            "fear": "frustration",
            "surprise": "neutral",
            "annoyance": "frustration",
            "frustration": "frustration",
        }
        self._intent_rules = [
            _IntentRule("support_request", ["help", "struggling", "overwhelmed", "advice", "support"]),
            _IntentRule("information_request", ["what", "how", "why", "when", "where", "can you", "could you"]),
            _IntentRule("complaint", ["annoyed", "frustrated", "hate", "upset", "angry", "terrible"]),
            _IntentRule("gratitude", ["thank", "thanks", "appreciate", "grateful"]),
            _IntentRule("greeting", ["hello", "hi", "hey", "good morning", "good evening"]),
        ]

        if load_model and pipeline is not None:
            try:
                self.classifier = pipeline(
                    task="text-classification",
                    model=self.model_name,
                    top_k=None,
                )
            except Exception:
                self.classifier = None

    def analyze(self, text: str) -> TextAnalysis:
        emotion = self._predict_emotion(text)
        intent = self._predict_intent(text)
        return TextAnalysis(emotion=emotion, intent=intent)

    def _predict_emotion(self, text: str) -> EmotionPrediction:
        if self.classifier is None:
            return self._fallback_emotion(text)

        try:
            raw = self.classifier(text)
            candidates = raw[0] if raw and isinstance(raw[0], list) else raw
            aggregated = {e: 0.0 for e in TARGET_EMOTIONS}

            for item in candidates:
                label = str(item.get("label", "")).lower()
                score = float(item.get("score", 0.0))
                target_label = self._label_map.get(label, "neutral")
                aggregated[target_label] += score

            total = sum(aggregated.values()) or 1.0
            normalized = {k: v / total for k, v in aggregated.items()}
            emotion = max(normalized, key=normalized.get)
            confidence = normalized[emotion]
            return EmotionPrediction(
                emotion=emotion,
                confidence=confidence,
                scores=normalized,
                source="text_transformer",
                details=f"Model={self.model_name}",
            )
        except Exception:
            return self._fallback_emotion(text)

    def _fallback_emotion(self, text: str) -> EmotionPrediction:
        text_l = text.lower()
        scores: Dict[str, float] = {e: 0.0 for e in TARGET_EMOTIONS}

        lexicon = {
            "joy": ["happy", "great", "excited", "glad", "wonderful"],
            "sadness": ["sad", "down", "depressed", "lonely", "hopeless"],
            "anger": ["angry", "furious", "mad", "rage", "irritated"],
            "frustration": ["frustrated", "stuck", "overwhelmed", "annoyed", "exhausted"],
            "neutral": ["okay", "fine", "normal", "update", "status"],
        }

        for emotion, keywords in lexicon.items():
            for keyword in keywords:
                if keyword in text_l:
                    scores[emotion] += 1.0

        if sum(scores.values()) == 0:
            scores["neutral"] = 1.0

        total = sum(scores.values())
        normalized = {k: v / total for k, v in scores.items()}
        top_emotion = max(normalized, key=normalized.get)
        return EmotionPrediction(
            emotion=top_emotion,
            confidence=normalized[top_emotion],
            scores=normalized,
            source="text_fallback_rules",
            details="Transformer unavailable; lexical fallback used.",
        )

    def _predict_intent(self, text: str) -> IntentPrediction:
        text_l = text.lower()
        tokens = set(re.findall(r"\b[a-z']+\b", text_l))

        def matches_keyword(keyword: str) -> bool:
            if " " in keyword:
                return keyword in text_l
            return keyword in tokens

        for rule in self._intent_rules:
            matches = sum(1 for keyword in rule.keywords if matches_keyword(keyword))
            if matches > 0:
                confidence = min(0.95, 0.6 + 0.1 * matches)
                return IntentPrediction(
                    label=rule.label,
                    confidence=confidence,
                    rationale=f"Matched keywords for {rule.label}.",
                )

        return IntentPrediction(
            label="other",
            confidence=0.5,
            rationale="No intent keywords matched.",
        )
