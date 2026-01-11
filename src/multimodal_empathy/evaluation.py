"""Evaluation utilities for qualitative and lightweight comparative analysis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .pipeline import MultimodalEmpathyAgent
from .schemas import EmotionPrediction


@dataclass
class EvaluationExample:
    example_id: str
    text: str
    image_path: Optional[str] = None
    vision_override: Optional[EmotionPrediction] = None
    note: str = ""


class EvaluationRunner:
    """Run text-only vs multimodal comparisons and log outputs for analysis."""

    def __init__(self, agent: MultimodalEmpathyAgent):
        self.agent = agent

    def run(
        self,
        examples: List[EvaluationExample],
        output_path: Optional[str] = None,
    ) -> Dict[str, object]:
        records = []
        for example in examples:
            text_only = self.agent.run(user_text=example.text, image_path=None, vision_override=None)
            multimodal = self.agent.run(
                user_text=example.text,
                image_path=example.image_path,
                vision_override=example.vision_override,
            )

            record = {
                "example_id": example.example_id,
                "note": example.note,
                "text": example.text,
                "image_path": example.image_path,
                "text_only": {
                    "emotion": text_only.fused_state.emotion,
                    "confidence": text_only.fused_state.confidence,
                    "response": text_only.response,
                },
                "multimodal": {
                    "emotion": multimodal.fused_state.emotion,
                    "confidence": multimodal.fused_state.confidence,
                    "response": multimodal.response,
                    "text_emotion": asdict(multimodal.text_analysis.emotion),
                    "vision_emotion": asdict(multimodal.vision_emotion) if multimodal.vision_emotion else None,
                    "conflict_detected": multimodal.fused_state.conflict_detected,
                    "fusion_notes": multimodal.fused_state.fusion_notes,
                },
            }
            records.append(record)

        shifted = sum(1 for r in records if r["text_only"]["emotion"] != r["multimodal"]["emotion"])
        summary = {
            "num_examples": len(records),
            "num_emotion_shifts": shifted,
            "shift_rate": shifted / len(records) if records else 0.0,
            "records": records,
        }

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return summary


def default_examples() -> List[EvaluationExample]:
    """Return required examples: neutral, emotionally charged, conflicting multimodal signals."""
    return [
        EvaluationExample(
            example_id="neutral_text",
            text="I reviewed the report and finished the updates. What should I prioritize next?",
            note="Neutral planning request.",
        ),
        EvaluationExample(
            example_id="emotionally_charged",
            text="I am overwhelmed and frustrated because nothing is working today.",
            note="Emotionally charged frustration-focused input.",
        ),
        EvaluationExample(
            example_id="conflicting_multimodal",
            text="I feel exhausted and sad after this argument.",
            vision_override=EmotionPrediction(
                emotion="joy",
                confidence=0.9,
                scores={"joy": 0.9, "neutral": 0.1},
                source="manual_override",
                details="Synthetic positive facial cue to force conflict analysis.",
            ),
            note="Conflict case: text is negative while visual signal is positive.",
        ),
    ]
