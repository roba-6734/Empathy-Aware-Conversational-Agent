#!/usr/bin/env python3
"""CLI demo for the multimodal empathy-aware conversational agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from multimodal_empathy.modules import (  # noqa: E402
    EmpathyAwareResponseGenerator,
    MultimodalFusionModule,
    TextEmotionIntentModule,
    VisionEmotionContextModule,
)
from multimodal_empathy.pipeline import MultimodalEmpathyAgent  # noqa: E402
from multimodal_empathy.schemas import EmotionPrediction  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single empathy-aware response generation turn.")
    parser.add_argument("--text", required=True, help="User text input.")
    parser.add_argument("--image", default=None, help="Optional path to image input.")
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="Disable HF model loading and use fallback behavior.",
    )
    parser.add_argument(
        "--vision-override-emotion",
        default=None,
        help="Override vision emotion label for controlled experiments.",
    )
    parser.add_argument(
        "--vision-override-confidence",
        type=float,
        default=0.0,
        help="Override vision confidence for controlled experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_models = not args.no_models

    agent = MultimodalEmpathyAgent(
        text_module=TextEmotionIntentModule(load_model=load_models),
        vision_module=VisionEmotionContextModule(load_model=load_models),
        fusion_module=MultimodalFusionModule(),
        response_module=EmpathyAwareResponseGenerator(load_model=load_models),
    )

    vision_override = None
    if args.vision_override_emotion:
        vision_override = EmotionPrediction(
            emotion=args.vision_override_emotion,
            confidence=args.vision_override_confidence,
            scores={args.vision_override_emotion: args.vision_override_confidence},
            source="manual_override",
            details="CLI-provided override.",
        )

    result = agent.run(user_text=args.text, image_path=args.image, vision_override=vision_override)

    payload = {
        "user_text": result.user_text,
        "image_path": result.image_path,
        "text_emotion": {
            "emotion": result.text_analysis.emotion.emotion,
            "confidence": result.text_analysis.emotion.confidence,
            "source": result.text_analysis.emotion.source,
        },
        "intent": {
            "label": result.text_analysis.intent.label,
            "confidence": result.text_analysis.intent.confidence,
        },
        "vision_emotion": (
            {
                "emotion": result.vision_emotion.emotion,
                "confidence": result.vision_emotion.confidence,
                "source": result.vision_emotion.source,
                "details": result.vision_emotion.details,
            }
            if result.vision_emotion
            else None
        ),
        "fused_emotion": {
            "emotion": result.fused_state.emotion,
            "confidence": result.fused_state.confidence,
            "conflict_detected": result.fused_state.conflict_detected,
            "fusion_notes": result.fused_state.fusion_notes,
        },
        "response": result.response,
    }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
