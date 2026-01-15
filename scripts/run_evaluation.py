#!/usr/bin/env python3
"""Run minimal evaluation for the multimodal empathy-aware MVP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from multimodal_empathy.evaluation import EvaluationRunner, default_examples  # noqa: E402
from multimodal_empathy.modules import (  # noqa: E402
    EmpathyAwareResponseGenerator,
    MultimodalFusionModule,
    TextEmotionIntentModule,
    VisionEmotionContextModule,
)
from multimodal_empathy.pipeline import MultimodalEmpathyAgent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation for text-only vs multimodal emotion inference.")
    parser.add_argument(
        "--output",
        default="outputs/evaluation_report.json",
        help="Path to write evaluation report JSON.",
    )
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="Disable pretrained model loading and use fallback behavior.",
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

    runner = EvaluationRunner(agent=agent)
    summary = runner.run(examples=default_examples(), output_path=args.output)

    print("Evaluation summary")
    print(json.dumps({
        "num_examples": summary["num_examples"],
        "num_emotion_shifts": summary["num_emotion_shifts"],
        "shift_rate": summary["shift_rate"],
        "report_path": args.output,
    }, indent=2))

    qualitative = next(
        (record for record in summary["records"] if record["example_id"] == "conflicting_multimodal"),
        None,
    )
    if qualitative:
        print("\nQualitative conflict example")
        print(json.dumps(qualitative, indent=2))


if __name__ == "__main__":
    main()
