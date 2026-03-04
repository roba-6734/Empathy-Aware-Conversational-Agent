# Multimodal Empathy-Aware Conversational Agent

Research-oriented Python prototype for empathetic dialogue with modular text emotion analysis, optional vision cues, explainable multimodal fusion, and emotion-adaptive response generation.

## 1) Project Structure

```text
multimodal/
├── README.md
├── requirements.txt
├── scripts/
│   ├── demo.py
│   └── run_evaluation.py
├── sample_images/
└── src/
    └── multimodal_empathy/
        ├── __init__.py
        ├── config.py
        ├── evaluation.py
        ├── pipeline.py
        ├── schemas.py
        └── modules/
            ├── __init__.py
            ├── fusion.py
            ├── response_generator.py
            ├── text_emotion_intent.py
            └── vision_emotion_context.py
```

## 2) System Design

### Text Emotion & Intent Module
- File: `src/multimodal_empathy/modules/text_emotion_intent.py`
- Model: `j-hartmann/emotion-english-distilroberta-base`
- Output: emotion in `{joy, sadness, anger, frustration, neutral}` + confidence + score distribution
- Includes lightweight rule-based intent prediction (`support_request`, `information_request`, `complaint`, etc.)
- If model loading fails, falls back to lexical heuristics

### Vision Emotion / Context Module (Optional)
- File: `src/multimodal_empathy/modules/vision_emotion_context.py`
- Model: `dima806/facial_emotions_image_detection`
- Input: optional image path
- Output: mapped emotion + confidence
- Graceful fallback: if no image -> text-only mode; if model unavailable -> neutral fallback with note

### Multimodal Fusion Module
- File: `src/multimodal_empathy/modules/fusion.py`
- Explainable strategy:
  - Weighted fusion: `0.7 * text + 0.3 * vision`
  - Conflict flag if high-confidence disagreement across modalities
- Output: fused emotional state + confidence + fusion notes

### Empathy-Aware Response Generator
- File: `src/multimodal_empathy/modules/response_generator.py`
- Model: `google/flan-t5-small`
- Generates 2-3 sentence empathetic responses conditioned on emotion + intent
- Adds tentative explainability phrasing (e.g., "I might be mistaken, but it seems...")
- If model loading fails, falls back to emotion-aware templates

### Orchestration
- File: `src/multimodal_empathy/pipeline.py`
- `MultimodalEmpathyAgent.run(text, image_path=None, vision_override=None)` returns structured outputs for all module predictions and final response.

## 3) How To Run

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Single-turn Demo

```bash
python scripts/demo.py --text "I am feeling overwhelmed and frustrated today."
```

Text + image mode:

```bash
python scripts/demo.py --text "I am okay." --image sample_images/face.jpg
```

Controlled multimodal conflict test:

```bash
python scripts/demo.py \
  --text "I feel sad and exhausted." \
  --vision-override-emotion joy \
  --vision-override-confidence 0.9
```

### Minimal Evaluation Pipeline

```bash
python scripts/run_evaluation.py --output outputs/evaluation_report.json
```

This compares text-only vs multimodal outputs and logs:
- predicted emotion labels
- confidence scores
- generated responses
- fusion conflict notes

## 4) Example Input/Output

These examples are produced by the included scripts (shown in `--no-models` mode for deterministic fallback behavior):

1. Neutral input
```bash
python scripts/demo.py --text "I reviewed the report and finished the updates. What should I prioritize next?" --no-models
```
Output highlights:
- predicted emotion: `neutral` (1.00)
- predicted intent: `information_request` (0.70)
- response: `I might be mistaken, but it seems you're looking at this in a steady, neutral way...`

2. Emotionally charged input
```bash
python scripts/demo.py --text "I am overwhelmed and frustrated because nothing is working today." --no-models
```
Output highlights:
- predicted emotion: `frustration` (1.00)
- predicted intent: `support_request` (0.70)
- response: `I might be mistaken, but it seems this situation is frustrating and draining...`

3. Conflicting multimodal signals
```bash
python scripts/demo.py \
  --text "I feel exhausted and sad after this argument." \
  --vision-override-emotion joy \
  --vision-override-confidence 0.9 \
  --no-models
```
Output highlights:
- text emotion: `sadness` (0.50)
- visual emotion (override): `joy` (0.90)
- fused emotion: `joy` (0.50), `conflict_detected=true`
- response includes explicit mixed-signal caveat

For full comparative logging across all three cases:

```bash
python scripts/run_evaluation.py --output outputs/evaluation_report.json --no-models
```

Recent evaluation summary:
- `num_examples=3`
- `num_emotion_shifts=1`
- `shift_rate=0.3333`

## 5) Known Limitations and Failure Cases

- Emotion taxonomies differ across pretrained models; mapping to the target label set may lose nuance.
- Vision model expects facial-expression-like imagery; social scene images may be weakly handled.
- Conflict resolution is intentionally simple (weighted average), not learned.
- Response generator can hallucinate or produce generic empathy phrasing.
- Cultural/linguistic variation in emotional expression is not explicitly modeled.
- No benchmark dataset metrics are reported in this MVP (qualitative and comparative logging only).

## 6) Research Readiness Notes

- Modular components support ablations (text-only, vision-only override, fusion strategy changes).
- Structured dataclasses and JSON logs are suitable for experiment tracking.
- Extend by swapping models, adding calibration, and formal dataset-based evaluation.
