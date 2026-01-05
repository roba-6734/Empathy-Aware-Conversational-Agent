"""Shared data structures across modules."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EmotionPrediction:
    emotion: str
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)
    source: str = "unknown"
    details: str = ""


@dataclass
class IntentPrediction:
    label: str
    confidence: float
    rationale: str = ""


@dataclass
class TextAnalysis:
    emotion: EmotionPrediction
    intent: IntentPrediction


@dataclass
class FusedEmotionState:
    emotion: str
    confidence: float
    fused_scores: Dict[str, float] = field(default_factory=dict)
    text_emotion: Optional[EmotionPrediction] = None
    vision_emotion: Optional[EmotionPrediction] = None
    conflict_detected: bool = False
    fusion_notes: str = ""


@dataclass
class AgentOutput:
    user_text: str
    image_path: Optional[str]
    text_analysis: TextAnalysis
    vision_emotion: Optional[EmotionPrediction]
    fused_state: FusedEmotionState
    response: str
