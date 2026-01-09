"""End-to-end multimodal empathy-aware conversational pipeline."""

from __future__ import annotations

from typing import Optional

from .modules.fusion import MultimodalFusionModule
from .modules.response_generator import EmpathyAwareResponseGenerator
from .modules.text_emotion_intent import TextEmotionIntentModule
from .modules.vision_emotion_context import VisionEmotionContextModule
from .schemas import AgentOutput, EmotionPrediction


class MultimodalEmpathyAgent:
    """Orchestrates text analysis, optional vision analysis, fusion, and response generation."""

    def __init__(
        self,
        text_module: Optional[TextEmotionIntentModule] = None,
        vision_module: Optional[VisionEmotionContextModule] = None,
        fusion_module: Optional[MultimodalFusionModule] = None,
        response_module: Optional[EmpathyAwareResponseGenerator] = None,
    ):
        self.text_module = text_module or TextEmotionIntentModule()
        self.vision_module = vision_module or VisionEmotionContextModule()
        self.fusion_module = fusion_module or MultimodalFusionModule()
        self.response_module = response_module or EmpathyAwareResponseGenerator()

    def run(
        self,
        user_text: str,
        image_path: Optional[str] = None,
        vision_override: Optional[EmotionPrediction] = None,
    ) -> AgentOutput:
        text_analysis = self.text_module.analyze(user_text)
        vision_prediction = vision_override or self.vision_module.analyze(image_path)
        fused_state = self.fusion_module.fuse(
            text_emotion=text_analysis.emotion,
            vision_emotion=vision_prediction,
        )
        response = self.response_module.generate(
            user_text=user_text,
            fused_state=fused_state,
            intent=text_analysis.intent,
        )

        return AgentOutput(
            user_text=user_text,
            image_path=image_path,
            text_analysis=text_analysis,
            vision_emotion=vision_prediction,
            fused_state=fused_state,
            response=response,
        )
