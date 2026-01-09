"""Empathy-aware conversational response generation."""

from __future__ import annotations

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional dependency in constrained envs
    pipeline = None

from ..config import DEFAULT_RESPONSE_MODEL
from ..schemas import FusedEmotionState, IntentPrediction


class EmpathyAwareResponseGenerator:
    """Generate emotion-adaptive responses using a lightweight LLM with fallback templates."""

    def __init__(self, model_name: str = DEFAULT_RESPONSE_MODEL, load_model: bool = True):
        self.model_name = model_name
        self.generator = None

        if load_model and pipeline is not None:
            try:
                self.generator = pipeline(
                    task="text2text-generation",
                    model=self.model_name,
                )
            except Exception:
                self.generator = None

    def generate(
        self,
        user_text: str,
        fused_state: FusedEmotionState,
        intent: IntentPrediction,
    ) -> str:
        if self.generator is None:
            return self._template_response(user_text, fused_state, intent)

        prompt = self._build_prompt(user_text, fused_state, intent)

        try:
            output = self.generator(
                prompt,
                max_new_tokens=96,
                do_sample=False,
            )[0]["generated_text"].strip()
            output = self._ensure_explainability_phrase(output)
            return self._inject_conflict_ack(output, fused_state)
        except Exception:
            return self._template_response(user_text, fused_state, intent)

    @staticmethod
    def _build_prompt(user_text: str, fused_state: FusedEmotionState, intent: IntentPrediction) -> str:
        return (
            "You are an empathetic assistant. "
            "Write 2-3 supportive sentences. "
            "Start with a tentative explainability phrase like 'I might be mistaken, but it seems...'. "
            "Avoid clinical language and avoid overconfidence. "
            f"Detected emotion: {fused_state.emotion} (confidence={fused_state.confidence:.2f}). "
            f"Detected intent: {intent.label}. "
            f"User message: {user_text}"
        )

    @staticmethod
    def _ensure_explainability_phrase(response: str) -> str:
        lower = response.lower()
        if "i might be mistaken" in lower or "it seems" in lower:
            return response
        return f"I might be mistaken, but it seems {response[0].lower() + response[1:] if response else ''}".strip()

    @staticmethod
    def _template_response(
        user_text: str,
        fused_state: FusedEmotionState,
        intent: IntentPrediction,
    ) -> str:
        openings = {
            "joy": "I might be mistaken, but it seems you're in a positive space right now.",
            "sadness": "I might be mistaken, but it seems this is weighing on you.",
            "anger": "I might be mistaken, but it seems you're feeling angry about this.",
            "frustration": "I might be mistaken, but it seems this situation is frustrating and draining.",
            "neutral": "I might be mistaken, but it seems you're looking at this in a steady, neutral way.",
        }

        intent_lines = {
            "support_request": "If helpful, we can break this down into one small next step together.",
            "information_request": "I can help with a direct, structured answer if you want to continue.",
            "complaint": "Your reaction makes sense, and we can work through what is under your control next.",
            "gratitude": "I appreciate you sharing that, and I am glad this helped.",
            "greeting": "Thanks for checking in. I am here and ready to help.",
            "other": "If you want, tell me a bit more and I can respond more precisely.",
        }

        opening = openings.get(fused_state.emotion, openings["neutral"])
        intent_line = intent_lines.get(intent.label, intent_lines["other"])
        base = f"{opening} {intent_line}"
        return EmpathyAwareResponseGenerator._inject_conflict_ack(base, fused_state)

    @staticmethod
    def _inject_conflict_ack(response: str, fused_state: FusedEmotionState) -> str:
        if not fused_state.conflict_detected:
            return response
        return (
            f"{response} I may be seeing mixed signals between your words and the visual cues, "
            "so please correct me if this feels off."
        )
