"""TypeSafe AI (Jev System One) Review Queue Automated Pre-Triage.

Evaluates validation anomalies in the Human-In-The-Loop review queue, assigning
calibrated severity scores and actionable resolution recommendations for curators.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.config import TYPESAFE_API_KEY, TYPESAFE_MODEL

logger = logging.getLogger(__name__)

try:
    from typesafe_sdk import Choice, Score, TypeSafeClient
    TYPESAFE_AVAILABLE = True
except ImportError:
    TYPESAFE_AVAILABLE = False
    Choice = None  
    Score = None  
    TypeSafeClient = None 


@dataclass
class TriageResult:
    """Outcome of an automated review queue triage evaluation."""
    question_id: str
    severity_level: str
    recommended_action: str
    confidence: float
    summary: str


class TypeSafeReviewTriager:
    """Evaluates questions in the review queue to prioritize curator intervention."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else TYPESAFE_API_KEY
        self.model = model or TYPESAFE_MODEL or "jev-latest"
        self._client: Any = None

        if self.is_configured:
            try:
                self._client = TypeSafeClient(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize TypeSafeClient for triage: {e}")
                self._client = None

    @property
    def is_configured(self) -> bool:
        """Returns True if TypeSafe SDK is available and an API key is configured."""
        return bool(TYPESAFE_AVAILABLE and self.api_key and self.api_key != "your_typesafe_api_key_here")

    def triage_question(self, question_data: dict[str, Any]) -> TriageResult:
        """Evaluates a flagged question and returns a prioritized triage judgment.

        Args:
            question_data: Question record from get_review_queue().

        Returns:
            TriageResult containing severity level and recommended curator action.
        """
        q_id = str(question_data.get("id", "UNKNOWN"))
        flags = question_data.get("flag_reasons_list") or []
        if isinstance(flags, str):
            flags = [flags]

        if not self.is_configured or self._client is None:
            # Deterministic fallback triage
            is_fatal = any("MISSING_MANDATORY_IMAGE" in f or "EMPTY_QUESTION_STEM" in f for f in flags)
            return TriageResult(
                question_id=q_id,
                severity_level="high_critical" if is_fatal else "medium_moderate",
                recommended_action="RE_EXTRACT_OR_REJECT" if is_fatal else "ADJUST_MARKS_MANUALLY",
                confidence=0.5,
                summary="Deterministic fallback triage (TypeSafe unconfigured)",
            )

        state = {
            "question_id": q_id,
            "category": question_data.get("category", "ESSAY"),
            "sub_type": question_data.get("sub_type", "SAQ"),
            "stem_snippet": (question_data.get("stem_text") or "")[:400],
            "total_marks": question_data.get("total_marks"),
            "items_count": len(question_data.get("items") or []),
            "diagnostic_flags": flags,
        }

        questions = {
            "severity": Score(
                instructions="What is the severity of the diagnostic validation flags on this medical question?",
                criteria=[
                    "low_minor_cosmetic_or_auto_resolvable",
                    "medium_manual_mark_or_text_check_needed",
                    "high_critical_data_missing_or_corrupted",
                ],
            ),
            "recommended_action": Choice(
                instructions="What action should the medical exam curator take on this flagged question?",
                criteria={
                    "AUTO_APPROVE": "Question content and items are complete and clinically sound despite the validation flag.",
                    "ADJUST_MARKS_MANUALLY": "The stem and items are sound, but the marks allocation needs manual arithmetic alignment.",
                    "RE_EXTRACT_OR_REJECT": "Content is severely corrupted, missing an essential diagram, or empty.",
                },
            ),
        }

        try:
            response = self._client.system_one(
                state=state,
                questions=questions,
                model=self.model,
            )

            sev_answer = response.answers["severity"]
            act_answer = response.answers["recommended_action"]

            score_val = sev_answer.score
            levels = ["low_minor", "medium_moderate", "high_critical"]
            sev_level = levels[min(int(score_val), len(levels) - 1)]

            action = str(act_answer.choice)
            confidence = float(act_answer.confidence)

            summary = f"Flagged with: {', '.join(flags)} -> Action: {action}"

            return TriageResult(
                question_id=q_id,
                severity_level=sev_level,
                recommended_action=action,
                confidence=confidence,
                summary=summary,
            )
        except Exception as e:
            logger.error(f"TypeSafe triage error for {q_id}: {e}")
            return TriageResult(
                question_id=q_id,
                severity_level="medium_moderate",
                recommended_action="ADJUST_MARKS_MANUALLY",
                confidence=0.5,
                summary=f"Fallback triage due to API error: {e}",
            )
