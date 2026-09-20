"""TypeSafe AI (Jev System One) Semantic Recurrence Engine.

Integrates TypeSafe's Jev model to evaluate semantic equivalence between medical
questions across academic years, bridging BMAS didactic recall questions and CCMAS
clinical vignettes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.config import TYPESAFE_API_KEY, TYPESAFE_MODEL

logger = logging.getLogger(__name__)

try:
    from typesafe_sdk import Choice, Noul, TypeSafeClient
    TYPESAFE_AVAILABLE = True
except ImportError:
    TYPESAFE_AVAILABLE = False
    Choice = None
    Noul = None
    TypeSafeClient = None


@dataclass
class RecurrenceDecision:
    """Outcome of a semantic equivalence judgment between two questions."""
    is_recurrence: bool
    probability: float
    recurrence_type: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    source: str = "typesafe_jev"


class TypeSafeRecurrenceJudge:
    """Manages System One semantic recurrence evaluation via TypeSafe's Jev model."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        probability_threshold: float = 0.80,
    ) -> None:
        self.api_key = api_key if api_key is not None else TYPESAFE_API_KEY
        self.model = model or TYPESAFE_MODEL or "jev-latest"
        self.probability_threshold = probability_threshold
        self._cache: dict[tuple[str, str], RecurrenceDecision] = {}
        self._client: Any = None

        if self.is_configured:
            try:
                self._client = TypeSafeClient(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize TypeSafeClient: {e}")
                self._client = None

    @property
    def is_configured(self) -> bool:
        """Returns True if TypeSafe SDK is installed and an API key is provided."""
        return bool(TYPESAFE_AVAILABLE and self.api_key and self.api_key != "your_typesafe_api_key_here")

    def evaluate_pair(
        self,
        q1: dict[str, Any],
        q2: dict[str, Any],
    ) -> RecurrenceDecision:
        """Evaluates whether two questions assess the same underlying medical concept.
        
        Args:
            q1: Dictionary containing question metadata and stem.
            q2: Dictionary containing question metadata and stem.
            
        Returns:
            RecurrenceDecision containing typed classification and calibrated probability.
        """
        id1 = str(q1.get("id") or hash(q1.get("stem_text", "")))
        id2 = str(q2.get("id") or hash(q2.get("stem_text", "")))
        cache_key = tuple(sorted([id1, id2]))

        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self.is_configured or self._client is None:
            decision = RecurrenceDecision(
                is_recurrence=False,
                probability=0.0,
                recurrence_type="UNAVAILABLE",
                confidence=0.0,
                source="fallback_unconfigured",
            )
            self._cache[cache_key] = decision
            return decision

        state = {
            "question_a": {
                "stem": q1.get("stem_text", ""),
                "discipline": q1.get("discipline", "Unknown"),
                "system_region": q1.get("system_region", "Unknown"),
                "topic": q1.get("topic", "General"),
                "curriculum_style": q1.get("curriculum_style", "BMAS_RECALL"),
            },
            "question_b": {
                "stem": q2.get("stem_text", ""),
                "discipline": q2.get("discipline", "Unknown"),
                "system_region": q2.get("system_region", "Unknown"),
                "topic": q2.get("topic", "General"),
                "curriculum_style": q2.get("curriculum_style", "BMAS_RECALL"),
            },
        }

        questions = {
            "is_recurrence": Noul(
                instructions=(
                    "Do Question A and Question B assess the same core anatomical, "
                    "physiological, biochemical, or clinical mechanism or concept "
                    "(including when one is a didactic essay and the other is a clinical scenario/vignette)?"
                ),
                criteria={
                    "true": "Both questions evaluate the same core medical structure, mechanism, or clinical condition.",
                    "false": "The questions evaluate distinctly different structures, physiological pathways, or clinical conditions.",
                },
            ),
            "recurrence_type": Choice(
                instructions="What is the pedagogical/clinical relationship between Question A and Question B?",
                criteria={
                    "EXACT_OR_VIGNETTE_RECURRENCE": (
                        "Questions assess the same underlying core medical knowledge, structure, or mechanism "
                        "(either identical/paraphrased or didactic vs clinical scenario)."
                    ),
                    "RELATED_TOPIC_ONLY": (
                        "Both questions relate to the same anatomical region or broad medical topic, "
                        "but test different specific details or separate pathways."
                    ),
                    "UNRELATED": "Different structures, topics, or biological mechanisms entirely.",
                },
            ),
        }

        try:
            response = self._client.system_one(
                state=state,
                questions=questions,
                model=self.model,
            )

            noul_ans = response.answers["is_recurrence"]
            choice_ans = response.answers["recurrence_type"]

            prob = float(noul_ans.noul)
            choice_label = str(choice_ans.choice)
            confidence = float(choice_ans.confidence)
            choice_probs = {k: float(v) for k, v in choice_ans.probabilities.items()}

            # High confidence acceptance:
            # Either high Noul probability OR primary choice is EXACT_OR_VIGNETTE_RECURRENCE
            is_match = (prob >= self.probability_threshold) or (
                choice_label == "EXACT_OR_VIGNETTE_RECURRENCE" and prob >= 0.65
            )

            decision = RecurrenceDecision(
                is_recurrence=is_match,
                probability=prob,
                recurrence_type=choice_label,
                confidence=confidence,
                probabilities=choice_probs,
                source="typesafe_jev",
            )
        except Exception as e:
            logger.error(f"TypeSafe API error evaluating pair ({id1}, {id2}): {e}")
            decision = RecurrenceDecision(
                is_recurrence=False,
                probability=0.0,
                recurrence_type="ERROR",
                confidence=0.0,
                source="fallback_error",
            )

        self._cache[cache_key] = decision
        return decision
