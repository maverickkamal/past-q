"""TypeSafe AI (Jev System One) Canonical Taxonomy Normalizer.

Automatically maps non-canonical or ambiguous organ systems and regions returned by
generative models into the strict canonical enums defined in curriculum_taxonomy.json.
Eliminates NON_CANONICAL_SYSTEM_REGION validation errors at sub-100ms latency.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import TYPESAFE_API_KEY, TYPESAFE_MODEL
from app.curriculum import get_canonical_system_regions

logger = logging.getLogger(__name__)

try:
    from typesafe_sdk import Choice, TypeSafeClient
    TYPESAFE_AVAILABLE = True
except ImportError:
    TYPESAFE_AVAILABLE = False
    Choice = None  
    TypeSafeClient = None  


class TypeSafeTaxonomyNormalizer:
    """Normalizes question taxonomy to canonical curriculum definitions using Jev Choice."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else TYPESAFE_API_KEY
        self.model = model or TYPESAFE_MODEL or "jev-latest"
        self._cache: dict[tuple[str, str], str] = {}
        self._client: Any = None

        if self.is_configured:
            try:
                self._client = TypeSafeClient(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize TypeSafeClient for taxonomy: {e}")
                self._client = None

    @property
    def is_configured(self) -> bool:
        """Returns True if TypeSafe SDK is available and an API key is configured."""
        return bool(TYPESAFE_AVAILABLE and self.api_key and self.api_key != "your_typesafe_api_key_here")

    def normalize_region(
        self,
        stem_text: str,
        discipline: str,
        raw_region: str,
        level: str | None = None,
    ) -> str:
        """Normalizes a system_region to its canonical form from curriculum_taxonomy.json.

        Args:
            stem_text: Question stem or prompt providing clinical/anatomical context.
            discipline: Medical discipline ('Anatomy', 'Physiology', 'Biochemistry').
            raw_region: The raw or potentially non-canonical system region string.
            level: Optional academic level ('200L', '300L').

        Returns:
            The canonical system region string.
        """
        canonical_regions = get_canonical_system_regions(discipline=discipline, level=level)
        if not canonical_regions:
            return raw_region

      
        if raw_region in canonical_regions:
            return raw_region

    
        lower_map = {r.lower().strip(): r for r in canonical_regions}
        cleaned_raw = raw_region.lower().strip()
        if cleaned_raw in lower_map:
            return lower_map[cleaned_raw]

        
        cache_key = (discipline, cleaned_raw)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if not self.is_configured or self._client is None:
            for canon in canonical_regions:
                if cleaned_raw in canon.lower() or canon.lower() in cleaned_raw:
                    self._cache[cache_key] = canon
                    return canon
            return raw_region

        state = {
            "question_stem": (stem_text or "")[:500],
            "discipline": discipline,
            "raw_region_detected": raw_region,
        }

        criteria = {region: f"Canonical {discipline} module: {region}" for region in canonical_regions}

        questions = {
            "canonical_system_region": Choice(
                instructions=(
                    f"Which of the following canonical {discipline} curriculum system/regions "
                    "does this medical examination question belong to?"
                ),
                criteria=criteria,
            )
        }

        try:
            response = self._client.system_one(
                state=state,
                questions=questions,
                model=self.model,
            )
            canonical_choice = str(response.answers["canonical_system_region"].choice)
            if canonical_choice in canonical_regions:
                self._cache[cache_key] = canonical_choice
                return canonical_choice
        except Exception as e:
            logger.error(f"TypeSafe taxonomy normalization error for '{raw_region}': {e}")

        for canon in canonical_regions:
            if cleaned_raw in canon.lower() or canon.lower() in cleaned_raw:
                self._cache[cache_key] = canon
                return canon

        return raw_region
