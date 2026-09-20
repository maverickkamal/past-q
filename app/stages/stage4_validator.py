"""Stage 4: Deterministic Validation Engine & HITL Gate.

Applies strict deterministic Python checks (zero LLM calls / zero hallucination risk)
to structured question batches:
1. Steeplechase mandatory visual asset presence & disk existence.
2. Mark arithmetic balance between item sum and question total_marks.
3. Stem text integrity (non-empty, length checks).
4. Taxonomy compliance (valid discipline, level, canonical system_region).
5. Zero-question exam guard.

Questions failing any check receive review_status='NEEDS_REVIEW' with diagnostic flags;
compliant questions receive review_status='APPROVED'.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any, Literal

from app.config import ASSETS_DIR, BASE_DIR
from app.curriculum import get_canonical_system_regions
from app.schemas import ExamPointer, QuestionBatch, StructuredQuestion


def compute_question_fingerprint(stem_text: str, items: list[Any] | None = None) -> str:
    """Computes a normalized SHA-256 fingerprint from question stem and items to detect verbatim duplicates."""
    norm_stem = re.sub(r"\s+", " ", (stem_text or "").lower().strip())
    content = norm_stem
    if items:
        for it in items:
            text = it.text if hasattr(it, "text") else (it.get("text", "") if isinstance(it, dict) else "")
            norm_it = re.sub(r"\s+", " ", (text or "").lower().strip())
            if norm_it:
                content += " | " + norm_it
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class ValidationResult(BaseModel):
    """Validation outcome for an individual question."""
    question_number: str
    review_status: Literal["APPROVED", "NEEDS_REVIEW"]
    flag_reasons: list[str] = Field(default_factory=list)


class BatchValidationResult(BaseModel):
    """Aggregated validation outcome for an entire exam batch."""
    exam_id: str
    total_questions: int
    approved_count: int
    needs_review_count: int
    results: list[ValidationResult]
    questions: list[StructuredQuestion]


def _check_diagram_exists(diagram_path: str | None) -> bool:
    """Checks if a diagram path exists on disk (checking relative and assets directory)."""
    if not diagram_path:
        return False

    candidate = Path(diagram_path)
    if candidate.is_absolute() and candidate.exists():
        return True

    if (BASE_DIR / candidate).exists():
        return True

    if (ASSETS_DIR / candidate.name).exists():
        return True

    return False


def validate_question(
    question: StructuredQuestion,
    check_asset_exists: bool = True,
) -> ValidationResult:
    """Performs deterministic validation rules on a single structured question.

    Args:
        question: The structured question to validate.
        check_asset_exists: Whether to verify that linked visual assets exist on disk.

    Returns:
        ValidationResult with 'APPROVED' or 'NEEDS_REVIEW' and diagnostic flag reasons.
    """
    flags: list[str] = []

    if question.category == "STEEPLECHASE":
        if not question.has_diagram or not question.diagram_path:
            flags.append("MISSING_MANDATORY_IMAGE")
        elif check_asset_exists and not _check_diagram_exists(question.diagram_path):
            flags.append(f"DIAGRAM_FILE_NOT_FOUND: {question.diagram_path}")

    if question.total_marks is not None and question.items:
        explicit_item_marks = [it.marks for it in question.items if it.marks is not None]
        if len(explicit_item_marks) == len(question.items) and len(explicit_item_marks) > 0:
            subparts_sum = sum(explicit_item_marks)
            if subparts_sum != question.total_marks:
                flags.append(
                    f"MARK_MISMATCH: subparts sum ({subparts_sum}) != total_marks ({question.total_marks})"
                )

    # Only flag as empty if both stem_text is missing AND no items/subparts exist
    has_valid_stem = question.stem_text and len(question.stem_text.strip()) >= 3
    has_valid_items = len(question.items) > 0 and any(len(it.text.strip()) >= 3 for it in question.items)
    if not has_valid_stem and not has_valid_items:
        flags.append("EMPTY_QUESTION_STEM")
    valid_disciplines = ("Anatomy", "Physiology", "Biochemistry")
    if question.discipline not in valid_disciplines:
        flags.append(f"INVALID_DISCIPLINE: {question.discipline}")

    if question.level not in ("200L", "300L"):
        flags.append(f"UNKNOWN_ACADEMIC_LEVEL: {question.level}")

    if question.discipline in valid_disciplines:
        canonical_regions = get_canonical_system_regions(
            discipline=question.discipline,
            level=question.level if question.level in ("200L", "300L") else None,
        )
        if canonical_regions and question.system_region not in canonical_regions:
            flags.append(f"NON_CANONICAL_SYSTEM_REGION: '{question.system_region}'")

    status: Literal["APPROVED", "NEEDS_REVIEW"] = "NEEDS_REVIEW" if flags else "APPROVED"

    return ValidationResult(
        question_number=question.question_number,
        review_status=status,
        flag_reasons=flags,
    )


def validate_batch(
    batch: QuestionBatch,
    exam_pointer: ExamPointer | None = None,
    check_asset_exists: bool = True,
    deduplicate_verbatim: bool = True,
) -> BatchValidationResult:
    """Validates an entire batch of structured questions.

    Args:
        batch: QuestionBatch produced by Stage 3.
        exam_pointer: Optional ExamPointer metadata for context and identification.
        check_asset_exists: Whether to verify visual asset file paths on disk.
        deduplicate_verbatim: Whether to detect and discard verbatim duplicate questions within the batch.

    Returns:
        BatchValidationResult summarizing batch statistics, individual question outcomes,
        and all validated questions.
    """
    exam_id = exam_pointer.exam_id if exam_pointer else "UNKNOWN_EXAM"
    results: list[ValidationResult] = []

    if not batch.questions:
        zero_result = ValidationResult(
            question_number="0",
            review_status="NEEDS_REVIEW",
            flag_reasons=["ZERO_QUESTIONS_IN_EXAM"],
        )
        return BatchValidationResult(
            exam_id=exam_id,
            total_questions=0,
            approved_count=0,
            needs_review_count=1,
            results=[zero_result],
            questions=[],
        )

    approved_count = 0
    needs_review_count = 0
    validated_questions: list[StructuredQuestion] = []
    seen_fingerprints: set[str] = set()

    for q in batch.questions:
        if deduplicate_verbatim:
            fp = compute_question_fingerprint(q.stem_text, q.items)
            # Deduplicate if question has meaningful content and was already seen in this exam
            if fp in seen_fingerprints and len((q.stem_text or "").strip()) >= 5:
                continue
            seen_fingerprints.add(fp)

        res = validate_question(q, check_asset_exists=check_asset_exists)
        results.append(res)
        validated_questions.append(q)
        if res.review_status == "APPROVED":
            approved_count += 1
        else:
            needs_review_count += 1

    return BatchValidationResult(
        exam_id=exam_id,
        total_questions=len(validated_questions),
        approved_count=approved_count,
        needs_review_count=needs_review_count,
        results=results,
        questions=validated_questions,
    )
