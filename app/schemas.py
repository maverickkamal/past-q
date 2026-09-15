"""Pydantic data contracts for the Medical Past Questions ingestion pipeline."""

from pydantic import BaseModel, Field
from typing import Literal


class QuestionItem(BaseModel):
    label: str = Field(description="Subpart label ('a', 'b', 'i') or MCQ option identifier ('A', 'B', 'C')")
    text: str = Field(description="Sub-question prompt or MCQ statement text")
    marks: int | None = Field(default=None, description="Explicit marks allocated to this item")


class StructuredQuestion(BaseModel):
    question_number: str = Field(description="Question number as printed, e.g., '1', '2(a)', 'Station 4'")
    category: Literal["ESSAY", "OBJECTIVE", "STEEPLECHASE", "CA_QUIZ"]
    sub_type: Literal["SEQ", "SAQ", "SBA", "MULTIPLE_TRUE_FALSE", "SPOTTER", "IN_COURSE_TEST", "QUIZ"]
    discipline: Literal["Anatomy", "Physiology", "Biochemistry"]
    level: Literal["200L", "300L", "UNKNOWN"] = Field(
        default="UNKNOWN", description="Preclinical academic level: 200L, 300L, or UNKNOWN"
    )
    course_code: str | None = Field(
        default=None, description="Preclinical course code if identifiable, e.g., 'ANA 201a', 'PIO 205'"
    )
    system_region: str = Field(description="Canonical organ system or anatomical region, e.g., 'Thorax and Abdomen', 'Lower Limb'")
    topic: str = Field(description="Specific topic, e.g., 'Brachial Plexus', 'Beta-Oxidation'")
    examiner: str = Field(default="UNKNOWN", description="Identified lecturer or section examiner")
    curriculum_style: Literal["BMAS_RECALL", "CCMAS_VIGNETTE"]
    stem_text: str = Field(description="Primary clinical scenario, question stem, or practical instruction")
    items: list[QuestionItem] = Field(default=[], description="Ordered list of sub-questions or MCQ options")
    total_marks: int | None = Field(default=None, description="Total marks allocated to this question")
    has_diagram: bool = Field(default=False)
    diagram_path: str | None = Field(default=None, description="Relative path to cropped visual asset")


class QuestionBatch(BaseModel):
    questions: list[StructuredQuestion]


class ExamPointer(BaseModel):
    exam_id: str = Field(description="Deterministic slug, e.g., 'ANA_2021_PROF_P1'")
    discipline: Literal["Anatomy", "Physiology", "Biochemistry"]
    level: Literal["200L", "300L", "UNKNOWN"] = Field(
        default="UNKNOWN", description="Preclinical academic level: 200L, 300L, or UNKNOWN"
    )
    paper_title: str = Field(description="e.g., 'Paper I (Gross Anatomy)'")
    session: str = Field(description="e.g., '2021/2022' or 'UNKNOWN'")
    examiner: str = Field(default="UNKNOWN", description="Primary examiner if identified")
    category: Literal["ESSAY", "OBJECTIVE", "STEEPLECHASE", "CA_QUIZ", "UNKNOWN"] = Field(
        default="UNKNOWN", description="Exam category: ESSAY, OBJECTIVE, STEEPLECHASE, or CA_QUIZ"
    )
    start_page: int = Field(description="1-based start page of this exam in the PDF booklet")
    end_page: int = Field(description="1-based end page of this exam in the PDF booklet")



class ExamManifest(BaseModel):
    exams: list[ExamPointer]


class PreScanMetadata(BaseModel):
    """Metadata derived from filename tokens and page-1 inspection in Stage 0."""
    source_file: str
    discipline: Literal["Anatomy", "Physiology", "Biochemistry", "UNKNOWN"] = "UNKNOWN"
    session: str = "UNKNOWN"
    examiner: str = "UNKNOWN"
    topic: str = "UNKNOWN"
    exam_type: str = "UNKNOWN"
    confidence_tier: Literal["TIER_1_FILENAME", "TIER_2_PAGE_INSPECTION", "FALLBACK"] = "FALLBACK"
