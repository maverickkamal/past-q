"""Core Master Orchestrator Agent for Medical Past Questions (PQ) Pipeline.

Implements the production entry point recommended in Best.md using ADK SequentialAgent.
Exposes root_agent for ADK CLI discovery and provides the complete end-to-end
ingestion pipeline execution from raw PDF to verified SQLite records.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any

from google.adk.agents import SequentialAgent

from app.config import DATABASE_PATH, PAGE_CHUNK_SIZE
from app.database import get_database_stats, init_db, persist_validated_exam
from app.schemas import ExamManifest, ExamPointer, PreScanMetadata, QuestionBatch, StructuredQuestion
from app.stages.stage0_scanner import scan_exam_file
from app.stages.stage1_transcriber import create_stage1_agent, run_stage1
from app.stages.stage2_manifest import (
    assemble_master_markdown,
    create_stage2_agent,
    generate_manifest,
    slice_exam_papers,
)
from app.stages.stage3_structuring import create_stage3_agent, structure_exam_paper
from app.stages.stage4_validator import BatchValidationResult, validate_batch

logging.getLogger("google_genai.models").setLevel(logging.ERROR)

# 1. Initialize Stage Sub-Agents
transcriber_agent = create_stage1_agent()
manifest_agent = create_stage2_agent()
structuring_agent = create_stage3_agent()

# 2. Define Master SequentialAgent Entry Point (ADK Convention)
root_agent = SequentialAgent(
    name="medical_past_questions_pipeline",
    description=(
        "Master preclinical medical past questions ingestion pipeline. "
        "Orchestrates vision transcription, manifest boundary slicing, and deep structuring in sequence."
    ),
    sub_agents=[
        transcriber_agent,
        manifest_agent,
        structuring_agent,
    ],
)


class PipelineReport(BaseModel):
    """Execution summary report for an end-to-end PDF ingestion run."""
    source_pdf: str
    pre_scan_metadata: PreScanMetadata
    manifest: ExamManifest
    total_exams: int
    total_questions: int
    approved_questions: int
    needs_review_questions: int
    duration_seconds: float
    persisted_db: str


async def run_pipeline(
    pdf_path: str | Path,
    exam_type: str = "MBBS_EXAM",
    db_path: str | Path | None = None,
    auto_recurrence: bool = False,
) -> PipelineReport:
    """Executes the full automated 5-stage ingestion pipeline on a past questions PDF.

    Workflow:
        Stage 0: Fast Pre-Scan & Metadata Extraction
        Stage 1: In-Memory PDF Chunking & Gemini Flash Vision Transcription
        Stage 2: Master Assembly & Pointer-Only Manifest Slicing
        Stage 3: Single-Turn Full Exam Paper Structuring with Taxonomy Guidance
        Stage 4: Deterministic Validation, Verbatim Deduplication & SQLite Atomic Persistence
        (Optional) Stage 6: Cross-Year Recurrence Matching via TypeSafe Jev

    Args:
        pdf_path: Path to the target medical past questions PDF.
        exam_type: General examination classification (e.g. 'MBBS_EXAM', 'CONTINUOUS_ASSESSMENT').
        db_path: Target SQLite database path (defaults to config.DATABASE_PATH).
        auto_recurrence: If True, automatically runs Jev cross-year recurrence clustering post-ingestion.

    Returns:
        PipelineReport summarizing the complete ingestion run.
    """
    start_time = time.time()
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"Source PDF file not found: {pdf_file}")

    target_db = str(db_path or DATABASE_PATH)
    init_db(target_db)

    print(f"\n=======================================================")
    print(f"STARTING MEDICAL PQ INGESTION PIPELINE")
    print(f"Source PDF: {pdf_file.name}")
    print(f"Target DB:  {target_db}")
    print(f"=======================================================\n")

    # --- STAGE 0: PRE-SCANNER ---
    print(f"[Stage 0] Pre-scanning {pdf_file.name}...")
    pre_scan_meta = scan_exam_file(pdf_file)
    print(
        f"  Discipline: {pre_scan_meta.discipline} | Session: {pre_scan_meta.session} | "
        f"Examiner: {pre_scan_meta.examiner} | Topic: {pre_scan_meta.topic}"
    )

    # --- STAGE 1: CHUNKED INGESTION & VISION TRANSCRIBER ---
    print(f"\n[Stage 1] Transcribing PDF via Gemini Flash (10-page in-memory chunks)...")
    markdown_chunks = await run_stage1(
        pdf_path=pdf_file,
        chunk_size=PAGE_CHUNK_SIZE,
        agent=transcriber_agent,
    )
    print(f"  Completed transcription of {len(markdown_chunks)} chunk(s).")

    # --- STAGE 2: MASTER ASSEMBLY & POINTER-ONLY MANIFEST ---
    print(f"\n[Stage 2] Assembling master markdown and generating pointer manifest...")
    master_md = assemble_master_markdown(markdown_chunks)
    manifest = await generate_manifest(master_md, pre_scan_meta=pre_scan_meta, agent=manifest_agent)
    print(f"  Identified {len(manifest.exams)} exam paper(s) in booklet:")
    for ep in manifest.exams:
        print(f"    - [{ep.exam_id}] {ep.paper_title} (Pages {ep.start_page}-{ep.end_page})")

    sliced_exams = slice_exam_papers(master_md, manifest)

    # --- STAGE 3 & 4: FULL EXAM STRUCTURING & DETERMINISTIC VALIDATION ---
    print(f"\n[Stage 3 & 4] Structuring and validating exam papers...")
    total_questions = 0
    total_approved = 0
    total_review = 0

    for pointer, isolated_md in sliced_exams.values():
        print(f"\n  Structuring [{pointer.exam_id}] ({pointer.discipline}, {pointer.level})...")
        structured_batch = await structure_exam_paper(isolated_md, pointer, agent=structuring_agent)
        print(f"    Extracted {len(structured_batch.questions)} questions.")

        # Stage 4: Deterministic Validation & Deduplication
        val_result = validate_batch(structured_batch, exam_pointer=pointer, check_asset_exists=True, deduplicate_verbatim=True)
        print(f"    Validation: {val_result.approved_count} APPROVED, {val_result.needs_review_count} NEEDS_REVIEW")

        # Stage 4: Persistence
        effective_exam_type = (
            "CONTINUOUS_ASSESSMENT" if pointer.category == "CA_QUIZ"
            else "STEEPLECHASE_OSPE" if pointer.category == "STEEPLECHASE"
            else exam_type
        )
        persist_validated_exam(
            exam_pointer=pointer,
            questions=val_result.questions,
            validation_results=val_result.results,
            source_document=pdf_file.name,
            exam_type=effective_exam_type,
            db_path=target_db,
        )

        total_questions += val_result.total_questions
        total_approved += val_result.approved_count
        total_review += val_result.needs_review_count

    duration = round(time.time() - start_time, 2)
    stats = get_database_stats(target_db)

    print(f"\n=======================================================")
    print(f"PIPELINE EXECUTION COMPLETE ({duration}s)")
    print(f"Exams Processed:         {len(manifest.exams)}")
    print(f"Total Questions Ingested:{total_questions}")
    print(f"Approved Questions:      {total_approved}")
    print(f"Questions For Review:    {total_review}")
    print(f"Database Total Exams:    {stats['total_exams']}")
    print(f"Database Total Questions:{stats['total_questions']}")
    print(f"=======================================================\n")

    if auto_recurrence:
        print("\n[Auto-Recurrence] Running TypeSafe Jev semantic recurrence clustering...")
        try:
            from app.recurrence import run_recurrence_matcher
            run_recurrence_matcher(db_path=target_db, use_typesafe=True)
        except Exception as e:
            print(f"Warning: Auto-recurrence run encountered an error: {e}")

    return PipelineReport(
        source_pdf=pdf_file.name,
        pre_scan_metadata=pre_scan_meta,
        manifest=manifest,
        total_exams=len(manifest.exams),
        total_questions=total_questions,
        approved_questions=total_approved,
        needs_review_questions=total_review,
        duration_seconds=duration,
        persisted_db=target_db,
    )


def main():
    parser = argparse.ArgumentParser(description="Medical Past Questions Ingestion Pipeline Orchestrator")
    parser.add_argument("pdf", type=str, help="Path to past questions PDF file")
    parser.add_argument("--exam-type", type=str, default="MBBS_EXAM", help="Default exam type classification")
    parser.add_argument("--db", type=str, default=None, help="Custom SQLite database output path")
    parser.add_argument("--auto-recurrence", action="store_true", help="Run Jev cross-year recurrence analysis post-ingestion")

    args = parser.parse_args()
    asyncio.run(
        run_pipeline(
            pdf_path=args.pdf,
            exam_type=args.exam_type,
            db_path=args.db,
            auto_recurrence=args.auto_recurrence,
        )
    )


if __name__ == "__main__":
    main()
