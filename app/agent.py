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
from app.database import (
    get_database_stats,
    get_ingested_source_documents,
    init_db,
    is_source_document_ingested,
    persist_validated_exam,
)
from app.schemas import ExamManifest, ExamPointer, PreScanMetadata, QuestionBatch, StructuredQuestion
from app.stages.stage0_scanner import is_pdf_password_protected, scan_exam_file
from app.stages.stage1_transcriber import create_stage1_agent, run_stage1
from app.stages.stage2_manifest import (
    assemble_master_markdown,
    create_stage2_agent,
    generate_manifest,
    slice_exam_papers,
)
from app.stages.stage3_structuring import create_stage3_agent, structure_exam_paper
from app.stages.stage4_validator import BatchValidationResult, validate_batch
from app.tools.retry_handler import get_retry_reason

logging.getLogger("google_genai.models").setLevel(logging.ERROR)

# 1. Initialize Stage Sub-Agents
transcriber_agent = create_stage1_agent()
manifest_agent = create_stage2_agent()
structuring_agent = create_stage3_agent()

# 2. Sequential ADK Coordinator
root_agent = SequentialAgent(
    name="medical_past_questions_pipeline",
    description="Sequential multi-agent orchestrator transforming medical past-question PDFs into structured SQLite records.",
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
    status: str = "COMPLETED"


async def run_pipeline(
    pdf_path: str | Path,
    exam_type: str = "MBBS_EXAM",
    db_path: str | Path | None = None,
    auto_recurrence: bool = False,
    force: bool = False,
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
        force: If True, re-ingests the PDF even if already recorded in SQLite.

    Returns:
        PipelineReport summarizing the complete ingestion run.
    """
    start_time = time.time()
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"Source PDF file not found: {pdf_file}")

    target_db = str(db_path or DATABASE_PATH)
    init_db(target_db)

    # 1. Skip if booklet is already ingested in database (unless force=True)
    if not force and is_source_document_ingested(pdf_file.name, db_path=target_db):
        print(
            f"\n[SKIP] Booklet '{pdf_file.name}' is already recorded in the database. "
            f"Skipping to save quota (pass --force to re-ingest).\n",
            flush=True,
        )
        return PipelineReport(
            source_pdf=pdf_file.name,
            pre_scan_metadata=PreScanMetadata(source_file=pdf_file.name),
            manifest=ExamManifest(exams=[]),
            total_exams=0,
            total_questions=0,
            approved_questions=0,
            needs_review_questions=0,
            duration_seconds=round(time.time() - start_time, 2),
            persisted_db=target_db,
            status="SKIPPED_ALREADY_INGESTED",
        )

    # 2. Password-protected / encrypted PDF check guard
    if is_pdf_password_protected(pdf_file):
        print(f"\n[WARNING] PDF '{pdf_file.name}' is password-protected or encrypted. Skipping ingestion gracefully.\n")
        return PipelineReport(
            source_pdf=pdf_file.name,
            pre_scan_metadata=PreScanMetadata(source_file=pdf_file.name),
            manifest=ExamManifest(exams=[]),
            total_exams=0,
            total_questions=0,
            approved_questions=0,
            needs_review_questions=0,
            duration_seconds=round(time.time() - start_time, 2),
            persisted_db=target_db,
            status="SKIPPED_PASSWORD_PROTECTED",
        )

    print(f"\n=======================================================")
    print(f"STARTING MEDICAL PQ INGESTION PIPELINE")
    print(f"Source PDF: {pdf_file.name}")
    print(f"Target DB:  {target_db}")
    print(f"=======================================================\n")

    # --- STAGE 0: PRE-SCANNER ---
    print(f"[Stage 0] Pre-scanning {pdf_file.name}...")
    pre_scan_meta = scan_exam_file(pdf_file)
    print(
        f"  Discipline: {pre_scan_meta.discipline} | Institution: {pre_scan_meta.institution} | "
        f"Session: {pre_scan_meta.session} | Examiner: {pre_scan_meta.examiner} | Topic: {pre_scan_meta.topic}"
    )

    # --- STAGE 1: CHUNKED INGESTION & VISION TRANSCRIBER ---
    print(f"\n[Stage 1] Transcribing PDF via Gemini Flash (adaptive in-memory chunking)...")
    master_md = await run_stage1(
        pdf_source=pdf_file,
        chunk_size=None,
        agent=transcriber_agent,
    )
    print(f"  Completed transcription ({len(master_md)} characters).")

    # --- STAGE 2: MASTER ASSEMBLY & POINTER-ONLY MANIFEST ---
    await asyncio.sleep(2)
    print(f"\n[Stage 2] Generating pointer manifest...")
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
        await asyncio.sleep(2)
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


async def run_batch_or_single(
    target_path: str | Path,
    exam_type: str = "MBBS_EXAM",
    db_path: str | Path | None = None,
    auto_recurrence: bool = False,
    delay_seconds: int = 60,
    force: bool = False,
) -> list[PipelineReport]:
    """Ingests a single PDF file or all PDFs within a target directory with rate-limit delays."""
    path = Path(target_path)
    if not path.exists():
        raise FileNotFoundError(f"Source path not found: {path}")

    target_db = str(db_path or DATABASE_PATH)
    init_db(target_db)
    ingested_set = get_ingested_source_documents(target_db) if not force else set()

    if path.is_dir():
        all_pdf_files = sorted([p for p in path.rglob("*.pdf") if not p.name.startswith("._")])
        pending_count = sum(1 for p in all_pdf_files if p.name.lower().strip() not in ingested_set)
        already_count = len(all_pdf_files) - pending_count
        print(f"\n=======================================================", flush=True)
        print(f"DISCOVERED {len(all_pdf_files)} PDF FILE(S) IN DIRECTORY: '{path.name}'", flush=True)
        print(f"  Already Ingested: {already_count} booklet(s) (will be skipped)", flush=True)
        print(f"  Pending Ingestion: {pending_count} booklet(s)", flush=True)
        print(f"  Queue Cooldown:    {delay_seconds}s (1 min) after actively processed booklets", flush=True)
        print(f"=======================================================\n", flush=True)
        pdf_files = all_pdf_files
    else:
        pdf_files = [path]

    reports: list[PipelineReport] = []
    for i, pdf_file in enumerate(pdf_files, 1):
        if len(pdf_files) > 1:
            print(f"\n=======================================================", flush=True)
            print(f"[{i}/{len(pdf_files)}] QUEUE PROCESSING: {pdf_file.name}", flush=True)
            print(f"=======================================================\n", flush=True)

        try:
            report = await run_pipeline(
                pdf_path=pdf_file,
                exam_type=exam_type,
                db_path=db_path,
                auto_recurrence=False,
                force=force,
            )
            reports.append(report)

            # Only pause for cooldown if the booklet was actively processed (not skipped)
            was_processed = getattr(report, "status", None) == "COMPLETED"
            if was_processed and len(pdf_files) > 1 and i < len(pdf_files) and delay_seconds > 0:
                # Check if any remaining files need processing
                remaining_pending = any(
                    p.name.lower().strip() not in ingested_set or force
                    for p in pdf_files[i:]
                )
                if remaining_pending:
                    print(
                        f"\n[Queue Cooldown] Pausing for {delay_seconds}s (1 min) before processing next booklet...",
                        flush=True,
                    )
                    await asyncio.sleep(delay_seconds)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n[Stopped] Batch queue interrupted by user. Exiting...", flush=True)
            break
        except Exception as e:
            reason = get_retry_reason(e)
            print(f"\n[ERROR] Failed ingesting '{pdf_file.name}': {reason}. Continuing queue...\n", flush=True)

    if auto_recurrence and any(r.total_questions > 0 for r in reports):
        print("\n[Auto-Recurrence] Running TypeSafe Jev semantic recurrence clustering across all ingested exams...", flush=True)
        try:
            from app.recurrence import run_recurrence_matcher
            run_recurrence_matcher(db_path=db_path, use_typesafe=True)
        except Exception as e:
            print(f"Warning: Auto-recurrence run encountered an error: {e}", flush=True)

    return reports


def main():
    parser = argparse.ArgumentParser(description="Medical Past Questions Ingestion Pipeline Orchestrator")
    parser.add_argument("path", type=str, help="Path to past questions PDF file or directory of PDFs")
    parser.add_argument("--exam-type", type=str, default="MBBS_EXAM", help="Default exam type classification")
    parser.add_argument("--db", type=str, default=None, help="Custom SQLite database output path")
    parser.add_argument("--delay", type=int, default=60, help="Delay in seconds between queued PDFs (default: 60s)")
    parser.add_argument("--auto-recurrence", action="store_true", help="Run Jev cross-year recurrence analysis post-ingestion")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion of PDF even if already present in database")

    args = parser.parse_args()
    try:
        asyncio.run(
            run_batch_or_single(
                target_path=args.path,
                exam_type=args.exam_type,
                db_path=args.db,
                auto_recurrence=args.auto_recurrence,
                delay_seconds=args.delay,
                force=args.force,
            )
        )
    except KeyboardInterrupt:
        print("\n[Stopped] Ingestion cancelled by user.\n", flush=True)


if __name__ == "__main__":
    main()
