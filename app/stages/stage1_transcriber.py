"""Stage 1: Multimodal 10-Page Ingestion Agent using Gemini Flash in Google ADK."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.config import PAGE_CHUNK_SIZE
from app.stages.stage1_chunker import PDFChunk, slice_pdf_chunks
from app.tools.diagram_tool import crop_diagram_tool
from app.tools.retry_handler import calculate_backoff, get_retry_reason, is_resource_exhausted_error


load_dotenv()

STAGE1_MODEL = os.getenv("STAGE1_MODEL", "gemini-3.5-flash-lite")

STAGE1_SYSTEM_INSTRUCTION = """\
You are an expert medical transcription agent specializing in preclinical exam booklets.
You will receive a PDF chunk along with its starting booklet page number.

Your responsibilities:
1. Transcribe the textual contents of all pages verbatim into clean GitHub-flavored Markdown.
2. Maintain question numbering ('1.', 'Question 2'), hierarchical subparts ('(a)', '(b)', '(i)', '(ii)'), \
marks allocations ('[5 marks]'), and MCQ options ('A' through 'E').
3. Retain all header information, exam titles, instructions, and section markers (e.g., 'SECTION A: PROF. ATIKU').
4. TABLE EXTRACTION ENFORCEMENT:
   - NEVER transcribe tables (lab ranges, comparative matrices, scoring rubrics, matching grids) into Markdown pipe tables (| ... |).
   - Treat EVERY table as a visual image asset.
   - Detect the normalized bounding box [ymin, xmin, ymax, xmax] (0-1000 scale) enclosing the table.
   - Assign a kebab-case label prefixed with 'table-' (e.g., 'table-q2-electrolytes').
   - Invoke 'crop_diagram_tool' with page_in_chunk, box_2d, and label.
5. DIAGRAM AND CLINICAL IMAGE ENFORCEMENT:
   - For all anatomical illustrations, cadaveric specimens, histology spotters, radiographs, graphs, or clinical schematics:
     - You MUST invoke 'crop_diagram_tool' for EVERY single image/specimen on EVERY page.
     - NEVER hallucinate or fabricate image URLs (such as fake cloud storage URLs, artifact URLs, or dummy links).
     - NEVER write markdown image links yourself for diagrams unless returned by 'crop_diagram_tool'.
     - Detect the 1-based page index relative to the chunk (1 to 10), detect the normalized bounding box [ymin, xmin, ymax, xmax] (0-1000 scale), assign a descriptive kebab-case label, and invoke 'crop_diagram_tool'.
6. Delineate all booklet page transitions using this exact delimiter:
   <!-- PAGE BREAK [N] -->
   where [N] is the true booklet page number. Insert this delimiter at the start of each page's content.
7. STRICT ANSWER KEY & SOLUTION SUPPRESSION:
   - If the exam paper or booklet includes printed answer keys, solutions, model answers, answer sheets, or handwritten student answers/ticks/markings:
     - DO NOT transcribe the answer keys, solutions, or student marks.
     - Transcribe ONLY the question stems, options, essay prompts, and official mark allocations.
     - NEVER output answers or solutions in the transcription.
"""


def create_stage1_agent(model_name: str | None = None) -> Agent:
    """Initializes the Stage 1 Gemini transcriber agent with diagram crop tool and HTTP retry options."""
    return Agent(
        name="stage1_transcriber",
        model=model_name or STAGE1_MODEL,
        instruction=STAGE1_SYSTEM_INSTRUCTION,
        tools=[crop_diagram_tool],
        generate_content_config=types.GenerateContentConfig(
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(initial_delay=2.0, attempts=3),
            )
        ),
    )


root_agent = create_stage1_agent()



async def transcribe_chunk(
    runner: Runner,
    artifact_service: InMemoryArtifactService,
    session_service: InMemorySessionService,
    chunk: PDFChunk,
    app_name: str = "medical_exam_pipeline",
) -> str:
    """Executes a multimodal transcription turn for a single PDF chunk."""
    user_id = "exam_pipeline_worker"
    session_id = f"chunk_session_{chunk.chunk_index:04d}"

    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    # Buffer raw PDF bytes as current_chunk.pdf in ADK artifact storage
    pdf_artifact = types.Part.from_bytes(
        data=chunk.pdf_bytes,
        mime_type="application/pdf",
    )
    await artifact_service.save_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename="current_chunk.pdf",
        artifact=pdf_artifact,
    )

    prompt_directive = (
        f"You are transcribing pages {chunk.start_page} through {chunk.end_page} of a medical exam booklet ({chunk.page_count} page(s)).\n"
        f"MANDATORY INSTRUCTIONS FOR EACH PAGE:\n"
        f"1. Start every page with <!-- PAGE BREAK [N] --> where [N] is the true booklet page number.\n"
        f"2. Visual Asset Extraction: For EVERY diagram, cadaveric specimen, histology slide, spotter photograph, graph, or data table, "
        f"you MUST call crop_diagram_tool(page_in_chunk=..., box_2d=[ymin, xmin, ymax, xmax], label='...') to crop it. "
        f"NEVER skip cropping visual assets. NEVER fabricate fake URLs or dummy image links yourself.\n"
        f"3. Verbatim Transcription: Transcribe all question text, subparts (A, B, C, etc.), options, and marks allocations exactly as written.\n"
        f"4. Multi-Page Continuation: Continue through all {chunk.page_count} pages in this chunk without stopping early!"
    )




    user_message = types.Content(
        role="user",
        parts=[
            types.Part.from_bytes(data=chunk.pdf_bytes, mime_type="application/pdf"),
            types.Part.from_text(text=prompt_directive),
        ],
    )

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            collected_content: list[str] = []
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=user_message,
            ):
                if not event.content:
                    continue

                if event.content.role == "model":
                    for part in event.content.parts:
                        if part.text:
                            collected_content.append(part.text)
                elif event.content.role == "user":
                    for part in event.content.parts:
                        if part.function_response and part.function_response.response:
                            res = part.function_response.response
                            img_md = res.get("result") if isinstance(res, dict) else str(res)
                            if img_md and img_md not in "".join(collected_content):
                                collected_content.append(f"\n\n{img_md}\n\n")

            return "".join(collected_content).strip()
        except Exception as exc:
            if is_resource_exhausted_error(exc) and attempt < max_retries:
                backoff = calculate_backoff(attempt=attempt, base=12.0)
                reason = get_retry_reason(exc)
                print(
                    f"\n  {reason}. "
                    f"Retrying Chunk {chunk.chunk_index + 1} in {int(backoff)}s (attempt {attempt}/{max_retries})...",
                    flush=True,
                )
                await asyncio.sleep(backoff)
            else:
                raise



async def run_stage1(
    pdf_source: str | Path | bytes | None = None,
    max_chunks: int | None = None,
    chunk_size: int | None = None,
    agent: Agent | None = None,
    app_name: str = "medical_exam_pipeline",
    pdf_path: str | Path | bytes | None = None,
) -> str:
    """Iterates through PDF chunks, transcribes content, and crops visuals using Google ADK.

    Args:
        pdf_source: Path to PDF or raw bytes.
        max_chunks: Optional limit on chunks to process (useful for testing).
        chunk_size: Pages per chunk (default from config: 10).
        agent: Optional custom Agent instance.
        app_name: ADK application name.
        pdf_path: Alias for pdf_source.

    Returns:
        Assembled master Markdown document.
    """
    source = pdf_source or pdf_path
    if source is None:
        raise ValueError("pdf_source or pdf_path must be provided to run_stage1.")

    agent_to_use = agent or create_stage1_agent()
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()

    runner = Runner(
        agent=agent_to_use,
        session_service=session_service,
        artifact_service=artifact_service,
        app_name=app_name,
    )

    assembled_markdown: list[str] = []
    chunk_generator: Generator[PDFChunk, None, None] = slice_pdf_chunks(source, chunk_size=chunk_size)

    for i, chunk in enumerate(chunk_generator):
        if max_chunks is not None and i >= max_chunks:
            break

        print(f"  [Stage 1] Transcribing Chunk {chunk.chunk_index + 1} (Pages {chunk.start_page}-{chunk.end_page})...", flush=True)
        transcription = await transcribe_chunk(
            runner=runner,
            artifact_service=artifact_service,
            session_service=session_service,
            chunk=chunk,
        )
        if transcription:
            assembled_markdown.append(transcription)
            print(f"    -> Chunk {chunk.chunk_index + 1} completed ({len(transcription)} chars).", flush=True)
            await asyncio.sleep(2)

    return "\n\n".join(assembled_markdown)
