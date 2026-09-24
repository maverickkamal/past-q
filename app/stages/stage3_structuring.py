"""Stage 3: Full Exam Paper Structuring & Taxonomy Agent.

This stage takes an isolated exam paper string and its ExamPointer metadata from Stage 2,
and transforms the unstructured text into a fully structured QuestionBatch Pydantic object
in a single API call leveraging Gemini Flash's 65,536 output token capacity.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.curriculum import (
    get_canonical_system_regions,
    get_taxonomy_prompt_context,
    resolve_course_code,
    resolve_system_region,
)
from app.schemas import (
    ExamPointer,
    ExamQuestionBatch,
    MultiExamBatch,
    QuestionBatch,
    StructuredQuestion,
)
from app.tools.retry_handler import (
    calculate_backoff,
    configure_clean_logging,
    get_retry_reason,
    is_resource_exhausted_error,
)

load_dotenv()
configure_clean_logging()

STAGE3_MODEL = os.getenv("STAGE3_MODEL", "gemini-3.8-flash")

STAGE3_SYSTEM_INSTRUCTION = """\
You are an expert preclinical medical examination structuring agent.
You will receive one or more examination papers transcribed in Markdown along with their metadata (discipline, session, title, examiner, level, exam_id).
You will also receive a CANONICAL CURRICULUM TAXONOMY specifying the exact courses, system regions, and topics for the relevant discipline(s).

Your task is to transform this raw examination text into a strictly structured `MultiExamBatch` containing the structured questions for EACH examination paper, grouped under its corresponding `exam_id`.

For each examination paper:
- Emit an `ExamQuestionBatch` with its exact `exam_id` (e.g. 'ANA_2012_CAT_P1').
- In `questions`, include every question from that examination paper transformed into a `StructuredQuestion`.
- Never mix questions from different exams under the same `exam_id`.

CRITICAL INVARIANTS & STRUCTURING RULES:
1. QUESTION IDENTIFICATION & NUMBERING:
   - Identify every distinct question, MCQ, essay prompt, or steeplechase station.
   - Retain original numbering in `question_number` (e.g. '1', '2', 'Question 3', 'Station 4', '2(a)').

2. CATEGORY & SUB_TYPE ASSIGNMENT (STRICT HIERARCHY RULES):
   - Category MUST be one of: 'CA_QUIZ', 'STEEPLECHASE', 'ESSAY', 'OBJECTIVE'.

   - RULE 1: CONTINUOUS ASSESSMENTS & QUIZZES ('CA_QUIZ'):
     If the paper title, header, or booklet context indicates a Continuous Assessment (C.A. / CA Test), \
In-Course Assessment (ICA), Departmental Test, Mid-Semester Test, or Modular Quiz:
     -> `category` MUST BE 'CA_QUIZ'.
     -> `sub_type` MUST BE 'IN_COURSE_TEST' (for continuous assessments, in-course tests, departmental tests) \
or 'QUIZ' (for rapid recall quizzes, topic quizzes).
     -> NEVER classify a Continuous Assessment or In-Course Test as 'ESSAY' or 'SAQ'/'SEQ', even if its questions are written in essay/short-answer format. \
In our preclinical database architecture, all in-course assessment items belong strictly under 'CA_QUIZ'.

   - RULE 2: STEEPLECHASE / OSPE PRACTICALS ('STEEPLECHASE'):
     If the paper consists of timed spotter stations, pinned specimens, anatomical slides, cadaveric photos:
     -> `category` MUST BE 'STEEPLECHASE'.
     -> `sub_type` MUST BE 'SPOTTER'.

   - RULE 3: SUMMATIVE ESSAYS ('ESSAY'):
     If the paper is a summative Professional examination (e.g. 1st or 2nd MBBS Professional Exam) or Semester Final Examination consisting of essay prompts:
     -> `category` MUST BE 'ESSAY'.
     -> `sub_type` MUST BE 'SAQ' (Short Answer Question / short notes, e.g. "Write short notes on...") or 'SEQ' (Structured Essay Question / long clinical vignette).

   - RULE 4: SUMMATIVE OBJECTIVE PAPERS ('OBJECTIVE'):
     If the paper is a summative Professional examination or Semester Final Examination consisting of multiple choice questions:
     -> `category` MUST BE 'OBJECTIVE'.
     -> `sub_type` MUST BE 'SBA' (Single Best Answer) or 'MULTIPLE_TRUE_FALSE' (Type X).
     -> SPECIAL PRECLINICAL INVARIANT ("KILLER MCQ"): In preclinical Nigerian medical curricula, 'MULTIPLE_TRUE_FALSE' (5-part True/False stems with negative marking, often nicknamed "killer MCQs") is predominantly exclusive to Physiology. Anatomy and Biochemistry almost universally default to 'SBA' (Single Best Answer) unless stems explicitly contain 5 individual (a)-(e) propositions each requiring an explicit True/False determination.

3. STEM VS. ITEMS DECOMPOSITION:
   - `stem_text`: The overarching clinical vignette, primary question prompt, or practical instruction.
   - `items`: The ordered list of subordinate components:
     - For MCQs: each option is an item: `label` = 'A', 'B', 'C', etc., `text` = option text.
     - For Essays: each marked subpart is an item: `label` = 'a', 'b', 'c', 'i', etc., `text` = subpart prompt, `marks` = explicit mark integer if indicated.
     - For Steeplechase: each question is an item: `label` = 'A', 'B', 'C', `text` = question prompt (e.g. "Identify the bone"), `marks` = marks if indicated.
   - For standalone essay questions without subparts, `items` may be empty `[]`.

4. TAXONOMY & CURRICULAR MAPPING:
   - `discipline`: Must be 'Anatomy', 'Physiology', or 'Biochemistry' (inherit or refine from context).
   - `level`: Academic preclinical level ('200L', '300L', or inherit from exam).
   - `course_code`: Map to the matching course code from the provided taxonomy (e.g., 'ANA 201a', 'PIO 205', 'BCH 201') whenever identifiable from the question topic/system.
   - `system_region`: MUST be chosen from the provided CANONICAL system regions list (e.g., 'General Anatomy and Upper Limb', 'Lower Limb', 'Thorax and Abdomen', 'Cardiovascular System', 'Renal System, Body Fluids, and Thermoregulation'). Do NOT invent non-canonical synonyms.
   - `topic`: Granular medical topic (e.g., 'Brachial Plexus', 'Cardiac Cycle', 'Beta-Oxidation', 'Femur Anatomy').
   - `curriculum_style`:
     - 'CCMAS_VIGNETTE': If the stem contains a clinical scenario (patient age, sex, clinical complaint, symptoms, surgical scenario, or lab findings).
     - 'BMAS_RECALL': If the stem is a direct didactic recall prompt ("List...", "Define...", "Which of the following is...").
   - `examiner`: If the question or section mentions an examiner (e.g. 'SECTION A: PROF. ATIKU'), extract their name; otherwise inherit the exam's default examiner or 'UNKNOWN'.

5. DIAGRAM & VISUAL ASSET BINDING:
   - If the markdown text contains an inline image reference (e.g. `![label](assets/filename.webp)` or similar):
     - Set `has_diagram = True`.
     - Extract the relative path into `diagram_path` (e.g. 'assets/filename.webp').
   - If no image is present, set `has_diagram = False` and `diagram_path = None`.

6. MARKS ARITHMETIC:
   - `total_marks`: Total explicit marks for the question if printed on the paper (e.g., '[15 marks]'). If not stated, set to `None`.
   - `marks` on items: Individual marks for each subpart if printed (e.g., '[5 marks]' -> 5). Otherwise `None`.

7. STRICT ANSWER SUPPRESSION & ANTI-HALLUCINATION CONSTRAINT:
   - IGNORE ALL ANSWERS, ANSWER KEYS, SOLUTIONS, OR ANNOTATIONS: If the paper contains printed answer keys, model answers, explanations, solutions at the end of questions/sections, or handwritten student ticks/marks/circled options, STRICTLY IGNORE AND OMIT THEM.
   - Do NOT transcribe or include answer keys or solutions in `stem_text`, `items`, or any other field.
   - NEVER invent, guess, solve, or output answers or answer keys (do NOT add "Answer: B", "Ans: True", or explanations).
   - Transcribe and structure only the questions and choices/subparts themselves as un-answered exam prompts.
"""


def create_stage3_agent(
    model_name: str | None = None,
    output_schema: type | None = None,
) -> Agent:
    """Initializes the Stage 3 structuring agent with strict Pydantic output schema and HTTP retry options."""
    return Agent(
        name="stage3_structuring_agent",
        model=model_name or STAGE3_MODEL,
        output_schema=output_schema or MultiExamBatch,
        instruction=STAGE3_SYSTEM_INSTRUCTION,
        generate_content_config=types.GenerateContentConfig(
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(initial_delay=2.0, attempts=3),
            )
        ),
    )


root_agent = create_stage3_agent()


def bundle_exams_by_page_budget(
    sliced_exams: list[tuple[ExamPointer, str]],
    max_pages: int = 25,
) -> list[list[tuple[ExamPointer, str]]]:
    """Groups consecutive whole exams into bundles up to max_pages.

    Exams are grouped so that:
    1. No exam is ever split across bundles.
    2. Only exams of the same discipline are grouped together (keeps taxonomy context cohesive).
    3. If a single exam is larger than max_pages, it is assigned to its own solo bundle.

    Args:
        sliced_exams: Ordered list of (ExamPointer, isolated_markdown) tuples.
        max_pages: Maximum cumulative pages per bundle (default: 25).

    Returns:
        List of exam bundles, where each bundle is a list of (ExamPointer, isolated_markdown) tuples.
    """
    bundles: list[list[tuple[ExamPointer, str]]] = []
    current_bundle: list[tuple[ExamPointer, str]] = []
    current_pages = 0
    current_discipline: str | None = None

    for pointer, md in sliced_exams:
        exam_pages = max(1, pointer.end_page - pointer.start_page + 1)

        if current_bundle and (
            (current_pages + exam_pages > max_pages)
            or (pointer.discipline != current_discipline)
        ):
            bundles.append(current_bundle)
            current_bundle = []
            current_pages = 0
            current_discipline = None

        current_bundle.append((pointer, md))
        current_pages += exam_pages
        current_discipline = pointer.discipline

    if current_bundle:
        bundles.append(current_bundle)

    return bundles


def normalize_structured_questions(
    questions: list[StructuredQuestion],
    pointer: ExamPointer,
) -> list[StructuredQuestion]:
    """Post-processing normalization against canonical curriculum taxonomy."""
    for q in questions:
        # 1. Inherit level and institution from pointer if unknown
        if (q.level == "UNKNOWN" or not q.level) and pointer.level != "UNKNOWN":
            q.level = pointer.level
        if (getattr(q, "institution", "UNKNOWN") == "UNKNOWN" or not getattr(q, "institution", None)) and pointer.institution != "UNKNOWN":
            q.institution = pointer.institution

        # 2. Normalize system_region deterministically
        effective_disc = q.discipline or pointer.discipline
        effective_lvl = q.level if q.level != "UNKNOWN" else None
        q.system_region = resolve_system_region(
            q.system_region,
            discipline=effective_disc,
            level=effective_lvl,
        )

        # 2b. Jev System One fallback if still non-canonical
        canonical_regions = get_canonical_system_regions(discipline=effective_disc, level=effective_lvl)
        if canonical_regions and q.system_region not in canonical_regions:
            try:
                from app.tools.typesafe_taxonomy import TypeSafeTaxonomyNormalizer
                normalizer = TypeSafeTaxonomyNormalizer()
                if normalizer.is_configured:
                    q.system_region = normalizer.normalize_region(
                        stem_text=q.stem_text,
                        discipline=effective_disc,
                        raw_region=q.system_region,
                        level=effective_lvl,
                    )
            except Exception as e:
                logging.getLogger(__name__).warning(f"Jev taxonomy normalization failed: {e}")

        # 3. Resolve course_code if unassigned
        if not q.course_code:
            q.course_code = resolve_course_code(
                q.system_region,
                discipline=effective_disc,
                level=effective_lvl,
            )

    return questions


async def structure_exam_bundle(
    bundle: list[tuple[ExamPointer, str]],
    agent: Agent | None = None,
    app_name: str = "medical_exam_pipeline",
) -> MultiExamBatch:
    """Submits a bundle of examination papers to Gemini Flash in a single call to produce structured questions.

    Args:
        bundle: List of (ExamPointer, isolated_markdown) tuples belonging to this bundle.
        agent: Optional custom Agent instance.
        app_name: ADK application name.

    Returns:
        Validated MultiExamBatch object containing all structured questions grouped by exam_id.
    """
    if not bundle:
        return MultiExamBatch(exams=[])

    structuring_agent = agent or create_stage3_agent()
    session_service = InMemorySessionService()
    runner = Runner(
        agent=structuring_agent,
        session_service=session_service,
        app_name=app_name,
    )

    primary_pointer = bundle[0][0]
    bundle_slug = f"{primary_pointer.exam_id}_bundle_{len(bundle)}"
    user_id = "exam_pipeline_worker"

    distinct_disciplines = {p.discipline for p, _ in bundle}
    taxonomy_sections = [
        get_taxonomy_prompt_context(
            discipline=disc,
            level=primary_pointer.level if primary_pointer.level != "UNKNOWN" else None,
        )
        for disc in sorted(distinct_disciplines)
    ]
    taxonomy_context = "\n\n".join(taxonomy_sections)

    exam_blocks: list[str] = []
    for pointer, isolated_md in bundle:
        category_hint = (
            f"Category: {pointer.category}\n"
            if pointer.category and pointer.category != "UNKNOWN"
            else ""
        )
        level_hint = (
            f"Academic Level: {pointer.level}\n"
            if pointer.level and pointer.level != "UNKNOWN"
            else ""
        )
        exam_block = (
            f"=== BEGIN EXAMINATION: {pointer.exam_id} ===\n"
            f"Exam ID: {pointer.exam_id}\n"
            f"Paper Title: {pointer.paper_title}\n"
            f"Discipline: {pointer.discipline}\n"
            f"{category_hint}"
            f"{level_hint}"
            f"Session: {pointer.session}\n"
            f"Default Examiner: {pointer.examiner}\n"
            f"Booklet Page Span: Pages {pointer.start_page} to {pointer.end_page}\n\n"
            f"{isolated_md}\n"
            f"=== END EXAMINATION: {pointer.exam_id} ==="
        )
        exam_blocks.append(exam_block)

    joined_exams = "\n\n".join(exam_blocks)
    prompt = (
        f"Please structure the following {len(bundle)} examination paper(s) into a MultiExamBatch.\n"
        f"For each exam, emit an ExamQuestionBatch with the exact matching `exam_id` and all its structured questions.\n\n"
        f"--- CANONICAL TAXONOMY GUIDANCE ---\n"
        f"{taxonomy_context}\n\n"
        f"--- BEGIN PAPERS ---\n"
        f"{joined_exams}\n"
        f"--- END PAPERS ---"
    )

    user_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)],
    )

    max_retries = 6
    raw_json = ""
    for attempt in range(1, max_retries + 1):
        try:
            session_id = f"stage3_{bundle_slug}_{attempt}"
            await session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
            )

            collected_json_text: list[str] = []
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=user_message,
            ):
                finish_reason = getattr(event, "finish_reason", None)
                if finish_reason and str(finish_reason).endswith("MAX_TOKENS"):
                    raise RuntimeError(
                        f"65,536 token ceiling breached on bundle {bundle_slug}!"
                    )

                if event.content and event.content.role == "model":
                    for part in event.content.parts:
                        if part.text:
                            collected_json_text.append(part.text)

            raw_json = "".join(collected_json_text).strip()
            if not raw_json:
                raise RuntimeError(
                    f"Stage 3 structuring agent returned an empty response for bundle {bundle_slug}."
                )
            break
        except Exception as exc:
            if is_resource_exhausted_error(exc) and attempt < max_retries:
                backoff = calculate_backoff(attempt=attempt, base=6.0)
                reason = get_retry_reason(exc)
                print(
                    f"\n  {reason} during Stage 3 structuring ({bundle_slug}). "
                    f"Retrying in {int(backoff)}s (attempt {attempt}/{max_retries})...",
                    flush=True,
                )
                await asyncio.sleep(backoff)
            else:
                raise

    if raw_json.startswith("```json"):
        raw_json = raw_json.removeprefix("```json").removesuffix("```").strip()
    elif raw_json.startswith("```"):
        raw_json = raw_json.removeprefix("```").removesuffix("```").strip()

    try:
        multi_batch = MultiExamBatch.model_validate_json(raw_json)
    except Exception:
        single_batch = QuestionBatch.model_validate_json(raw_json)
        multi_batch = MultiExamBatch(
            exams=[ExamQuestionBatch(exam_id=primary_pointer.exam_id, questions=single_batch.questions)]
        )

    pointer_map = {p.exam_id: p for p, _ in bundle}
    for exam_batch in multi_batch.exams:
        matching_pointer = pointer_map.get(exam_batch.exam_id, primary_pointer)
        exam_batch.questions = normalize_structured_questions(
            questions=exam_batch.questions,
            pointer=matching_pointer,
        )

    returned_ids = {eb.exam_id for eb in multi_batch.exams}
    for pointer, _ in bundle:
        if pointer.exam_id not in returned_ids:
            multi_batch.exams.append(ExamQuestionBatch(exam_id=pointer.exam_id, questions=[]))

    return multi_batch


async def structure_exam_paper(
    isolated_markdown: str,
    pointer: ExamPointer,
    agent: Agent | None = None,
    app_name: str = "medical_exam_pipeline",
) -> QuestionBatch:
    """Backward-compatible single-exam structuring function.

    Args:
        isolated_markdown: Markdown segment representing the single isolated examination.
        pointer: ExamPointer containing exam metadata and page ranges.
        agent: Optional custom Agent instance.
        app_name: ADK application name.

    Returns:
        Validated QuestionBatch object containing all structured questions.
    """
    multi_batch = await structure_exam_bundle(
        bundle=[(pointer, isolated_markdown)],
        agent=agent,
        app_name=app_name,
    )
    for eb in multi_batch.exams:
        if eb.exam_id == pointer.exam_id:
            return QuestionBatch(questions=eb.questions)
    if multi_batch.exams:
        return QuestionBatch(questions=multi_batch.exams[0].questions)
    return QuestionBatch(questions=[])

