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
from app.schemas import ExamPointer, QuestionBatch, StructuredQuestion
from app.tools.retry_handler import calculate_backoff, get_retry_reason, is_resource_exhausted_error

logging.getLogger("google_genai.models").setLevel(logging.ERROR)

load_dotenv()

STAGE3_MODEL = os.getenv("STAGE3_MODEL", "gemini-3.8-flash")

STAGE3_SYSTEM_INSTRUCTION = """\
You are an expert preclinical medical examination structuring agent.
You will receive an isolated examination paper transcribed in Markdown along with its metadata (discipline, session, title, examiner, level).
You will also receive a CANONICAL CURRICULUM TAXONOMY specifying the exact courses, system regions, and topics for this discipline.

Your task is to transform this raw examination text into a strictly structured `QuestionBatch` containing every question in the paper.

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


def create_stage3_agent(model_name: str | None = None) -> Agent:
    """Initializes the Stage 3 structuring agent with strict Pydantic output schema and HTTP retry options."""
    return Agent(
        name="stage3_structuring_agent",
        model=model_name or STAGE3_MODEL,
        output_schema=QuestionBatch,
        instruction=STAGE3_SYSTEM_INSTRUCTION,
        generate_content_config=types.GenerateContentConfig(
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(initial_delay=2.0, attempts=3),
            )
        ),
    )


root_agent = create_stage3_agent()


async def structure_exam_paper(
    isolated_markdown: str,
    pointer: ExamPointer,
    agent: Agent | None = None,
    app_name: str = "medical_exam_pipeline",
) -> QuestionBatch:
    """Submits an entire exam paper to Gemini Flash in a single call to produce structured questions.

    Args:
        isolated_markdown: Markdown segment representing the single isolated examination.
        pointer: ExamPointer containing exam metadata and page ranges.
        agent: Optional custom Agent instance.
        app_name: ADK application name.

    Returns:
        Validated QuestionBatch object containing all structured questions.
    """
    structuring_agent = agent or create_stage3_agent()
    session_service = InMemorySessionService()
    runner = Runner(
        agent=structuring_agent,
        session_service=session_service,
        app_name=app_name,
    )

    user_id = "exam_pipeline_worker"
    session_id = f"stage3_structuring_{pointer.exam_id}"

    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    category_hint = (
        f"Designated Category: {pointer.category}\n"
        if pointer.category and pointer.category != "UNKNOWN"
        else ""
    )
    level_hint = (
        f"Designated Academic Level: {pointer.level}\n"
        if pointer.level and pointer.level != "UNKNOWN"
        else ""
    )

    taxonomy_context = get_taxonomy_prompt_context(
        discipline=pointer.discipline,
        level=pointer.level if pointer.level != "UNKNOWN" else None,
    )

    prompt = (
        f"Please structure the following {pointer.discipline} examination paper:\n"
        f"Exam ID: {pointer.exam_id}\n"
        f"Paper Title: {pointer.paper_title}\n"
        f"{category_hint}"
        f"{level_hint}"
        f"Session: {pointer.session}\n"
        f"Default Examiner: {pointer.examiner}\n"
        f"Booklet Page Span: Pages {pointer.start_page} to {pointer.end_page}\n\n"
        f"--- TAXONOMY GUIDANCE ---\n"
        f"{taxonomy_context}\n\n"
        f"--- BEGIN EXAM TEXT ---\n"
        f"{isolated_markdown}\n"
        f"--- END EXAM TEXT ---"
    )

    user_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)],
    )

    max_retries = 5
    raw_json = ""
    for attempt in range(1, max_retries + 1):
        try:
            session_id = f"stage3_structuring_{pointer.exam_id}_{attempt}"
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
                # Method B Safety Check: 3-line truncation assertion
                finish_reason = getattr(event, "finish_reason", None)
                if finish_reason and str(finish_reason).endswith("MAX_TOKENS"):
                    raise RuntimeError(
                        f"Unbelievable: 65,536 token ceiling breached on exam {pointer.exam_id}!"
                    )

                if event.content and event.content.role == "model":
                    for part in event.content.parts:
                        if part.text:
                            collected_json_text.append(part.text)

            raw_json = "".join(collected_json_text).strip()
            if not raw_json:
                raise RuntimeError(
                    f"Stage 3 structuring agent returned an empty response for exam {pointer.exam_id}."
                )
            break
        except Exception as exc:
            if is_resource_exhausted_error(exc) and attempt < max_retries:
                backoff = calculate_backoff(attempt=attempt, base=12.0)
                reason = get_retry_reason(exc)
                print(
                    f"\n  {reason} during Stage 3 structuring ({pointer.exam_id}). "
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

    batch = QuestionBatch.model_validate_json(raw_json)

    # Post-processing normalization against canonical taxonomy
    for q in batch.questions:
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

    return batch
