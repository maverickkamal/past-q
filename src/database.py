"""SQLite database operations and deterministic entity storage."""

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from src.config import DATABASE_PATH, SCHEMA_PATH
from src.schemas import StructuredQuestion, ExamPointer


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Creates a sqlite3 connection with Row factory enabled."""
    target_path = str(db_path or DATABASE_PATH)
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str | Path | None = None, schema_file: str | Path | None = None) -> None:
    """Initializes SQLite database tables and indexes from schema.sql."""
    schema_to_run = Path(schema_file or SCHEMA_PATH).read_text(encoding="utf-8")
    with get_connection(db_path) as conn:
        conn.executescript(schema_to_run)
        conn.commit()


def make_question_id(exam_id: str, question_number: str) -> str:
    """Generates a clean deterministic ID for a question, e.g. ANA_2021_PROF_P1_Q2a."""
    clean_q = re.sub(r"[^a-zA-Z0-9_]", "", question_number.replace(" ", "_"))
    return f"{exam_id}_Q{clean_q}"


def save_exam(
    exam_pointer: ExamPointer,
    source_document: str,
    exam_type: str = "MBBS_EXAM",
    db_path: str | Path | None = None,
) -> None:
    """Inserts or updates an exam record in the exams table."""
    query = """
    INSERT INTO exams (id, source_document, academic_year, discipline, level, paper_title, exam_type, examiner)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
        source_document = excluded.source_document,
        academic_year = excluded.academic_year,
        discipline = excluded.discipline,
        level = excluded.level,
        paper_title = excluded.paper_title,
        exam_type = excluded.exam_type,
        examiner = excluded.examiner;
    """
    with get_connection(db_path) as conn:
        conn.execute(
            query,
            (
                exam_pointer.exam_id,
                source_document,
                exam_pointer.session,
                exam_pointer.discipline,
                exam_pointer.level,
                exam_pointer.paper_title,
                exam_type,
                exam_pointer.examiner,
            ),
        )
        conn.commit()


def save_question(
    exam_id: str,
    question: StructuredQuestion,
    review_status: str = "APPROVED",
    flag_reasons: str | None = None,
    db_path: str | Path | None = None,
) -> str:
    """Inserts or replaces a structured question in the questions table."""
    q_id = make_question_id(exam_id, question.question_number)
    items_serialized = json.dumps([item.model_dump() for item in question.items])

    query = """
    INSERT INTO questions (
        id, exam_id, question_number, category, sub_type, discipline,
        level, course_code, system_region, topic, examiner, curriculum_style, stem_text,
        items_json, total_marks, has_diagram, diagram_path, review_status, flag_reasons
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
        category = excluded.category,
        sub_type = excluded.sub_type,
        discipline = excluded.discipline,
        level = excluded.level,
        course_code = excluded.course_code,
        system_region = excluded.system_region,
        topic = excluded.topic,
        examiner = excluded.examiner,
        curriculum_style = excluded.curriculum_style,
        stem_text = excluded.stem_text,
        items_json = excluded.items_json,
        total_marks = excluded.total_marks,
        has_diagram = excluded.has_diagram,
        diagram_path = excluded.diagram_path,
        review_status = excluded.review_status,
        flag_reasons = excluded.flag_reasons;
    """

    with get_connection(db_path) as conn:
        conn.execute(
            query,
            (
                q_id,
                exam_id,
                question.question_number,
                question.category,
                question.sub_type,
                question.discipline,
                question.level,
                question.course_code,
                question.system_region,
                question.topic,
                question.examiner,
                question.curriculum_style,
                question.stem_text,
                items_serialized,
                question.total_marks,
                int(question.has_diagram),
                question.diagram_path,
                review_status,
                flag_reasons,
            ),
        )
        conn.commit()

    return q_id


def get_questions_by_filter(
    discipline: str | None = None,
    level: str | None = None,
    category: str | None = None,
    system_region: str | None = None,
    course_code: str | None = None,
    topic: str | None = None,
    examiner: str | None = None,
    review_status: str | None = None,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Retrieves questions matching optional filtering criteria."""
    query = "SELECT * FROM questions WHERE 1=1"
    params: list[Any] = []

    if discipline:
        query += " AND discipline = ?"
        params.append(discipline)
    if level and level != "UNKNOWN":
        query += " AND level = ?"
        params.append(level)
    if category and category != "UNKNOWN":
        query += " AND category = ?"
        params.append(category)
    if system_region:
        query += " AND system_region LIKE ?"
        params.append(f"%{system_region}%")
    if course_code:
        query += " AND course_code = ?"
        params.append(course_code)
    if topic:
        query += " AND topic LIKE ?"
        params.append(f"%{topic}%")
    if examiner and examiner != "UNKNOWN":
        query += " AND examiner LIKE ?"
        params.append(f"%{examiner}%")
    if review_status:
        query += " AND review_status = ?"
        params.append(review_status)

    query += " ORDER BY created_at ASC"

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]

    # Parse items_json
    for row in rows:
        if row.get("items_json"):
            row["items"] = json.loads(row["items_json"])
        else:
            row["items"] = []
        if row.get("flag_reasons"):
            try:
                row["flag_reasons_list"] = json.loads(row["flag_reasons"])
            except Exception:
                row["flag_reasons_list"] = [row["flag_reasons"]]
        else:
            row["flag_reasons_list"] = []

    return rows


def persist_validated_exam(
    exam_pointer: ExamPointer,
    questions: list[StructuredQuestion],
    validation_results: list[Any],
    source_document: str,
    exam_type: str = "MBBS_EXAM",
    db_path: str | Path | None = None,
) -> None:
    """Atomically persists an exam and its validated structured questions into SQLite."""
    with get_connection(db_path) as conn:
        # 1. Save or update parent exam
        save_exam(
            exam_pointer=exam_pointer,
            source_document=source_document,
            exam_type=exam_type,
            db_path=db_path,
        )

        # 2. Save each question with its specific validation status & flag reasons
        for q, res in zip(questions, validation_results):
            flag_str = json.dumps(res.flag_reasons) if getattr(res, "flag_reasons", None) else None
            status = getattr(res, "review_status", "APPROVED")
            save_question(
                exam_id=exam_pointer.exam_id,
                question=q,
                review_status=status,
                flag_reasons=flag_str,
                db_path=db_path,
            )


def get_review_queue(
    discipline: str | None = None,
    exam_id: str | None = None,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Retrieves all questions flagged with 'NEEDS_REVIEW' for human inspection."""
    query = "SELECT * FROM questions WHERE review_status = 'NEEDS_REVIEW'"
    params: list[Any] = []

    if discipline:
        query += " AND discipline = ?"
        params.append(discipline)
    if exam_id:
        query += " AND exam_id = ?"
        params.append(exam_id)

    query += " ORDER BY created_at ASC"

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]

    for row in rows:
        if row.get("items_json"):
            row["items"] = json.loads(row["items_json"])
        else:
            row["items"] = []
        if row.get("flag_reasons"):
            try:
                row["flag_reasons_list"] = json.loads(row["flag_reasons"])
            except Exception:
                row["flag_reasons_list"] = [row["flag_reasons"]]
        else:
            row["flag_reasons_list"] = []

    return rows


def get_question_by_id(question_id: str, db_path: str | Path | None = None) -> dict[str, Any] | None:
    """Retrieves a single question by its primary key ID."""
    query = "SELECT * FROM questions WHERE id = ?"
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, (question_id,))
        row = cursor.fetchone()
        if not row:
            return None
        data = dict(row)

    if data.get("items_json"):
        data["items"] = json.loads(data["items_json"])
    else:
        data["items"] = []

    if data.get("flag_reasons"):
        try:
            data["flag_reasons_list"] = json.loads(data["flag_reasons"])
        except Exception:
            data["flag_reasons_list"] = [data["flag_reasons"]]
    else:
        data["flag_reasons_list"] = []

    return data


def approve_question(question_id: str, db_path: str | Path | None = None) -> bool:
    """Updates a question's review_status to 'APPROVED' and clears flags."""
    query = """
    UPDATE questions 
    SET review_status = 'APPROVED', flag_reasons = NULL 
    WHERE id = ?
    """
    with get_connection(db_path) as conn:
        cursor = conn.execute(query, (question_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_database_stats(db_path: str | Path | None = None) -> dict[str, Any]:
    """Returns total counts of exams and questions grouped by status and discipline."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        total_exams = cursor.execute("SELECT COUNT(*) FROM exams").fetchone()[0]
        total_questions = cursor.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
        approved_questions = cursor.execute(
            "SELECT COUNT(*) FROM questions WHERE review_status = 'APPROVED'"
        ).fetchone()[0]
        review_questions = cursor.execute(
            "SELECT COUNT(*) FROM questions WHERE review_status = 'NEEDS_REVIEW'"
        ).fetchone()[0]

    return {
        "total_exams": total_exams,
        "total_questions": total_questions,
        "approved_questions": approved_questions,
        "needs_review_questions": review_questions,
    }
