"""Dynamic Institution Normalizer using Gemini Flash Lite.

Scans all distinct institutions recorded in the database and normalizes variants
(e.g., 'BUK', 'Bayero University', 'Bayero Univ Kano' -> 'Bayero University, Kano (BUK)')
dynamically without requiring a rigid hardcoded institution list ahead of time.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.config import DATABASE_PATH
from app.database import get_connection

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_NORMALIZER_MODEL = os.getenv("INSTITUTION_NORMALIZER_MODEL", "gemini-3.5-flash-lite")

SYSTEM_PROMPT = """\
You are an expert university database registrar and entity resolution specialist specializing in Nigerian and international medical colleges.
You will receive a JSON list of raw institution strings discovered across medical examination booklets.

Your task:
1. Identify strings that refer to the same university or medical college (e.g. 'BUK', 'Bayero University', 'Bayero Univ Kano').
2. Normalize each group into a single canonical institution name with its official acronym, formatted as:
   'Full Official University Name, City/Location (ACRONYM)' or 'Full Official University Name (ACRONYM)'
   Examples:
   - 'BUK', 'Bayero University', 'bayero' -> 'Bayero University, Kano (BUK)'
   - 'ABU', 'Ahmadu Bello University', 'abu zaria' -> 'Ahmadu Bello University, Zaria (ABU)'
   - 'UNILAG', 'University of Lagos' -> 'University of Lagos (UNILAG)'
   - 'UI', 'University of Ibadan' -> 'University of Ibadan (UI)'
3. If an institution does not match any other and is already a distinct valid college or university, format it cleanly with its canonical title and acronym.
4. Output STRICT JSON: an object where each key is the EXACT raw string from the input, and the value is the canonical normalized name.
   Example:
   {
     "BUK": "Bayero University, Kano (BUK)",
     "Bayero University": "Bayero University, Kano (BUK)",
     "abu": "Ahmadu Bello University, Zaria (ABU)"
   }
"""


def get_distinct_institutions(db_path: str | Path | None = None) -> list[str]:
    """Retrieves all distinct non-unknown institution strings from exams and questions."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        exam_insts = [
            row[0]
            for row in cursor.execute(
                "SELECT DISTINCT institution FROM exams WHERE institution IS NOT NULL AND trim(institution) != '' AND institution != 'UNKNOWN'"
            ).fetchall()
        ]
        q_insts = [
            row[0]
            for row in cursor.execute(
                "SELECT DISTINCT institution FROM questions WHERE institution IS NOT NULL AND trim(institution) != '' AND institution != 'UNKNOWN'"
            ).fetchall()
        ]
    all_unique = sorted(set(exam_insts + q_insts))
    return all_unique


def normalize_institutions_heuristic(raw_institutions: list[str]) -> dict[str, str]:
    """Fallback heuristic normalization using acronym detection and common aliases."""
    common_acronyms = {
        "buk": "Bayero University, Kano (BUK)",
        "bayero": "Bayero University, Kano (BUK)",
        "abu": "Ahmadu Bello University, Zaria (ABU)",
        "ahmadu bello": "Ahmadu Bello University, Zaria (ABU)",
        "unilag": "University of Lagos (UNILAG)",
        "lagos": "University of Lagos (UNILAG)",
        "ui": "University of Ibadan (UI)",
        "ibadan": "University of Ibadan (UI)",
        "oau": "Obafemi Awolowo University (OAU)",
        "ife": "Obafemi Awolowo University (OAU)",
        "unn": "University of Nigeria, Nsukka (UNN)",
        "uniben": "University of Benin (UNIBEN)",
        "benin": "University of Benin (UNIBEN)",
        "unilorin": "University of Ilorin (UNILORIN)",
        "ilorin": "University of Ilorin (UNILORIN)",
        "udus": "Usmanu Danfodiyo University, Sokoto (UDUS)",
        "sokoto": "Usmanu Danfodiyo University, Sokoto (UDUS)",
        "lasu": "Lagos State University (LASU)",
        "unical": "University of Calabar (UNICAL)",
        "calabar": "University of Calabar (UNICAL)",
        "uniport": "University of Port Harcourt (UNIPORT)",
    }

    mapping: dict[str, str] = {}
    for raw in raw_institutions:
        norm_key = re.sub(r"[^\w\s]", "", raw.lower()).strip()
        matched = False
        for prefix, canonical in common_acronyms.items():
            if norm_key == prefix or prefix in norm_key.split():
                mapping[raw] = canonical
                matched = True
                break
        if not matched:
            mapping[raw] = raw.strip()
    return mapping


def normalize_institutions_with_gemini(
    raw_institutions: list[str],
    model_name: str = DEFAULT_NORMALIZER_MODEL,
) -> dict[str, str]:
    """Invokes Gemini Flash Lite to cluster and canonicalize institution names."""
    if not raw_institutions:
        return {}

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    prompt = f"Here is the list of raw institution strings to normalize:\n{json.dumps(raw_institutions, indent=2)}"

    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"{SYSTEM_PROMPT}\n\n{prompt}")],
            )
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )

    text = (response.text or "").strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").removesuffix("```").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").removesuffix("```").strip()

    mapping: dict[str, str] = json.loads(text)
    # Ensure every input is mapped
    for raw in raw_institutions:
        if raw not in mapping:
            mapping[raw] = raw
    return mapping


def run_institution_normalization(
    db_path: str | Path | None = None,
    dry_run: bool = False,
    model_name: str = DEFAULT_NORMALIZER_MODEL,
) -> dict[str, str]:
    """Runs end-to-end institution normalization on the SQLite database."""
    target_db = str(db_path or DATABASE_PATH)
    distinct_institutions = get_distinct_institutions(target_db)

    if not distinct_institutions:
        print(f"No non-unknown institutions discovered in {target_db} to normalize.")
        return {}

    print(f"\n--- Discovered {len(distinct_institutions)} distinct institution names ---")
    for inst in distinct_institutions:
        print(f"  - {inst}")

    try:
        print(f"\nRequesting normalization from {model_name}...")
        mapping = normalize_institutions_with_gemini(distinct_institutions, model_name=model_name)
    except Exception as e:
        print(f"[WARNING] Gemini normalization call failed ({e}). Falling back to heuristic consolidator.")
        mapping = normalize_institutions_heuristic(distinct_institutions)

    print("\nProposed Normalization Map:")
    for raw, canonical in mapping.items():
        print(f"  '{raw}' -> '{canonical}'")

    if dry_run:
        print("\n[DRY RUN] No database modifications written.")
        return mapping

    # Apply database updates
    with get_connection(target_db) as conn:
        cursor = conn.cursor()
        for raw, canonical in mapping.items():
            if raw != canonical:
                cursor.execute(
                    "UPDATE exams SET institution = ? WHERE institution = ?",
                    (canonical, raw),
                )
                cursor.execute(
                    "UPDATE questions SET institution = ? WHERE institution = ?",
                    (canonical, raw),
                )
        conn.commit()

    print(f"\nSuccessfully applied canonical institution updates to {target_db}!")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical PQ Dynamic Institution Normalizer")
    parser.add_argument("--db", type=str, default=str(DATABASE_PATH), help="Path to SQLite database")
    parser.add_argument("--dry-run", action="store_true", help="Preview proposed mappings without writing to DB")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_NORMALIZER_MODEL,
        help="Gemini model to use (default: gemini-3.5-flash-lite)",
    )
    args = parser.parse_args()

    run_institution_normalization(
        db_path=args.db,
        dry_run=args.dry_run,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
