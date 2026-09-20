"""Stage 0: Pre-Ingestion File & Metadata Scanner.

Extracts discipline, topic, session, examiner, and exam type from:
- Tier 1: File name tokens and regex patterns
- Tier 2: Page 1 header inscriptions via PyMuPDF
"""

import re
from pathlib import Path
from typing import Any
import pymupdf

from app.schemas import PreScanMetadata

# Discipline keyword maps
DISCIPLINE_MAP = {
    "anatomy": "Anatomy",
    "ana": "Anatomy",
    "gross": "Anatomy",
    "histology": "Anatomy",
    "embryology": "Anatomy",
    "physiology": "Physiology",
    "physio": "Physiology",
    "pio": "Physiology",
    "phs": "Physiology",
    "biochemistry": "Biochemistry",
    "biochem": "Biochemistry",
    "bch": "Biochemistry",
    "bic": "Biochemistry",
}

COMMON_INSTITUTION_MAP = {
    "buk": "Bayero University Kano (BUK)",
    "abu": "Ahmadu Bello University (ABU)",
    "unilag": "University of Lagos (UNILAG)",
    "ui": "University of Ibadan (UI)",
    "oau": "Obafemi Awolowo University (OAU)",
    "unn": "University of Nigeria, Nsukka (UNN)",
    "uniben": "University of Benin (UNIBEN)",
    "unilorin": "University of Ilorin (UNILORIN)",
    "udus": "Usmanu Danfodiyo University Sokoto (UDUS)",
    "lasu": "Lagos State University (LASU)",
    "unical": "University of Calabar (UNICAL)",
    "uniport": "University of Port Harcourt (UNIPORT)",
    "futo": "Federal University of Technology Owerri (FUTO)",
    "futa": "Federal University of Technology Akure (FUTA)",
    "delsu": "Delta State University (DELSU)",
    "eksu": "Ekiti State University (EKSU)",
    "oou": "Olabisi Onabanjo University (OOU)",
}

EXAM_TYPE_PATTERNS = [
    (r"\b(2nd\s*mbbs|professional|prof\s*exam)\b", "2nd_MBBS_PROFESSIONAL"),
    (r"\b(ca|in-course|in_course|test|continuous\s*assessment)\b", "IN_COURSE_ASSESSMENT"),
    (r"\b(ospe|steeplechase|practical)\b", "PRACTICAL_OSPE"),
    (r"\b(200l|300l|400l)\b", "MBBS_LEVEL_EXAM"),
]


def is_pdf_password_protected(pdf_path: str | Path) -> bool:
    """Checks whether a PDF file is encrypted and password-protected."""
    path = Path(pdf_path)
    if not path.exists():
        return False
    try:
        doc = pymupdf.open(str(path))
        if doc.is_encrypted:
            is_locked = not doc.authenticate("")
            doc.close()
            return is_locked
        doc.close()
        return False
    except Exception:
        return False


def extract_filename_tokens(file_path: str | Path) -> dict[str, str]:
    """Tier 1: Parses filename tokens using heuristics and regex."""
    stem = Path(file_path).stem
    normalized_stem = re.sub(r"[-_.]+", " ", stem)
    tokens = normalized_stem.split()

    result = {
        "discipline": "UNKNOWN",
        "session": "UNKNOWN",
        "examiner": "UNKNOWN",
        "topic": "UNKNOWN",
        "exam_type": "UNKNOWN",
        "institution": "UNKNOWN",
    }

    # 1. Check Discipline
    for token in tokens:
        clean = token.lower()
        if clean in DISCIPLINE_MAP:
            result["discipline"] = DISCIPLINE_MAP[clean]
            break

    # 1b. Check Institution
    for token in tokens:
        clean = token.lower()
        if clean in COMMON_INSTITUTION_MAP:
            result["institution"] = COMMON_INSTITUTION_MAP[clean]
            break

    # 2. Check Session / Year (e.g. 2021 2022, 2021/2022, or 2022)
    session_match = re.search(r"\b(20\d{2}\s*(?:[/]|\s+)?\s*20\d{2}|20\d{2})\b", normalized_stem)
    if session_match:
        raw_sess = session_match.group(1).strip()
        parts = re.split(r"[\s/]+", raw_sess)
        if len(parts) == 2 and len(parts[0]) == 4 and len(parts[1]) == 4:
            result["session"] = f"{parts[0]}/{parts[1]}"
        elif len(parts) == 1 and len(parts[0]) == 4:
            yr = int(parts[0])
            result["session"] = f"{yr}/{yr + 1}"
        else:
            result["session"] = raw_sess

    # 3. Check Examiner (e.g. Prof Atiku, Dr Bello)
    examiner_match = re.search(r"\b(Prof\.?|Dr\.?)\s+([A-Za-z]+)", normalized_stem, re.IGNORECASE)
    if examiner_match:
        title = examiner_match.group(1).capitalize()
        name = examiner_match.group(2).capitalize()
        result["examiner"] = f"{title}. {name}"

    # 4. Check Exam Type
    stem_lower = normalized_stem.lower()
    for pattern, exam_type in EXAM_TYPE_PATTERNS:
        if re.search(pattern, stem_lower):
            result["exam_type"] = exam_type
            break


    # 5. Extract residual descriptive token as candidate topic
    topic_candidates = []
    for token in tokens:
        lower = token.lower()
        if (
            lower not in DISCIPLINE_MAP
            and not re.search(r"\b(20\d{2}|mbbs|prof|dr|paper|exam|p1|p2|seq|mcq|ospe)\b", lower)
            and len(token) > 2
        ):
            topic_candidates.append(token)

    if topic_candidates:
        result["topic"] = " ".join(topic_candidates)

    return result


def inspect_page_one_headers(pdf_path: str | Path) -> dict[str, str]:
    """Tier 2: Inspects Page 1 for printed metadata in the header block."""
    result: dict[str, str] = {}
    path = Path(pdf_path)
    if not path.exists():
        return result

    try:
        doc = pymupdf.open(str(path))
        if len(doc) == 0:
            return result

        page1 = doc[0]
        text = page1.get_text("text")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        header_text = " ".join(lines[:15]).lower()

        # Check for discipline
        for kw, disc in DISCIPLINE_MAP.items():
            if re.search(rf"\b{kw}\b", header_text):
                result["discipline"] = disc
                break

        # Check for session (e.g. 2021/2022 session)
        session_match = re.search(r"\b(20\d{2}\s*/\s*20\d{2})\b", header_text)
        if session_match:
            result["session"] = re.sub(r"\s+", "", session_match.group(1))

        # Check for examiner lines (e.g. Examiner: Prof. Gangadharan or Lecturer: Dr. X)
        examiner_match = re.search(
            r"(?:examiner|lecturer|course lecturer)[:\s]+(?:prof\.?|dr\.?)?\s*([A-Za-z\s]+)",
            header_text,
            re.IGNORECASE,
        )
        if examiner_match:
            cand = examiner_match.group(1).strip().split("  ")[0]
            if len(cand) < 40:
                result["examiner"] = cand.title()

        # Check for Paper Title
        for line in lines[:8]:
            if re.search(r"(paper\s+[i|v|x\d]+|2nd\s+mbbs|in-course)", line, re.IGNORECASE):
                result["paper_title"] = line
                break

        # Check for Institution
        for kw, inst in COMMON_INSTITUTION_MAP.items():
            if re.search(rf"\b{kw}\b", header_text, re.IGNORECASE):
                result["institution"] = inst
                break

        if "institution" not in result:
            univ_match = re.search(
                r"(?:university\s+of\s+[a-z]+|[a-z]+\s+university(?:\s+[a-z]+)?|college\s+of\s+(?:health|medical)\s+sciences)",
                header_text,
                re.IGNORECASE,
            )
            if univ_match:
                result["institution"] = univ_match.group(0).strip().title()

        doc.close()
    except Exception:
        pass

    return result


def run_stage0(pdf_path: str | Path) -> PreScanMetadata:
    """Executes Stage 0 inspection combining Tier 1 and Tier 2."""
    path = Path(pdf_path)
    if is_pdf_password_protected(path):
        return PreScanMetadata(
            source_file=path.name,
            confidence_tier="FALLBACK",
            exam_type="SKIPPED_PASSWORD_PROTECTED",
        )

    tier1 = extract_filename_tokens(pdf_path)
    tier2 = inspect_page_one_headers(pdf_path)

    # Merge results, letting Tier 2 refine Tier 1
    discipline = tier2.get("discipline") or tier1.get("discipline") or "UNKNOWN"
    session = tier2.get("session") or tier1.get("session") or "UNKNOWN"
    examiner = tier2.get("examiner") or tier1.get("examiner") or "UNKNOWN"
    topic = tier1.get("topic") or "UNKNOWN"
    exam_type = tier2.get("paper_title") or tier1.get("exam_type") or "UNKNOWN"
    institution = tier2.get("institution") or tier1.get("institution") or "UNKNOWN"

    confidence = "TIER_1_FILENAME"
    if tier2:
        confidence = "TIER_2_PAGE_INSPECTION"

    return PreScanMetadata(
        source_file=Path(pdf_path).name,
        discipline=discipline,  # type: ignore[arg-type]
        session=session,
        examiner=examiner,
        topic=topic,
        exam_type=exam_type,
        institution=institution,
        confidence_tier=confidence,
    )


scan_exam_file = run_stage0
