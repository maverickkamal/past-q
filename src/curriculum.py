"""Curriculum taxonomy loader, resolver, and context generator for preclinical levels and disciplines."""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import CURRICULUM_TAXONOMY_PATH


@lru_cache(maxsize=1)
def load_curriculum(file_path: Path | str | None = None) -> dict[str, Any]:
    """Loads and caches the curriculum taxonomy JSON data."""
    target_path = Path(file_path or CURRICULUM_TAXONOMY_PATH)
    if not target_path.exists():
        raise FileNotFoundError(f"Curriculum taxonomy file not found at: {target_path}")
    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_valid_levels() -> list[str]:
    """Returns the supported level identifiers, e.g. ['200L', '300L']."""
    curriculum = load_curriculum()
    return list(curriculum.get("levels", {}).keys())


def get_disciplines(level: str | None = None) -> list[str]:
    """Returns disciplines available under the given level, or across all levels."""
    curriculum = load_curriculum()
    levels = curriculum.get("levels", {})

    if level and level in levels:
        return list(levels[level].get("disciplines", {}).keys())

    all_disciplines: set[str] = set()
    for lvl_data in levels.values():
        all_disciplines.update(lvl_data.get("disciplines", {}).keys())
    return sorted(all_disciplines)


def get_courses(discipline: str | None = None, level: str | None = None) -> list[dict[str, Any]]:
    """Returns a list of course dictionaries matching the discipline and/or level."""
    curriculum = load_curriculum()
    levels = curriculum.get("levels", {})
    results: list[dict[str, Any]] = []

    for lvl_key, lvl_data in levels.items():
        if level and level != lvl_key:
            continue
        disciplines = lvl_data.get("disciplines", {})
        for disc_key, disc_data in disciplines.items():
            if discipline and discipline.lower() != disc_key.lower():
                continue
            for course in disc_data.get("courses", []):
                course_copy = dict(course)
                course_copy["level"] = lvl_key
                course_copy["discipline"] = disc_key
                results.append(course_copy)
    return results


def get_canonical_system_regions(discipline: str | None = None, level: str | None = None) -> list[str]:
    """Returns a sorted list of unique canonical system_region strings."""
    courses = get_courses(discipline=discipline, level=level)
    regions = {c["system_region"] for c in courses if "system_region" in c}
    return sorted(regions)


def resolve_system_region(raw_system: str, discipline: str | None = None, level: str | None = None) -> str:
    """Deterministically normalizes a raw system/region string to a canonical system_region.
    
    Checks exact matches, case-insensitive matches, aliases, course codes, and keyword containment.
    Falls back to cleaned raw_system if no canonical match is identified.
    """
    if not raw_system or not raw_system.strip():
        return "General"

    clean_raw = raw_system.strip()
    courses = get_courses(discipline=discipline, level=level)
    if not courses and (discipline or level):
        # Fallback to searching all courses if specific filter yielded nothing
        courses = get_courses()

    clean_raw_lower = clean_raw.lower()

    # 1. Exact or case-insensitive match on canonical system_region
    for course in courses:
        canonical_region = course.get("system_region", "")
        if clean_raw_lower == canonical_region.lower():
            return canonical_region

    # 2. Match against aliases
    for course in courses:
        aliases = course.get("aliases", [])
        for alias in aliases:
            if clean_raw_lower == alias.lower():
                return course.get("system_region", clean_raw)

    # 3. Match against course code (e.g. 'ANA 201a' -> 'General Anatomy and Upper Limb')
    for course in courses:
        code = course.get("course_code", "").lower()
        if clean_raw_lower == code or clean_raw_lower == code.replace(" ", ""):
            return course.get("system_region", clean_raw)

    # 4. Partial / Substring containment in canonical region or aliases
    for course in courses:
        canonical_region = course.get("system_region", "")
        canonical_lower = canonical_region.lower()
        if clean_raw_lower in canonical_lower or canonical_lower in clean_raw_lower:
            return canonical_region

        for alias in course.get("aliases", []):
            alias_lower = alias.lower()
            if clean_raw_lower in alias_lower or alias_lower in clean_raw_lower:
                return canonical_region

    # Fallback: return cleaned raw string
    return clean_raw


def resolve_course_code(raw_input: str, discipline: str | None = None, level: str | None = None) -> str | None:
    """Attempts to resolve a course_code (e.g. 'ANA 201a') from a system region, course name, or topic."""
    if not raw_input or not raw_input.strip():
        return None

    clean_input = raw_input.strip().lower()
    courses = get_courses(discipline=discipline, level=level)
    if not courses:
        courses = get_courses()

    # 1. Direct match on course_code
    for course in courses:
        code = course.get("course_code", "")
        if clean_input == code.lower() or clean_input == code.lower().replace(" ", ""):
            return code

    # 2. Match on system_region
    for course in courses:
        if clean_input == course.get("system_region", "").lower():
            return course.get("course_code")

    # 3. Match on canonical_name or alias
    for course in courses:
        if clean_input == course.get("canonical_name", "").lower():
            return course.get("course_code")
        for alias in course.get("aliases", []):
            if clean_input == alias.lower():
                return course.get("course_code")

    # 4. Containment match
    for course in courses:
        canonical_region = course.get("system_region", "").lower()
        if clean_input in canonical_region or canonical_region in clean_input:
            return course.get("course_code")

    return None


def get_taxonomy_prompt_context(discipline: str | None = None, level: str | None = None) -> str:
    """Builds a compact context block for agent prompts detailing canonical courses and system regions."""
    courses = get_courses(discipline=discipline, level=level)
    if not courses and (discipline or level):
        courses = get_courses(discipline=discipline)

    lines: list[str] = [
        "CANONICAL CURRICULUM TAXONOMY (Strictly map 'system_region' to one of the canonical names below, and include 'course_code' if clear):"
    ]

    for c in courses:
        code = c.get("course_code", "")
        name = c.get("canonical_name", "")
        region = c.get("system_region", "")
        lvl = c.get("level", "")
        topics_preview = ", ".join(c.get("core_topics", [])[:5])
        lines.append(
            f"- [{code}] System Region: \"{region}\" (Course: {name} [{lvl}])\n  Key Topics: {topics_preview}"
        )

    return "\n".join(lines)
