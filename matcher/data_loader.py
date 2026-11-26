"""Dataset loading, skill normalization, and text skill extraction helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SKILL_ALIASES: dict[str, str] = {
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "k8s": "kubernetes",
    "js": "javascript",
    "ts": "typescript",
    "tf": "tensorflow",
    "postgres": "postgresql",
    "ml": "machine learning",
    "dl": "deep learning",
    "rest": "rest api",
    "rest apis": "rest api",
    "restful api": "rest api",
    "cicd": "ci/cd",
    "ci cd": "ci/cd",
    "github actions": "ci/cd",
    "powerbi": "power bi",
    "huggingface": "hugging face",
    "nodejs": "node.js",
    "nextjs": "next.js",
    "tailwindcss": "tailwind css",
    "ab testing": "a/b testing",
    "spark": "apache spark",
    "pyspark": "apache spark",
    "data viz": "data visualization",
    "visualization": "data visualization",
    "unit tests": "unit testing",
    "pytest": "unit testing",
}


@dataclass(frozen=True)
class RoleRecord:
    """Normalized representation of a target technical job role."""

    role_id: str
    role_title: str
    domain: str
    experience_level: str
    core_skills: tuple[str, ...]
    bonus_skills: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class SkillResource:
    """Learning time and study focus metadata for a single skill."""

    skill: str
    category: str
    est_hours: int
    learning_focus: str


@dataclass(frozen=True)
class CandidateSample:
    """Sample student/candidate profile loaded from CSV."""

    resume_id: str
    candidate_name: str
    target_role_id: str
    current_background: str
    resume_text: str


def normalize_skill(skill: str) -> str:
    """Normalize whitespace, casing, and common industry aliases for a skill."""
    cleaned = re.sub(r"\s+", " ", skill.strip().lower())
    return SKILL_ALIASES.get(cleaned, cleaned)


def parse_skill_list(raw_value: object) -> tuple[str, ...]:
    """Parse a semicolon-separated cell of skills into a deduplicated tuple."""
    if raw_value is None:
        return ()
    seen: set[str] = set()
    parsed: list[str] = []
    for item in str(raw_value).split(";"):
        norm = normalize_skill(item)
        if norm and norm not in seen:
            seen.add(norm)
            parsed.append(norm)
    return tuple(parsed)


def load_datasets(
    data_dir: str | Path | None = None,
) -> tuple[list[RoleRecord], dict[str, SkillResource], list[CandidateSample]]:
    """Load job roles, learning resources, and sample candidate resumes via stdlib csv."""
    base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    roles: list[RoleRecord] = []
    with (base / "job_roles.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            roles.append(
                RoleRecord(
                    role_id=row["role_id"].strip(),
                    role_title=row["role_title"].strip(),
                    domain=row["domain"].strip(),
                    experience_level=row["experience_level"].strip(),
                    core_skills=parse_skill_list(row["core_skills"]),
                    bonus_skills=parse_skill_list(row["bonus_skills"]),
                    description=row["description"].strip(),
                )
            )

    resources: dict[str, SkillResource] = {}
    with (base / "skill_learning_resources.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            norm_skill = normalize_skill(row["skill"])
            resources[norm_skill] = SkillResource(
                skill=norm_skill,
                category=row["category"].strip(),
                est_hours=max(1, int(row["est_hours"])),
                learning_focus=row["learning_focus"].strip(),
            )

    resumes: list[CandidateSample] = []
    with (base / "sample_resumes.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            resumes.append(
                CandidateSample(
                    resume_id=row["resume_id"].strip(),
                    candidate_name=row["candidate_name"].strip(),
                    target_role_id=row["target_role_id"].strip(),
                    current_background=row["current_background"].strip(),
                    resume_text=row["resume_text"].strip(),
                )
            )

    return roles, resources, resumes


def build_skill_vocabulary(roles: Iterable[RoleRecord]) -> set[str]:
    """Collect all canonical skill tokens across core and bonus role fields."""
    vocab: set[str] = set()
    for role in roles:
        vocab.update(role.core_skills)
        vocab.update(role.bonus_skills)
    return vocab


def extract_skills_from_text(
    text: str,
    vocabulary: Iterable[str],
    extra_skills: Iterable[str] = (),
) -> list[str]:
    """Extract canonical skills from free-form resume text plus explicit selections."""
    lowered = f" {text.lower()} "
    vocab_set = {normalize_skill(s) for s in vocabulary}
    matched: set[str] = set()

    candidates: list[tuple[str, str]] = [(term, term) for term in vocab_set]
    for alias, canonical in SKILL_ALIASES.items():
        if canonical in vocab_set:
            candidates.append((alias, canonical))

    for surface, canonical in candidates:
        pattern = r"(?<![a-z0-9])" + re.escape(surface) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            matched.add(canonical)

    for skill in extra_skills:
        norm = normalize_skill(skill)
        if norm:
            matched.add(norm)

    return sorted(matched)
