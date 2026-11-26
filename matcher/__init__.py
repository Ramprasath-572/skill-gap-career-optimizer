"""Core package for the Skill-Gap & Career Path Optimizer."""

from matcher.data_loader import (
    SKILL_ALIASES,
    extract_skills_from_text,
    load_datasets,
    normalize_skill,
)
from matcher.engine import RoleMatchResult, SkillGapMatcher
from matcher.optimizer import LearningPlan, RoadmapItem, optimize_learning_path

__all__ = [
    "SKILL_ALIASES",
    "LearningPlan",
    "RoadmapItem",
    "RoleMatchResult",
    "SkillGapMatcher",
    "extract_skills_from_text",
    "load_datasets",
    "normalize_skill",
    "optimize_learning_path",
]
