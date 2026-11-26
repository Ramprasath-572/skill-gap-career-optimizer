"""Unit tests for skill extraction, matching, clustering, and roadmap optimization."""

from __future__ import annotations

import pytest

from matcher import (
    SkillGapMatcher,
    extract_skills_from_text,
    load_datasets,
    normalize_skill,
    optimize_learning_path,
)


@pytest.fixture(scope="module")
def datasets():
    return load_datasets()


def test_normalize_aliases():
    assert normalize_skill("  K8S ") == "kubernetes"
    assert normalize_skill("sklearn") == "scikit-learn"
    assert normalize_skill("PySpark") == "apache spark"


def test_extract_skills_boundary_safe(datasets):
    roles, _, _ = datasets
    matcher = SkillGapMatcher(roles)
    extracted = extract_skills_from_text(
        "Worked on Python, sklearn, k8s, and CI/CD pipelines. Not Java.",
        matcher.vocabulary,
    )
    assert "python" in extracted
    assert "scikit-learn" in extracted
    assert "kubernetes" in extracted
    assert "ci/cd" in extracted
    assert "r" not in extracted


def test_ranking_prefers_relevant_role(datasets):
    roles, _, _ = datasets
    matcher = SkillGapMatcher(roles)
    _, results = matcher.rank_roles(
        "Experienced in Python, pandas, numpy, scikit-learn, machine learning, docker, git, model evaluation."
    )
    top_titles = [r.role_title for r in results[:3]]
    assert "Machine Learning Engineer" in top_titles
    assert results[0].readiness_score >= results[1].readiness_score


def test_knapsack_respects_budget_and_improves_readiness(datasets):
    roles, resources, _ = datasets
    matcher = SkillGapMatcher(roles)
    _, results = matcher.rank_roles("Beginner with Python and git.")
    ml_role = next(r for r in results if r.role_id == "R01")

    plan = optimize_learning_path(
        role_match=ml_role,
        resources=resources,
        skill_demand=matcher.skill_demand,
        budget_hours=30,
        hours_per_week=10,
    )
    assert plan.total_allocated_hours <= 30
    assert len(plan.items) >= 1
    assert plan.projected_readiness > plan.current_readiness
    assert plan.projected_readiness >= plan.naive_projected_readiness


def test_role_landscape_contains_candidate(datasets):
    roles, _, _ = datasets
    matcher = SkillGapMatcher(roles)
    points = matcher.compute_role_landscape("Python SQL Docker")
    assert len(points) == len(roles) + 1
    assert any(p["label"] == "Your Current Profile" for p in points)
