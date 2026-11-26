"""Time-budgeted learning roadmap optimizer using 0/1 Knapsack dynamic programming."""

from __future__ import annotations

from dataclasses import dataclass
import math

from matcher.data_loader import SkillResource, normalize_skill
from matcher.engine import RoleMatchResult


@dataclass(frozen=True)
class RoadmapItem:
    """Single recommended skill module within an optimized study plan."""

    skill: str
    priority: str
    category: str
    est_hours: int
    impact_score: float
    roi_per_hour: float
    week: int
    learning_focus: str


@dataclass(frozen=True)
class LearningPlan:
    """Optimized roadmap output comparing baseline readiness vs. projected readiness."""

    target_role_title: str
    budget_hours: int
    hours_per_week: int
    total_allocated_hours: int
    total_missing_hours: int
    current_readiness: float
    projected_readiness: float
    naive_projected_readiness: float
    items: tuple[RoadmapItem, ...]
    deferred_skills: tuple[str, ...]


def _lookup_skill_metadata(
    skill: str,
    resources: dict[str, SkillResource],
) -> tuple[str, int, str]:
    norm = normalize_skill(skill)
    info = resources.get(norm)
    if info:
        return info.category, info.est_hours, info.learning_focus
    return ("General Engineering", 10, f"Hands-on practice and mini-project using {norm}")


def _simulate_readiness_gain(
    role_match: RoleMatchResult,
    acquired_skills: set[str],
    skill_weight: float = 0.65,
    core_importance: float = 2.0,
) -> float:
    """Estimate updated readiness % after acquiring a subset of missing skills."""
    total_core = len(role_match.matched_core) + len(role_match.missing_core)
    total_bonus = len(role_match.matched_bonus) + len(role_match.missing_bonus)

    new_core = len(role_match.matched_core) + sum(
        1 for s in role_match.missing_core if s in acquired_skills
    )
    new_bonus = len(role_match.matched_bonus) + sum(
        1 for s in role_match.missing_bonus if s in acquired_skills
    )

    total_weight = core_importance * total_core + total_bonus
    old_weight = core_importance * len(role_match.matched_core) + len(role_match.matched_bonus)
    new_weight = core_importance * new_core + new_bonus

    if total_weight <= 0:
        return role_match.readiness_score

    structured_gain = ((new_weight - old_weight) / total_weight) * 100.0
    semantic_lift = ((new_weight - old_weight) / total_weight) * (100.0 - role_match.tfidf_similarity) * 0.6
    delta = skill_weight * structured_gain + (1.0 - skill_weight) * semantic_lift
    return round(min(99.5, role_match.readiness_score + delta), 1)


def optimize_learning_path(
    role_match: RoleMatchResult,
    resources: dict[str, SkillResource],
    skill_demand: dict[str, int],
    budget_hours: int = 40,
    hours_per_week: int = 10,
) -> LearningPlan:
    """Select the highest-impact missing skills fitting within budget_hours via 0/1 Knapsack."""
    budget_hours = max(0, int(budget_hours))
    hours_per_week = max(1, int(hours_per_week))

    candidates: list[dict[str, object]] = []
    for priority, skills, base_value in (
        ("Core", role_match.missing_core, 200),
        ("Bonus", role_match.missing_bonus, 90),
    ):
        for skill in skills:
            category, est_hours, focus = _lookup_skill_metadata(skill, resources)
            demand_bonus = min(40, skill_demand.get(skill, 1) * 5)
            impact = base_value + demand_bonus
            candidates.append(
                {
                    "skill": skill,
                    "priority": priority,
                    "category": category,
                    "est_hours": est_hours,
                    "impact": impact,
                    "focus": focus,
                }
            )

    total_missing_hours = sum(int(c["est_hours"]) for c in candidates)
    if not candidates or budget_hours == 0:
        return LearningPlan(
            target_role_title=role_match.role_title,
            budget_hours=budget_hours,
            hours_per_week=hours_per_week,
            total_allocated_hours=0,
            total_missing_hours=total_missing_hours,
            current_readiness=role_match.readiness_score,
            projected_readiness=role_match.readiness_score,
            naive_projected_readiness=role_match.readiness_score,
            items=(),
            deferred_skills=tuple(str(c["skill"]) for c in candidates),
        )

    n = len(candidates)
    dp = [[0] * (budget_hours + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost = int(candidates[i - 1]["est_hours"])
        val = int(candidates[i - 1]["impact"])
        for h in range(budget_hours + 1):
            dp[i][h] = dp[i - 1][h]
            if cost <= h and dp[i - 1][h - cost] + val > dp[i][h]:
                dp[i][h] = dp[i - 1][h - cost] + val

    chosen_indices: list[int] = []
    remaining = budget_hours
    for i in range(n, 0, -1):
        if dp[i][remaining] != dp[i - 1][remaining]:
            chosen_indices.append(i - 1)
            remaining -= int(candidates[i - 1]["est_hours"])
    chosen_indices.reverse()

    chosen_set = set(chosen_indices)
    chosen_raw = [candidates[idx] for idx in chosen_indices]
    chosen_raw.sort(
        key=lambda c: (0 if c["priority"] == "Core" else 1, -int(c["impact"]) / int(c["est_hours"]))
    )

    cumulative_hours = 0
    roadmap_items: list[RoadmapItem] = []
    for c in chosen_raw:
        hrs = int(c["est_hours"])
        cumulative_hours += hrs
        week_num = max(1, math.ceil(cumulative_hours / hours_per_week))
        impact_score = round(int(c["impact"]) / 10.0, 1)
        roadmap_items.append(
            RoadmapItem(
                skill=str(c["skill"]),
                priority=str(c["priority"]),
                category=str(c["category"]),
                est_hours=hrs,
                impact_score=impact_score,
                roi_per_hour=round(impact_score / hrs, 2),
                week=week_num,
                learning_focus=str(c["focus"]),
            )
        )

    naive_hours = 0
    naive_skills: set[str] = set()
    for c in candidates:
        hrs = int(c["est_hours"])
        if naive_hours + hrs <= budget_hours:
            naive_hours += hrs
            naive_skills.add(str(c["skill"]))

    acquired = {item.skill for item in roadmap_items}
    deferred = tuple(
        str(candidates[i]["skill"]) for i in range(n) if i not in chosen_set
    )

    return LearningPlan(
        target_role_title=role_match.role_title,
        budget_hours=budget_hours,
        hours_per_week=hours_per_week,
        total_allocated_hours=cumulative_hours,
        total_missing_hours=total_missing_hours,
        current_readiness=role_match.readiness_score,
        projected_readiness=_simulate_readiness_gain(role_match, acquired),
        naive_projected_readiness=_simulate_readiness_gain(role_match, naive_skills),
        items=tuple(roadmap_items),
        deferred_skills=deferred,
    )
