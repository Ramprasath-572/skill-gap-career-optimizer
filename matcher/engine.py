"""Pure-Python TF-IDF matching, skill-gap analysis, K-Means clustering, and 2D PCA."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Iterable

from matcher.data_loader import (
    RoleRecord,
    build_skill_vocabulary,
    extract_skills_from_text,
    normalize_skill,
)

STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "build", "by", "for", "from",
        "in", "including", "into", "is", "of", "on", "or", "that", "the", "to",
        "using", "with", "work", "worked",
    }
)


@dataclass(frozen=True)
class RoleMatchResult:
    """Structured readiness evaluation for one target job role."""

    role_id: str
    role_title: str
    domain: str
    experience_level: str
    cluster_id: int
    readiness_score: float
    tfidf_similarity: float
    core_coverage: float
    bonus_coverage: float
    matched_core: tuple[str, ...]
    missing_core: tuple[str, ...]
    matched_bonus: tuple[str, ...]
    missing_bonus: tuple[str, ...]


def _tokenize_ngrams(text: str) -> list[str]:
    """Tokenize text into unigrams and bigrams while preserving tech terms like c++ or ci/cd."""
    raw_tokens = re.findall(r"[a-z0-9][a-z0-9+.#/-]*", text.lower())
    tokens = [t for t in raw_tokens if t not in STOP_WORDS and len(t) > 1 or t == "r"]
    bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
    return tokens + bigrams


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        return [0.0] * len(vec)
    return [v / norm for v in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _sq_distance(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _kmeans_cluster(vectors: list[list[float]], k: int, max_iter: int = 30) -> list[int]:
    """Deterministic farthest-first K-Means clustering on L2-normalized TF-IDF vectors."""
    n = len(vectors)
    if n == 0:
        return []
    k = max(2, min(k, n))
    dim = len(vectors[0])

    # Farthest-first deterministic seeding for reproducible clusters.
    centroids: list[list[float]] = [list(vectors[0])]
    for _ in range(1, k):
        best_idx = max(
            range(n),
            key=lambda i: min(_sq_distance(vectors[i], c) for c in centroids),
        )
        centroids.append(list(vectors[best_idx]))

    assignments = [0] * n
    for _ in range(max_iter):
        changed = False
        for i, vec in enumerate(vectors):
            nearest = min(range(k), key=lambda c_idx: _sq_distance(vec, centroids[c_idx]))
            if nearest != assignments[i]:
                assignments[i] = nearest
                changed = True

        new_centroids = [[0.0] * dim for _ in range(k)]
        counts = [0] * k
        for vec, cid in zip(vectors, assignments):
            counts[cid] += 1
            for d in range(dim):
                new_centroids[cid][d] += vec[d]

        for cid in range(k):
            if counts[cid] > 0:
                centroids[cid] = _l2_normalize([val / counts[cid] for val in new_centroids[cid]])
        if not changed:
            break

    return assignments


def _power_iteration_top_component(
    centered_rows: list[list[float]],
    iterations: int = 40,
) -> list[float]:
    """Compute top principal axis of centered data without forming a huge covariance matrix."""
    dim = len(centered_rows[0])
    # Seed deterministically from feature variance so zero-variance axes are skipped.
    v = [sum(abs(row[j]) for row in centered_rows) + 1e-6 for j in range(dim)]
    v = _l2_normalize(v)

    for _ in range(iterations):
        # Compute X^T (X v) directly in O(n * d) time.
        projections = [_dot(row, v) for row in centered_rows]
        next_v = [0.0] * dim
        for row, proj in zip(centered_rows, projections):
            for j in range(dim):
                next_v[j] += row[j] * proj
        v = _l2_normalize(next_v)
    return v


def _fit_pca_2d(vectors: list[list[float]]) -> tuple[list[float], list[float], list[float]]:
    """Return mean vector and first 2 principal axes via power iteration + deflation."""
    n = len(vectors)
    dim = len(vectors[0])
    mean = [sum(row[j] for row in vectors) / n for j in range(dim)]
    centered = [[row[j] - mean[j] for j in range(dim)] for row in vectors]

    axis1 = _power_iteration_top_component(centered)
    deflated = []
    for row in centered:
        proj = _dot(row, axis1)
        deflated.append([row[j] - proj * axis1[j] for j in range(dim)])
    axis2 = _power_iteration_top_component(deflated)
    return mean, axis1, axis2


def _project_2d(
    vector: list[float],
    mean: list[float],
    axis1: list[float],
    axis2: list[float],
) -> tuple[float, float]:
    centered = [v - m for v, m in zip(vector, mean)]
    return round(_dot(centered, axis1), 4), round(_dot(centered, axis2), 4)


class SkillGapMatcher:
    """Matches a candidate profile against job roles using TF-IDF, clustering, and skill graphs."""

    def __init__(
        self,
        roles: list[RoleRecord],
        n_clusters: int = 5,
        skill_weight: float = 0.65,
        core_importance: float = 2.0,
    ) -> None:
        self.roles = list(roles)
        self.vocabulary = build_skill_vocabulary(self.roles)
        self.skill_weight = skill_weight
        self.core_importance = core_importance

        role_docs = [self._build_role_doc(role) for role in self.roles]
        tokenized_docs = [_tokenize_ngrams(doc) for doc in role_docs]

        # Build vocabulary and smooth IDF weights across the role corpus.
        n_docs = len(tokenized_docs)
        df_counts: Counter[str] = Counter()
        for toks in tokenized_docs:
            df_counts.update(set(toks))

        self._term_to_idx = {term: idx for idx, term in enumerate(sorted(df_counts))}
        self._idf = {
            term: math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
            for term, df in df_counts.items()
        }

        self.role_vectors = [self._vectorize_tokens(toks) for toks in tokenized_docs]
        self.cluster_ids = _kmeans_cluster(self.role_vectors, k=n_clusters)
        self._pca_mean, self._pca_axis1, self._pca_axis2 = _fit_pca_2d(self.role_vectors)
        self.role_coords = [
            _project_2d(vec, self._pca_mean, self._pca_axis1, self._pca_axis2)
            for vec in self.role_vectors
        ]
        self.skill_demand = self._compute_skill_demand()

    @staticmethod
    def _build_role_doc(role: RoleRecord) -> str:
        core_repeat = " ".join(role.core_skills * 2)
        bonus = " ".join(role.bonus_skills)
        return f"{role.role_title} {role.domain} {core_repeat} {bonus} {role.description}"

    def _vectorize_tokens(self, tokens: list[str]) -> list[float]:
        vec = [0.0] * len(self._term_to_idx)
        counts = Counter(tokens)
        for term, tf in counts.items():
            idx = self._term_to_idx.get(term)
            if idx is not None:
                sublinear_tf = 1.0 + math.log(tf)
                vec[idx] = sublinear_tf * self._idf[term]
        return _l2_normalize(vec)

    def _vectorize_text(self, text: str) -> list[float]:
        return self._vectorize_tokens(_tokenize_ngrams(text))

    def _compute_skill_demand(self) -> dict[str, int]:
        demand: dict[str, int] = {}
        for role in self.roles:
            for skill in set(role.core_skills) | set(role.bonus_skills):
                demand[skill] = demand.get(skill, 0) + 1
        return demand

    def rank_roles(
        self,
        resume_text: str,
        selected_skills: Iterable[str] = (),
    ) -> tuple[list[str], list[RoleMatchResult]]:
        """Score and rank all roles from highest to lowest candidate readiness."""
        candidate_skills = extract_skills_from_text(
            resume_text, self.vocabulary, extra_skills=selected_skills
        )
        candidate_set = set(candidate_skills)

        candidate_doc = f"{resume_text} {' '.join(candidate_skills * 2)}".strip()
        cand_vec = self._vectorize_text(candidate_doc) if candidate_doc else [0.0] * len(self._term_to_idx)

        results: list[RoleMatchResult] = []
        for idx, role in enumerate(self.roles):
            matched_core = tuple(s for s in role.core_skills if s in candidate_set)
            missing_core = tuple(s for s in role.core_skills if s not in candidate_set)
            matched_bonus = tuple(s for s in role.bonus_skills if s in candidate_set)
            missing_bonus = tuple(s for s in role.bonus_skills if s not in candidate_set)

            core_cov = len(matched_core) / len(role.core_skills) if role.core_skills else 1.0
            bonus_cov = len(matched_bonus) / len(role.bonus_skills) if role.bonus_skills else 0.0

            total_weight = self.core_importance * len(role.core_skills) + len(role.bonus_skills)
            earned_weight = self.core_importance * len(matched_core) + len(matched_bonus)
            structured_score = earned_weight / total_weight if total_weight > 0 else 0.0

            tfidf_sim = max(0.0, min(1.0, _dot(cand_vec, self.role_vectors[idx])))
            blended = (
                self.skill_weight * structured_score
                + (1.0 - self.skill_weight) * tfidf_sim
            )

            results.append(
                RoleMatchResult(
                    role_id=role.role_id,
                    role_title=role.role_title,
                    domain=role.domain,
                    experience_level=role.experience_level,
                    cluster_id=self.cluster_ids[idx],
                    readiness_score=round(blended * 100.0, 1),
                    tfidf_similarity=round(tfidf_sim * 100.0, 1),
                    core_coverage=round(core_cov * 100.0, 1),
                    bonus_coverage=round(bonus_cov * 100.0, 1),
                    matched_core=matched_core,
                    missing_core=missing_core,
                    matched_bonus=matched_bonus,
                    missing_bonus=missing_bonus,
                )
            )

        results.sort(key=lambda r: (r.readiness_score, r.core_coverage), reverse=True)
        return candidate_skills, results

    def compute_role_landscape(
        self,
        resume_text: str,
        selected_skills: Iterable[str] = (),
    ) -> list[dict[str, object]]:
        """Return 2D PCA coordinates for all roles plus the candidate profile."""
        candidate_skills = extract_skills_from_text(
            resume_text, self.vocabulary, extra_skills=selected_skills
        )
        candidate_doc = f"{resume_text} {' '.join(candidate_skills * 2)}".strip() or "python"
        cand_vec = self._vectorize_text(candidate_doc)
        cand_x, cand_y = _project_2d(cand_vec, self._pca_mean, self._pca_axis1, self._pca_axis2)

        points: list[dict[str, object]] = []
        for role, cid, (rx, ry) in zip(self.roles, self.cluster_ids, self.role_coords):
            points.append(
                {
                    "label": role.role_title,
                    "domain": role.domain,
                    "cluster": f"Cluster {cid + 1}",
                    "x": rx,
                    "y": ry,
                    "kind": "Job Role",
                }
            )
        points.append(
            {
                "label": "Your Current Profile",
                "domain": "Candidate",
                "cluster": "Your Profile",
                "x": cand_x,
                "y": cand_y,
                "kind": "Candidate",
            }
        )
        return points

    def get_all_known_skills(self) -> list[str]:
        """Return sorted list of all canonical skills known to the matcher."""
        return sorted({normalize_skill(s) for s in self.vocabulary})
