"""Interactive Streamlit application for the Skill-Gap & Career Path Optimizer."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from matcher import (
    LearningPlan,
    RoleMatchResult,
    SkillGapMatcher,
    load_datasets,
    optimize_learning_path,
)

st.set_page_config(
    page_title="Skill-Gap & Career Path Optimizer",
    page_icon="🧭",
    layout="wide",
)


@st.cache_data
def get_data():
    return load_datasets()


def _format_skill_badges(skills: tuple[str, ...], empty_text: str = "None") -> str:
    if not skills:
        return empty_text
    return ", ".join(f"`{s}`" for s in skills)


def _plan_to_markdown(plan: LearningPlan, candidate_skills: list[str]) -> str:
    lines = [
        f"# Personalized Learning Roadmap: {plan.target_role_title}",
        "",
        f"- **Extracted / Current Skills:** {', '.join(candidate_skills) or 'None'}",
        f"- **Study Budget:** {plan.total_allocated_hours} / {plan.budget_hours} hours ({plan.hours_per_week} hrs/week)",
        f"- **Current Readiness:** {plan.current_readiness}%",
        f"- **Projected Readiness (Optimized):** {plan.projected_readiness}% (vs. {plan.naive_projected_readiness}% naive baseline)",
        "",
        "## Recommended Schedule",
        "",
        "| Week | Skill | Priority | Est. Hours | Impact | Focus |",
        "| :--- | :--- | :--- | :---: | :---: | :--- |",
    ]
    for item in plan.items:
        lines.append(
            f"| Week {item.week} | {item.skill} | {item.priority} | {item.est_hours}h | {item.impact_score} | {item.learning_focus} |"
        )
    if plan.deferred_skills:
        lines.extend(["", f"**Deferred (beyond current hour budget):** {', '.join(plan.deferred_skills)}"])
    return "\n".join(lines)


def main() -> None:
    roles, resources, resumes = get_data()

    st.title("🧭 Skill-Gap & Career Path Optimizer")
    st.caption(
        "Real-life prescriptive ML problem solver: combines **TF-IDF Semantic Matching**, "
        "**K-Means + PCA Role Clustering**, and **0/1 Knapsack Study-Plan Optimization** "
        "(no prediction or detection)."
    )

    with st.sidebar:
        st.header("1. Candidate Profile")
        input_mode = st.radio(
            "Choose profile input",
            ["Sample Student Profile", "Custom Resume / Skills"],
        )

        default_target_id: str | None = None
        if input_mode == "Sample Student Profile":
            profile_options = {
                f"{sample.candidate_name} — {sample.current_background}": sample
                for sample in resumes
            }
            chosen_label = st.selectbox("Select sample profile", list(profile_options.keys()))
            chosen_sample = profile_options[chosen_label]
            resume_text = st.text_area("Resume / Bio text", value=chosen_sample.resume_text, height=140)
            default_target_id = chosen_sample.target_role_id
        else:
            uploaded = st.file_uploader("Upload resume (.txt)", type=["txt"])
            default_text = (
                uploaded.read().decode("utf-8", errors="ignore")
                if uploaded is not None
                else "CS undergraduate skilled in Python, SQL, pandas, git, and basic data visualization."
            )
            resume_text = st.text_area("Paste your resume or skills summary", value=default_text, height=140)

        base_matcher = SkillGapMatcher(roles)
        extra_skills = st.multiselect(
            "Add extra skills manually (optional)",
            options=base_matcher.get_all_known_skills(),
        )

        st.header("2. Optimization Settings")
        skill_weight = st.slider(
            "Structured skill vs. TF-IDF text weight",
            min_value=0.30,
            max_value=0.90,
            value=0.65,
            step=0.05,
        )
        budget_hours = st.slider("Total study hours budget", min_value=10, max_value=120, value=40, step=5)
        hours_per_week = st.slider("Study hours per week", min_value=5, max_value=30, value=10, step=1)

    matcher = SkillGapMatcher(roles, skill_weight=skill_weight)
    candidate_skills, ranked_roles = matcher.rank_roles(resume_text, selected_skills=extra_skills)

    role_by_id = {r.role_id: r for r in ranked_roles}
    role_labels = [f"{r.role_title} ({r.readiness_score}% match)" for r in ranked_roles]
    default_idx = 0
    if default_target_id and default_target_id in role_by_id:
        default_idx = ranked_roles.index(role_by_id[default_target_id])

    selected_role_idx = st.selectbox(
        "🎯 Target Job Role to Optimize For",
        options=range(len(ranked_roles)),
        index=default_idx,
        format_func=lambda idx: role_labels[idx],
    )
    target_role: RoleMatchResult = ranked_roles[selected_role_idx]

    plan = optimize_learning_path(
        role_match=target_role,
        resources=resources,
        skill_demand=matcher.skill_demand,
        budget_hours=budget_hours,
        hours_per_week=hours_per_week,
    )

    st.info(
        f"**Detected Profile Skills ({len(candidate_skills)}):** "
        + (", ".join(candidate_skills) if candidate_skills else "None detected yet — add text or select skills in the sidebar.")
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Readiness", f"{plan.current_readiness}%")
    col2.metric(
        "Projected Readiness",
        f"{plan.projected_readiness}%",
        delta=f"+{round(plan.projected_readiness - plan.current_readiness, 1)}% (vs {plan.naive_projected_readiness}% naive)",
    )
    total_core = len(target_role.matched_core) + len(target_role.missing_core)
    col3.metric("Core Skills Matched", f"{len(target_role.matched_core)} / {total_core}")
    col4.metric("Planned Study Hours", f"{plan.total_allocated_hours} / {plan.budget_hours} hrs")

    tab_rank, tab_plan, tab_map = st.tabs(
        ["📊 Role Match & Skill Gaps", "🗓️ Optimized Study Roadmap", "🗺️ 2D Career Landscape (Clustering)"]
    )

    with tab_rank:
        top_df = pd.DataFrame(
            [
                {
                    "Role": r.role_title,
                    "Domain": r.domain,
                    "Readiness (%)": r.readiness_score,
                    "Core Coverage (%)": r.core_coverage,
                    "TF-IDF Similarity (%)": r.tfidf_similarity,
                }
                for r in ranked_roles[:10]
            ]
        )
        fig_bar = px.bar(
            top_df.iloc[::-1],
            x="Readiness (%)",
            y="Role",
            color="Domain",
            orientation="h",
            title="Top 10 Role Matches by Hybrid Readiness Score",
            hover_data=["Core Coverage (%)", "TF-IDF Similarity (%)"],
        )
        fig_bar.update_layout(height=420, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.subheader("✅ Matched Skills")
            st.markdown(f"**Core:** {_format_skill_badges(target_role.matched_core)}")
            st.markdown(f"**Bonus:** {_format_skill_badges(target_role.matched_bonus)}")
        with right:
            st.subheader("🧩 Missing Skills (Gap)")
            st.markdown(f"**Missing Core:** {_format_skill_badges(target_role.missing_core)}")
            st.markdown(f"**Missing Bonus:** {_format_skill_badges(target_role.missing_bonus)}")

    with tab_plan:
        st.subheader(f"Optimal {plan.budget_hours}-Hour Roadmap for {plan.target_role_title}")
        st.write(
            "Using **0/1 Knapsack Dynamic Programming**, the solver selects missing skills that "
            "maximize readiness gain and cross-role transferability without exceeding your study budget."
        )
        if plan.items:
            plan_df = pd.DataFrame(
                [
                    {
                        "Week": f"Week {item.week}",
                        "Skill": item.skill,
                        "Priority": item.priority,
                        "Category": item.category,
                        "Est. Hours": item.est_hours,
                        "Impact Score": item.impact_score,
                        "ROI / Hour": item.roi_per_hour,
                        "Learning Focus": item.learning_focus,
                    }
                    for item in plan.items
                ]
            )
            st.dataframe(plan_df, use_container_width=True, hide_index=True)

            fig_road = px.bar(
                plan_df,
                x="Skill",
                y="Est. Hours",
                color="Priority",
                facet_col="Week",
                title="Weekly Study Hour Allocation",
                hover_data=["Impact Score", "ROI / Hour"],
            )
            fig_road.update_layout(height=360, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig_road, use_container_width=True)

            dl_col1, dl_col2 = st.columns(2)
            dl_col1.download_button(
                "⬇️ Download Roadmap (Markdown)",
                data=_plan_to_markdown(plan, candidate_skills),
                file_name="learning_roadmap.md",
                mime="text/markdown",
            )
            dl_col2.download_button(
                "⬇️ Download Roadmap (CSV)",
                data=plan_df.to_csv(index=False),
                file_name="learning_roadmap.csv",
                mime="text/csv",
            )
        else:
            st.success("You already cover the target skills or need a higher hour budget to fit the next skill.")

        if plan.deferred_skills:
            st.caption(
                f"Deferred beyond {plan.budget_hours}h budget (total missing = {plan.total_missing_hours}h): "
                + ", ".join(plan.deferred_skills)
            )

    with tab_map:
        st.subheader("Unsupervised Role Clusters & Candidate Position (K-Means + PCA)")
        landscape_df = pd.DataFrame(matcher.compute_role_landscape(resume_text, selected_skills=extra_skills))
        fig_map = px.scatter(
            landscape_df,
            x="x",
            y="y",
            color="cluster",
            symbol="kind",
            text="label",
            hover_data=["domain"],
            title="2D Career Skill-Space Projection",
        )
        fig_map.update_traces(textposition="top center", marker=dict(size=12))
        fig_map.update_layout(height=500, xaxis_title="PCA Component 1", yaxis_title="PCA Component 2")
        st.plotly_chart(fig_map, use_container_width=True)


if __name__ == "__main__":
    main()
