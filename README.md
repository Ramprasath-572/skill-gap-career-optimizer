# Skill-Gap & Career Path Optimizer

A **prescriptive Machine Learning mini-project** that solves a real-life problem: helping students and job seekers figure out **which tech roles fit them best** and **which missing skills to learn first under a limited time budget**.

Unlike prediction or detection projects, this system focuses on **semantic matching, unsupervised clustering, and constrained optimization**.

---

## Key ML & Problem-Solving Features

1. **Hybrid TF-IDF + Skill-Graph Matching (`matcher/engine.py`)**
   - Implements unigram + bigram TF-IDF vectorization (sublinear TF, smooth IDF) and cosine similarity blended with weighted structured skill coverage (`core_skills` weighted `2x` over `bonus_skills`).
   - Includes alias normalization (`sklearn` → `scikit-learn`, `k8s` → `kubernetes`, `pyspark` → `apache spark`, etc.) in `matcher/data_loader.py`.
2. **Unsupervised Career Landscape Mapping (`matcher/engine.py`)**
   - Uses deterministic **K-Means clustering** to group 18 technical roles into career neighborhoods and **2D PCA (power iteration + deflation)** to project both roles and the candidate's current profile onto a 2D skill-space map.
3. **Time-Budgeted Learning Path Optimizer (`matcher/optimizer.py`)**
   - Models upskilling as a **0/1 Knapsack optimization problem**: given a user's study-hour budget (e.g., 40 hours), Dynamic Programming selects the subset of missing skills that maximizes readiness gain and cross-role transferability, outperforming a naive sequential baseline.
4. **Dual Web Interfaces (`app.py` & `web_app.py`)**
   - `app.py`: Streamlit + Plotly app ready for 1-click cloud hosting on **Streamlit Community Cloud**.
   - `web_app.py`: Zero-dependency local web server using Python's built-in `http.server` (runs anywhere with standard Python 3.11+, even on machines restricting third-party DLLs).

---

## Project Structure

```text
skill-gap-career-matcher/
├── app.py                           # Streamlit + Plotly cloud web app
├── web_app.py                       # Zero-dependency local Python web app
├── matcher/
│   ├── __init__.py
│   ├── data_loader.py               # CSV loading, alias normalization, skill extraction
│   ├── engine.py                    # TF-IDF matching, K-Means clustering, 2D PCA
│   └── optimizer.py                 # 0/1 Knapsack weekly roadmap optimizer
├── data/
│   ├── job_roles.csv                # 18 structured technical roles
│   ├── skill_learning_resources.csv # Estimated learning hours & focus per skill
│   └── sample_resumes.csv           # 8 sample student profiles
├── tests/
│   └── test_matcher.py              # Pytest verification suite
└── requirements.txt
```

---

## Run Locally

```bash
# Run automated unit tests
.\.venv\Scripts\python.exe -m pytest -v

# Option A: Zero-dependency local web app (works offline / without pandas DLLs)
python web_app.py
# Open http://127.0.0.1:8501 in your browser

# Option B: Streamlit web app (for cloud deployment or standard local Python)
streamlit run app.py
```

---

## Deploy for Free to Get a Shareable Link (Streamlit Community Cloud)

1. Create a new **public repository** on your GitHub account (e.g., `skill-gap-career-optimizer`).
2. Upload all project files (`app.py`, `web_app.py`, `requirements.txt`, `README.md`, `.gitignore`, `matcher/`, `data/`, `tests/`) — **do not upload `.venv/`**.
3. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click **Create app**, select your repository, branch `main`, and main file `app.py`.
4. Click **Deploy** — in ~2 minutes you get a public `https://<your-app>.streamlit.app` link to share alongside your GitHub repository link.
