"""Zero-dependency local web UI powered by Python's built-in http.server."""

from __future__ import annotations

from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys
from urllib.parse import urlparse

from matcher import SkillGapMatcher, load_datasets, optimize_learning_path

ROLES, RESOURCES, RESUMES = load_datasets()
BASE_MATCHER = SkillGapMatcher(ROLES)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Skill-Gap & Career Path Optimizer</title>
  <style>
    :root { --bg:#0f172a; --card:#1e293b; --text:#e2e8f0; --muted:#94a3b8; --accent:#38bdf8; --ok:#34d399; --warn:#fbbf24; }
    * { box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }
    body { margin: 0; background: var(--bg); color: var(--text); padding: 24px; }
    .container { max-width: 1180px; margin: 0 auto; }
    h1 { margin: 0 0 6px; font-size: 1.75rem; }
    .subtitle { color: var(--muted); margin-bottom: 20px; }
    .grid { display: grid; grid-template-columns: 340px 1fr; gap: 20px; align-items: start; }
    .card { background: var(--card); border-radius: 12px; padding: 18px; border: 1px solid #334155; }
    label { display: block; font-size: 0.85rem; color: var(--muted); margin: 12px 0 6px; }
    select, textarea, input[type=range] { width: 100%; border-radius: 8px; border: 1px solid #475569; background: #0f172a; color: var(--text); padding: 8px; }
    textarea { min-height: 110px; resize: vertical; }
    button { margin-top: 14px; width: 100%; padding: 10px 14px; border: 0; border-radius: 8px; background: var(--accent); color: #0f172a; font-weight: 700; cursor: pointer; }
    .kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
    .kpi { background: #0f172a; padding: 12px; border-radius: 10px; border: 1px solid #334155; }
    .kpi .val { font-size: 1.35rem; font-weight: 700; color: var(--accent); }
    .kpi .lbl { font-size: 0.78rem; color: var(--muted); }
    .bar-row { display: grid; grid-template-columns: 220px 1fr 58px; gap: 10px; align-items: center; margin: 7px 0; font-size: 0.88rem; }
    .bar-track { background: #0f172a; border-radius: 999px; height: 12px; overflow: hidden; }
    .bar-fill { background: linear-gradient(90deg, #38bdf8, #34d399); height: 100%; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.88rem; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #334155; }
    th { color: var(--muted); }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.78rem; margin: 2px; background: #0f172a; border: 1px solid #475569; }
    .pill.ok { border-color: var(--ok); color: var(--ok); }
    .pill.miss { border-color: var(--warn); color: var(--warn); }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } .kpis { grid-template-columns: repeat(2, 1fr); } }
  </style>
</head>
<body>
<div class="container">
  <h1>Skill-Gap & Career Path Optimizer</h1>
  <div class="subtitle">Prescriptive ML Solver: TF-IDF Semantic Matching + K-Means/PCA Landscape + 0/1 Knapsack Study Optimizer</div>
  <div class="grid">
    <div class="card">
      <label for="sampleSelect">Sample Student Profile</label>
      <select id="sampleSelect"></select>
      <label for="resumeText">Resume / Skills Text</label>
      <textarea id="resumeText"></textarea>
      <label for="targetRole">Target Role</label>
      <select id="targetRole"></select>
      <label>Study Budget: <span id="budgetVal">40</span> hrs</label>
      <input id="budgetHours" type="range" min="10" max="100" step="5" value="40" />
      <label>Weekly Pace: <span id="weekVal">10</span> hrs/week</label>
      <input id="hoursPerWeek" type="range" min="5" max="30" step="1" value="10" />
      <button onclick="runAnalysis()">Optimize Career Roadmap</button>
    </div>
    <div>
      <div class="kpis">
        <div class="kpi"><div class="lbl">Current Readiness</div><div class="val" id="kpiCurrent">-</div></div>
        <div class="kpi"><div class="lbl">Projected (Knapsack)</div><div class="val" id="kpiProjected">-</div></div>
        <div class="kpi"><div class="lbl">Core Skills Matched</div><div class="val" id="kpiCore">-</div></div>
        <div class="kpi"><div class="lbl">Study Hours Used</div><div class="val" id="kpiHours">-</div></div>
      </div>
      <div class="card" style="margin-bottom:16px">
        <strong>Detected Skills:</strong> <span id="detectedSkills"></span>
        <div style="margin-top:10px"><strong>Matched Core:</strong> <span id="matchedCore"></span></div>
        <div style="margin-top:6px"><strong>Missing Core:</strong> <span id="missingCore"></span></div>
      </div>
      <div class="card" style="margin-bottom:16px">
        <h3 style="margin-top:0">Top Role Matches</h3>
        <div id="roleBars"></div>
      </div>
      <div class="card" style="margin-bottom:16px">
        <h3 style="margin-top:0">Optimized Study Roadmap (0/1 Knapsack)</h3>
        <table>
          <thead><tr><th>Week</th><th>Skill</th><th>Priority</th><th>Hours</th><th>Impact</th><th>Focus</th></tr></thead>
          <tbody id="roadmapBody"></tbody>
        </table>
      </div>
      <div class="card">
        <h3 style="margin-top:0">2D Career Landscape (K-Means + PCA)</h3>
        <svg id="landscapeSvg" viewBox="0 0 640 320" style="width:100%; background:#0f172a; border-radius:8px;"></svg>
      </div>
    </div>
  </div>
</div>
<script>
let bootData = null;
async function init() {
  const res = await fetch('/api/bootstrap');
  bootData = await res.json();
  const sSel = document.getElementById('sampleSelect');
  bootData.resumes.forEach((r, i) => {
    const opt = document.createElement('option');
    opt.value = i; opt.textContent = `${r.candidate_name} (${r.current_background})`;
    sSel.appendChild(opt);
  });
  const rSel = document.getElementById('targetRole');
  bootData.roles.forEach(r => {
    const opt = document.createElement('option');
    opt.value = r.role_id; opt.textContent = r.role_title;
    rSel.appendChild(opt);
  });
  sSel.onchange = () => {
    const sample = bootData.resumes[sSel.value];
    document.getElementById('resumeText').value = sample.resume_text;
    document.getElementById('targetRole').value = sample.target_role_id;
    runAnalysis();
  };
  document.getElementById('budgetHours').oninput = e => document.getElementById('budgetVal').textContent = e.target.value;
  document.getElementById('hoursPerWeek').oninput = e => document.getElementById('weekVal').textContent = e.target.value;
  sSel.onchange();
}
async function runAnalysis() {
  const payload = {
    resume_text: document.getElementById('resumeText').value,
    target_role_id: document.getElementById('targetRole').value,
    budget_hours: Number(document.getElementById('budgetHours').value),
    hours_per_week: Number(document.getElementById('hoursPerWeek').value)
  };
  const res = await fetch('/api/analyze', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
  const data = await res.json();
  document.getElementById('kpiCurrent').textContent = data.plan.current_readiness + '%';
  document.getElementById('kpiProjected').textContent = data.plan.projected_readiness + '%';
  document.getElementById('kpiCore').textContent = `${data.target.matched_core.length} / ${data.target.matched_core.length + data.target.missing_core.length}`;
  document.getElementById('kpiHours').textContent = `${data.plan.total_allocated_hours} / ${data.plan.budget_hours}h`;
  document.getElementById('detectedSkills').innerHTML = data.candidate_skills.map(s => `<span class="pill">${s}</span>`).join('') || 'None';
  document.getElementById('matchedCore').innerHTML = data.target.matched_core.map(s => `<span class="pill ok">${s}</span>`).join('') || 'None';
  document.getElementById('missingCore').innerHTML = data.target.missing_core.map(s => `<span class="pill miss">${s}</span>`).join('') || 'None';
  document.getElementById('roleBars').innerHTML = data.ranked_roles.slice(0, 8).map(r => `
    <div class="bar-row"><div>${r.role_title}</div><div class="bar-track"><div class="bar-fill" style="width:${r.readiness_score}%"></div></div><div>${r.readiness_score}%</div></div>
  `).join('');
  document.getElementById('roadmapBody').innerHTML = data.plan.items.map(it => `
    <tr><td>Week ${it.week}</td><td><strong>${it.skill}</strong></td><td>${it.priority}</td><td>${it.est_hours}h</td><td>${it.impact_score}</td><td>${it.learning_focus}</td></tr>
  `).join('') || '<tr><td colspan="6">Increase hour budget or target skills are already covered.</td></tr>';
  renderScatter(data.landscape);
}
function renderScatter(pts) {
  const svg = document.getElementById('landscapeSvg');
  const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const colors = {'Cluster 1':'#38bdf8','Cluster 2':'#34d399','Cluster 3':'#a78bfa','Cluster 4':'#fbbf24','Cluster 5':'#f472b6','Your Profile':'#ef4444'};
  svg.innerHTML = pts.map(p => {
    const cx = 50 + ((p.x - minX) / (maxX - minX || 1)) * 540;
    const cy = 280 - ((p.y - minY) / (maxY - minY || 1)) * 240;
    const r = p.kind === 'Candidate' ? 8 : 5;
    const col = colors[p.cluster] || '#cbd5e1';
    return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${col}"></circle><text x="${cx+8}" y="${cy+3}" fill="#cbd5e1" font-size="9">${p.label}</text>`;
  }).join('');
}
init();
</script>
</body>
</html>
"""


class AppHandler(BaseHTTPRequestHandler):
    """HTTP handler serving the static page and JSON optimization endpoints."""

    def _send_json(self, payload: object, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            raw = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif path == "/api/bootstrap":
            self._send_json(
                {
                    "roles": [asdict(r) for r in ROLES],
                    "resumes": [asdict(s) for s in RESUMES],
                }
            )
        else:
            self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/analyze":
            self._send_json({"error": "Not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        resume_text = str(body.get("resume_text", ""))
        budget_hours = int(body.get("budget_hours", 40))
        hours_per_week = int(body.get("hours_per_week", 10))
        target_role_id = str(body.get("target_role_id", ""))

        candidate_skills, ranked = BASE_MATCHER.rank_roles(resume_text)
        target = next((r for r in ranked if r.role_id == target_role_id), ranked[0])
        plan = optimize_learning_path(
            role_match=target,
            resources=RESOURCES,
            skill_demand=BASE_MATCHER.skill_demand,
            budget_hours=budget_hours,
            hours_per_week=hours_per_week,
        )
        landscape = BASE_MATCHER.compute_role_landscape(resume_text)
        self._send_json(
            {
                "candidate_skills": candidate_skills,
                "target": asdict(target),
                "ranked_roles": [asdict(r) for r in ranked],
                "plan": asdict(plan),
                "landscape": landscape,
            }
        )


def run_server(port: int = 8501) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), AppHandler)
    print(f"Serving Skill-Gap & Career Path Optimizer at http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    chosen_port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    run_server(chosen_port)
