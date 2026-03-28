# TraceLens — Backend incident diagnosis environment

An OpenEnv-compatible environment where agents inspect backend logs and metrics to pinpoint incident root causes.

---

### 1. Project Title + One-Line Description
TraceLens — An environment for training and evaluating agents on backend incident diagnosis through log and metric investigation.

### 2. Overview (What this is)
TraceLens simulates noisy backend outages. Agents start from an alert, inspect service logs and metrics, and submit a structured diagnosis. The focus is on step-by-step debugging under uncertainty: partial views, noisy signals, and limited steps.

### 3. Why This Environment Is Realistic
- Partial observability: agents only see the current log window or selected metrics.
- Noisy logs: INFO/WARN/ERROR lines are interleaved; true signals are embedded.
- Indirect root causes: symptoms may surface in non-affected services.
- Cross-service dependencies: hard tasks require looking beyond the entry service.

### 4. Core Interaction Loop
reset → observe alert → open_logs / view_metrics / scroll → gather signals → submit_diagnosis.

### 5. Action Space
- open_logs(service): fetch the latest window (size 3) for a service.
- scroll_logs: move backward to older windows for the current service.
- view_metrics(service): retrieve metrics snapshot for a service.
- submit_diagnosis(service, root_cause, severity): end the episode with a structured answer.

### 6. Observation Format
Observations are Pydantic models with:
- message: alert text, log window, or metrics dump.
- available_tools: actions allowed next.
- reward: numeric reward for the step (populated by the environment).
- done: whether the episode ended.
- signals_discovered, services_explored, progress_score: simple telemetry for progress visibility.

### 7. Reward Design
- Small positive reward for first-time signal discovery (error lines or abnormal metrics markers).
- Step penalty: -0.01; repeat-action penalty: -0.02.
- Final reward: 1.0 for correct root_cause; 0.0 otherwise.
- Evidence scaling: ×0.7 if no signals were found before submission.
- Invalid diagnosis options yield -1.0 and terminate.
Reward encourages investigation, not guessing.

### 8. Difficulty Levels
- Easy: direct signal visible in initial logs.
- Medium: requires multiple steps or metrics to confirm.
- Hard: cross-service reasoning with misleading signals in the entry service.

### 9. Dataset Design
Incidents are grouped into easy/medium/hard buckets in [server/incidents.json](server/incidents.json). Logs are ordered (latest first), and signals are inferred from ERROR lines or abnormal metrics markers. The provided dataset is minimal and does not yet cover a full corpus.

### 10. API Endpoints
- POST /reset — start a new episode (optional seed, difficulty).
- POST /step — send an action payload and receive observation, reward, done.
- GET /tasks — list tasks with difficulty metadata and success criteria.
- POST /grader — deterministic scorer for submitted diagnoses.
- GET /baseline — returns oracle scores and optional OpenAI scores when an API key is available.

Example (reset):
```bash
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"difficulty": "medium", "seed": 42}'
```

Example (step):
```bash
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" -d '{"action": {"type": "open_logs", "service": "payments"}}'
```

### 11. Baseline
- Oracle baseline: submits ground truth for grading.
- OpenAI baseline: runs only when OPENAI_API_KEY is set; otherwise omitted.
- Both are reproducible via fixed seeds.

### 12. Example Run
ALERT → open_logs(service A) → scroll_logs() for deeper context → view_metrics(service B) → submit_diagnosis(service, root_cause, severity) → receive reward and done.

### 13. Setup Instructions
1) Install dependencies (from repo root):
```bash
pip install -r server/requirements.txt
```
2) Run the server:
```bash
uvicorn server.app:app --reload --port 8000
```
3) Run the baseline client (oracle by default):
```bash
python client.py --base-url http://localhost:8000
```

### 14. Design Philosophy
Emphasize reasoning over memorization: agents must gather evidence, cope with noise, and decide when they have enough to submit. Reward shaping is minimal but pushes toward thoughtful investigation.

### 15. Limitations
- Logs are simplified and synthetic; not production scale.
- Limited number of services per incident.
- Dataset is minimal and may be expanded; not all incident types are represented yet.

---

Repository name: TraceLens.
