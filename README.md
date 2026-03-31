---
title: TraceLens
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - debugging
  - agents
base_path: /docs
---

TraceLens — Diagnosing Real Backend Incidents with Agents

TraceLens is an OpenEnv environment that simulates real backend failures — where agents must debug production-like issues using logs and metrics under uncertainty.

Unlike toy environments, TraceLens focuses on how engineers actually investigate incidents:

* starting from incomplete alerts
* navigating noisy logs
* correlating signals across services
* deciding when enough evidence is gathered

This environment is designed to evaluate whether an agent can *debug*, not just answer.

---

### Overview

TraceLens simulates backend outages where agents must investigate and identify root causes. Each episode begins with an alert. The agent then interacts with the system using tools like log inspection and metric queries to gather evidence before submitting a structured diagnosis.

The environment emphasizes:

* multi-step reasoning
* partial observability
* noisy, real-world-like signals
* decision-making under uncertainty

---

### Why This Matters

Modern AI agents are increasingly expected to operate in real-world systems — assisting with incident response, debugging failures, and supporting engineers.

However, most benchmarks evaluate static reasoning or simplified tasks.

TraceLens fills this gap by introducing:

* an interactive investigation loop instead of one-shot answers
* noisy and incomplete information
* multi-step reasoning with consequences

This makes it suitable not only for evaluation, but also for training agents capable of assisting in real DevOps workflows.

---

### Core Interaction Loop

reset → observe alert → open_logs / view_metrics / scroll_logs → gather signals → submit_diagnosis

Agents must decide:

* what to inspect next
* when to switch services
* when they have enough evidence to conclude

---

### Action Space

* open_logs(service): fetch the latest log window (size = 3)
* scroll_logs: move backward to older log windows
* view_metrics(service): retrieve metrics snapshot
* submit_diagnosis(service, root_cause, severity): finalize answer

---

### Observation Format

Each observation includes:

* message: alert text, logs, or metrics
* available_tools: actions allowed at current step
* available_services: valid services for the incident
* diagnosis_options: valid root cause candidates
* reward: reward for the current step
* done: whether the episode has ended
* signals_discovered: number of unique signals found
* services_explored: services investigated so far
* progress_score: simple proxy for investigation progress

---

### Reward Design

The reward function reflects realistic debugging behavior:

* Agents are rewarded for discovering *new evidence*, not repeating actions
* Exploration yields small positive rewards for first-time signals
* Inefficient behavior (repetition, redundant actions) is penalized
* Final correctness dominates reward (correct diagnosis = 1.0)
* Submitting without evidence reduces reward (evidence scaling)
* Invalid diagnoses result in immediate penalties

This prevents both blind guessing and reward exploitation while encouraging structured investigation.

---

### Difficulty Design

* Easy: Direct signal visible immediately; solvable in 1–2 steps
* Medium: Requires combining logs and metrics
* Hard: Requires cross-service reasoning with misleading signals

Hard tasks intentionally introduce *false leads*, requiring agents to verify assumptions rather than rely on first impressions.

---

### Example Investigation (Medium Task)

ALERT: Elevated 500s in checkout service

1. Agent checks metrics → sees error_rate: spiking
2. Opens logs → finds template render failures
3. Confirms issue originates in checkout service
4. Submits diagnosis:
   service = "checkout", root_cause = "TEMPLATE_ERROR", severity = "high"

This requires combining multiple signals rather than relying on a single observation.

---

### Dataset Design

Incidents are grouped into three difficulty buckets:

easy | medium | hard

Each incident follows this structure:

```json
{
   "incident_id": "unique_id",
   "alert": "text",
   "entry_service": "service_name",
   "max_steps": 10,
   "ground_truth": {
      "root_cause": "...",
      "affected_service": "...",
      "severity": "..."
   },
   "services": {
      "service_name": {
         "logs": ["INFO ...", "WARN ...", "ERROR ..."],
         "metrics": { "k": "v" }
      }
   },
   "diagnosis_options": ["..."]
}
```

Key properties:

* Logs are ordered oldest → newest (latest entries last)
* Signals are embedded in noisy logs (INFO/WARN/ERROR mix)
* Metrics include abnormal markers such as: high, spiking, 100%, maxed
* Each incident has exactly one correct diagnosis
* Difficulty increases via noise, indirection, and cross-service dependencies

The current dataset is minimal and designed for demonstration; it can be extended for broader evaluation.

---

### API Endpoints

* POST /reset — start a new episode (optional: difficulty, seed)
* POST /step — perform an action and receive observation, reward, done, info
* GET /tasks — list available tasks and difficulty levels
* POST /grader — deterministic scoring of a submitted diagnosis
* POST /baseline — returns oracle scores and optional OpenAI-based baseline

Example (reset):

```bash
curl -X POST http://localhost:7860/reset \
   -H "Content-Type: application/json" \
   -d '{"difficulty": "medium", "seed": 42}'
```

Example (step):

```bash
curl -X POST http://localhost:7860/step \
   -H "Content-Type: application/json" \
   -d '{
      "session_id": "...",
      "action": {"type": "open_logs", "service": "payments"}
   }'
```

---

### Baseline

* Oracle baseline: submits ground truth for deterministic scoring
* OpenAI baseline: runs an LLM agent using OpenAI-compatible APIs (supports OpenRouter)
* Both baselines are reproducible using fixed seeds

---

### Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r server/requirements.txt
   ```

2. Run the server:
   ```bash
   uvicorn server.app:app --port 7860
   ```

3. Run baseline client (optional):
   ```bash
   python client.py --base-url http://localhost:7860
   ```

---

### Design Philosophy

TraceLens prioritizes **reasoning over memorization**.

Agents must:

* gather evidence incrementally
* deal with noisy and incomplete data
* decide when they have enough information

The environment is intentionally designed to mirror real debugging workflows rather than simplified benchmarks.

---

### What Makes TraceLens Different

TraceLens is not a static QA benchmark — it is an *interactive debugging environment* where agents must decide:

* what to inspect next
* how to interpret signals
* when to stop investigating and submit

This makes it closer to real engineering systems than traditional evaluation setups.

---

### Limitations

* Logs are synthetic and simplified compared to production systems
* Limited number of services per incident
* Dataset size is currently small
* Not all failure types are represented

These are intentional trade-offs to keep the environment interpretable and extensible.

---

Repository name: TraceLens
