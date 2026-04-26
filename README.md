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
# TraceLens 🔍

**An interactive environment for training LLMs to debug real backend incidents.**

> Most benchmarks ask whether a model can *answer* a question.  
> TraceLens asks whether it can *figure one out*.

---

## Links

| | |
|---|---|
| 🌐 **Environment (HF Space)** | [sammyurfen/tracelens](https://huggingface.co/spaces/sammyurfen/tracelens) |
| 📝 **Blog post** | [TraceLens: Training an LLM to Debug Backend Incidents](https://huggingface.co/spaces/sammyurfen/tracelens-blog) |
| 📓 **Training notebook** | [`tracelens-training.ipynb`](https://colab.research.google.com/drive/14KkxvsDpTCo5JkxxUMxgRvxCy_2ECi4z?usp=sharing) |

---

## The Problem

Incident diagnosis is one of the hardest things engineers do under pressure. They start with a vague alert, dig through logs, cross-reference metrics across services, discard false leads, and piece together a root cause — all without seeing the full picture at once.

LLMs are increasingly expected to assist in these scenarios. But the benchmarks used to evaluate them don't test this at all. They test static recall, one-shot answers, and reasoning over complete information. Real debugging is none of those things.

TraceLens fills that gap. It's an environment where an agent must *investigate* — not just answer.

---

## Hackathon Themes Addressed

TraceLens was designed to directly address three hackathon themes:

**Theme 3.1 — World Modeling (Professional Tasks)**  
The core environment models a backend system as a partially observable world. The agent interacts with real tool calls, receives noisy real-world-like signals, and must maintain an internal picture of the system across multiple steps. There are no shortcuts: the correct answer isn't in the prompt — it has to be derived from evidence.

**Theme 2 — Long-Horizon Planning**  
Hard incidents require the agent to delay commitment. It must explore multiple services, discount misleading signals, and only submit when evidence is sufficient. Submitting early is penalized. This directly trains durable multi-step planning behavior, not shallow pattern-matching.

**Theme 1 — Multi-Agent Interactions**  
Our baseline (`inference.py`) implements a multi-agent architecture over the single-agent environment. Three specialist explorers (latency-focused, error-focused, resource-focused) run in parallel and propose actions. A coordinator aggregates their evidence through a phase machine (`EXPLORE → FOCUS → DECIDE`) and selects the final action. This demonstrates that TraceLens naturally supports — and benefits from — multi-agent reasoning strategies, even without native multi-agent environment support.

---

## The Environment

Each episode begins with an alert:

```
"Elevated 500 errors in checkout service"
```

The agent has four tools:

| Tool | What it does |
|---|---|
| `open_logs(service)` | Opens the latest 3-line log window for a service |
| `scroll_logs()` | Moves backward through the currently open service's logs — **irreversible** |
| `view_metrics(service)` | Returns a metrics snapshot for a service |
| `submit_diagnosis(service, root_cause, severity)` | Finalizes the diagnosis and ends the episode |

The constraints are deliberate. `scroll_logs()` is irreversible — you can't go back. `Max steps` per task is capped and agent is cut off from uploading more steps after. The agent must decide where to look, in what order, and when it has seen enough.

### Difficulty Levels

| Level | What makes it hard |
|---|---|
| Easy | Root cause is visible in a single service's logs |
| Medium | Requires combining logs and metrics across two services |
| Hard | Includes false leads — a metric spike or error that points the wrong way — requiring cross-service verification |

### Reward Design

The reward function is shaped to produce real investigative behavior, not gaming:

- **New signal discovered** → small positive reward
- **Redundant / repeated action** → penalty
- **Submitting without evidence** → scaled penalty
- **Correct final diagnosis** → primary reward (dominates)
- **Invalid action** → immediate penalty

This prevents the two failure modes that plague naive RL setups: blind random exploration and premature commitment.

### Dataset

Incidents are stored in a structured JSON bank across three difficulty buckets. Each incident includes an alert, per-service logs and metrics, a ground truth diagnosis, and a set of valid root-cause candidates. Root causes include `DB_OVERLOAD`, `CACHE_STALE`, `NETWORK_PARTITION`, `TEMPLATE_ERROR`, `DEPLOY_REGRESSION`, `MEMORY_LEAK`, and more.

---

## Baseline: Multi-Agent vs Single-Agent

To evaluate what the environment actually tests, we ran both a single-agent and a multi-agent baseline using GPT-4o-mini, 5 episodes per difficulty level.

| | Easy | Medium | Hard |
|---|---|---|---|
| **Single agent** | 0.60 | 0.64 | 0.56 |
| **Multi-agent** (3 explorers + coordinator) | **0.94** | **0.82** | **0.50** |

Easy and medium tasks benefit substantially from the multi-agent setup — parallel exploration and hypothesis comparison reduces the chance of latching onto the wrong signal. Hard tasks remain challenging because the false leads require deep verification that even multiple agents struggle with.

The multi-agent architecture (`inference.py`) runs three specialist explorers in parallel, each focused on a different signal type, with a coordinator that manages phase transitions and enforces evidence-quality gates before allowing a `submit_diagnosis` action.

---

## Training

We trained `Qwen2.5-0.5B-Instruct` using GRPO on hard incidents, with the model collecting its own experience by interacting with the live environment over HTTP.

**Setup:**
- Model: Qwen2.5-0.5B-Instruct with LoRA (Unsloth)
- Algorithm: GRPO (TRL)
- Training: 30 steps × 4 episodes = 120 rollouts
- Combined reward: `0.85 × grader_score + 0.15 × normalized_step_reward`
- Runtime: ~35 minutes on A100 40GB

**Results on hard incidents:**

| | Grader Score |
|---|---|
| Before training | 0.490 |
| After training | **0.618** |
| Delta | **+0.128** |

The post-training score (0.618) exceeds the GPT-4o-mini multi-agent baseline (0.504) on hard tasks — using a model that is orders of magnitude smaller.

![Training results](moving-avg.png)

**Left: grader score over 30 training steps with 7-step moving average. Right: per-episode before/after comparison.**

The behavioral change is meaningful: the trained model explores more services before committing and is less likely to submit on the first plausible signal it finds.

---

## Why This Matters

Backend incident diagnosis is a real, high-stakes task that existing LLM benchmarks don't evaluate. TraceLens makes it trainable and measurable. The environment is:

- **Interactive** — the agent queries the system, it doesn't just read a prompt
- **Partially observable** — no single action reveals the full picture
- **Reward-shaped** — the signal teaches investigation strategy, not just answer correctness
- **Extensible** — new incidents, services, and failure modes can be added without changing the environment logic

The result we care about isn't just the +0.128 delta. It's that a small model, trained for 35 minutes on a synthetic environment, learns to delay decisions and gather more evidence before committing. That's a behavior change — and it's the kind of behavior that makes an agent actually useful in a real system.

---

## API Reference

The environment runs as a REST API on Hugging Face Spaces.

```bash
# Start a new episode
curl -X POST https://sammyurfen-tracelens.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "hard", "seed": 42}'

# Take a step
curl -X POST https://sammyurfen-tracelens.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "...", "action": {"type": "open_logs", "service": "payments"}}'

# Score a diagnosis independently
curl -X POST https://sammyurfen-tracelens.hf.space/grader \
  -H "Content-Type: application/json" \
  -d '{"seed": 42, "difficulty": "hard", "service": "payments", "root_cause": "DB_OVERLOAD", "severity": "high"}'
```

**Endpoints:** `/reset` · `/step` · `/state` · `/tasks` · `/grader`

---

## Running Locally

```bash
# Install dependencies
pip install -r server/requirements.txt

# Start the server
uvicorn server.app:app --port 7860

# Run the multi-agent baseline (requires OpenAI-compatible API key)
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="<your_key>"
python inference.py --mode openai --base-url http://127.0.0.1:7860 --episodes 5

# Run oracle baseline (ground truth, for sanity check)
python inference.py --mode oracle --base-url http://127.0.0.1:7860 --episodes 5
```

---

## Limitations

- Logs and incidents are synthetic — real production telemetry is noisier and more varied
- The incident dataset is currently small; designed for demonstration and extensibility
- The trained model is small; long-horizon reasoning remains shallow compared to a real engineer
- Grading evaluates only the final diagnosis tuple, not the quality of the investigation path

These are intentional trade-offs that keep the environment interpretable, fast to run, and easy to extend.

---

## Repository Structure

```
├── server/
│   ├── app.py                          # FastAPI session wrapper
│   ├── backend_diagnosis_environment.py # Core environment logic
│   ├── incidents.json                   # Incident dataset (easy/medium/hard)
│   └── requirements.txt
├── models.py                            # Action/observation schema
├── inference.py                         # Multi-agent baseline
├── tracelens-training.ipynb        # GRPO training notebook (< 1hr on A100)
├── moving-avg.png       # Training results plot
└── README.md
```