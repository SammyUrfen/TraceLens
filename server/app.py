# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Backend Diagnosis Environment.

This module creates an HTTP server that exposes the BackendDiagnosisEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 4

    # Or run directly:
    python -m server.app
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple
from uuid import uuid4
import difflib

from fastapi import Body, HTTPException, Query
from fastapi.responses import RedirectResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from models import BackendDiagnosisAction, BackendDiagnosisObservation
    from .backend_diagnosis_environment import BackendDiagnosisEnvironment
except ModuleNotFoundError:
    from models import BackendDiagnosisAction, BackendDiagnosisObservation
    from server.backend_diagnosis_environment import BackendDiagnosisEnvironment


# Create the app with web interface and README integration
app = create_app(
    BackendDiagnosisEnvironment,
    BackendDiagnosisAction,
    BackendDiagnosisObservation,
    env_name="backend_diagnosis",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

# Session-backed env store for HTTP reset/step lifecycle
env_store: Dict[str, BackendDiagnosisEnvironment] = {}


# Remove create_app-provided reset/step routes so we can enforce session persistence.
app.router.routes = [
    route
    for route in app.router.routes
    if not (route.path in {"/reset", "/step"} and getattr(route, "methods", None) and "POST" in route.methods)
]

# Root endpoint: redirect to /docs for better UX
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")

@app.post("/reset")
def reset(payload: Dict[str, object] = Body(default_factory=dict)) -> Dict[str, object]:
    seed = payload.get("seed")
    difficulty = payload.get("difficulty")
    session_id = str(uuid4())
    env = BackendDiagnosisEnvironment()
    obs = env.reset(seed=seed, difficulty=difficulty)
    env_store[session_id] = env
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
        "session_id": session_id,
    }


@app.post("/step")
def step(payload: Dict[str, object] = Body(...)) -> Dict[str, object]:
    session_id = payload.get("session_id")
    if not session_id or session_id not in env_store:
        raise HTTPException(status_code=400, detail="Invalid or missing session_id")

    env = env_store[session_id]
    action_payload = payload.get("action") or {}
    action = BackendDiagnosisAction(**action_payload)
    obs = env.step(
        action,
        seed=payload.get("seed"),
        difficulty=payload.get("difficulty"),
    )

    done = bool(obs.done)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": done,
        "info": {
            "signals_discovered": obs.signals_discovered,
            "services_explored": obs.services_explored,
            "progress_score": obs.progress_score,
        },
        "session_id": session_id,
    }


@app.get("/tasks")
def list_tasks() -> Dict[str, List[Dict[str, object]]]:
    action_schema = {
        "type": ["open_logs", "scroll_logs", "view_metrics", "submit_diagnosis"],
        "service": "string (optional depending on action)",
        "root_cause": "string (required for submit)",
        "severity": "string (optional)",
    }

    actions = {
        "open_logs": {"requires": ["service"]},
        "scroll_logs": {"requires": []},
        "view_metrics": {"requires": ["service"]},
        "submit_diagnosis": {"requires": ["root_cause", "service", "severity"]},
    }

    tasks: List[Dict[str, object]] = [
        {
            "id": "easy",
            "difficulty": "easy",
            "description": "Simple incident where root cause is directly visible in logs",
            "objective": "Identify the root cause of the backend incident",
            "success_criteria": "Correct root_cause, affected_service, and severity",
            "action_schema": action_schema,
            "actions": actions,
        },
        {
            "id": "medium",
            "difficulty": "medium",
            "description": "Requires multiple steps or metrics to diagnose",
            "objective": "Identify the root cause of the backend incident",
            "success_criteria": "Correct root_cause, affected_service, and severity",
            "action_schema": action_schema,
            "actions": actions,
        },
        {
            "id": "hard",
            "difficulty": "hard",
            "description": "Requires cross-service reasoning and handling misleading signals",
            "objective": "Identify the root cause of the backend incident",
            "success_criteria": "Correct root_cause, affected_service, and severity",
            "action_schema": action_schema,
            "actions": actions,
        },
    ]

    return {"tasks": tasks}


@app.post("/grader")
def grade(final_action: dict = Body(...)) -> Dict[str, float]:
    env = BackendDiagnosisEnvironment()
    seed = final_action.get("seed")
    difficulty = final_action.get("difficulty")
    env.reset(seed=seed, difficulty=difficulty)

    action_payload = final_action.get("action") or final_action
    action = BackendDiagnosisAction(
        type="submit_diagnosis",
        service=action_payload.get("service"),
        root_cause=action_payload.get("root_cause"),
        severity=action_payload.get("severity"),
    )
    print("GRADER INCIDENT:", env._current_incident["incident_id"])
    print("EXPECTED:", env._current_incident["ground_truth"])
    score = env.grade_episode(action)
    return {"score": score}


@app.get("/baseline")
@app.post("/baseline")
def baseline(
    mode: str = Query("oracle", enum=["oracle", "openai"]),
    payload: Dict[str, object] | None = Body(default=None),
) -> Dict[str, object]:
    if payload and isinstance(payload, dict):
        mode = payload.get("mode", mode)
    seeds = [42, 43, 44]
    episodes_per_difficulty = int(payload.get("episodes", 1)) if payload else 1
    max_steps = int(payload.get("max_steps", BackendDiagnosisEnvironment.MAX_STEPS)) if payload else BackendDiagnosisEnvironment.MAX_STEPS
    model = payload.get("model", "gpt-4o-mini") if payload else "gpt-4o-mini"
    base_url = payload.get("base_url", "https://api.openai.com/v1") if payload else "https://api.openai.com/v1"
    api_key_payload = payload.get("api_key") if payload else None
    openai_api_key = api_key_payload or os.environ.get("OPENAI_API_KEY")

    def run_oracle(diff: str) -> float:
        scores: List[float] = []
        for idx in range(episodes_per_difficulty):
            s = seeds[idx % len(seeds)]
            env = BackendDiagnosisEnvironment()
            env.reset(seed=s, difficulty=diff)
            gt = env._current_incident.get("ground_truth", {})
            action = BackendDiagnosisAction(
                type="submit_diagnosis",
                service=gt.get("affected_service"),
                root_cause=gt.get("root_cause"),
                severity=gt.get("severity"),
            )
            env.step(action)
            scores.append(env.grade_episode(action))
        return sum(scores) / len(scores)

    def run_openai(diff: str) -> float:
        try:
            from openai import OpenAI
        except Exception as e:
            print(f"OpenAI client import failed: {e}")
            return run_oracle(diff)

        client = OpenAI(api_key=openai_api_key, base_url=base_url)
        scores: List[float] = []
        for idx in range(episodes_per_difficulty):
            s = seeds[idx % len(seeds)]
            env = BackendDiagnosisEnvironment()
            obs = env.reset(seed=s, difficulty=diff)
            final_action: BackendDiagnosisAction | None = None
            last_service: Optional[str] = None
            consecutive_invalid = 0
            last_actions: List[Tuple[Optional[str], Optional[str]]] = []
            last_signal_count: int = obs.signals_discovered or 0
            no_progress_count = 0
            best_signal_message: str = ""
            best_signal_service: Optional[str] = None
            service_visit_count: Dict[str, int] = {}
            service_signal_map: Dict[str, int] = {}
            visited_services = set()
            episode_history: List[Dict[str, object]] = []
            last_step_had_strong_signal = False
            revisiting_without_new_evidence = False

            system_prompt = (
                "You are a backend diagnosis agent.\n\n"
                "Your job is to identify the root cause of an incident using logs and metrics.\n\n"
                "---\n\n"
                "HOW TO THINK:\n\n"
                "1. Look at the current observation\n"
                "2. Identify the most likely cause based on signals\n"
                "3. If unsure, check another service\n"
                "4. Submit when reasonably confident\n\n"
                "---\n\n"
                "IMPORTANT:\n\n"
                "- Do NOT overthink\n"
                "- Do NOT invent complex explanations\n"
                "- Prefer the simplest explanation that fits the signals\n"
                "- One strong signal is often enough in easy tasks\n"
                "- Multiple signals are needed in medium/hard tasks\n\n"
                "---\n\n"
                "CONFIRMATION RULE:\n\n"
                "- If signals are weak or ambiguous -> check another service\n"
                "- If signals are strong and clear -> you may submit\n\n"
                "---\n\n"
                "SERVICE RULE:\n\n"
                "- Do not stay on one service if no useful signal is found\n"
                "- Move to related services when needed\n\n"
                "ROOT CAUSE MAPPING (VERY IMPORTANT):\n\n"
                "Use these exact mappings:\n\n"
                "NETWORK_PARTITION:\n"
                "- high packet loss\n"
                "- degraded probes\n"
                "- connection failures across zones\n\n"
                "RATE_LIMITED:\n"
                "- retry_rate high\n"
                "- errors + retries + saturation\n"
                "- signs of throttling or backoff\n\n"
                "Important: High retry_rate indicates RATE_LIMITED, not TIMEOUT.\n\n"
                "TIMEOUT:\n"
                "- high latency + queue/backlog\n"
                "- slow downstream response\n"
                "- no explicit rate limit signals\n\n"
                "DB_OVERLOAD:\n"
                "- high connections\n"
                "- queue buildup\n"
                "- slow queries / DB saturation\n\n"
                "CACHE_MISS:\n"
                "- high miss rate\n"
                "- latency increase due to fallback\n\n"
                "CACHE_STALE:\n"
                "- stale data errors\n"
                "- inconsistent results\n\n"
                "TEMPLATE_ERROR:\n"
                "- render failures\n"
                "- template processing errors\n\n"
                "DEPENDENCY_DOWN:\n"
                "- upstream healthy, downstream failing\n"
                "- connection refused / dependency unavailable\n\n"
                "SERVICE_CRASH:\n"
                "- restarts\n"
                "- crash logs\n\n"
                "MEMORY_LEAK:\n"
                "- memory steadily increasing\n"
                "- OOM patterns\n\n"
                "---\n\n"
                "CRITICAL:\n\n"
                "- Do NOT confuse RATE_LIMITED and TIMEOUT\n"
                "- Do NOT guess randomly\n"
                "- Choose the closest exact label from taxonomy\n\n"
                "OUTPUT FORMAT:\n\n"
                "{\"thought\": \"short reasoning (1-2 sentences)\", \"action\": {\"type\": \"...\", \"service\": \"...\", \"root_cause\": \"...\", \"severity\": \"...\"}}\n\n"
                "Only include root_cause when submitting."
            )

            def _has_observation_strong_signal(message: str) -> bool:
                text = (message or "").lower()
                latency = any(k in text for k in ["latency", "p95", "p99", "slow"])
                errors = any(k in text for k in ["error", "5xx", "failed", "failure"])
                saturation = any(k in text for k in ["saturation", "backlog", "queue", "maxed", "utilization", "100%", "connection wait"])
                network_deg = any(k in text for k in ["packet loss", "degraded probe", "network instability", "network partition"])
                return (latency and errors and saturation) or network_deg

            def _has_strong_signal(message: str) -> bool:
                text = (message or "").lower()
                markers = ["error", "spiking", "100%", "maxed", "timeout", "down", "failed"]
                return any(marker in text for marker in markers)

            def build_context(history: List[Dict[str, object]]) -> str:
                recent = history[-5:]
                if not recent:
                    return "Previous steps: (none yet)"
                lines = ["Previous steps:"]
                for item in recent:
                    lines.append(f"Step {item.get('step')}:")
                    lines.append(f"Observation: {item.get('observation', '')}")
                    lines.append(f"Action: {item.get('action', '')}")
                    lines.append(f"Outcome: reward={item.get('reward', 0.0)}")
                    lines.append("")
                return "\n".join(lines).strip()

            episode_max_steps = max_steps + 2 if diff == "hard" else max_steps

            for step_idx in range(1, episode_max_steps + 1):
                hints: List[str] = []
                if (
                    last_service
                    and service_visit_count.get(last_service, 0) >= 3
                    and service_signal_map.get(last_service, 0) == 0
                ):
                    unexplored = [s for s in (obs.available_services or []) if s not in visited_services]
                    if unexplored:
                        hints.append(
                            f"If you are repeatedly exploring the same service without new signals, "
                            f"consider switching from '{last_service}' to '{unexplored[0]}'."
                        )

                if no_progress_count >= 3:
                    hints.append("No useful signal found. Try a different service.")

                if consecutive_invalid > 0:
                    hints.append(
                        "Your previous action was invalid. Ensure required fields are present and action/service are compatible before retrying."
                    )

                recent_actions = last_actions[-4:]
                if len(recent_actions) == 4 and len(set(recent_actions)) == 1:
                    hints.append("You are repeating the same action without progress. Consider a different strategy.")

                if last_step_had_strong_signal:
                    hints.append(
                        "You are seeing strong signals. Consider whether you now have enough evidence to submit diagnosis."
                    )

                if _has_observation_strong_signal(obs.message):
                    hints.append(
                        "You are seeing strong signals. If consistent with your hypothesis, you may submit."
                    )

                if revisiting_without_new_evidence:
                    hints.append(
                        "You are revisiting a service without new evidence. Consider eliminating this hypothesis and exploring alternatives."
                    )

                if len(visited_services) <= 1 and step_idx >= 2:
                    hints.append("Consider checking another related service if unsure.")

                if _has_observation_strong_signal(obs.message):
                    hints.append("Signals look strong. You can submit if confident.")

                explored_services = sorted(list(visited_services))
                hints_text = "\n".join(f"- {h}" for h in hints) if hints else "- No additional hints."
                recent_action_text = [
                    {"type": a_type, "service": a_service}
                    for a_type, a_service in last_actions[-5:]
                ]

                user_prompt = (
                    f"{build_context(episode_history)}\n\n"
                    "Current observation:\n"
                    f"{obs.message}\n\n"
                    f"Available tools: {obs.available_tools}\n"
                    f"Available services: {getattr(obs, 'available_services', [])}\n"
                    f"Diagnosis options: {getattr(obs, 'diagnosis_options', [])}\n"
                    f"Services explored so far: {explored_services}\n"
                    f"Recent actions: {recent_action_text}\n"
                    "Guidance:\n"
                    f"{hints_text}\n\n"
                    "If action.type is submit_diagnosis, include severity as one of: low, medium, high.\n"
                    "Return ONLY JSON in this exact form:\n"
                    "{\"thought\":\"step-by-step reasoning\",\"action\":{\"type\":\"...\",\"service\":\"...\",\"root_cause\":\"...\",\"severity\":\"...\"}}"
                )

                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0,
                    )
                    content = completion.choices[0].message.content if completion.choices else ""
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and isinstance(parsed.get("action"), dict):
                        action_payload = parsed.get("action")
                    else:
                        action_payload = {}
                except Exception as e:
                    print(f"OpenAI call failed: {e}")
                    action_payload = {}

                action = _safe_action_from_payload(action_payload, last_service, obs.diagnosis_options, obs.message)
                if action.type in {"open_logs", "view_metrics", "submit_diagnosis"} and not action.service:
                    action.service = last_service or (obs.available_services[0] if (obs.available_services or []) else None)
                obs = env.step(action)
                reward = float(obs.reward or 0.0)
                done = bool(obs.done)

                episode_history.append(
                    {
                        "step": step_idx,
                        "observation": obs.message,
                        "action": json.dumps(
                            {
                                "type": action.type,
                                "service": action.service,
                                "root_cause": action.root_cause,
                                "severity": action.severity,
                            }
                        ),
                        "reward": reward,
                    }
                )

                current_signals = obs.signals_discovered or 0
                new_signal_found = current_signals > last_signal_count
                last_signal_count = max(last_signal_count, current_signals)
                last_step_had_strong_signal = bool(new_signal_found or reward > 0)
                no_progress_count = 0 if new_signal_found else no_progress_count + 1

                if _has_strong_signal(obs.message) or new_signal_found:
                    best_signal_message = obs.message or best_signal_message
                    best_signal_service = action.service or best_signal_service

                current_service = action.service or last_service
                if current_service:
                    revisiting_without_new_evidence = current_service in visited_services and not new_signal_found
                    visited_services.add(current_service)
                    service_visit_count[current_service] = service_visit_count.get(current_service, 0) + 1
                    service_signal_map[current_service] = max(service_signal_map.get(current_service, 0), current_signals)
                else:
                    revisiting_without_new_evidence = False

                last_actions.append((action.type, action.service))
                if len(last_actions) > 6:
                    last_actions.pop(0)

                is_invalid = "Invalid action" in obs.message
                consecutive_invalid = consecutive_invalid + 1 if is_invalid else 0
                if is_invalid and not done:
                    continue

                if action.service:
                    last_service = action.service
                if action.type == "submit_diagnosis":
                    final_action = action
                if done:
                    break

            if final_action is None:
                final_action = BackendDiagnosisAction(
                    type="submit_diagnosis",
                    service=best_signal_service or last_service or env.state.current_service,
                    root_cause=_infer_root_cause(best_signal_message or obs.message, getattr(obs, "diagnosis_options", None)),
                    severity=_infer_severity_from_text(best_signal_message or obs.message),
                )
                env.step(final_action)
            scores.append(env.grade_episode(final_action))
        return sum(scores) / len(scores)

    oracle_results = {
        "easy": run_oracle("easy"),
        "medium": run_oracle("medium"),
        "hard": run_oracle("hard"),
    }

    openai_results: Optional[Dict[str, float]] = None
    openai_error: Optional[str] = None
    if mode == "openai":
        if not openai_api_key:
            openai_error = "OPENAI_API_KEY is required for openai baseline"
        else:
            openai_results = {
                "easy": run_openai("easy"),
                "medium": run_openai("medium"),
                "hard": run_openai("hard"),
            }

    response: Dict[str, object] = {
        "oracle": oracle_results,
        "openai": openai_results,
    }
    if openai_error:
        response["error"] = openai_error
    return response


def _infer_root_cause(message: str, diagnosis_options: Optional[List[str]]) -> Optional[str]:
    """Pick a root cause from options using simple keyword matching."""

    if not diagnosis_options:
        return None

    text = (message or "").lower()

    def _allowed(root: str) -> bool:
        return root in diagnosis_options

    retry_rate_value: Optional[float] = None
    match = re.search(r"retry[_ ]rate[^0-9]*([0-9]*\.?[0-9]+)", text)
    if match:
        try:
            retry_rate_value = float(match.group(1))
        except ValueError:
            retry_rate_value = None

    has_retry_pattern = any(k in text for k in ["retries", "retry", "backoff", "jitter"])
    has_errors = any(k in text for k in ["error_rate", "error rate", "errors", "5xx", "failed", "failure"])
    has_saturation = any(k in text for k in ["saturation", "maxed", "100%", "throttl"])
    has_backlog = any(k in text for k in ["backlog", "queue", "queue_depth", "queue depth"])
    has_latency = any(k in text for k in ["latency", "slow", "timed out", "timeout", "p95", "p99"])
    has_packet_loss = any(k in text for k in ["packet loss", "degraded probe", "network instability", "connection failures across zones", "network partition"])
    has_connections_or_queue = any(k in text for k in ["high connections", "connection wait", "queue buildup", "slow queries", "db saturation", "pool saturation"])
    retry_rate_high = retry_rate_value is not None and retry_rate_value > 0.1

    if has_packet_loss and _allowed("NETWORK_PARTITION"):
        return "NETWORK_PARTITION"

    if (retry_rate_high and has_errors and has_retry_pattern and (has_saturation or has_backlog)) and _allowed("RATE_LIMITED"):
        return "RATE_LIMITED"

    if (has_latency and has_backlog and not retry_rate_high) and _allowed("TIMEOUT"):
        return "TIMEOUT"

    if has_connections_or_queue and _allowed("DB_OVERLOAD"):
        return "DB_OVERLOAD"

    keyword_map = [
        ("cache stale", "CACHE_STALE"),
        ("stale data", "CACHE_STALE"),
        ("inconsistent results", "CACHE_STALE"),
        ("cache miss", "CACHE_MISS"),
        ("miss rate", "CACHE_MISS"),
        ("template render failed", "TEMPLATE_ERROR"),
        ("template processing error", "TEMPLATE_ERROR"),
        ("template", "TEMPLATE_ERROR"),
        ("dependency unavailable", "DEPENDENCY_DOWN"),
        ("connection refused", "DEPENDENCY_DOWN"),
        ("downstream failing", "DEPENDENCY_DOWN"),
        ("service unavailable", "DEPENDENCY_DOWN"),
        ("503", "DEPENDENCY_DOWN"),
        ("rate limit", "RATE_LIMITED"),
        ("too many requests", "RATE_LIMITED"),
        ("429", "RATE_LIMITED"),
        ("timeout", "TIMEOUT"),
        ("timed out", "TIMEOUT"),
        ("service crash", "SERVICE_CRASH"),
        ("restarts", "SERVICE_CRASH"),
        ("crash", "SERVICE_CRASH"),
        ("memory leak", "MEMORY_LEAK"),
        ("out of memory", "MEMORY_LEAK"),
        ("oom", "MEMORY_LEAK"),
        ("disk full", "DISK_FULL"),
        ("no space left", "DISK_FULL"),
        ("storage full", "DISK_FULL"),
        ("config error", "CONFIG_ERROR"),
        ("misconfigured", "CONFIG_ERROR"),
        ("invalid config", "CONFIG_ERROR"),
        ("auth failed", "AUTH_FAILURE"),
        ("unauthorized", "AUTH_FAILURE"),
        ("forbidden", "AUTH_FAILURE"),
        ("401", "AUTH_FAILURE"),
        ("403", "AUTH_FAILURE"),
        ("deploy regression", "DEPLOY_REGRESSION"),
        ("regression", "DEPLOY_REGRESSION"),
        ("payments down", "PAYMENTS_DOWN"),
        ("payments unavailable", "PAYMENTS_DOWN"),
    ]

    for keyword, root in keyword_map:
        if keyword in text and _allowed(root):
            return root

    return None


def _safe_action_from_payload(
    payload: Dict,
    fallback_service: Optional[str],
    diagnosis_options: Optional[List[str]] = None,
    observation_message: Optional[str] = None,
) -> BackendDiagnosisAction:
    """Parse action JSON safely; keep a simple fallback when invalid."""

    try:
        action_type = payload.get("type")
    except Exception:
        action_type = None

    if action_type not in {"open_logs", "scroll_logs", "view_metrics", "submit_diagnosis"}:
        return BackendDiagnosisAction(type="open_logs", service=fallback_service)

    service = payload.get("service")
    root_cause = payload.get("root_cause")
    severity = payload.get("severity")

    if diagnosis_options and root_cause not in diagnosis_options:
        closest = difflib.get_close_matches(str(root_cause), diagnosis_options, n=1, cutoff=0.0)
        root_cause = closest[0] if closest else diagnosis_options[0]

    if action_type == "scroll_logs":
        service = service or fallback_service
    if action_type in {"open_logs", "view_metrics", "submit_diagnosis"}:
        service = service or fallback_service

    if action_type == "submit_diagnosis":
        if severity not in {"low", "medium", "high"}:
            severity = _infer_severity_from_text(observation_message)
    else:
        root_cause = None
        severity = None

    return BackendDiagnosisAction(
        type=action_type,
        service=service,
        root_cause=root_cause,
        severity=severity,
    )


def _infer_severity_from_text(message: Optional[str]) -> str:
    text = (message or "").lower()

    high_markers = [
        "outage",
        "down",
        "crash",
        "oom",
        "connection refused",
        "service unavailable",
        "packet loss",
    ]
    if any(m in text for m in high_markers):
        return "high"

    medium_markers = [
        "latency",
        "backlog",
        "queue",
        "retry",
        "saturation",
        "error_rate",
        "error rate",
        "timeout",
    ]
    if any(m in text for m in medium_markers):
        return "medium"

    return "medium"


def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m backend_diagnosis.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 7860)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn backend_diagnosis.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)
