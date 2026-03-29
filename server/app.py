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
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from uuid import uuid4
import difflib

from fastapi import Body, HTTPException, Query

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
    model = payload.get("model", "openai/gpt-4o-mini") if payload else "openai/gpt-4o-mini"
    api_key = payload.get("api_key") if payload else None
    base_url = payload.get("base_url", "https://openrouter.ai/api/v1") if payload else "https://openrouter.ai/api/v1"

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
        if not api_key:
            raise HTTPException(status_code=400, detail="api_key is required for openai baseline")
        try:
            from openai import OpenAI
        except Exception as e:
            print(f"OpenAI client import failed: {e}")
            return run_oracle(diff)

        client = OpenAI(api_key=api_key, base_url=base_url)
        scores: List[float] = []
        for idx in range(episodes_per_difficulty):
            s = seeds[idx % len(seeds)]
            env = BackendDiagnosisEnvironment()
            obs = env.reset(seed=s, difficulty=diff)
            final_action: BackendDiagnosisAction | None = None
            last_service: Optional[str] = None
            consecutive_invalid = 0
            EARLY_SUBMIT_SIGNALS: Optional[int] = None
            last_actions: List[Tuple[Optional[str], Optional[str]]] = []
            last_messages: List[str] = []
            last_signal_count: int = obs.signals_discovered or 0
            system_prompt = (
                "You are diagnosing a backend incident.\n"
                "You MUST:\n"
                "- use only provided services\n"
                "- use only provided diagnosis options\n"
                "- follow the exact JSON format\n"
                "- not invent new services or root causes\n\n"
                "Behavior:\n"
                "- avoid repeating the same action on the same service\n"
                "- submit as soon as any strong signal is found\n"
                "- if metrics are uninformative, pivot to logs\n"
                "- if the same action yields the same result, switch tools\n"
                "- if you see an ERROR or abnormal metric, consider submitting\n"
                "- do not repeat actions without new information\n"
                "- explore logs when metrics are insufficient\n"
                "- after seeing an abnormal metric, your next step should be open_logs on the same service\n"
                "- if logs do not show errors, check related services\n"
                "- do not randomly switch services\n"
                "- prefer logs over repeated metrics\n\n"
                "- severity must be one of: low, medium, high\n\n"
                "- root_cause MUST be selected from diagnosis_options\n\n"
                "Always return ONLY JSON."
            )
            for _ in range(max_steps):
                user_prompt = (
                    "You must return a single JSON object with the next action.\n"
                    "Fields:\n"
                    "- type: one of open_logs, scroll_logs, view_metrics, submit_diagnosis\n"
                    "- service: only include when required (open_logs, view_metrics, submit_diagnosis) and must be from available_services\n"
                    "- root_cause: only for submit_diagnosis and must be from diagnosis_options\n"
                    "- severity: only for submit_diagnosis\n\n"
                    f"Observation: {obs.message}\n"
                    f"Available tools: {obs.available_tools}\n"
                    f"Available services: {getattr(obs, 'available_services', [])}\n"
                    f"Diagnosis options: {getattr(obs, 'diagnosis_options', [])}\n"
                    "Return ONLY the JSON object, no extra text.\n"
                    "Format:\n"
                    "{\n"
                    '  "type": "open_logs|scroll_logs|view_metrics|submit_diagnosis",\n'
                    '  "service": "... (only if required)",\n'
                    '  "root_cause": "... (only for submit)",\n'
                    '  "severity": "... (only for submit)"\n'
                    "}"
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
                    action_payload = json.loads(content) if content else {}
                except Exception as e:
                    print(f"OpenAI call failed: {e}")
                    action_payload = {}

                action = _safe_action_from_payload(action_payload, last_service, obs.diagnosis_options)
                obs = env.step(action)

                current_signals = obs.signals_discovered or 0
                new_signal_found = current_signals > last_signal_count
                last_signal_count = max(last_signal_count, current_signals)

                last_actions.append((action.type, action.service))
                last_messages.append(obs.message or "")
                if len(last_actions) > 4:
                    last_actions.pop(0)
                if len(last_messages) > 4:
                    last_messages.pop(0)

                if (
                    EARLY_SUBMIT_SIGNALS is not None
                    and not obs.done
                    and action.type != "submit_diagnosis"
                    and getattr(obs, "signals_discovered", 0) is not None
                    and obs.signals_discovered >= EARLY_SUBMIT_SIGNALS
                ):
                    submit_service = last_service or (obs.available_services[0] if getattr(obs, "available_services", None) else None)
                    submit_root = (obs.diagnosis_options[0] if getattr(obs, "diagnosis_options", None) else None)
                    early_action = BackendDiagnosisAction(
                        type="submit_diagnosis",
                        service=submit_service,
                        root_cause=submit_root,
                        severity=None,
                    )
                    obs = env.step(early_action)
                    final_action = early_action
                    break

                if (
                    not obs.done
                    and action.type != "submit_diagnosis"
                    and len(last_actions) == 4
                    and len(last_messages) == 4
                    and len(set(last_actions)) == 1
                    and len(set(last_messages)) == 1
                    and not new_signal_found
                ):
                    submit_service = last_service or (obs.available_services[0] if getattr(obs, "available_services", None) else None)
                    submit_root = _infer_root_cause(obs.message, getattr(obs, "diagnosis_options", None))
                    stall_action = BackendDiagnosisAction(
                        type="submit_diagnosis",
                        service=submit_service,
                        root_cause=submit_root,
                        severity=None,
                    )
                    obs = env.step(stall_action)
                    final_action = stall_action
                    break

                is_invalid = "Invalid action" in obs.message
                consecutive_invalid = consecutive_invalid + 1 if is_invalid else 0
                if consecutive_invalid >= 3 and not obs.done:
                    forced_action = BackendDiagnosisAction(
                        type="submit_diagnosis",
                        service=last_service or (obs.available_services[0] if getattr(obs, "available_services", None) else None),
                        root_cause=_infer_root_cause(obs.message, getattr(obs, "diagnosis_options", None)),
                        severity=None,
                    )
                    obs = env.step(forced_action)
                    final_action = forced_action
                    break

                if action.service:
                    last_service = action.service
                if action.type == "submit_diagnosis":
                    final_action = action
                if obs.done:
                    break

            if final_action is None:
                final_action = BackendDiagnosisAction(
                    type="submit_diagnosis",
                    service=env.state.current_service,
                    root_cause=_infer_root_cause(obs.message, getattr(obs, "diagnosis_options", None)),
                    severity=None,
                )
                env.step(final_action)
            scores.append(env.grade_episode(final_action))
        return sum(scores) / len(scores)

    oracle_results = {
        "easy": run_oracle("easy"),
        "medium": run_oracle("medium"),
        "hard": run_oracle("hard"),
    }

    openai_results = None
    if mode == "openai":
        openai_results = {
            "easy": run_openai("easy"),
            "medium": run_openai("medium"),
            "hard": run_openai("hard"),
        }

    return {
        "oracle": oracle_results,
        "openai": openai_results,
    }


def _infer_root_cause(message: str, diagnosis_options: Optional[List[str]]) -> Optional[str]:
    """Pick a root cause from options using simple keyword matching."""

    if not diagnosis_options:
        return None

    text = (message or "").lower()
    keyword_map = [
        ("db timeout", "DB_OVERLOAD"),
        ("database", "DB_OVERLOAD"),
        ("cache stale", "CACHE_STALE"),
        ("stale cache", "CACHE_STALE"),
        ("cache miss", "CACHE_MISS"),
        ("network partition", "NETWORK_PARTITION"),
        ("upstream timeout", "NETWORK_PARTITION"),
        ("template render failed", "TEMPLATE_ERROR"),
        ("template", "TEMPLATE_ERROR"),
        ("deploy regression", "DEPLOY_REGRESSION"),
        ("regression", "DEPLOY_REGRESSION"),
        ("payments down", "PAYMENTS_DOWN"),
        ("payments unavailable", "PAYMENTS_DOWN"),
        ("service crash", "SERVICE_CRASH"),
        ("crash", "SERVICE_CRASH"),
        ("memory leak", "MEMORY_LEAK"),
        ("out of memory", "MEMORY_LEAK"),
        ("oom", "MEMORY_LEAK"),
    ]

    for keyword, root in keyword_map:
        if keyword in text and root in diagnosis_options:
            return root

    return diagnosis_options[0]


def _safe_action_from_payload(
    payload: Dict,
    fallback_service: Optional[str],
    diagnosis_options: Optional[List[str]] = None,
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

    if severity not in {"low", "medium", "high"}:
        severity = "high"

    if diagnosis_options and root_cause not in diagnosis_options:
        closest = difflib.get_close_matches(str(root_cause), diagnosis_options, n=1, cutoff=0.0)
        root_cause = closest[0] if closest else diagnosis_options[0]

    if action_type == "scroll_logs":
        service = service or fallback_service
    if action_type in {"open_logs", "view_metrics", "submit_diagnosis"}:
        service = service or fallback_service

    return BackendDiagnosisAction(
        type=action_type,
        service=service,
        root_cause=root_cause,
        severity=severity,
    )


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m backend_diagnosis.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn backend_diagnosis.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
