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

from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import Body, Query
from fastapi.responses import RedirectResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from models import BackendDiagnosisAction, BackendDiagnosisObservation, BackendDiagnosisState
    from .backend_diagnosis_environment import BackendDiagnosisEnvironment
except ModuleNotFoundError:
    from models import BackendDiagnosisAction, BackendDiagnosisObservation, BackendDiagnosisState
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
sessions: Dict[str, BackendDiagnosisEnvironment] = {}
env_store = sessions


def _round2(value: object) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return 0.0


def _normalize_grader_score(score: object) -> float:
    eps = 0.01
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        numeric_score = 0.0
    numeric_score = max(eps, min(1.0 - eps, numeric_score))
    return round(numeric_score, 2)


# Remove create_app-provided reset/step routes so we can enforce session persistence.
app.router.routes = [
    route
    for route in app.router.routes
    if not (
        route.path in {"/reset", "/step", "/state"}
        and getattr(route, "methods", None)
        and (("POST" in route.methods) or ("GET" in route.methods and route.path == "/state"))
    )
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
    try:
        obs = env.reset(seed=seed, difficulty=difficulty)
    except Exception as e:
        return {
            "observation": {
                "message": f"Invalid reset: {str(e)}"
            },
            "reward": -0.1,
            "done": False,
        }
    sessions[session_id] = env
    return {
        "observation": obs.model_dump(),
        "reward": _round2(obs.reward),
        "done": obs.done,
        "session_id": session_id,
    }


@app.get("/reset")
def reset_get() -> Dict[str, object]:
    return reset({})


@app.post("/step")
def step(payload: Dict[str, object] = Body(...)) -> Dict[str, object]:
    def _invalid_step(reason: str) -> Dict[str, object]:
        return {
            "observation": {"message": f"Invalid action: {reason}"},
            "reward": -0.1,
            "done": False,
            "info": {},
        }

    session_id = payload.get("session_id")
    if session_id not in sessions:
        return {
            "observation": {"message": "Invalid session_id"},
            "reward": -0.1,
            "done": False,
            "info": {},
        }

    env = sessions[session_id]
    action_payload = payload.get("action") or {}

    if not isinstance(action_payload, dict):
        return _invalid_step("action must be an object")

    try:
        action_type = action_payload.get("type")
        if action_type not in {"open_logs", "scroll_logs", "view_metrics", "submit_diagnosis"}:
            # Keep invalid action type penalty behavior inside environment logic.
            action = BackendDiagnosisAction.model_construct(
                type=action_type,
                service=action_payload.get("service"),
                root_cause=action_payload.get("root_cause"),
                severity=action_payload.get("severity"),
            )
        else:
            action = BackendDiagnosisAction(**action_payload)

        obs = env.step(action)
    except Exception as e:
        return _invalid_step(str(e))

    done = bool(obs.done)
    return {
        "observation": obs.model_dump(),
        "reward": _round2(obs.reward),
        "done": done,
        "info": {
            "signals_discovered": obs.signals_discovered,
            "services_explored": obs.services_explored,
            "progress_score": _round2(obs.progress_score),
        },
        "session_id": session_id,
    }


def _empty_state() -> Dict[str, object]:
    state = BackendDiagnosisState(
        incident_id="no_active_session",
        current_service=None,
        log_pointers={},
        logs_seen={},
        metrics_seen={},
        discovered_signals_count=0,
        steps_taken=0,
        done=False,
        difficulty=None,
        services_visited=set(),
        max_possible_signals=0,
        seen_signals=set(),
        last_action=(None, None),
        action_history={},
    )
    return state.model_dump()


@app.get("/state")
def state(session_id: Optional[str] = Query(default=None)) -> Dict[str, object]:
    env: Optional[BackendDiagnosisEnvironment] = None

    if session_id and session_id in env_store:
        env = env_store[session_id]
    elif env_store:
        latest_session_id = list(env_store.keys())[-1]
        env = env_store[latest_session_id]

    if env is None or env.state is None:
        return {"state": _empty_state(), "session_id": session_id}

    return {"state": env.state.model_dump(), "session_id": session_id}


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
def grade(final_action: dict = Body(...)) -> Dict[str, object]:
    env = BackendDiagnosisEnvironment()
    seed = final_action.get("seed")
    difficulty = final_action.get("difficulty")
    try:
        env.reset(seed=seed, difficulty=difficulty)
    except Exception as e:
        return {
            "score": _normalize_grader_score(0.0),
            "error": f"Invalid difficulty: {str(e)}"
        }

    action_payload = final_action.get("action") or final_action
    if not isinstance(action_payload, dict):
        return {
            "score": _normalize_grader_score(0.0),
            "error": "Invalid action payload"
        }
    action = BackendDiagnosisAction(
        type="submit_diagnosis",
        service=action_payload.get("service"),
        root_cause=action_payload.get("root_cause"),
        severity=action_payload.get("severity"),
    )

    gt = env._current_incident.get("ground_truth", {}) if env._current_incident else {}
    related_root_causes = {
        "TIMEOUT": ["RATE_LIMITED", "DB_OVERLOAD"],
        "RATE_LIMITED": ["TIMEOUT"],
        "DB_OVERLOAD": ["TIMEOUT"],
    }

    root_component = 0.0
    if action.root_cause == gt.get("root_cause"):
        root_component = 0.6
    elif action.root_cause in related_root_causes.get(gt.get("root_cause"), []):
        root_component = 0.6 * 0.6

    service_component = 0.3 if action.service == gt.get("affected_service") else 0.0
    severity_component = 0.1 if action.severity == gt.get("severity") else 0.0

    score = env.grade_episode(action)
    return {
        "score": _normalize_grader_score(score),
        "breakdown": {
            "service": _round2(service_component),
            "root_cause": _round2(root_component),
            "severity": _round2(severity_component),
        },
    }


def main():
    import uvicorn
    uvicorn.run(
        "backend_diagnosis.server.app:app",
        host="0.0.0.0",
        port=7860,
    )


if __name__ == "__main__":
    main()
