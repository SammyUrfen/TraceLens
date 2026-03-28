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
from typing import Dict, List

from fastapi import Body, Query

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


@app.get("/tasks")
def list_tasks() -> Dict[str, List[Dict[str, object]]]:
    action_schema = {
        "type": ["open_logs", "scroll_logs", "view_metrics", "submit_diagnosis"],
        "service": "string (optional depending on action)",
        "root_cause": "string (required for submit)",
        "severity": "string (optional)",
    }

    tasks: List[Dict[str, object]] = [
        {
            "id": "easy",
            "difficulty": "easy",
            "description": "Simple incident where root cause is directly visible in logs",
            "objective": "Identify the root cause of the backend incident",
            "success_criteria": "Correct root_cause, affected_service, and severity",
            "action_schema": action_schema,
        },
        {
            "id": "medium",
            "difficulty": "medium",
            "description": "Requires multiple steps or metrics to diagnose",
            "objective": "Identify the root cause of the backend incident",
            "success_criteria": "Correct root_cause, affected_service, and severity",
            "action_schema": action_schema,
        },
        {
            "id": "hard",
            "difficulty": "hard",
            "description": "Requires cross-service reasoning and handling misleading signals",
            "objective": "Identify the root cause of the backend incident",
            "success_criteria": "Correct root_cause, affected_service, and severity",
            "action_schema": action_schema,
        },
    ]

    return {"tasks": tasks}


@app.post("/grader")
def grade(final_action: dict = Body(...)) -> Dict[str, float]:
    env = BackendDiagnosisEnvironment()
    seed = final_action.get("seed")
    difficulty = final_action.get("difficulty")
    env.reset(seed=seed, difficulty=difficulty)
    action = BackendDiagnosisAction(
        type="submit_diagnosis",
        service=final_action.get("service"),
        root_cause=final_action.get("root_cause"),
        severity=final_action.get("severity"),
    )
    score = env.grade_episode(action)
    return {"score": score}


@app.get("/baseline")
def baseline(mode: str = Query("oracle", enum=["oracle", "openai"])) -> Dict[str, object]:
    seeds = [42, 43, 44]

    def run_oracle(diff: str) -> float:
        scores: List[float] = []
        for s in seeds:
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
        if "OPENAI_API_KEY" not in os.environ:
            return run_oracle(diff)
        try:
            from openai import OpenAI
        except Exception:
            return run_oracle(diff)

        client = OpenAI()
        scores: List[float] = []
        for s in seeds:
            env = BackendDiagnosisEnvironment()
            obs = env.reset(seed=s, difficulty=diff)
            final_action: BackendDiagnosisAction | None = None
            system_prompt = (
                "You are an AI diagnosing backend incidents.\n"
                "Use tools to inspect logs and metrics before submitting a diagnosis."
            )
            for _ in range(env.MAX_STEPS):
                user_prompt = (
                    f"Observation: {getattr(obs, 'message', '')}\n"
                    f"Available tools: {getattr(obs, 'available_tools', [])}\n"
                    "Return your next action in JSON format:\n"
                    "{\n"
                    '  "type": "open_logs|scroll_logs|view_metrics|submit_diagnosis",\n'
                    '  "service": "...",\n'
                    '  "root_cause": "...",\n'
                    '  "severity": "..."\n'
                    "}"
                )

                try:
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0,
                    )
                    content = completion.choices[0].message.content if completion.choices else ""
                    action_payload = json.loads(content) if content else {}
                except Exception:
                    action_payload = {}

                action = _safe_action_from_payload(action_payload, env)
                obs, reward, done, _info = env.step(action)

                if action.type == "submit_diagnosis":
                    final_action = action
                if done:
                    break

            if final_action is None:
                final_action = BackendDiagnosisAction(
                    type="submit_diagnosis",
                    service=env.state.current_service,
                    root_cause=None,
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
    if mode == "openai" and os.environ.get("OPENAI_API_KEY"):
        openai_results = {
            "easy": run_openai("easy"),
            "medium": run_openai("medium"),
            "hard": run_openai("hard"),
        }

    return {
        "oracle": oracle_results,
        "openai": openai_results,
    }


def _safe_action_from_payload(payload: Dict, env: BackendDiagnosisEnvironment) -> BackendDiagnosisAction:
    try:
        action_type = payload.get("type")
    except Exception:
        action_type = None

    if action_type not in {"open_logs", "scroll_logs", "view_metrics", "submit_diagnosis"}:
        fallback_service = env.state.current_service or env._current_incident.get("entry_service")
        return BackendDiagnosisAction(type="open_logs", service=fallback_service)

    service = payload.get("service")
    root_cause = payload.get("root_cause")
    severity = payload.get("severity")

    if action_type == "open_logs" and service is None:
        service = env.state.current_service or env._current_incident.get("entry_service")
    if action_type == "view_metrics" and service is None:
        service = env.state.current_service or env._current_incident.get("entry_service")
    if action_type == "scroll_logs":
        service = env.state.current_service
    if action_type == "submit_diagnosis":
        service = service or env.state.current_service or env._current_incident.get("entry_service")

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
