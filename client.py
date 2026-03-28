# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HTTP baseline client for Backend Diagnosis Environment."""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from openai import OpenAI

from models import BackendDiagnosisAction, BackendDiagnosisObservation


def _post_json(session: requests.Session, url: str, payload: Dict) -> Dict:
    response = session.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def _reset(session: requests.Session, base_url: str, seed: Optional[int], difficulty: Optional[str]) -> Tuple[BackendDiagnosisObservation, Dict[str, object]]:
    payload: Dict[str, object] = {}
    if seed is not None:
        payload["seed"] = seed
    if difficulty is not None:
        payload["difficulty"] = difficulty

    data = _post_json(session, f"{base_url}/reset", payload)
    obs_data = data.get("observation", {})
    meta = {k: v for k, v in data.items() if k not in {"observation", "reward", "done"}}
    observation = BackendDiagnosisObservation(
        message=obs_data.get("message", ""),
        available_tools=obs_data.get("available_tools", []),
    )
    return observation, meta


def _step(session: requests.Session, base_url: str, action: BackendDiagnosisAction, meta: Dict[str, object]) -> Tuple[BackendDiagnosisObservation, float, bool]:
    payload: Dict[str, object] = {"action": {
        "type": action.type,
        "service": action.service,
        "root_cause": action.root_cause,
        "severity": action.severity,
    }}
    # Preserve any server-provided metadata (episode/session IDs) if present
    payload.update({k: v for k, v in meta.items() if k not in payload})

    data = _post_json(session, f"{base_url}/step", payload)
    obs_data = data.get("observation", {})
    observation = BackendDiagnosisObservation(
        message=obs_data.get("message", ""),
        available_tools=obs_data.get("available_tools", []),
    )
    reward = data.get("reward") or 0.0
    done = data.get("done", False)
    return observation, reward, done


def _grade(session: requests.Session, base_url: str, action: BackendDiagnosisAction, seed: Optional[int], difficulty: Optional[str]) -> float:
    payload: Dict[str, object] = {
        "seed": seed,
        "difficulty": difficulty,
        "service": action.service,
        "root_cause": action.root_cause,
        "severity": action.severity,
    }
    data = _post_json(session, f"{base_url}/grader", payload)
    return float(data.get("score", 0.0))


def _load_incidents() -> Dict[str, List[Dict[str, object]]]:
    dataset_path = Path(__file__).parent / "server" / "incidents.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sample_ground_truth(seed: Optional[int], difficulty: Optional[str]) -> Dict[str, Optional[str]]:
    incidents = _load_incidents()
    pool = incidents.get(difficulty or "", [])
    if not pool:
        return {"service": None, "root_cause": None, "severity": None}

    rnd = random.Random(seed)
    incident = rnd.choice(pool)
    gt = incident.get("ground_truth", {})
    return {
        "service": gt.get("affected_service"),
        "root_cause": gt.get("root_cause"),
        "severity": gt.get("severity"),
    }


def run_baseline_agent(
    model: str = "gpt-4o-mini",
    max_steps: int = 10,
    seeds: Optional[List[int]] = None,
    episodes_per_difficulty: int = 3,
    mode: str = "oracle",
    base_url: str = "http://localhost:8000",
) -> Dict[str, float]:
    """Run baseline episodes per difficulty with reproducible seeds.

    If use_openai is False, a deterministic oracle baseline submits ground truth for grading
    (no external API key required). If use_openai is True, requires OPENAI_API_KEY and
    will run the interactive policy.
    """

    seeds = seeds or [42, 43, 44]
    difficulties = ["easy", "medium", "hard"]
    results: Dict[str, float] = {}

    use_openai = mode == "openai"

    for difficulty in difficulties:
        scores: List[float] = []
        for idx in range(episodes_per_difficulty):
            seed = seeds[idx % len(seeds)]
            if use_openai:
                score = _run_openai_episode(
                    model=model,
                    max_steps=max_steps,
                    seed=seed,
                    difficulty=difficulty,
                    base_url=base_url,
                )
            else:
                score = _run_oracle_episode(seed=seed, difficulty=difficulty, base_url=base_url)
            scores.append(score)
        results[difficulty] = sum(scores) / len(scores)

    print(json.dumps(results, indent=2))
    return results


def _run_openai_episode(model: str, max_steps: int, seed: Optional[int], difficulty: Optional[str], base_url: str) -> float:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set when use_openai=True")

    client = OpenAI()
    session = requests.Session()
    obs, meta = _reset(session, base_url, seed=seed, difficulty=difficulty)
    meta.setdefault("seed", seed)
    meta.setdefault("difficulty", difficulty)
    final_action: Optional[BackendDiagnosisAction] = None
    last_service: Optional[str] = None

    system_prompt = (
        "You are an AI agent diagnosing backend incidents.\n"
        "You can use tools to inspect logs and metrics.\n"
        "Your goal is to find the root cause and submit a correct diagnosis."
    )

    for _ in range(max_steps):
        user_prompt = (
            f"Observation: {obs.message}\n"
            f"Available tools: {obs.available_tools}\n"
            "Return your next action in JSON format:\n"
            "{\n"
            '  "type": ...,\n'
            '  "service": ...,\n'
            '  "root_cause": ...,\n'
            '  "severity": ...\n'
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
            action_payload = json.loads(content)
        except Exception:
            action_payload = {}

        action = _safe_action_from_payload(action_payload, last_service)

        obs, reward, done = _step(session, base_url, action, meta)

        if action.service:
            last_service = action.service
        if action.type == "submit_diagnosis":
            final_action = action

        if done:
            break

    if final_action is None:
        final_action = BackendDiagnosisAction(
            type="submit_diagnosis",
            root_cause=None,
            service=last_service,
        )
        obs, reward, done = _step(session, base_url, final_action, meta)

    return _grade(session, base_url, final_action, seed=seed, difficulty=difficulty)


def _run_oracle_episode(seed: Optional[int], difficulty: Optional[str], base_url: str) -> float:
    """Deterministic oracle via HTTP: submit ground truth through step, grade via /grader."""

    session = requests.Session()
    obs, meta = _reset(session, base_url, seed=seed, difficulty=difficulty)
    meta.setdefault("seed", seed)
    meta.setdefault("difficulty", difficulty)
    gt = _sample_ground_truth(seed, difficulty)
    action = BackendDiagnosisAction(
        type="submit_diagnosis",
        service=gt.get("service"),
        root_cause=gt.get("root_cause"),
        severity=gt.get("severity"),
    )
    _step(session, base_url, action, meta)
    return _grade(session, base_url, action, seed=seed, difficulty=difficulty)


def _safe_action_from_payload(payload: Dict, fallback_service: Optional[str]) -> BackendDiagnosisAction:
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

    # Provide sensible defaults for required fields per action type
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


def main():
    parser = argparse.ArgumentParser(description="Run baseline agent over difficulties.")
    parser.add_argument("--mode", choices=["oracle", "openai"], default="oracle", help="Baseline mode")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per difficulty")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--max-steps", type=int, default=10, help="Max steps per episode")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model for openai mode")
    args = parser.parse_args()

    run_baseline_agent(
        mode=args.mode,
        episodes_per_difficulty=args.episodes,
        base_url=args.base_url,
        max_steps=args.max_steps,
        model=args.model,
    )


if __name__ == "__main__":
    main()
