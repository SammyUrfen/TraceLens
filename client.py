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
import difflib

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
        available_services=obs_data.get("available_services"),
        diagnosis_options=obs_data.get("diagnosis_options"),
        signals_discovered=obs_data.get("signals_discovered"),
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
        available_services=obs_data.get("available_services"),
        diagnosis_options=obs_data.get("diagnosis_options"),
        signals_discovered=obs_data.get("signals_discovered"),
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
    model: str = "openai/gpt-3.5-turbo",
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
            print(f"[baseline] mode={mode} difficulty={difficulty} episode={idx + 1}/{episodes_per_difficulty} seed={seed}")
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
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    client = OpenAI(base_url=openai_base_url)
    session = requests.Session()
    obs, meta = _reset(session, base_url, seed=seed, difficulty=difficulty)
    meta.setdefault("seed", seed)
    meta.setdefault("difficulty", difficulty)
    final_action: Optional[BackendDiagnosisAction] = None
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

    for step_idx in range(1, max_steps + 1):
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
            print(f"[openai] step={step_idx} raw_content={content[:200]!r}")
            action_payload = json.loads(content)
        except Exception as e:
            print(f"[openai] step={step_idx} parse_error={e}")
            action_payload = {}

        action = _safe_action_from_payload(action_payload, last_service, obs.diagnosis_options)
        print(f"[openai] step={step_idx} action={action}")

        obs, reward, done = _step(session, base_url, action, meta)
        print(f"[openai] step={step_idx} reward={reward} done={done} msg_preview={obs.message[:120]!r}")

        # Track signals progression
        current_signals = obs.signals_discovered or 0
        new_signal_found = current_signals > last_signal_count
        last_signal_count = max(last_signal_count, current_signals)

        # Track recent actions and messages for stall detection
        last_actions.append((action.type, action.service))
        last_messages.append(obs.message or "")
        if len(last_actions) > 4:
            last_actions.pop(0)
        if len(last_messages) > 4:
            last_messages.pop(0)

        # Early submit when enough evidence is found
        if (
            EARLY_SUBMIT_SIGNALS is not None
            and not done
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
            print(f"[openai] step={step_idx} early_submit signals={obs.signals_discovered} action={early_action}")
            obs, reward, done = _step(session, base_url, early_action, meta)
            print(f"[openai] early submission reward={reward} done={done}")
            final_action = early_action
            break

        # Stall detection: same action and identical message repeated
        if (
            not done
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
            print(f"[openai] step={step_idx} stall_detected last_actions={last_actions} forcing submit {stall_action}")
            obs, reward, done = _step(session, base_url, stall_action, meta)
            print(f"[openai] forced submission (stall) reward={reward} done={done}")
            final_action = stall_action
            break

        # Track invalid actions (reward penalty pattern and message check)
        is_invalid = "Invalid action" in obs.message
        consecutive_invalid = consecutive_invalid + 1 if is_invalid else 0
        if consecutive_invalid >= 3 and not done:
            print(f"[openai] step={step_idx} forced submit after {consecutive_invalid} invalid actions")
            forced_action = BackendDiagnosisAction(
                type="submit_diagnosis",
                service=last_service or (obs.available_services[0] if getattr(obs, "available_services", None) else None),
                root_cause=_infer_root_cause(obs.message, getattr(obs, "diagnosis_options", None)),
                severity=None,
            )
            obs, reward, done = _step(session, base_url, forced_action, meta)
            print(f"[openai] forced submission reward={reward} done={done}")
            final_action = forced_action
            break

        if action.service:
            last_service = action.service
        if action.type == "submit_diagnosis":
            final_action = action

        if done:
            print(f"[openai] step={step_idx} episode_done=True final_action={final_action}")
            break

    if final_action is None:
        print("[openai] no final_action; submitting fallback")
        final_action = BackendDiagnosisAction(
            type="submit_diagnosis",
            root_cause=_infer_root_cause(obs.message, getattr(obs, "diagnosis_options", None)),
            service=last_service,
        )
        obs, reward, done = _step(session, base_url, final_action, meta)
        print(f"[openai] fallback_submission reward={reward} done={done}")

    score = _grade(session, base_url, final_action, seed=seed, difficulty=difficulty)
    print(f"[openai] final score difficulty={difficulty} seed={seed} score={score} final_action={final_action}")
    return score


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
