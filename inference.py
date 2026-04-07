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
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import difflib

import requests
from openai import OpenAI

from models import BackendDiagnosisAction, BackendDiagnosisObservation


def _emit(event: str, fields: Dict[str, object]) -> None:
    if event == "START":
        print(
            f"[START] task={fields.get('task')} env=backend_diagnosis model={fields.get('model')}",
            flush=True,
        )
        return
    if event == "END":
        success = str(bool(fields.get("success", False))).lower()
        steps = int(fields.get("steps", 0) or 0)
        score = float(fields.get("score", 0.0) or 0.0)
        rewards = str(fields.get("rewards", ""))
        print(f"[END] success={success} steps={steps} score={score:.2f} rewards={rewards}", flush=True)
        return
    return


def _format_action_str(action: BackendDiagnosisAction) -> str:
    if action.type == "submit_diagnosis":
        return (
            "submit_diagnosis("
            f"service={action.service}, root_cause={action.root_cause}, severity={action.severity}"
            ")"
        )
    if action.type in {"open_logs", "view_metrics"}:
        return f"{action.type}(service={action.service})"
    if action.type == "scroll_logs":
        if action.service:
            return f"scroll_logs(service={action.service})"
        return "scroll_logs()"
    return f"{action.type}()"


def _post_json(session: requests.Session, url: str, payload: Dict) -> Dict:
    response = session.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def _reset(session: requests.Session, env_url: str, seed: Optional[int], difficulty: Optional[str]) -> Tuple[BackendDiagnosisObservation, Dict[str, object]]:
    payload: Dict[str, object] = {}
    if seed is not None:
        payload["seed"] = seed
    if difficulty is not None:
        payload["difficulty"] = difficulty

    data = _post_json(session, f"{env_url}/reset", payload)
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


def _step(session: requests.Session, env_url: str, action: BackendDiagnosisAction, meta: Dict[str, object]) -> Tuple[BackendDiagnosisObservation, float, bool]:
    payload: Dict[str, object] = {"action": {
        "type": action.type,
        "service": action.service,
        "root_cause": action.root_cause,
        "severity": action.severity,
    }}
    # Preserve any server-provided metadata (episode/session IDs) if present
    payload.update({k: v for k, v in meta.items() if k not in payload})

    data = _post_json(session, f"{env_url}/step", payload)
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


def _grade(session: requests.Session, env_url: str, action: BackendDiagnosisAction, seed: Optional[int], difficulty: Optional[str]) -> float:
    payload: Dict[str, object] = {
        "seed": seed,
        "difficulty": difficulty,
        "service": action.service,
        "root_cause": action.root_cause,
        "severity": action.severity,
    }
    data = _post_json(session, f"{env_url}/grader", payload)
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
    base_url: str = "http://localhost:7860",
    llm_base_url: str = "https://api.openai.com/v1",
    api_key: Optional[str] = None,
) -> Dict[str, float]:
    """
    Run baseline episodes per difficulty with reproducible seeds.
    """

    seeds = seeds or [42, 43, 44]
    difficulties = ["easy", "medium", "hard"]
    results: Dict[str, float] = {}

    use_openai = mode == "openai"

    if use_openai and not api_key:
        _emit("END", {"success": False, "steps": 0, "score": 0.0, "rewards": ""})
        return {}

    for difficulty in difficulties:
        scores: List[float] = []
        for idx in range(episodes_per_difficulty):
            seed = seeds[idx % len(seeds)]
            episode_step_rewards: List[float] = []
            _emit(
                "START",
                {
                    "task": difficulty,
                    "model": model,
                },
            )
            if use_openai:
                score = _run_openai_episode(
                    model=model,
                    max_steps=max_steps,
                    seed=seed,
                    difficulty=difficulty,
                    base_url=base_url,
                    llm_base_url=llm_base_url,
                    api_key=api_key,
                    step_rewards=episode_step_rewards,
                )
            else:
                score = _run_oracle_episode(
                    seed=seed,
                    difficulty=difficulty,
                    base_url=base_url,
                    step_rewards=episode_step_rewards,
                )
            scores.append(score)
            rewards_csv = ",".join(f"{r:.2f}" for r in episode_step_rewards)
            _emit(
                "END",
                {
                    "success": score > 0,
                    "steps": len(episode_step_rewards),
                    "score": score,
                    "rewards": rewards_csv,
                },
            )
        results[difficulty] = sum(scores) / len(scores)
    return results


def _run_openai_episode(
    model: str,
    max_steps: int,
    seed: Optional[int],
    difficulty: Optional[str],
    base_url: str,
    llm_base_url: str,
    api_key: str,
    step_rewards: Optional[List[float]] = None,
) -> float:
    client = OpenAI(api_key=api_key, base_url=llm_base_url)
    llm_temperature = 0
    session = requests.Session()
    obs, meta = _reset(session, base_url, seed=seed, difficulty=difficulty)
    meta.setdefault("seed", seed)
    meta.setdefault("difficulty", difficulty)
    final_action: Optional[BackendDiagnosisAction] = None
    last_service: Optional[str] = None
    consecutive_invalid = 0
    last_actions: List[Tuple[Optional[str], Optional[str]]] = []
    last_signal_count: int = obs.signals_discovered or 0
    no_progress_count = 0
    best_signal_message: str = ""
    best_signal_service: Optional[str] = None
    service_visit_count: Dict[str, int] = {}
    service_signal_map: Dict[str, int] = {}
    service_signal_hypothesis: Dict[str, Optional[str]] = {}
    visited_services = set()
    episode_history: List[Dict[str, object]] = []
    last_step_had_strong_signal = False
    revisiting_without_new_evidence = False

    system_prompt = (
        "You are a backend diagnosis agent.\n\n"
        "Your job is to identify the root cause of an incident using logs and metrics.\n\n"
        "Do not just find signals. You must confirm or reject hypotheses.\n\n"
        "CORE THINKING RULE (apply every step):\n"
        "1. Form one likely hypothesis.\n"
        "2. Check what evidence supports it.\n"
        "3. Check what evidence contradicts it.\n"
        "4. If contradicted, discard it and pick a better hypothesis.\n"
        "5. If evidence is insufficient, explore another service.\n\n"
        "ELIMINATION RULE (before submit):\n"
        "- Consider at least one alternative root cause.\n"
        "- Explain why the alternative is less likely than the chosen cause.\n"
        "- Example: latency + backlog suggests TIMEOUT, but high retry_rate makes RATE_LIMITED more likely.\n\n"
        "CROSS-SERVICE RULE:\n"
        "- Symptoms in one service may be caused by another.\n"
        "- If a service shows symptoms, consider upstream/downstream dependencies.\n\n"
        "CONFIDENCE RULE:\n"
        "- Weak or conflicting evidence: explore.\n"
        "- Strong and consistent evidence with no contradiction: submit.\n"
        "- Do not submit unless evidence is consistent and not contradicted.\n\n"
        "Keep reasoning short: 1-2 sentences only. No long chain-of-thought.\n"
        "Do not hallucinate explanations not present in observations.\n\n"
        "ROOT CAUSE MAPPING (VERY IMPORTANT):\n"
        "NETWORK_PARTITION: packet loss, degraded probes, cross-zone connection failures.\n"
        "RATE_LIMITED: high retry_rate with retries/errors and saturation/backoff.\n"
        "TIMEOUT: high latency with queue/backlog and no clear high retry_rate.\n"
        "DB_OVERLOAD: high connections, queue buildup, slow queries/saturation.\n"
        "CACHE_MISS: high miss rate with latency increase.\n"
        "CACHE_STALE: stale data errors, inconsistent results.\n"
        "TEMPLATE_ERROR: render/template processing failures.\n"
        "DEPENDENCY_DOWN: upstream healthy, downstream unavailable/refused.\n"
        "SERVICE_CRASH: restarts/crash logs.\n"
        "MEMORY_LEAK: steadily increasing memory, OOM patterns.\n"
        "Important: High retry_rate indicates RATE_LIMITED, not TIMEOUT.\n\n"
        "OUTPUT FORMAT:\n"
        "Return JSON only: {\"thought\": \"short reasoning with hypothesis, evidence, and decision\", \"action\": {...}}\n"
        "Thought must include: hypothesis, key evidence, and decision (explore vs submit).\n"
        "Only include root_cause when submitting."
    )

    def _has_observation_strong_signal(message: str) -> bool:
        text = (message or "").lower()
        latency = any(k in text for k in ["latency", "p95", "p99", "slow"]) 
        errors = any(k in text for k in ["error", "5xx", "failed", "failure"]) 
        saturation = any(k in text for k in ["saturation", "backlog", "queue", "maxed", "utilization", "100%", "connection wait"]) 
        network_deg = any(k in text for k in ["packet loss", "degraded probe", "network instability", "network partition"]) 
        return (latency and errors and saturation) or network_deg

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

    def _parse_action_payload_strict(raw_text: str) -> Dict[str, object]:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and isinstance(parsed.get("action"), dict):
            return parsed.get("action")
        raise ValueError("missing_action_object")

    def _parse_action_payload_with_repair(raw_text: str) -> Dict[str, object]:
        try:
            return _parse_action_payload_strict(raw_text)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", raw_text or "")
            if not match:
                raise
            return _parse_action_payload_strict(match.group(0))

    def _request_action_payload(user_prompt_text: str) -> Optional[Dict[str, object]]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_text},
        ]
        for attempt in range(2):
            retry_suffix = "\nReturn ONLY valid JSON." if attempt == 1 else ""
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages if attempt == 0 else [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt_text + retry_suffix},
                    ],
                    temperature=llm_temperature,
                )
                content = completion.choices[0].message.content if completion.choices else ""
                return _parse_action_payload_with_repair(content)
            except Exception:
                if attempt == 0:
                    continue
                return None
        return None

    def _is_valid_action_payload(payload: Dict[str, object]) -> bool:
        action_type = payload.get("type")
        if action_type == "scroll":
            payload["type"] = "scroll_logs"
            action_type = "scroll_logs"
        allowed = {"view_metrics", "open_logs", "scroll_logs", "submit_diagnosis"}
        return action_type in allowed

    def _safe_recovery_action(current_obs: BackendDiagnosisObservation, current_last_service: Optional[str]) -> Optional[BackendDiagnosisAction]:
        services = list(current_obs.available_services or [])
        service = current_last_service or (services[0] if services else None)
        tools = set(current_obs.available_tools or [])

        if service and "view_metrics" in tools:
            return BackendDiagnosisAction(type="view_metrics", service=service)
        if service and "open_logs" in tools:
            return BackendDiagnosisAction(type="open_logs", service=service)
        if "scroll_logs" in tools:
            return BackendDiagnosisAction(type="scroll_logs")
        return None

    episode_max_steps = max_steps + 2 if difficulty == "hard" else max_steps

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
            "Do not repeat the same hypothesis without new evidence.\n"
            "If action.type is submit_diagnosis, include severity as one of: low, medium, high.\n"
            "Return ONLY JSON in this exact form:\n"
            "{\"thought\":\"step-by-step reasoning\",\"action\":{\"type\":\"...\",\"service\":\"...\",\"root_cause\":\"...\",\"severity\":\"...\"}}"
        )

        parse_error: Optional[str] = None
        action_payload = _request_action_payload(user_prompt)
        if action_payload is None:
            recovery_action = _safe_recovery_action(obs, last_service)
            if recovery_action is None:
                action_str = "unknown_action()"
                print(
                    f"[STEP] step={step_idx} action={action_str} reward={float(0.0):.2f} done={str(False).lower()} error=parse_failed",
                    flush=True,
                )
                if step_rewards is not None:
                    step_rewards.append(float(0.0))
                continue

            action = recovery_action
            parse_error = "parse_failed_recovered"

        elif not _is_valid_action_payload(action_payload):
            recovery_action = _safe_recovery_action(obs, last_service)
            if recovery_action is None:
                action_str = "invalid_action()"
                print(
                    f"[STEP] step={step_idx} action={action_str} reward={float(0.0):.2f} done={str(False).lower()} error=Invalid action",
                    flush=True,
                )
                if step_rewards is not None:
                    step_rewards.append(float(0.0))
                continue

            action = recovery_action
            parse_error = "Invalid action"

        else:
            try:
                action = _safe_action_from_payload(action_payload, last_service, obs.diagnosis_options, obs.message)
            except ValueError:
                recovery_action = _safe_recovery_action(obs, last_service)
                if recovery_action is None:
                    if step_rewards is not None:
                        step_rewards.append(float(0.0))
                    continue
                action = recovery_action
                parse_error = "Invalid action"

        # Control Rule 1: service switching after repeated no-signal visits.
        if parse_error is None and action.type != "submit_diagnosis":
            candidate_service = action.service or last_service
            if (
                candidate_service
                and service_visit_count.get(candidate_service, 0) >= 2
                and service_signal_map.get(candidate_service, 0) == 0
            ):
                available_services = list(obs.available_services or [])
                unexplored = [s for s in available_services if s not in visited_services and s != candidate_service]
                if unexplored:
                    next_service = unexplored[0]
                else:
                    alternatives = [s for s in available_services if s != candidate_service]
                    next_service = min(alternatives, key=lambda s: service_visit_count.get(s, 0)) if alternatives else None
                if next_service:
                    action = BackendDiagnosisAction(type="open_logs", service=next_service)

        # Control Rule 2: smart loop breaker.
        if parse_error is None and action.type != "submit_diagnosis":
            recent3 = last_actions[-3:]
            if len(recent3) == 3 and len(set(recent3)) == 1 and no_progress_count >= 2:
                available_services = list(obs.available_services or [])
                current_forced = action.service or last_service
                alternatives = [s for s in available_services if s != current_forced]
                if alternatives:
                    forced_service = min(alternatives, key=lambda s: service_visit_count.get(s, 0))
                    action = BackendDiagnosisAction(type="open_logs", service=forced_service)

        # Control Rule 3: prevent random exploration by preferring unseen services when stuck.
        if parse_error is None and action.type != "submit_diagnosis":
            available_services = list(obs.available_services or [])
            if len(visited_services) < len(available_services) and no_progress_count >= 2:
                unexplored = [s for s in available_services if s not in visited_services]
                if unexplored:
                    action = BackendDiagnosisAction(type="open_logs", service=unexplored[0])

        # Control Rule 4: confirmation gate before submit, difficulty-aware.
        if parse_error is None and action.type == "submit_diagnosis":
            signal_services = [svc for svc, val in service_signal_map.items() if val > 0]
            signal_service_count = len(signal_services)
            total_signals = obs.signals_discovered or 0
            strong_signal_seen = bool(last_step_had_strong_signal or _has_observation_strong_signal(obs.message) or (best_signal_message and _has_strong_signal(best_signal_message)))

            allow_submit = False
            if difficulty == "easy":
                allow_submit = strong_signal_seen or total_signals >= 1
            elif difficulty == "medium":
                allow_submit = (signal_service_count >= 2) or (strong_signal_seen and total_signals >= 2)
            else:
                hypotheses = [v for v in service_signal_hypothesis.values() if v]
                no_conflict = len(set(hypotheses)) <= 1
                allow_submit = signal_service_count >= 2 and no_conflict

            if not allow_submit:
                available_services = list(obs.available_services or [])
                unexplored = [s for s in available_services if s not in visited_services]
                next_service = unexplored[0] if unexplored else (min(available_services, key=lambda s: service_visit_count.get(s, 0)) if available_services else None)
                if next_service:
                    action = BackendDiagnosisAction(type="view_metrics", service=next_service)

        if parse_error is None and action.type == "submit_diagnosis" and (
            action.root_cause is None
            or action.service is None
            or action.severity not in {"low", "medium", "high"}
        ):
            repair_service = action.service or last_service or ((obs.available_services or [None])[0])
            if repair_service is None:
                if step_rewards is not None:
                    step_rewards.append(float(0.0))
                continue
            action = BackendDiagnosisAction(type="view_metrics", service=repair_service)

        if action.type in {"open_logs", "view_metrics", "submit_diagnosis"} and not action.service:
            action.service = last_service or (obs.available_services[0] if (obs.available_services or []) else None)
        obs, reward, done = _step(session, base_url, action, meta)
        action_str = _format_action_str(action)
        print(
            f"[STEP] step={step_idx} action={action_str} reward={float(reward):.2f} done={str(done).lower()} error={parse_error or 'null'}",
            flush=True,
        )
        if step_rewards is not None:
            step_rewards.append(float(reward))

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
                "reward": float(reward),
            }
        )

        # Track signals progression
        current_signals = obs.signals_discovered or 0
        new_signal_found = current_signals > last_signal_count
        last_step_had_strong_signal = bool(new_signal_found or float(reward) > 0)
        last_signal_count = max(last_signal_count, current_signals)
        no_progress_count = 0 if new_signal_found else no_progress_count + 1
        if _has_strong_signal(obs.message) or new_signal_found:
            best_signal_message = obs.message or best_signal_message
            best_signal_service = action.service or best_signal_service

        current_service = action.service or last_service
        if current_service:
            revisiting_without_new_evidence = (
                current_service in visited_services and not new_signal_found
            )
            visited_services.add(current_service)
            service_visit_count[current_service] = service_visit_count.get(current_service, 0) + 1
            service_signal_map[current_service] = max(service_signal_map.get(current_service, 0), current_signals)
            if new_signal_found:
                service_signal_hypothesis[current_service] = _infer_root_cause(
                    best_signal_message or obs.message,
                    getattr(obs, "diagnosis_options", None),
                )
        else:
            revisiting_without_new_evidence = False

        # Track recent actions for loop detection
        last_actions.append((action.type, action.service))
        if len(last_actions) > 6:
            last_actions.pop(0)

        # Track invalid actions (reward penalty pattern and message check)
        is_invalid = "Invalid action" in obs.message
        consecutive_invalid = consecutive_invalid + 1 if is_invalid else 0
        if is_invalid and not done:
            # Reject invalid action behavior and let the model retry next step.
            continue

        if action.service:
            last_service = action.service
        if action.type == "submit_diagnosis":
            final_action = action

        if done:
            break

    if final_action is None:
        fallback_service = best_signal_service or last_service
        if fallback_service is None and getattr(obs, "available_services", None):
            fallback_service = obs.available_services[0]
        final_action = BackendDiagnosisAction(
            type="submit_diagnosis",
            root_cause=_infer_root_cause(best_signal_message or obs.message, getattr(obs, "diagnosis_options", None)),
            service=fallback_service,
            severity="high",
        )
        obs, reward, done = _step(session, base_url, final_action, meta)
        action_str = _format_action_str(final_action)
        print(
            f"[STEP] step={episode_max_steps + 1} action={action_str} reward={float(reward):.2f} done={str(done).lower()} error=null",
            flush=True,
        )
        if step_rewards is not None:
            step_rewards.append(float(reward))

    score = _grade(session, base_url, final_action, seed=seed, difficulty=difficulty)
    return score


def _run_oracle_episode(
    seed: Optional[int],
    difficulty: Optional[str],
    base_url: str,
    step_rewards: Optional[List[float]] = None,
) -> float:
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
    _obs, reward, done = _step(session, base_url, action, meta)
    action_str = _format_action_str(action)
    print(
        f"[STEP] step={1} action={action_str} reward={float(reward):.2f} done={str(done).lower()} error=null",
        flush=True,
    )
    if step_rewards is not None:
        step_rewards.append(float(reward))
    return _grade(session, base_url, action, seed=seed, difficulty=difficulty)


def _infer_root_cause(message: str, diagnosis_options: Optional[List[str]]) -> Optional[str]:
    """Pick a root cause from options using simple keyword matching."""

    if not diagnosis_options:
        return None

    text = (message or "").lower()
    def _allowed(root: str) -> bool:
        return root in diagnosis_options

    # Prioritize precise distinctions first (especially RATE_LIMITED vs TIMEOUT).
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
    """Parse action JSON safely and reject invalid action types explicitly."""

    try:
        action_type = payload.get("type")
    except Exception:
        action_type = None

    if action_type == "scroll":
        action_type = "scroll_logs"
    if action_type not in {"open_logs", "scroll_logs", "view_metrics", "submit_diagnosis"}:
        raise ValueError("invalid_action_type")

    service = payload.get("service")
    root_cause = payload.get("root_cause")
    severity = payload.get("severity")

    if diagnosis_options and root_cause not in diagnosis_options:
        closest = difflib.get_close_matches(str(root_cause), diagnosis_options, n=1, cutoff=0.0)
        root_cause = closest[0] if closest else diagnosis_options[0]

    # Provide sensible defaults for required fields per action type
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


def _pick_alternate_service(current_service: Optional[str], available_services: Optional[List[str]]) -> Optional[str]:
    if not available_services:
        return None
    for service in available_services:
        if service != current_service:
            return service
    return None


def _has_strong_signal(message: str) -> bool:
    text = (message or "").lower()
    markers = ["error", "spiking", "100%", "maxed", "timeout", "down", "failed"]
    return any(marker in text for marker in markers)


def main():
    parser = argparse.ArgumentParser(description="Run baseline agent over difficulties.")
    parser.add_argument("--mode", choices=["oracle", "openai"], default="openai", help="Baseline mode")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per difficulty")
    parser.add_argument("--base-url", default="http://localhost:7860", help="Environment server base URL (e.g., http://localhost:7860 or https://<space>.hf.space)")
    parser.add_argument("--env-url", default=None, help="Environment server base URL (deprecated; prefer --base-url)")
    parser.add_argument("--llm-base-url", default="https://api.openai.com/v1", help="LLM base URL (OpenAI-compatible)")
    parser.add_argument("--max-steps", type=int, default=10, help="Max steps per episode")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name (e.g., gpt-4o, gpt-4o-mini)")
    args = parser.parse_args()

    api_key = os.environ.get("HF_TOKEN")
    if args.mode == "openai" and not api_key:
        _emit("END", {"success": False, "steps": 0, "score": 0.0, "rewards": ""})
        sys.exit(1)

    env_base_url = args.base_url if args.base_url else "http://localhost:7860"
    if args.env_url:
        env_base_url = args.env_url

    run_baseline_agent(
        mode=args.mode,
        episodes_per_difficulty=args.episodes,
        base_url=env_base_url,
        llm_base_url=os.environ.get("API_BASE_URL", args.llm_base_url),
        max_steps=args.max_steps,
        model=os.environ.get("MODEL_NAME", args.model),
        api_key=api_key,
    )


if __name__ == "__main__":
    main()
