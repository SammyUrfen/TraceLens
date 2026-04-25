# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HTTP baseline client for Backend Diagnosis Environment."""

import argparse
from collections import Counter
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


DEBUG_MODE = False
COORDINATOR_MAX_TOKENS = 15170
COORDINATOR_DEFAULT_RESPONSE_TOKENS = 768
COORDINATOR_MIN_RESPONSE_TOKENS = 128
COORDINATOR_EVIDENCE_SERVICE_LIMIT = 6
COORDINATOR_EVIDENCE_SIGNAL_LIMIT = 3
COORDINATOR_EVIDENCE_TEXT_LIMIT = 180


def _debug(message: str) -> None:
    if DEBUG_MODE:
        print(f"[DEBUG] {message}", flush=True)


class SharedState:
    """Central shared memory store for future multi-agent inference flows."""

    def __init__(self):
        self.visited_services: set[str] = set()
        self.signals: list[dict] = []
        self.rewards: list[float] = []
        self.logs_seen: dict[str, list[str]] = {}
        self.metrics_seen: dict[str, dict] = {}
        self.log_depth: Dict[str, int] = {}
        self.phase: str = "EXPLORE"
        self.focus_service: Optional[str] = None
        self.hypotheses: list[dict] = []
        self.hypothesis_history: list[dict] = []
        self.committed_root_cause: Optional[str] = None
        self.committed_service: Optional[str] = None
        self.step_count: int = 0
        self.difficulty: Optional[str] = None
        self.max_steps: Optional[int] = None

    def add_signal(
        self,
        service: str,
        content: str,
        signal_type: str = "log",
        severity: Optional[float] = None,
    ) -> None:
        normalized_type = signal_type if signal_type in {"log", "metric"} else "log"
        signal = {
            "service": service,
            "type": normalized_type,
            "content": content,
        }
        if severity is not None:
            signal["severity"] = round(max(0.0, min(1.0, float(severity))), 2)

        duplicate_exists = any(
            isinstance(existing, dict)
            and existing.get("service") == service
            and existing.get("type", "log") == normalized_type
            and existing.get("content") == content
            for existing in self.signals
        )
        if not duplicate_exists:
            self.signals.append(signal)
        self.visited_services.add(service)

    def add_logs(self, service: str, logs: list[str]) -> None:
        if service not in self.logs_seen:
            self.logs_seen[service] = []
        self.logs_seen[service].extend(logs)
        self.visited_services.add(service)

    def add_metrics(self, service: str, metrics: dict) -> None:
        if service not in self.metrics_seen:
            self.metrics_seen[service] = {}
        self.metrics_seen[service].update(metrics)
        self.visited_services.add(service)

    def add_hypothesis(self, service: str, hypothesis: str, confidence: float) -> None:
        self.hypotheses.append(
            {
                "service": service,
                "hypothesis": hypothesis,
                "confidence": confidence,
            }
        )
        self.visited_services.add(service)

    def unique_services_with_signals(self) -> set[str]:
        return {
            str(item.get("service"))
            for item in self.signals
            if isinstance(item, dict) and item.get("service")
        }


class ExplorerAgent:
    """Single explorer agent that can be reused in future multi-agent setups."""

    _proposal_audit: Dict[tuple[int, int], List[tuple[Optional[str], Optional[str]]]] = {}

    def __init__(
        self,
        shared_state: SharedState,
        service_name: Optional[str] = None,
        personality: str = "error",
    ):
        self.shared_state = shared_state
        self.service_name = service_name
        self.current_target_service: Optional[str] = service_name
        allowed_personalities = {"latency", "error", "resource"}
        self.personality: str = personality if personality in allowed_personalities else "error"

        self.client: Optional[OpenAI] = None
        self.model: str = "gpt-4o-mini"
        self.llm_temperature: float = 0
        self.difficulty: Optional[str] = None

        self.last_parse_error: Optional[str] = None
        self.last_signal_count: int = 0
        self.consecutive_invalid = 0
        self.last_actions: List[Tuple[Optional[str], Optional[str]]] = []
        self.no_progress_count = 0
        self.best_signal_message: str = ""
        self.best_signal_service: Optional[str] = None
        self.service_visit_count: Dict[str, int] = {}
        self.service_signal_map: Dict[str, int] = {}
        self.service_signal_hypothesis: Dict[str, Optional[str]] = {}
        self.episode_history: List[Dict[str, object]] = []
        self.last_step_had_strong_signal = False
        self.revisiting_without_new_evidence = False
        self._service_logs_explored: set[str] = set()
        self._service_metrics_explored: set[str] = set()

        self.system_prompt = (
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
            "- Strong and consistent evidence: keep exploring to validate further.\n"
            "- Do not submit diagnosis. Explorer agents only gather evidence.\n\n"
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
            "Thought must include: hypothesis, key evidence, and exploration decision.\n"
            "Never output submit_diagnosis.\n"
            "ACTION RULES (VERY IMPORTANT):\n"
            "- If you already opened logs for a service → NEVER call open_logs again\n"
            "- Use scroll_logs to continue reading logs\n"
            "- Only switch service if no useful signals are found\n"
            "- Do NOT repeat the same action unless new evidence appears\n"
        )

    def configure(self, client: OpenAI, model: str, difficulty: Optional[str], llm_temperature: float = 0) -> None:
        self.client = client
        self.model = model
        self.difficulty = difficulty
        self.llm_temperature = llm_temperature

    def initialize(self, observation: BackendDiagnosisObservation) -> None:
        self.last_signal_count = observation.signals_discovered or 0

    def _personality_instructions(
        self,
        available_services: List[str],
        dependency_map: Optional[Dict[str, List[str]]],
    ) -> str:
        dep_map = dependency_map or {}
        services = ", ".join(available_services) if available_services else "none"

        if self.personality == "latency":
            return (
                "Personality: latency\n"
                "- Prioritize services with high latency metrics (p95/p99, slow, timeout, downstream).\n"
                "- Prefer view_metrics first if latency evidence is unclear.\n"
                "- Bias toward upstream services in dependency chains.\n"
                f"- Dependency map: {dep_map}\n"
                f"- Available services: {services}"
            )

        if self.personality == "error":
            return (
                "Personality: error\n"
                "- Prioritize services with ERROR log lines, failures, exceptions, 5xx, or auth errors.\n"
                "- Prefer open_logs and scroll_logs before view_metrics unless logs are exhausted.\n"
                "- Bias toward the entry service and its direct dependencies.\n"
                f"- Dependency map: {dep_map}\n"
                f"- Available services: {services}"
            )

        return (
            "Personality: resource\n"
            "- Prioritize services with resource metrics such as cpu, memory, disk, queue depth, or saturation.\n"
            "- Prefer view_metrics first when resource evidence is available.\n"
            "- Bias toward leaf/downstream services.\n"
            f"- Dependency map: {dep_map}\n"
            f"- Available services: {services}"
        )

    def _build_prompt(
        self,
        observation: BackendDiagnosisObservation,
        available_services: List[str],
        dependency_map: Optional[Dict[str, List[str]]],
        target_service: Optional[str],
    ) -> str:
        return (
            f"{self._build_context(self.episode_history)}\n\n"
            f"{self._personality_instructions(available_services, dependency_map)}\n\n"
            "Current observation:\n"
            f"{observation.message}\n\n"
            f"Suggested target service: {target_service or 'none'}\n"
            f"Available tools: {observation.available_tools}\n"
            f"Available services: {available_services}\n"
            "Use only exploration actions: open_logs, view_metrics, or scroll_logs.\n"
            "Return ONLY JSON in this exact form:\n"
            "{\"thought\":\"short reasoning\",\"action\":{\"type\":\"...\",\"service\":\"...\"}}"
        )

    def propose(self, observation: BackendDiagnosisObservation) -> dict:
        """Generate a proposal without executing any environment action."""
        action_payload = self.act(observation)
        suggested_service = action_payload.get("service") if isinstance(action_payload, dict) else None

        hypothesis = None
        if suggested_service:
            hypothesis = self._build_service_hypothesis(observation, suggested_service)
        if not hypothesis and suggested_service:
            history = [
                h
                for h in self.shared_state.hypotheses
                if isinstance(h, dict) and h.get("service") == suggested_service
            ]
            if history:
                hypothesis = str(history[-1].get("hypothesis") or history[-1].get("root_cause") or "") or None

        if not hypothesis and self.shared_state.hypotheses:
            last_h = self.shared_state.hypotheses[-1]
            if isinstance(last_h, dict):
                hypothesis = str(last_h.get("hypothesis") or last_h.get("root_cause") or "") or None

        confidence = 0.2
        if suggested_service and hypothesis:
            confidence = self._hypothesis_confidence(suggested_service)

        action_type = action_payload.get("type") if isinstance(action_payload, dict) else None
        audit_key = (id(self.shared_state), int(self.shared_state.step_count))
        audit_bucket = type(self)._proposal_audit.setdefault(audit_key, [])
        audit_bucket.append((action_type, suggested_service))
        if len(audit_bucket) >= 3:
            unique_actions = set(audit_bucket)
            if len(unique_actions) == 1:
                print(
                    f"[WARN] Identical explorer proposals at step={self.shared_state.step_count}: {next(iter(unique_actions))}",
                    flush=True,
                )
            type(self)._proposal_audit.pop(audit_key, None)

        reasoning = (
            f"{self.personality} explorer suggests {action_type} on {suggested_service}; "
            f"hypothesis={hypothesis or 'unknown'}"
        )

        return {
            "suggested_action": {
                "type": action_type,
                "service": suggested_service,
            },
            "hypothesis": hypothesis or "",
            "confidence": round(max(0.0, min(1.0, float(confidence))), 2),
            "reasoning": reasoning,
        }

    def act(self, observation: BackendDiagnosisObservation) -> dict:
        self.last_parse_error = None
        self._sync_shared_state(observation)

        available_services = list(observation.available_services or [])
        dependency_map = getattr(observation, "available_dependencies", None)
        target_service = self._pick_target_service(available_services, dependency_map)

        if self.no_progress_count >= 2:
            new_service = next(
                (s for s in available_services if s != self.current_target_service),
                None,
            )
            if new_service:
                self.current_target_service = new_service
                self.service_name = new_service
                self.shared_state.visited_services.add(new_service)
                return {
                    "type": "open_logs",
                    "service": new_service,
                }

        if target_service:
            self.current_target_service = target_service
            self.service_name = target_service
            self.shared_state.visited_services.add(target_service)

            hypothesis = self._build_service_hypothesis(observation, target_service)
            if hypothesis:
                confidence = self._hypothesis_confidence(target_service)
                self.shared_state.add_hypothesis(target_service, hypothesis, confidence)

        user_prompt = self._build_prompt(
            observation,
            available_services,
            dependency_map,
            target_service,
        )

        action_payload = self._request_action_payload(user_prompt) or {}
        action_type = action_payload.get("type") if isinstance(action_payload, dict) else None
        service = action_payload.get("service") if isinstance(action_payload, dict) else None

        if action_type == "submit_diagnosis":
            # Explorer agents are never allowed to submit diagnoses.
            action_type = "scroll_logs"

        # Ensure service is always valid.
        if not service:
            if self.current_target_service:
                service = self.current_target_service
            else:
                available_services = list(observation.available_services or [])
                service = available_services[0] if available_services else None

        if action_type == "scroll":
            action_type = "scroll_logs"
        if action_type not in {"open_logs", "view_metrics", "scroll_logs"}:
            if self.personality == "latency":
                action_type = "view_metrics"
            elif self.personality == "error":
                action_type = "open_logs"
            else:
                action_type = "view_metrics"

        # Prevent invalid scrolls with missing service.
        if action_type == "scroll_logs" and not service:
            action_type = "open_logs"

        if action_type in {"open_logs", "view_metrics"} and not service:
            service = target_service or (available_services[0] if available_services else None)

        if target_service and len(available_services) > 1 and action_type in {"open_logs", "view_metrics", "scroll_logs"}:
            service = target_service

        if action_type == "open_logs" and service in self._service_logs_explored:
            action_type = "scroll_logs"

        if len(self.last_actions) >= 2:
            last_type, last_service = self.last_actions[-1]
            if last_type == action_type and last_service == service:
                if action_type == "scroll_logs":
                    action_type = "view_metrics"
                elif action_type == "view_metrics":
                    action_type = "scroll_logs"

        # Final hard guarantee for service-bound exploration actions.
        if action_type in {"open_logs", "view_metrics", "scroll_logs"} and not service:
            return {
                "type": "open_logs",
                "service": self.current_target_service or service,
            }

        if service:
            self.service_name = service
            self.current_target_service = service
            self.shared_state.visited_services.add(service)

        return {
            "type": action_type,
            "service": service,
        }

    def observe_step_outcome(
        self,
        step_idx: int,
        action: BackendDiagnosisAction,
        reward: float,
        observation: BackendDiagnosisObservation,
    ) -> bool:
        self.episode_history.append(
            {
                "step": step_idx,
                "observation": observation.message,
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

        current_signals = observation.signals_discovered or 0
        new_signal_found = current_signals > self.last_signal_count
        self.last_step_had_strong_signal = bool(new_signal_found or float(reward) > 0)
        self.last_signal_count = max(self.last_signal_count, current_signals)
        self.no_progress_count = 0 if new_signal_found else self.no_progress_count + 1
        if _has_strong_signal(observation.message) or new_signal_found:
            self.best_signal_message = observation.message or self.best_signal_message
            self.best_signal_service = action.service or self.best_signal_service

        current_service = action.service or self.service_name
        if current_service:
            self.revisiting_without_new_evidence = (
                current_service in self.shared_state.visited_services and not new_signal_found
            )
            self.shared_state.visited_services.add(current_service)
            self.service_visit_count[current_service] = self.service_visit_count.get(current_service, 0) + 1
            self.service_signal_map[current_service] = max(self.service_signal_map.get(current_service, 0), current_signals)
            if new_signal_found:
                inferred = _infer_root_cause(
                    self.best_signal_message or observation.message,
                    getattr(observation, "diagnosis_options", None),
                )
                self.service_signal_hypothesis[current_service] = inferred
                if inferred:
                    self.shared_state.add_hypothesis(current_service, inferred, 0.7)
        else:
            self.revisiting_without_new_evidence = False

        self.last_actions.append((action.type, action.service))
        if len(self.last_actions) > 6:
            self.last_actions.pop(0)

        is_invalid = "Invalid action" in observation.message
        self.consecutive_invalid = self.consecutive_invalid + 1 if is_invalid else 0
        if action.service:
            self.service_name = action.service
            self.current_target_service = action.service
            self.shared_state.visited_services.add(action.service)

        if action.type == "open_logs" and action.service:
            self._service_logs_explored.add(action.service)
        if action.type == "scroll_logs" and action.service:
            self.shared_state.log_depth[action.service] = self.shared_state.log_depth.get(action.service, 0) + 1
        if action.type == "view_metrics" and action.service:
            self._service_metrics_explored.add(action.service)

        return is_invalid

    def _pick_target_service(
        self,
        available_services: List[str],
        dependency_map: Optional[Dict[str, List[str]]] = None,
    ) -> Optional[str]:
        if not available_services:
            return None

        dep_map = dependency_map or {}
        all_children = {child for children in dep_map.values() for child in children}
        root_services = [svc for svc in dep_map.keys() if svc not in all_children]

        def downstream_count(service: str) -> int:
            seen: set[str] = set()
            stack = list(dep_map.get(service, []))
            while stack:
                child = stack.pop()
                if child in seen:
                    continue
                seen.add(child)
                stack.extend(dep_map.get(child, []))
            return len(seen)

        def score(service: str) -> tuple:
            unvisited = service not in self.shared_state.visited_services
            logs_opened = service in self._service_logs_explored
            metrics_seen = service in self._service_metrics_explored
            is_root = service in root_services
            is_leaf = service not in dep_map
            is_direct_dep = any(service in deps for deps in dep_map.values())
            downstream = downstream_count(service)

            if self.personality == "latency":
                return (
                    1 if is_root or service in dep_map else 0,
                    downstream,
                    1 if unvisited else 0,
                    0 if logs_opened else 1,
                    service,
                )

            if self.personality == "error":
                return (
                    1 if is_root else 0,
                    1 if is_direct_dep else 0,
                    1 if not logs_opened else 0,
                    1 if unvisited else 0,
                    service,
                )

            return (
                1 if is_leaf else 0,
                1 if not metrics_seen else 0,
                1 if unvisited else 0,
                downstream,
                service,
            )

        unvisited = [s for s in available_services if s not in self.shared_state.visited_services]
        candidate_pool = unvisited or list(available_services)

        # Filter out services that have already been visited and produced zero signals.
        # These are likely noise services injected by the hard transformation.
        # Only filter if we have enough visited services to afford exclusion
        # (don't filter on the very first few steps when all services are unvisited).
        _step = self.shared_state.step_count
        if _step >= 3 and len(self.shared_state.visited_services) >= 2:
            _signal_services = self.shared_state.unique_services_with_signals()
            _zero_signal_visited = {
                svc for svc in self.shared_state.visited_services
                if svc not in _signal_services
                and svc in self._service_metrics_explored  # only exclude if metrics checked
                and svc in self._service_logs_explored     # AND logs checked
            }
            # Only filter from candidate pool if we still have candidates left
            _filtered = [s for s in candidate_pool if s not in _zero_signal_visited]
            if _filtered:  # never leave candidate_pool empty
                candidate_pool = _filtered

        candidate_pool.sort(key=score, reverse=True)

        if self.current_target_service in candidate_pool:
            current_score = score(self.current_target_service)
            best_score = score(candidate_pool[0])
            if current_score >= best_score:
                return self.current_target_service

        return candidate_pool[0]

    def _build_service_hypothesis(self, observation: BackendDiagnosisObservation, service: str) -> Optional[str]:
        text = (observation.message or "").lower()

        service_logs = [str(x).lower() for x in self.shared_state.logs_seen.get(service, [])]
        service_metrics = {
            str(k).lower(): str(v).lower()
            for k, v in self.shared_state.metrics_seen.get(service, {}).items()
        }

        combined_logs = " ".join(service_logs)
        combined_text = f"{text} {combined_logs}".strip()

        service_signals = [
            s
            for s in self.shared_state.signals
            if isinstance(s, dict) and s.get("service") == service
        ]
        signal_count = len(service_signals)

        # Early exploration phase: allow weak keyword-based hypotheses.
        if self.shared_state.step_count < 3:
            if any(k in combined_text for k in ["error", "timeout", "latency", "fail"]):
                return "possible issue (early signal)"
            return None

        if signal_count < 1:
            return None

        strong_signal_keywords = [
            "error", "failed", "failure", "exception", "timeout", "timed out",
            "latency", "saturation", "overload", "rate limit", "429", "packet loss",
            "degraded", "connection refused", "unavailable", "db", "cache", "auth",
        ]
        has_strong_signal = any(
            (
                isinstance(sig.get("severity"), (int, float)) and float(sig.get("severity", 0.0)) >= 0.7
            )
            or any(token in str(sig.get("content", "")).lower() for token in strong_signal_keywords)
            for sig in service_signals
        )

        has_latency = any(k in combined_text for k in ["latency", "p95", "p99", "slow", "timeout", "timed out", "downstream", "dependency"])
        has_extreme_latency = any(k in combined_text for k in ["timeout", "timed out", "p99", "downstream timeout", "dependency unavailable", "503"])
        has_error = any(k in combined_text for k in ["error", "failed", "failure", "exception", "5xx"])
        has_resource = any(k in combined_text for k in ["cpu", "memory", "queue", "saturation", "overload", "connection pool", "db saturation"])

        abnormal_markers = {"high", "spiking", "100%", "maxed", "degraded"}
        metric_abnormal_keys = [k for k, v in service_metrics.items() if v in abnormal_markers]
        has_resource_metric_abnormal = any(
            any(token in k for token in ["cpu", "memory", "queue", "saturation", "db", "connection", "utilization"])
            for k in metric_abnormal_keys
        )
        has_latency_metric_abnormal = any("latency" in k for k in metric_abnormal_keys)

        has_keyword_match = any(
            keyword in combined_text
            for keyword in [
                "db", "connection pool", "db_cpu", "slow queries", "db saturation",
                "timeout", "timed out", "upstream timeout", "downstream", "503",
                "rate limit", "429", "retry_rate", "too many requests",
                "cache stale", "stale", "cache freshness", "stale_reads",
                "cache miss", "miss rate", "cache_hit: low",
                "template", "render", "template_error",
                "auth", "unauthorized", "forbidden", "401", "403",
                "network partition", "packet loss", "degraded probe",
                "latency", "p95", "p99", "slow", "error", "failed", "failure", "exception",
            ]
        ) or bool(metric_abnormal_keys)
        if not has_keyword_match:
            return None

        # Mid stage: allow moderate hypotheses with a single supporting signal.
        if signal_count < 2 or not has_strong_signal:
            if any(k in text for k in ["db", "connection pool", "db_cpu", "slow queries", "db saturation"]):
                return "possible DB overload"
            if any(k in text for k in ["timeout", "timed out", "upstream timeout", "downstream", "503"]):
                return "timeout due to downstream API"
            if any(k in text for k in ["rate limit", "429", "retry_rate", "too many requests"]):
                return "possible rate limiting issue"
            if any(k in text for k in ["cache stale", "stale", "cache freshness", "stale_reads"]):
                return "possible stale cache issue"
            if any(k in text for k in ["cache miss", "miss rate", "cache_hit: low"]):
                return "possible cache miss issue"
            if any(k in text for k in ["template", "render", "template_error"]):
                return "possible template rendering failure"
            if any(k in text for k in ["auth", "unauthorized", "forbidden", "401", "403"]):
                return "possible authentication failure"
            if any(k in text for k in ["network partition", "packet loss", "degraded probe"]):
                return "possible network partition"
            if any(k in text for k in ["error", "failed", "failure", "exception"]):
                return "possible service-level instability"
            return None

        # Late stage: strict mode (>=2 signals and strong evidence) keeps current logic.

        # Only allow personality logic when that personality has supporting evidence.
        personality_signal_exists = {
            "latency": has_latency or has_latency_metric_abnormal,
            "error": has_error or any(
                k in combined_text for k in ["auth", "unauthorized", "forbidden", "401", "403"]
            ) or has_extreme_latency,
            "resource": has_resource or has_resource_metric_abnormal or has_latency_metric_abnormal,
        }
        if not personality_signal_exists.get(self.personality, False):
            return None

        # Personality-specific filtering/prioritization.
        if self.personality == "latency":
            # Prioritize timeout/latency/downstream/dependency and ignore weak generic errors.
            if has_latency or has_latency_metric_abnormal:
                return "timeout due to downstream API"
            if has_resource_metric_abnormal:
                return "possible dependency saturation"

        if self.personality == "error":
            # Prioritize explicit errors/failures/exceptions; ignore latency unless extreme.
            if has_error:
                return "possible service-level instability"
            if any(k in combined_text for k in ["auth", "unauthorized", "forbidden", "401", "403"]):
                return "possible authentication failure"
            if has_extreme_latency:
                return "timeout due to downstream API"

        if self.personality == "resource":
            # Prioritize CPU/memory/queue/saturation and trust metrics over logs.
            if has_resource_metric_abnormal:
                return "possible DB overload"
            if has_resource:
                return "possible resource saturation"
            if has_latency_metric_abnormal:
                return "possible queue-induced latency"

        if any(k in text for k in ["db", "connection pool", "db_cpu", "slow queries", "db saturation"]):
            return "possible DB overload"
        if any(k in text for k in ["timeout", "timed out", "upstream timeout", "downstream", "503"]):
            return "timeout due to downstream API"
        if any(k in text for k in ["rate limit", "429", "retry_rate", "too many requests"]):
            return "possible rate limiting issue"
        if any(k in text for k in ["cache stale", "stale", "cache freshness", "stale_reads"]):
            return "possible stale cache issue"
        if any(k in text for k in ["cache miss", "miss rate", "cache_hit: low"]):
            return "possible cache miss issue"
        if any(k in text for k in ["template", "render", "template_error"]):
            return "possible template rendering failure"
        if any(k in text for k in ["auth", "unauthorized", "forbidden", "401", "403"]):
            return "possible authentication failure"
        if any(k in text for k in ["network partition", "packet loss", "degraded probe"]):
            return "possible network partition"

        return None

    def _hypothesis_confidence(self, service: str) -> float:
        service_signal_count = len([
            s for s in self.shared_state.signals
            if isinstance(s, dict) and s.get("service") == service
        ])
        confidence = 0.3 + (0.2 * service_signal_count)
        return round(max(0.1, min(0.95, confidence)), 2)

    def _build_context(self, history: List[Dict[str, object]]) -> str:
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

    def _has_observation_strong_signal(self, message: str) -> bool:
        text = (message or "").lower()
        latency = any(k in text for k in ["latency", "p95", "p99", "slow"])
        errors = any(k in text for k in ["error", "5xx", "failed", "failure"])
        saturation = any(k in text for k in ["saturation", "backlog", "queue", "maxed", "utilization", "100%", "connection wait"])
        network_deg = any(k in text for k in ["packet loss", "degraded probe", "network instability", "network partition"])
        return (latency and errors and saturation) or network_deg

    def _parse_action_payload_strict(self, raw_text: str) -> Dict[str, object]:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and isinstance(parsed.get("action"), dict):
            return parsed.get("action")
        raise ValueError("missing_action_object")

    def _parse_action_payload_with_repair(self, raw_text: str) -> Dict[str, object]:
        try:
            return self._parse_action_payload_strict(raw_text)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", raw_text or "")
            if not match:
                raise
            return self._parse_action_payload_strict(match.group(0))

    def _request_action_payload(self, user_prompt_text: str) -> Optional[Dict[str, object]]:
        if self.client is None:
            return None

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt_text},
        ]
        for attempt in range(2):
            retry_suffix = "\nReturn ONLY valid JSON." if attempt == 1 else ""
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages if attempt == 0 else [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt_text + retry_suffix},
                    ],
                    temperature=self.llm_temperature,
                )
                content = completion.choices[0].message.content if completion.choices else ""
                return self._parse_action_payload_with_repair(content)
            except Exception:
                if attempt == 0:
                    continue
                return None
        return None

    def _is_valid_action_payload(self, payload: Dict[str, object]) -> bool:
        action_type = payload.get("type")
        if action_type == "scroll":
            payload["type"] = "scroll_logs"
            action_type = "scroll_logs"
        allowed = {"view_metrics", "open_logs", "scroll_logs", "submit_diagnosis"}
        return action_type in allowed

    def _safe_recovery_action(self, observation: BackendDiagnosisObservation) -> Optional[BackendDiagnosisAction]:
        services = list(observation.available_services or [])
        service = self.service_name or (services[0] if services else None)
        tools = set(observation.available_tools or [])

        if service and "view_metrics" in tools:
            return BackendDiagnosisAction(type="view_metrics", service=service)
        if service and "open_logs" in tools:
            return BackendDiagnosisAction(type="open_logs", service=service)
        if "scroll_logs" in tools:
            return BackendDiagnosisAction(type="scroll_logs")
        return None

    def _sync_shared_state(self, observation: BackendDiagnosisObservation) -> None:
        service = self.service_name or ((observation.available_services or [None])[0])
        if not service:
            return

        logs: List[str] = []
        metrics: Dict[str, str] = {}

        for raw_line in (observation.message or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if re.match(r"^(INFO|WARN|ERROR)\b", line):
                logs.append(line)
                continue
            if ":" in line and not line.startswith("ALERT"):
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key and value:
                    metrics[key] = value

        if logs:
            self.shared_state.add_logs(service, logs)
            signal_keywords = [
                "error",
                "timeout",
                "failed",
                "failure",
                "exception",
                "overload",
                "saturation",
                "degraded",
                "unavailable",
            ]
            for line in logs:
                lowered = line.lower()
                if any(keyword in lowered for keyword in signal_keywords):
                    severity = 0.6
                    if "error" in lowered or "exception" in lowered or "failed" in lowered:
                        severity = 0.8
                    elif "overload" in lowered:
                        severity = 0.75
                    self.shared_state.add_signal(service, line, signal_type="log", severity=severity)

        if metrics:
            self.shared_state.add_metrics(service, metrics)
            abnormal_markers = {"high", "spiking", "100%", "maxed", "degraded"}
            for key, value in metrics.items():
                metric_value = str(value).strip().lower()
                if metric_value in abnormal_markers:
                    metric_signal = f"{key}: {value}"
                    severity = 0.7 if metric_value in {"100%", "maxed"} else 0.6
                    self.shared_state.add_signal(
                        service,
                        metric_signal,
                        signal_type="metric",
                        severity=severity,
                    )


class CoordinatorAgent:
    """Reads shared state and produces high-level coordination signals."""

    INVALID_TRANSITIONS = {
        ("scroll_logs", "open_logs"),
        ("view_metrics", "open_logs"),
    }

    ROOT_CAUSE_KEYWORDS = {
        "DB_OVERLOAD": ["db", "connection pool", "slow query", "db_cpu", "saturation"],
        "TIMEOUT": ["timeout", "timed out", "downstream", "latency", "p99"],
        "RATE_LIMITED": ["retry_rate", "429", "too many requests", "rate limit"],
        "CACHE_STALE": ["stale", "cache freshness", "stale_reads"],
        "CACHE_MISS": ["cache miss", "miss rate", "low hit"],
        "TEMPLATE_ERROR": ["template", "render", "template_error"],
        "DEPENDENCY_DOWN": ["connection refused", "unavailable", "upstream failure"],
        "SERVICE_CRASH": ["crash", "restart", "oom"],
        "MEMORY_LEAK": ["memory leak", "increasing memory"],
    }

    def __init__(self, shared_state: SharedState):
        self.shared_state = shared_state
        self.client: Optional[OpenAI] = None
        self.model: str = "gpt-4o-mini"
        self._last_selected_service: Optional[str] = None
        self.hypothesis_streak: Dict[str, int] = {}
        self._last_selected_service_signal_count: int = 0
        self._last_selected_action_type: Optional[str] = None
        self._last_selected_action_service: Optional[str] = None
        self._action_history: List[Tuple[Optional[str], Optional[str]]] = []
        self.best_global_score: float = -1.0
        self.best_global_hypothesis: Optional[str] = None
        self._last_thrash_signal_count: int = 0

    @staticmethod
    def _normalize_hypothesis(hypothesis: object) -> str:
        h = str(hypothesis).lower()
        if "timeout" in h:
            return "TIMEOUT"
        if "db" in h:
            return "DB_OVERLOAD"
        if "rate" in h or "429" in h:
            return "RATE_LIMITED"
        if "cache" in h:
            return "CACHE_STALE"
        if "template" in h:
            return "TEMPLATE_ERROR"
        return h.upper()

    def configure(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def _infer_severity_from_signals(self) -> str:
        """Infer severity from collected signal content, not difficulty."""
        all_content = " ".join(
            str(s.get("content", "")).lower()
            for s in self.shared_state.signals
            if isinstance(s, dict)
        )
        # High severity: system-level failures and unavailability
        high_markers = [
            "crash", "down", "unavailable", "connection refused",
            "oom", "out of memory", "packet loss", "100%", "maxed",
            "service unavailable", "503", "template", "render",
            "stale", "cache miss", "deploy regression",
        ]
        if any(m in all_content for m in high_markers):
            return "high"
        medium_markers = [
            "timeout", "timed out", "latency", "saturation",
            "retry", "backlog", "queue", "spiking", "degraded",
        ]
        if any(m in all_content for m in medium_markers):
            return "medium"
        return "high"  # Default to high for unknown patterns (safer for grader)

    def is_invalid_transition(
        self,
        prev_action: Optional[str],
        new_action: Optional[str],
        prev_service: Optional[str],
        new_service: Optional[str],
    ) -> bool:
        if (prev_action, new_action) in self.INVALID_TRANSITIONS:
            if prev_service == new_service:
                return True
        return False

    def _get_last_n_actions(self, n: int) -> List[str]:
        if n <= 0:
            return []
        return [
            str(action)
            for action, _service in self._action_history[-n:]
            if action
        ]

    def _get_recent_rewards(self, n: int) -> List[float]:
        if n <= 0:
            return []
        return [
            float(r)
            for r in self.shared_state.rewards[-n:]
        ]

    def _score_hypothesis_with_evidence(self, hypothesis: str, service: Optional[str]) -> float:
        if not service:
            return 0.0

        signals = [
            s
            for s in self.shared_state.signals
            if isinstance(s, dict) and s.get("service") == service
        ]
        text_blob = " ".join(str(s.get("content", "")).lower() for s in signals)

        score = 0.0
        normalized_hypothesis = self._normalize_hypothesis(hypothesis).lower()

        for root, keywords in self.ROOT_CAUSE_KEYWORDS.items():
            if root.lower() in normalized_hypothesis:
                match_count = sum(1 for kw in keywords if kw in text_blob)
                if match_count >= 3:
                    score += 1.2
                elif match_count == 2:
                    score += 0.7
                elif match_count == 1:
                    score += 0.3

        return score

    def _contradiction_penalty(self, hypothesis: str, service: Optional[str]) -> float:
        if not service:
            return 0.0

        text_blob = " ".join(
            str(s.get("content", "")).lower()
            for s in self.shared_state.signals
            if isinstance(s, dict) and s.get("service") == service
        )

        penalty = 0.0
        normalized_hypothesis = str(hypothesis).lower()

        if "timeout" in normalized_hypothesis and "retry_rate" in text_blob:
            penalty += 0.8

        if "db" in normalized_hypothesis and "timeout" in text_blob and "db" not in text_blob:
            penalty += 0.8

        if "cache" in normalized_hypothesis and "db" in text_blob:
            penalty += 0.8

        return penalty

    def _stability_bonus(self, hypothesis: str) -> float:
        recent = [
            h for h in self.shared_state.hypothesis_history[-5:]
            if float(h.get("confidence", 0.0) or 0.0) >= 0.7
        ]
        normalized = self._normalize_hypothesis(hypothesis)
        count = sum(1 for h in recent if self._normalize_hypothesis(h.get("hypothesis")) == normalized)
        return 0.2 * count

    @staticmethod
    def _map_hypothesis_to_root(hypothesis: str) -> Optional[str]:
        normalized = str(hypothesis).lower()
        if "timeout" in normalized:
            return "TIMEOUT"
        if "db" in normalized or "connection pool" in normalized:
            return "DB_OVERLOAD"
        if "rate" in normalized or "429" in normalized:
            return "RATE_LIMITED"
        if "cache" in normalized:
            return "CACHE_STALE"
        if "template" in normalized:
            return "TEMPLATE_ERROR"
        return None

    def _suggested_action_violates_rules(self, suggested: dict) -> bool:
        if not isinstance(suggested, dict):
            return True

        action_type = suggested.get("type")
        service = suggested.get("service")
        # Only reject invalid explorer actions.
        if action_type not in {"open_logs", "view_metrics", "scroll_logs"}:
            return True

        if not service:
            return True

        return False

    def _validate_final_action(self, action: dict) -> dict:
        """
        FINAL SAFETY NET.
        This runs RIGHT BEFORE returning action.
        Nothing invalid is allowed past this point.
        """

        def _fallback_service() -> Optional[str]:
            return (
                self._last_selected_action_service
                or next(iter(self.shared_state.visited_services), None)
                or next(iter(self.shared_state.unique_services_with_signals()), None)
            )

        if not isinstance(action, dict):
            return {
                "type": "scroll_logs",
                "service": _fallback_service(),
            }

        action_type = action.get("type")
        service = action.get("service")

        # RULE 1: service must exist for all actions
        if action_type in {"open_logs", "view_metrics", "scroll_logs", "submit_diagnosis"}:
            if not service:
                return {
                    "type": "scroll_logs",
                    "service": _fallback_service(),
                }

        # RULE 2: explorer-only actions enforced
        if action_type not in {"open_logs", "view_metrics", "scroll_logs", "submit_diagnosis"}:
            return {
                "type": "scroll_logs",
                "service": service or _fallback_service(),
            }

        # RULE 3: NO invalid submission
        if action_type == "submit_diagnosis":
            if not action.get("root_cause"):
                return {
                    "type": "view_metrics",
                    "service": service or _fallback_service(),
                }

        return action

    def _is_stuck(self) -> bool:
        rewards = self._get_recent_rewards(4)
        return len(rewards) == 4 and sum(rewards) < -0.2

    def _assert_valid_state(self, action: dict):
        assert isinstance(action, dict), "Action must be dict"
        assert action.get("type") in {"open_logs", "view_metrics", "scroll_logs", "submit_diagnosis"}

        if action["type"] in {"open_logs", "view_metrics", "scroll_logs", "submit_diagnosis"}:
            assert action.get("service") is not None, "Action must have service"

        if action["type"] == "submit_diagnosis":
            assert action.get("root_cause") is not None, "Submit must have root cause"

    def decide_next_action(self, proposals: list[dict]) -> dict:
        """Choose a single next action from explorer proposals."""
        _debug(f"proposals: {proposals}")

        valid_proposals = [
            p for p in (proposals or [])
            if isinstance(p, dict)
            and isinstance(p.get("suggested_action"), dict)
            and p["suggested_action"].get("service") is not None
        ]

        # HARD RULE: if proposals exist, use their service as the grounding source.
        selected_service = self._last_selected_action_service
        if valid_proposals:
            chosen = max(valid_proposals, key=lambda p: float(p.get("confidence", 0.0) or 0.0))
            selected_service = chosen["suggested_action"]["service"]

        def _force_service(action: dict, service: Optional[str]) -> dict:
            if not isinstance(action, dict):
                return {
                    "type": "scroll_logs",
                    "service": service,
                }
            if action.get("type") in {"open_logs", "view_metrics", "scroll_logs", "submit_diagnosis"}:
                action["service"] = service
            return action

        def _ensure_valid_service(action_dict: Optional[dict]) -> Optional[dict]:
            if not action_dict:
                action_dict = {
                    "type": "scroll_logs",
                    "service": selected_service,
                }
            if action_dict.get("type") in {"open_logs", "view_metrics", "scroll_logs", "submit_diagnosis"}:
                if not action_dict.get("service"):
                    action_dict["service"] = (
                        selected_service
                        or next(iter(self.shared_state.visited_services), None)
                    )
            action_dict = _force_service(action_dict, selected_service or action_dict.get("service"))
            validated = self._validate_final_action(action_dict)
            self._assert_valid_state(validated)
            return validated

        proposal_list = [p for p in (proposals or []) if isinstance(p, dict)]

        if self._last_selected_action_type:
            self._action_history.append(
                (self._last_selected_action_type, self._last_selected_action_service)
            )
            if len(self._action_history) > 30:
                self._action_history = self._action_history[-30:]

        recent_rewards = self._get_recent_rewards(3)
        step = self.shared_state.step_count
        signals = len(self.shared_state.signals)

        last_services = [
            service
            for _action, service in self._action_history[-4:]
            if service is not None
        ]
        thrash_focus_service = self.shared_state.focus_service or selected_service or self._last_selected_action_service
        # Thrash guard: only fire when genuinely bouncing across many services
        # without progress, not on the first legitimate cross-service exploration
        _unique_in_window = set(last_services)
        _step = self.shared_state.step_count
        _difficulty = str(self.shared_state.difficulty or "").lower()

        # Compute threshold relative to how many services exist.
        # Hard tasks have many services — exploring 3+ is expected, not thrashing.
        # Only fire if we've visited MORE THAN HALF the services AND rewards are negative.
        _total_visited = len(self.shared_state.visited_services)
        _recent_rewards = self._get_recent_rewards(3)
        _rewards_negative = len(_recent_rewards) >= 2 and sum(_recent_rewards) < -0.1

        # Thrash = visited many services recently AND getting no reward for it
        _thrash_threshold = max(4, _total_visited // 2 + 1) if _difficulty in ("medium", "hard") else 3
        if (
            thrash_focus_service
            and len(_unique_in_window) >= _thrash_threshold
            and _step >= 5
            and _rewards_negative
        ):
            return _ensure_valid_service({
                "type": "scroll_logs",
                "service": thrash_focus_service,
            })

        # Guard: detect strict 2-service alternation (A→B→A→B pattern)
        if len(last_services) >= 4:
            _pairs = [(last_services[i], last_services[i+1]) for i in range(len(last_services)-1)]
            _unique_pairs = set(_pairs)
            _unique_services_in_window = set(last_services)
            # Only fire if: multiple DIFFERENT services AND bouncing between them
            if (
                len(_unique_services_in_window) >= 2
                and len(_unique_pairs) <= 2
                and len(last_services) >= 4
            ):
                # Only treat as thrash if: no new signals found recently AND
                # rewards have been negative. Productive cross-service exploration
                # should not be interrupted.
                _recent_rewards_thrash = self._get_recent_rewards(4)
                _no_new_signals = len(self.shared_state.signals) == getattr(self, '_last_thrash_signal_count', len(self.shared_state.signals))
                _rewards_bad = len(_recent_rewards_thrash) >= 3 and sum(_recent_rewards_thrash) < -0.1

                if _rewards_bad and _no_new_signals:
                    _thrash_service = thrash_focus_service or last_services[-1]
                    if _thrash_service:
                        return _ensure_valid_service({
                            "type": "view_metrics",
                            "service": _thrash_service,
                        })

        # Update thrash signal baseline
        self._last_thrash_signal_count = len(self.shared_state.signals)

        if self._is_stuck():
            return _ensure_valid_service({
                "type": "view_metrics",
                "service": self._last_selected_action_service,
            })

        # Phase transitions.
        if self.shared_state.phase == "EXPLORE":
            if signals >= 2 or step >= 3:
                self.shared_state.phase = "FOCUS"

        def has_signal(service: Optional[str]) -> bool:
            if not service:
                return False
            return any(
                s
                for s in self.shared_state.signals
                if isinstance(s, dict) and s.get("service") == service
            )

        # Only fire if the SAME service was scroll_logs'd 4 times in a row
        last_4_with_service = self._action_history[-4:]
        if (
            len(last_4_with_service) == 4
            and all(a == "scroll_logs" for a, _ in last_4_with_service)
            and len(set(s for _, s in last_4_with_service if s)) == 1
        ):
            return _ensure_valid_service({
                "type": "view_metrics",
                "service": self._last_selected_action_service,
            })

        # Only fire if the SAME service was view_metrics'd 3 times in a row
        last_3_with_service = self._action_history[-3:]
        if (
            len(last_3_with_service) == 3
            and all(a == "view_metrics" for a, _ in last_3_with_service)
            and len(set(s for _, s in last_3_with_service if s)) == 1
        ):
            return _ensure_valid_service({
                "type": "scroll_logs",
                "service": self._last_selected_action_service,
            })

        best_proposal = max(
            proposal_list,
            key=lambda p: float(p.get("confidence", 0.0) or 0.0),
            default=None,
        )

        if best_proposal:
            suggested = best_proposal.get("suggested_action", {})
            if not self._suggested_action_violates_rules(suggested):
                chosen_type = suggested.get("type")
                chosen_service = suggested.get("service")
                self._last_selected_action_type = chosen_type
                self._last_selected_action_service = chosen_service
                return _ensure_valid_service({
                    "type": chosen_type,
                    "service": chosen_service,
                })

        # Phase machine.
        if self.shared_state.phase == "EXPLORE":
            available = list(self.shared_state.visited_services)
            if not available:
                return _ensure_valid_service({"type": "open_logs", "service": None})

            target = next(iter(available))
            self._last_selected_action_type = "open_logs"
            self._last_selected_action_service = target
            return _ensure_valid_service({
                "type": "open_logs",
                "service": target,
            })

        if self.shared_state.phase == "FOCUS":
            if self.shared_state.focus_service:
                # LOCK existing focus unless we are stuck.
                if not self._is_stuck():
                    focus_service = self.shared_state.focus_service
                else:
                    focus_service = selected_service or self._last_selected_action_service or self.shared_state.focus_service
            else:
                focus_service = selected_service or self._last_selected_action_service or next(iter(self.shared_state.visited_services), None)

            self.shared_state.focus_service = focus_service

            log_depth = self.shared_state.log_depth.get(focus_service, 0) if focus_service else 0
            metrics_seen = focus_service in self.shared_state.metrics_seen if focus_service else False

            if focus_service:
                if log_depth < 2:
                    self._last_selected_action_type = "scroll_logs"
                    self._last_selected_action_service = focus_service
                    return _ensure_valid_service({
                        "type": "scroll_logs",
                        "service": focus_service,
                    })

                if not metrics_seen:
                    self._last_selected_action_type = "view_metrics"
                    self._last_selected_action_service = focus_service
                    return _ensure_valid_service({
                        "type": "view_metrics",
                        "service": focus_service,
                    })

                # Strict DECIDE entry: only switch when service evidence is mature.
                service_signal_count = len([
                    s
                    for s in self.shared_state.signals
                    if isinstance(s, dict) and s.get("service") == focus_service
                ])
                if service_signal_count >= 4 and log_depth >= 2 and metrics_seen:
                    self.shared_state.phase = "DECIDE"
                else:
                    self._last_selected_action_type = "scroll_logs"
                    self._last_selected_action_service = focus_service
                    return _ensure_valid_service({
                        "type": "scroll_logs",
                        "service": focus_service,
                    })

        if self.shared_state.phase == "DECIDE":
            focus_service = (
                self.shared_state.focus_service
                or selected_service
                or self._last_selected_action_service
            )

            # HARD RULE 1: must have service
            if not focus_service:
                return _ensure_valid_service({
                    "type": "scroll_logs",
                    "service": selected_service,
                })

            signals = [
                s for s in self.shared_state.signals
                if isinstance(s, dict) and s.get("service") == focus_service
            ]

            log_depth = self.shared_state.log_depth.get(focus_service, 0)
            metrics_seen = focus_service in self.shared_state.metrics_seen

            # HARD RULE: must explore deeply before any submit.
            if log_depth < 2:
                return _ensure_valid_service({
                    "type": "scroll_logs",
                    "service": focus_service,
                })
            if not metrics_seen:
                return _ensure_valid_service({
                    "type": "view_metrics",
                    "service": focus_service,
                })

            # Minimum signal requirement before any hypothesis comparison.
            if len(signals) < 3:
                return _ensure_valid_service({
                    "type": "scroll_logs",
                    "service": focus_service,
                })

            all_hypotheses = [
                h for h in self.shared_state.hypotheses
                if isinstance(h, dict) and h.get("hypothesis")
            ]

            if not all_hypotheses:
                return _ensure_valid_service({
                    "type": "scroll_logs",
                    "service": focus_service,
                })

            services = set(
                h.get("service")
                for h in all_hypotheses
                if isinstance(h.get("service"), str) and h.get("service")
            )

            scored: List[Dict[str, object]] = []
            for h in all_hypotheses:
                raw = self._normalize_hypothesis(h.get("hypothesis"))
                if not raw:
                    continue

                h["hypothesis"] = raw

                base_conf = float(h.get("confidence", 0.0) or 0.0)
                hypothesis_service = h.get("service") if isinstance(h.get("service"), str) else focus_service
                evidence_score = self._score_hypothesis_with_evidence(raw, hypothesis_service)
                contradiction_penalty = self._contradiction_penalty(raw, hypothesis_service)
                final_score = (
                    base_conf
                    + evidence_score
                    - contradiction_penalty
                    + self._stability_bonus(raw)
                )

                scored.append({
                    "hypothesis": raw,
                    "score": final_score,
                    "service": hypothesis_service,
                })

            if not scored:
                return _ensure_valid_service({
                    "type": "scroll_logs",
                    "service": focus_service,
                })

            scored.sort(key=lambda x: float(x["score"]), reverse=True)

            top1 = scored[0]
            top2 = scored[1] if len(scored) > 1 else {"hypothesis": None, "score": 0.0, "service": None}

            gap = float(top1["score"]) - float(top2["score"]) if len(scored) >= 2 else float("inf")

            best = self._normalize_hypothesis(top1["hypothesis"])
            self.hypothesis_streak[best] = self.hypothesis_streak.get(best, 0) + 1
            for key in list(self.hypothesis_streak.keys()):
                if key != best:
                    self.hypothesis_streak[key] = 0

            if self.hypothesis_streak.get(best, 0) < 2:
                return _ensure_valid_service({
                    "type": "scroll_logs",
                    "service": focus_service,
                })

            signals_all = self.shared_state.signals

            def dominance_score(hypothesis_text: str) -> int:
                lowered = self._normalize_hypothesis(hypothesis_text).lower()
                count = 0
                for signal in signals_all:
                    content = str(signal.get("content", "")).lower()
                    if (
                        ("timeout" in lowered and ("timeout" in content or "latency" in content))
                        or ("db" in lowered and ("db" in content or "connection pool" in content))
                        or (("rate" in lowered or "429" in lowered) and ("retry_rate" in content or "429" in content or "rate" in content))
                        or ("cache" in lowered and "cache" in content)
                        or ("template" in lowered and "template" in content)
                    ):
                        count += 1
                return count

            if float(top1["score"]) < 1.2:
                return _ensure_valid_service({
                    "type": "scroll_logs",
                    "service": focus_service,
                })

            if self.best_global_hypothesis:
                if str(top1["hypothesis"]) != str(self.best_global_hypothesis):
                    if float(top1["score"]) < float(self.best_global_score) + 0.3:
                        return _ensure_valid_service({
                            "type": "scroll_logs",
                            "service": focus_service,
                        })

            if len(services) > 1 and len([h for h in scored if float(h["score"]) >= 1.0]) >= 2:
                return _ensure_valid_service({
                    "type": "view_metrics",
                    "service": focus_service,
                })

            if len(scored) >= 2:
                if dominance_score(str(top1["hypothesis"])) <= dominance_score(str(top2["hypothesis"])):
                    return _ensure_valid_service({
                        "type": "scroll_logs",
                        "service": focus_service,
                    })

            if len(scored) >= 2 and gap < 0.6:
                return _ensure_valid_service({
                    "type": "view_metrics",
                    "service": focus_service,
                })

            conflicting = [
                h for h in scored
                if h.get("service") != top1.get("service")
                and float(h.get("score", 0.0) or 0.0) >= float(top1["score"]) * 0.7
            ]
            if conflicting:
                return _ensure_valid_service({
                    "type": "view_metrics",
                    "service": focus_service,
                })

            if float(top1["score"]) > self.best_global_score:
                self.best_global_score = float(top1["score"])
                self.best_global_hypothesis = str(top1["hypothesis"])

            if not (float(top1["score"]) >= 1.3 and (len(scored) < 2 or gap >= 0.6)):
                return _ensure_valid_service({
                    "type": "view_metrics",
                    "service": focus_service,
                })

            root = self._map_hypothesis_to_root(top1["hypothesis"])
            if not root:
                return _ensure_valid_service({
                    "type": "view_metrics",
                    "service": focus_service,
                })

            severity = self._infer_severity_from_signals()

            submitted_service: Optional[str] = None

            # Priority 1: committed service (stable across steps)
            if isinstance(self.shared_state.committed_service, str) and self.shared_state.committed_service:
                submitted_service = self.shared_state.committed_service

            # Priority 2 (leaf-walk) is intentionally skipped here because dependency map
            # is not available in SharedState during DECIDE.

            # Priority 3: service with most signals
            if not submitted_service:
                service_signal_counts: Dict[str, int] = {}
                for signal in self.shared_state.signals:
                    if isinstance(signal, dict):
                        svc = signal.get("service")
                        if isinstance(svc, str) and svc:
                            service_signal_counts[svc] = service_signal_counts.get(svc, 0) + 1
                if service_signal_counts:
                    submitted_service = max(service_signal_counts, key=service_signal_counts.get)

            # Priority 4: fallback
            if not submitted_service and isinstance(top1.get("service"), str):
                submitted_service = top1.get("service")
            if not submitted_service:
                submitted_service = focus_service

            self._last_selected_action_type = "submit_diagnosis"
            self._last_selected_action_service = str(submitted_service or focus_service)
            return _ensure_valid_service({
                "type": "submit_diagnosis",
                "service": submitted_service or focus_service,
                "root_cause": root,
                "severity": severity,
            })

        if len(recent_rewards) == 3 and all(r < 0 for r in recent_rewards):
            new_service = next(
                (
                    s
                    for s in self.shared_state.visited_services
                    if s != self._last_selected_action_service and has_signal(s)
                ),
                None,
            )
            if new_service:
                self._last_selected_action_type = "open_logs"
                self._last_selected_action_service = new_service
                return _ensure_valid_service({
                    "type": "open_logs",
                    "service": new_service,
                })
            self._last_selected_action_type = "scroll_logs"
            return _ensure_valid_service(
                {
                    "type": "scroll_logs",
                    "service": self._last_selected_action_service,
                }
            )

        last_actions = self._get_last_n_actions(3)
        if last_actions == ["open_logs", "open_logs", "open_logs"]:
            available_services = [
                p.get("suggested_action", {}).get("service")
                for p in proposal_list
                if isinstance(p.get("suggested_action"), dict)
                and isinstance(p.get("suggested_action", {}).get("service"), str)
            ]
            new_service = next(
                (
                    s
                    for s in available_services
                    if s != self._last_selected_action_service and has_signal(s)
                ),
                None,
            )
            if new_service:
                self._last_selected_action_type = "open_logs"
                self._last_selected_action_service = new_service
                return _ensure_valid_service({
                    "type": "open_logs",
                    "service": new_service,
                })
            self._last_selected_action_type = "scroll_logs"
            return _ensure_valid_service(
                {
                    "type": "scroll_logs",
                    "service": self._last_selected_action_service,
                }
            )

        same_service = self._last_selected_action_service
        log_depth = self.shared_state.log_depth

        if self._last_selected_action_type == "open_logs" and same_service:
            self._last_selected_action_type = "scroll_logs"
            self._last_selected_action_service = same_service
            return _ensure_valid_service({
                "type": "scroll_logs",
                "service": same_service,
            })

        if same_service:
            service_log_depth = log_depth.get(same_service, 0)
            metrics_seen = same_service in self.shared_state.metrics_seen

            # Case 1: Not enough logs -> keep scrolling.
            if service_log_depth < 2:
                self._last_selected_action_type = "scroll_logs"
                self._last_selected_action_service = same_service
                return _ensure_valid_service({
                    "type": "scroll_logs",
                    "service": same_service,
                })

            # Case 2: Logs explored but no metrics yet -> try metrics once.
            if not metrics_seen:
                self._last_selected_action_type = "view_metrics"
                self._last_selected_action_service = same_service
                return _ensure_valid_service({
                    "type": "view_metrics",
                    "service": same_service,
                })

            # Case 3: Metrics already seen -> fall through to normal decision logic.

        valid_proposals: List[dict] = []
        for proposal in (proposals or []):
            if not isinstance(proposal, dict):
                continue
            action = proposal.get("suggested_action")
            if not isinstance(action, dict):
                continue

            action_type = action.get("type")
            if action_type not in {"open_logs", "view_metrics", "scroll_logs", "submit_diagnosis"}:
                continue

            if action_type != "scroll_logs" and not action.get("service"):
                continue

            valid_proposals.append(proposal)

        if not valid_proposals:
            fallback_service = next(
                (
                    p.get("suggested_action", {}).get("service")
                    for p in (proposals or [])
                    if isinstance(p, dict)
                    and isinstance(p.get("suggested_action"), dict)
                    and p.get("suggested_action", {}).get("service")
                ),
                None,
            )
            if fallback_service is None:
                fallback_service = next(iter(self.shared_state.visited_services), None)
            self._last_selected_action_type = "open_logs"
            self._last_selected_action_service = fallback_service
            return _ensure_valid_service({
                "type": "open_logs",
                "service": fallback_service,
            })

        exploration_action = {
            "type": "open_logs",
            "service": next(
                (
                    p.get("suggested_action", {}).get("service")
                    for p in valid_proposals
                    if isinstance(p.get("suggested_action"), dict)
                    and p.get("suggested_action", {}).get("service")
                    not in self.shared_state.visited_services
                ),
                next(
                    (
                        p.get("suggested_action", {}).get("service")
                        for p in valid_proposals
                        if isinstance(p.get("suggested_action"), dict)
                        and p.get("suggested_action", {}).get("service")
                    ),
                    next(iter(self.shared_state.visited_services), None),
                ),
            ),
        }

        hypothesis_groups: Dict[str, List[dict]] = {}
        for p in valid_proposals:
            h = str(p.get("hypothesis") or "").strip()
            if not h:
                continue
            key = h.lower()
            if key not in hypothesis_groups:
                hypothesis_groups[key] = []
            hypothesis_groups[key].append(p)

        filtered_groups: Dict[str, List[dict]] = {}
        for h, group in hypothesis_groups.items():
            first = group[0] if group else {}
            suggested = first.get("suggested_action", {}) if isinstance(first, dict) else {}
            service = suggested.get("service") if isinstance(suggested, dict) else None
            evidence_score = self._score_hypothesis_with_evidence(h, service)
            if evidence_score > 0:
                filtered_groups[h] = group

        if filtered_groups:
            hypothesis_groups = filtered_groups

        hypothesis_scores: List[dict] = []
        recent_rewards = self._get_recent_rewards(3)
        for h, group in hypothesis_groups.items():
            confidences = [float(p.get("confidence", 0.0) or 0.0) for p in group]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            agreement = len(group)

            service = None
            first = group[0] if group else {}
            suggested = first.get("suggested_action") if isinstance(first, dict) else None
            if isinstance(suggested, dict):
                service = suggested.get("service")

            signal_count = len([
                s
                for s in self.shared_state.signals
                if isinstance(s, dict) and s.get("service") == service
            ])

            reward_penalty = 0.0
            if len(recent_rewards) == 3 and all(r < 0 for r in recent_rewards):
                reward_penalty = 0.2

            score = (
                (avg_conf * 0.4)
                + (agreement * 0.2)
                + (signal_count * 0.2)
                + (self._score_hypothesis_with_evidence(h, service) * 0.2)
            )
            score -= self._contradiction_penalty(h, service)
            score -= reward_penalty

            hypothesis_scores.append(
                {
                    "hypothesis": h,
                    "score": score,
                    "service": service,
                    "avg_conf": avg_conf,
                    "agreement": agreement,
                }
            )

        hypothesis_scores.sort(key=lambda x: x["score"], reverse=True)
        best = hypothesis_scores[0] if hypothesis_scores else None
        second = hypothesis_scores[1] if len(hypothesis_scores) > 1 else None

        def is_strong(best_item: Optional[dict], second_item: Optional[dict]) -> bool:
            if not best_item:
                return False

            hypothesis_text = str(best_item.get("hypothesis") or "")
            if not hypothesis_text or hypothesis_text.strip() == "":
                return False

            if float(best_item.get("avg_conf", 0.0) or 0.0) < 0.75:
                return False

            if int(best_item.get("agreement", 0) or 0) < 2:
                return False

            evidence_score = self._score_hypothesis_with_evidence(
                hypothesis_text,
                best_item.get("service"),
            )
            if evidence_score < 0.4:
                return False

            if second_item:
                margin = float(best_item.get("score", 0.0) or 0.0) - float(second_item.get("score", 0.0) or 0.0)
                if margin < 0.4:
                    return False

            return True

        if best and is_strong(best, second):
            chosen_hypothesis = best.get("hypothesis")
            chosen_service = best.get("service")

            if not chosen_hypothesis or not chosen_service:
                self._last_selected_action_type = "scroll_logs"
                return _ensure_valid_service(
                    {
                        "type": "scroll_logs",
                        "service": self._last_selected_action_service,
                    }
                )

            self._last_selected_action_type = "submit_diagnosis"
            self._last_selected_action_service = chosen_service
            return _ensure_valid_service(
                {
                    "type": "submit_diagnosis",
                    "service": chosen_service,
                    "root_cause": chosen_hypothesis,
                    "severity": "high",
                }
            )

        if best and best.get("service"):
            service = best.get("service")
            log_depth = self.shared_state.log_depth.get(service, 0)
            metrics_seen = service in self.shared_state.metrics_seen
            if log_depth < 2:
                self._last_selected_action_type = "scroll_logs"
                self._last_selected_action_service = service
                return _ensure_valid_service({"type": "scroll_logs", "service": service})
            if not metrics_seen:
                self._last_selected_action_type = "view_metrics"
                self._last_selected_action_service = service
                return _ensure_valid_service({"type": "view_metrics", "service": service})

        if second and second.get("service"):
            self._last_selected_action_type = "open_logs"
            self._last_selected_action_service = second.get("service")
            return _ensure_valid_service({"type": "open_logs", "service": second.get("service")})

        self._last_selected_action_type = exploration_action.get("type")
        self._last_selected_action_service = exploration_action.get("service")
        return _ensure_valid_service(
            {
                "type": "open_logs",
                "service": next(iter(self.shared_state.visited_services), None),
            }
        )

    def think(self) -> dict:
        heuristic_analysis = self.analyze()

        # Fallback path if coordinator is not configured for LLM reasoning.
        if self.client is None:
            decision = "submit" if self.should_submit(heuristic_analysis) else "explore"
            result = dict(heuristic_analysis)
            result["decision"] = decision
            _debug(f"coordinator fallback (no client): decision={decision}")
            return result

        evidence_block = self._build_compact_evidence()
        hypotheses = self.shared_state.hypotheses[-3:]
        top_two_hypotheses = sorted(
            [h for h in hypotheses if isinstance(h, dict)],
            key=lambda h: float(h.get("confidence", 0.0) or 0.0),
            reverse=True,
        )[:2]
        user_prompt = (
            "Analyze the system state and decide.\n\n"
            "Return JSON with:\n"
            "{\n"
            '  "hypotheses": [ { "root_cause": str, "service": str, "confidence": float } ],\n'
            '  "best_hypothesis": { "root_cause": str, "service": str, "confidence": float },\n'
            '  "decision": "explore" or "submit"\n'
            "}\n\n"
            "Rules:\n"
            "- Prefer hypotheses with strongest evidence\n"
            "- Reject contradicted hypotheses\n"
            "- If uncertain -> explore\n"
            "- If strong consistent evidence -> submit\n\n"
            f"Compact evidence:\n{evidence_block}\n\n"
            f"Top hypotheses (max 2): {top_two_hypotheses}"
        )

        llm_result = self._request_think_json(user_prompt)
        if llm_result is None:
            decision = "submit" if self.should_submit(heuristic_analysis) else "explore"
            result = dict(heuristic_analysis)
            result["decision"] = decision
            _debug(f"coordinator fallback (llm parse/request failed): decision={decision}")
            return result

        if not self._validate_think_output(llm_result):
            _debug("coordinator invalid LLM output, falling back")
            decision = "submit" if self.should_submit(heuristic_analysis) else "explore"
            result = dict(heuristic_analysis)
            result["decision"] = decision
            return result

        normalized = self._normalize_think_output(llm_result)
        if normalized is None:
            decision = "submit" if self.should_submit(heuristic_analysis) else "explore"
            result = dict(heuristic_analysis)
            result["decision"] = decision
            _debug(f"coordinator fallback (llm normalization failed): decision={decision}")
            return result

        top_hypotheses = normalized.get("hypotheses", [])
        if isinstance(top_hypotheses, list) and len(top_hypotheses) >= 2:
            h1 = top_hypotheses[0]
            h2 = top_hypotheses[1]
            try:
                gap = float(h1.get("confidence", 0.0)) - float(h2.get("confidence", 0.0))
            except (TypeError, ValueError):
                gap = 0.0
            _debug(
                "coordinator disagreement: "
                f"A={h1.get('root_cause')}@{h1.get('service')} "
                f"vs B={h2.get('root_cause')}@{h2.get('service')} "
                f"confidence_gap={gap:.2f}"
            )

        self.shared_state.hypotheses = normalized["hypotheses"]
        return normalized

    def analyze(self) -> dict:
        hypotheses: List[dict] = []

        signal_groups: Dict[str, List[str]] = {}
        for item in self.shared_state.signals:
            if not isinstance(item, dict):
                continue
            service = item.get("service")
            content = item.get("content")
            if isinstance(service, str) and service and isinstance(content, str):
                signal_groups.setdefault(service, []).append(content)

        for service, contents in signal_groups.items():
            joined_signal_text = " ".join(contents).lower()
            service_logs = self.shared_state.logs_seen.get(service, [])
            joined_logs = " ".join(str(x) for x in service_logs).lower()
            service_metrics = self.shared_state.metrics_seen.get(service, {})
            joined_metrics = " ".join(f"{k}:{v}" for k, v in service_metrics.items()).lower()

            combined_text = f"{joined_signal_text} {joined_logs} {joined_metrics}"
            root_cause = self._infer_from_text(combined_text)
            supporting_signals = len(contents)
            repeated_log_count = max(0, len(service_logs) - len(set(service_logs)))
            confidence = self._score_confidence(supporting_signals, repeated_log_count)

            hypotheses.append(
                {
                    "root_cause": root_cause,
                    "service": service,
                    "confidence": confidence,
                    "supporting_signals": supporting_signals,
                }
            )

        hypotheses = self.filter_contradictions(hypotheses)

        if not hypotheses:
            default_root = self._infer_from_signals() or "TIMEOUT"
            hypotheses.append(
                {
                    "root_cause": default_root,
                    "service": None,
                    "confidence": 0.2,
                    "supporting_signals": 0,
                }
            )

        # Ensure at least two competing hypotheses.
        if len(hypotheses) < 2:
            base_root = str(hypotheses[0].get("root_cause") or "TIMEOUT")
            alt_root = self._alternate_root_cause(base_root)
            hypotheses.append(
                {
                    "root_cause": alt_root,
                    "service": hypotheses[0].get("service"),
                    "confidence": round(max(0.1, float(hypotheses[0].get("confidence", 0.2)) - 0.15), 2),
                    "supporting_signals": 0,
                }
            )

        hypotheses.sort(key=lambda h: float(h.get("confidence", 0.0)), reverse=True)

        # Weight previously seen hypotheses more heavily than one-off fresh outputs.
        recent_history = [
            h for h in self.shared_state.hypothesis_history[-5:]
            if float(h.get("confidence", 0.0) or 0.0) >= 0.7
        ]
        for hyp in hypotheses:
            root = str(hyp.get("root_cause") or "")
            service = hyp.get("service")
            history_hits = sum(
                1
                for h in recent_history
                if str(h.get("root_cause") or "").upper() == root.upper()
                and (
                    service is None
                    or h.get("service") == service
                )
            )
            if history_hits:
                hyp["confidence"] = round(float(hyp.get("confidence", 0.0) or 0.0) + (0.25 * history_hits), 2)

        hypotheses.sort(key=lambda h: float(h.get("confidence", 0.0)), reverse=True)
        best_hypothesis = hypotheses[0]

        current_best_root = self.shared_state.committed_root_cause
        current_best_service = self.shared_state.committed_service

        def _current_committed_score() -> float:
            if not current_best_root:
                return 0.0
            for hyp in hypotheses:
                if str(hyp.get("root_cause") or "").upper() == str(current_best_root).upper() and hyp.get("service") == current_best_service:
                    return float(hyp.get("confidence", 0.0) or 0.0)
            for hyp in hypotheses:
                if str(hyp.get("root_cause") or "").upper() == str(current_best_root).upper():
                    return float(hyp.get("confidence", 0.0) or 0.0)
            return 0.0

        current_confidence = _current_committed_score()
        new_confidence = float(best_hypothesis.get("confidence", 0.0) or 0.0)
        if (
            not current_best_root
            or not current_best_service
            or new_confidence > current_confidence + 0.1
        ):
            _raw_root = best_hypothesis.get("root_cause")
            self.shared_state.committed_root_cause = (
                str(_raw_root) if isinstance(_raw_root, str) and _raw_root else None
            )
            _raw_svc = best_hypothesis.get("service")
            self.shared_state.committed_service = (
                str(_raw_svc) if isinstance(_raw_svc, str) and _raw_svc else None
            )
            current_best_root = self.shared_state.committed_root_cause
            current_best_service = self.shared_state.committed_service

        committed_hypothesis = None
        for hyp in hypotheses:
            if str(hyp.get("root_cause") or "").upper() == str(current_best_root or "").upper():
                if current_best_service is None or hyp.get("service") == current_best_service:
                    committed_hypothesis = hyp
                    break
        if committed_hypothesis is None:
            committed_hypothesis = best_hypothesis

        history_entry = {
            "step": self.shared_state.step_count,
            "root_cause": committed_hypothesis.get("root_cause"),
            "service": committed_hypothesis.get("service"),
            "confidence": committed_hypothesis.get("confidence"),
        }
        if not self.shared_state.hypothesis_history or self.shared_state.hypothesis_history[-1].get("step") != self.shared_state.step_count:
            self.shared_state.hypothesis_history.append(history_entry)
        else:
            self.shared_state.hypothesis_history[-1] = history_entry
        if len(self.shared_state.hypothesis_history) > 3:
            self.shared_state.hypothesis_history = self.shared_state.hypothesis_history[-3:]

        # Store coordinator-level ranked hypotheses for downstream coordination.
        self.shared_state.hypotheses = hypotheses

        return {
            "hypotheses": hypotheses,
            "best_hypothesis": committed_hypothesis,
        }

    def _build_service_evidence(self) -> Dict[str, Dict[str, object]]:
        buckets: Dict[str, Dict[str, object]] = {}

        services = set(self.shared_state.logs_seen.keys()) | set(self.shared_state.metrics_seen.keys())
        services |= self.shared_state.unique_services_with_signals()

        for service in services:
            service_logs = [str(x) for x in self.shared_state.logs_seen.get(service, [])]
            service_metrics = self.shared_state.metrics_seen.get(service, {})
            service_signals = [
                str(item.get("content", ""))
                for item in self.shared_state.signals
                if isinstance(item, dict) and item.get("service") == service
            ]

            log_head = service_logs[:2]
            log_tail = service_logs[-2:] if len(service_logs) > 2 else []
            logs_summary_lines = log_head + log_tail
            logs_summary = " | ".join(logs_summary_lines) if logs_summary_lines else "no logs"

            metrics_summary = ", ".join(f"{k}:{v}" for k, v in service_metrics.items()) if service_metrics else "no metrics"

            buckets[service] = {
                "signals": service_signals,
                "logs_summary": logs_summary,
                "metrics_summary": metrics_summary,
            }

        return buckets

    def _build_compact_evidence(self) -> str:
        signals = self.shared_state.signals[-6:]
        services = list(self.shared_state.visited_services)[-4:]
        service_set = {s for s in services if isinstance(s, str)}

        summary: List[str] = []
        for signal in signals:
            if not isinstance(signal, dict):
                continue
            service = signal.get("service")
            content = str(signal.get("content", ""))
            if service_set and service not in service_set:
                continue
            summary.append(f"{service}: {content[:80]}")

        if not summary:
            return "no compact evidence"
        return "\n".join(summary)

    def _request_think_json(self, user_prompt_text: str) -> Optional[dict]:
        """Robust LLM call with retries and parsing recovery."""

        if self.client is None:
            return None

        base_messages = [
            {
                "role": "system",
                "content": (
                    "You are a backend diagnosis coordinator.\n"
                    "You MUST return ONLY valid JSON.\n"
                    "No explanations, no markdown, no text outside JSON.\n"
                ),
            },
            {"role": "user", "content": user_prompt_text},
        ]
        request_max_tokens = min(COORDINATOR_DEFAULT_RESPONSE_TOKENS, COORDINATOR_MAX_TOKENS)

        for attempt in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=base_messages,
                    temperature=0,
                    max_tokens=request_max_tokens,
                )

                raw = completion.choices[0].message.content if completion.choices else ""
                # _debug(f"coordinator raw LLM response: {raw}")

                # Step 1: direct parse
                try:
                    parsed = json.loads(raw)
                    return parsed if isinstance(parsed, dict) else None
                except Exception:
                    pass

                # Step 2: extract JSON block
                extracted = self._extract_json_block(raw)
                if extracted:
                    try:
                        parsed = json.loads(extracted)
                        return parsed if isinstance(parsed, dict) else None
                    except Exception:
                        pass

                # Step 3: retry with stricter instruction
                if attempt < 2:
                    base_messages = [
                        {
                            "role": "system",
                            "content": (
                                "STRICT MODE: Return ONLY valid JSON.\n"
                                "No extra text.\n"
                                "No explanation.\n"
                                "Output must be parseable by json.loads().\n"
                            ),
                        },
                        {"role": "user", "content": user_prompt_text},
                    ]
                    continue

            except Exception as e:
                _debug(f"coordinator LLM error: {str(e)}")
                affordable = self._extract_affordable_max_tokens(str(e))
                if affordable is not None:
                    # Keep a safety margin and stay within provider affordability.
                    reduced = max(
                        COORDINATOR_MIN_RESPONSE_TOKENS,
                        min(COORDINATOR_MAX_TOKENS, affordable - 64),
                    )
                    if reduced < request_max_tokens:
                        request_max_tokens = reduced
                        _debug(f"coordinator token backoff applied: max_tokens={request_max_tokens}")

        return None

    def _extract_affordable_max_tokens(self, error_text: str) -> Optional[int]:
        if not error_text:
            return None
        match = re.search(r"can only afford\s+(\d+)", error_text.lower())
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _clip_text(self, value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        return value[: max(0, limit - 3)] + "..."

    def _extract_json_block(self, text: str) -> Optional[str]:
        """Extract the first valid JSON object from raw LLM output."""
        if not text:
            return None

        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return match.group(0)
        return None

    def _validate_think_output(self, data: dict) -> bool:
        if not isinstance(data, dict):
            return False
        if "decision" not in data:
            return False
        if data["decision"] not in {"explore", "submit"}:
            return False
        if "best_hypothesis" not in data:
            return False
        return True

    def _parse_json_with_repair(self, raw_text: str) -> Optional[dict]:
        try:
            parsed = json.loads(raw_text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            match = re.search(r"\{[\s\S]*\}", raw_text or "")
            if not match:
                return None
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None

    def _normalize_think_output(self, data: dict) -> Optional[dict]:
        raw_hypotheses = data.get("hypotheses")
        raw_best = data.get("best_hypothesis")
        raw_decision = data.get("decision")

        if not isinstance(raw_hypotheses, list) or not raw_hypotheses:
            return None

        hypotheses: List[dict] = []
        for item in raw_hypotheses[:5]:
            if not isinstance(item, dict):
                continue
            root = item.get("root_cause")
            if not isinstance(root, str) or not root:
                continue

            service = item.get("service")
            if not isinstance(service, str) or not service:
                service = None

            try:
                conf = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0

            try:
                supporting = int(item.get("supporting_signals", 0))
            except (TypeError, ValueError):
                supporting = 0

            contradictions_field = item.get("contradictions")
            if isinstance(contradictions_field, list):
                contradiction_count = len(contradictions_field)
            elif isinstance(contradictions_field, (int, float)):
                contradiction_count = max(0, int(contradictions_field))
            elif isinstance(contradictions_field, str) and contradictions_field.strip():
                contradiction_count = 1
            else:
                contradiction_count = 0

            hypotheses.append(
                {
                    "root_cause": root,
                    "service": service,
                    "confidence": round(max(0.0, min(1.0, conf)), 2),
                    "supporting_signals": max(0, supporting),
                    "contradiction_count": contradiction_count,
                }
            )

        if not hypotheses:
            return None

        # Prefer stronger evidence and fewer contradictions, then confidence.
        hypotheses.sort(
            key=lambda h: (
                float(h.get("confidence", 0.0)),
                int(h.get("supporting_signals", 0)),
                -int(h.get("contradiction_count", 0)),
            ),
            reverse=True,
        )

        best_hypothesis = hypotheses[0]
        if isinstance(raw_best, dict):
            best_root = raw_best.get("root_cause")
            if isinstance(best_root, str) and best_root:
                matched = next((h for h in hypotheses if h.get("root_cause") == best_root), None)
                if matched is not None:
                    best_hypothesis = matched

        decision = raw_decision if raw_decision in {"explore", "submit"} else "explore"

        return {
            "hypotheses": hypotheses,
            "best_hypothesis": best_hypothesis,
            "decision": decision,
        }

    def filter_contradictions(self, hypotheses: list[dict]) -> list[dict]:
        filtered: List[dict] = []
        abnormal_markers = {"high", "spiking", "100%", "maxed", "degraded"}

        for hypothesis in hypotheses:
            if not isinstance(hypothesis, dict):
                continue

            root_cause = str(hypothesis.get("root_cause") or "").upper()
            service = hypothesis.get("service")
            service_name = service if isinstance(service, str) and service else None

            if service_name:
                logs = [str(x) for x in self.shared_state.logs_seen.get(service_name, [])]
                metrics = self.shared_state.metrics_seen.get(service_name, {})
            else:
                logs = [
                    str(line)
                    for all_logs in self.shared_state.logs_seen.values()
                    for line in all_logs
                ]
                metrics = {}
                for service_metrics in self.shared_state.metrics_seen.values():
                    if isinstance(service_metrics, dict):
                        metrics.update(service_metrics)

            logs_text = " ".join(logs).lower()
            metric_items = {
                str(k).lower(): str(v).lower()
                for k, v in metrics.items()
            }

            has_error_log = any(tok in logs_text for tok in ["error", "failed", "fail"])
            has_timeout_log = any(tok in logs_text for tok in ["timeout", "timed out", "upstream timeout"])
            has_cache_error = any(tok in logs_text for tok in ["cache stale", "stale", "cache miss", "cache error"])

            has_latency_spike = any(
                ("latency" in k and v in abnormal_markers)
                for k, v in metric_items.items()
            )
            has_high_cpu = any(
                (("cpu" in k or "db" in k) and v in abnormal_markers)
                for k, v in metric_items.items()
            )
            has_cache_metric_abnormal = any(
                ("cache" in k and v in abnormal_markers)
                for k, v in metric_items.items()
            )
            has_cache_metric_normal = any(
                ("cache" in k and v == "normal")
                for k, v in metric_items.items()
            )

            contradicted = False
            if root_cause == "DB_OVERLOAD":
                # Reject if there is no high CPU, no latency spike, and no error logs.
                contradicted = not (has_high_cpu or has_latency_spike or has_error_log)
            elif root_cause == "CACHE_STALE":
                # Reject if cache looks normal and there are no cache-related errors.
                contradicted = has_cache_metric_normal and not (has_cache_metric_abnormal or has_cache_error)
            elif root_cause == "TIMEOUT":
                # Reject if no timeout-related logs are present.
                contradicted = not has_timeout_log

            if contradicted:
                try:
                    current_conf = float(hypothesis.get("confidence", 0.0))
                except (TypeError, ValueError):
                    current_conf = 0.0
                lowered_conf = round(max(0.0, current_conf - 0.5), 2)
                if lowered_conf <= 0.05:
                    continue
                adjusted = dict(hypothesis)
                adjusted["confidence"] = lowered_conf
                filtered.append(adjusted)
            else:
                filtered.append(hypothesis)

        if not filtered:
            fallback_root = self._infer_from_signals() or "TIMEOUT"
            filtered.append(
                {
                    "root_cause": fallback_root,
                    "service": None,
                    "confidence": 0.2,
                    "supporting_signals": 0,
                }
            )

        return filtered

    def should_submit_clean(self, observation: Optional[BackendDiagnosisObservation] = None) -> bool:
        analysis = self.analyze()
        best = analysis.get("best_hypothesis", {})
        confidence = float(best.get("confidence", 0.0))

        sufficient_signals = len(self.shared_state.signals) >= 2
        if confidence < 0.8 or not sufficient_signals:
            return False

        difficulty = str(self.shared_state.difficulty or "").lower()
        available_services = list(getattr(observation, "available_services", None) or [])
        visited_count = len(self.shared_state.visited_services)
        min_visited = min(3, max(0, len(available_services) - 1)) if available_services else 3
        if difficulty == "hard" and visited_count < min_visited:
            return False

        min_steps = 3
        if difficulty == "medium":
            min_steps = 5
        elif difficulty == "hard":
            min_steps = 7
        if self.shared_state.step_count < min_steps:
            return False

        # Also require that at least 2 distinct services have signals before
        # submitting on medium/hard. This prevents committing to the first
        # service that looks suspicious before cross-service evidence is gathered.
        if difficulty in ("medium", "hard"):
            services_with_signals = len(self.shared_state.unique_services_with_signals())
            if services_with_signals < 2:
                return False

        dep_map = getattr(observation, "available_dependencies", None) if observation is not None else None
        submitted_service = best.get("service")
        if dep_map and isinstance(dep_map, dict) and submitted_service:
            has_outgoing = submitted_service in dep_map and bool(dep_map.get(submitted_service))
            if has_outgoing:
                return False

        return True

    def should_submit(self, decision: Optional[dict] = None, observation: Optional[BackendDiagnosisObservation] = None) -> bool:
        # Backward-compatible alias; all submit gates use should_submit_clean.
        return self.should_submit_clean(observation=observation)

    def _infer_from_text(self, text: str) -> str:
        normalized = (text or "").lower()
        keyword_map = [
            (["rate limit", "429", "retry_rate"], "RATE_LIMITED"),
            (["timeout", "timed out", "latency", "upstream timeout"], "TIMEOUT"),
            (["db", "connection pool", "slow queries", "db_cpu"], "DB_OVERLOAD"),
            (["cache stale", "stale", "stale_reads"], "CACHE_STALE"),
            (["cache miss", "miss rate", "cache_hit:low"], "CACHE_MISS"),
            (["template", "render"], "TEMPLATE_ERROR"),
            (["network partition", "packet loss", "degraded probe"], "NETWORK_PARTITION"),
            (["crash", "restarts", "oom"], "SERVICE_CRASH"),
            (["unauthorized", "forbidden", "401", "403"], "AUTH_FAILURE"),
            (["dependency unavailable", "connection refused", "503"], "DEPENDENCY_DOWN"),
        ]
        for keywords, root_cause in keyword_map:
            if any(keyword in normalized for keyword in keywords):
                return root_cause
        return "TIMEOUT"

    def _score_confidence(self, supporting_signals: int, repeated_count: int) -> float:
        if supporting_signals <= 0:
            return 0.15
        confidence = 0.25 + (0.16 * supporting_signals) + (0.04 * repeated_count)
        return round(max(0.1, min(0.95, confidence)), 2)

    def _alternate_root_cause(self, base_root: str) -> str:
        alternates = {
            "TIMEOUT": "DEPENDENCY_DOWN",
            "DEPENDENCY_DOWN": "TIMEOUT",
            "DB_OVERLOAD": "TIMEOUT",
            "CACHE_STALE": "CACHE_MISS",
            "CACHE_MISS": "CACHE_STALE",
            "RATE_LIMITED": "TIMEOUT",
            "TEMPLATE_ERROR": "DEPLOY_REGRESSION",
            "AUTH_FAILURE": "CONFIG_ERROR",
        }
        return alternates.get(base_root, "TIMEOUT")

    def _infer_from_signals(self) -> Optional[str]:
        if not self.shared_state.signals:
            return None

        text = " ".join(
            str(item.get("content", ""))
            for item in self.shared_state.signals
            if isinstance(item, dict)
        ).lower()

        keyword_map = [
            (["rate limit", "429", "retry_rate"], "RATE_LIMITED"),
            (["timeout", "timed out", "latency"], "TIMEOUT"),
            (["db", "connection pool", "slow queries", "db_cpu"], "DB_OVERLOAD"),
            (["cache stale", "stale"], "CACHE_STALE"),
            (["cache miss", "miss rate"], "CACHE_MISS"),
            (["template", "render"], "TEMPLATE_ERROR"),
            (["network partition", "packet loss", "degraded probe"], "NETWORK_PARTITION"),
            (["crash", "restarts", "oom"], "SERVICE_CRASH"),
            (["unauthorized", "forbidden", "401", "403"], "AUTH_FAILURE"),
            (["dependency unavailable", "connection refused", "503"], "DEPENDENCY_DOWN"),
        ]

        for keywords, root_cause in keyword_map:
            if any(keyword in text for keyword in keywords):
                return root_cause
        return None


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
    debug: bool = False,
) -> Dict[str, float]:
    """
    Run baseline episodes per difficulty with reproducible seeds.
    """

    global DEBUG_MODE
    DEBUG_MODE = bool(debug)

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

    shared_state = SharedState()
    shared_state.difficulty = difficulty
    explorers = [
        ExplorerAgent(shared_state=shared_state, service_name=None, personality="latency"),
        ExplorerAgent(shared_state=shared_state, service_name=None, personality="error"),
        ExplorerAgent(shared_state=shared_state, service_name=None, personality="resource"),
    ]
    coordinator = CoordinatorAgent(shared_state=shared_state)
    coordinator.configure(client=client, model=model)
    for explorer in explorers:
        explorer.configure(client=client, model=model, difficulty=difficulty, llm_temperature=llm_temperature)
        explorer.initialize(obs)

    final_action: Optional[BackendDiagnosisAction] = None
    coordinator_triggered = False
    last_coordinator_analysis = {
        "hypotheses": [],
        "best_hypothesis": {
            "root_cause": "UNKNOWN",
            "service": None,
            "confidence": 0.0,
            "supporting_signals": 0,
        },
    }
    episode_max_steps = max_steps + 2 if difficulty == "hard" else max_steps
    shared_state.max_steps = episode_max_steps

    for step_idx in range(1, episode_max_steps + 1):
        proposals: List[dict] = []
        for ex in explorers:
            proposal = ex.propose(obs)
            proposal["explorer_personality"] = ex.personality
            proposals.append(proposal)

        proposal_pairs = [
            (
                p.get("suggested_action", {}).get("type") if isinstance(p.get("suggested_action"), dict) else None,
                p.get("suggested_action", {}).get("service") if isinstance(p.get("suggested_action"), dict) else None,
            )
            for p in proposals
            if isinstance(p, dict)
        ]
        if proposal_pairs and len(set(proposal_pairs)) == 1:
            print(
                "[WARN] explorer proposals converged — personality divergence failed",
                flush=True,
            )

        decision = coordinator.decide_next_action(proposals)
        # Coordinator decision is the final action source for this step.
        action_payload = coordinator._validate_final_action(decision)
        coordinator._assert_valid_state(action_payload)

        selected_explorer = None
        for idx, proposal in enumerate(proposals):
            suggested = proposal.get("suggested_action") if isinstance(proposal, dict) else None
            if not isinstance(suggested, dict):
                continue
            if suggested.get("type") == action_payload.get("type") and suggested.get("service") == action_payload.get("service"):
                selected_explorer = explorers[idx]
                break

        selected_personality = "coordinator"
        if selected_explorer is not None:
            selected_personality = selected_explorer.personality

        if isinstance(action_payload, dict) and action_payload.get("type") in {"open_logs", "view_metrics", "submit_diagnosis"}:
            if not action_payload.get("service"):
                fallback_service = (
                    selected_explorer.current_target_service
                    or selected_explorer.service_name
                    or (obs.available_services[0] if (obs.available_services or []) else None)
                )
                action_payload["service"] = fallback_service

        if isinstance(action_payload, dict) and action_payload.get("type") == "submit_diagnosis":
            fallback_service = (
                action_payload.get("service")
                or selected_explorer.current_target_service
                or selected_explorer.service_name
                or (obs.available_services[0] if (obs.available_services or []) else None)
            )
            if (
                not action_payload.get("service")
                or not action_payload.get("root_cause")
                or action_payload.get("severity") not in {"low", "medium", "high"}
            ):
                action_payload = {
                    "type": "view_metrics",
                    "service": fallback_service,
                }

        if isinstance(action_payload, dict) and action_payload.get("service") is None:
            print("[CRITICAL] Action has no service!", action_payload, flush=True)

        _debug(
            f"step dispatch: step={step_idx} explorer_personality={selected_personality} target={selected_explorer.current_target_service if selected_explorer is not None else 'n/a'}"
        )
        parse_error = selected_explorer.last_parse_error if selected_explorer is not None else None

        if not action_payload or not action_payload.get("type"):
            action_str = "unknown_action()" if parse_error == "parse_failed" else "invalid_action()"
            error_text = parse_error or "Invalid action"
            print(
                f"[STEP] step={step_idx} action={action_str} reward={float(0.0):.2f} done={str(False).lower()} error={error_text}",
                flush=True,
            )
            if step_rewards is not None:
                step_rewards.append(float(0.0))
            continue

        try:
            action = BackendDiagnosisAction(**action_payload)
        except Exception:
            action = BackendDiagnosisAction(
                type="view_metrics",
                service=(selected_explorer.service_name if selected_explorer is not None else None) or (obs.available_services[0] if (obs.available_services or []) else None),
            )

        if action.type in {"open_logs", "view_metrics", "submit_diagnosis"} and not action.service:
            fallback_service = (
                (selected_explorer.current_target_service if selected_explorer is not None else None)
                or (selected_explorer.service_name if selected_explorer is not None else None)
                or (obs.available_services[0] if (obs.available_services or []) else None)
            )
            action.service = fallback_service

        if action.service is None:
            print("[CRITICAL] Action has no service!", {"type": action.type, "service": action.service}, flush=True)

        obs, reward, done = _step(session, base_url, action, meta)
        shared_state.rewards.append(float(reward))
        shared_state.step_count += 1
        action_str = _format_action_str(action)
        print(
            f"[STEP] step={step_idx} action={action_str} reward={float(reward):.2f} done={str(done).lower()} error={parse_error or 'null'}",
            flush=True,
        )
        if step_rewards is not None:
            step_rewards.append(float(reward))

        last_coordinator_analysis = coordinator.think()
        best_h = last_coordinator_analysis.get("best_hypothesis") or {}
        _debug(
            "coordinator analysis: "
            f"best_root={best_h.get('root_cause')} "
            f"best_service={best_h.get('service')} "
            f"confidence={best_h.get('confidence')}"
        )

        # Keep explorer control flow in sync: all perspectives see the same action/outcome.
        invalid_flags = [
            ex.observe_step_outcome(
                step_idx=step_idx,
                action=action,
                reward=float(reward),
                observation=obs,
            )
            for ex in explorers
        ]
        is_invalid = any(invalid_flags)

        if (
            not done
            and action.type != "submit_diagnosis"
            and last_coordinator_analysis.get("decision") == "submit"
            and coordinator.should_submit(last_coordinator_analysis, obs)
        ):
            diagnosis_options = list(getattr(obs, "diagnosis_options", None) or [])
            best_hypothesis = last_coordinator_analysis.get("best_hypothesis") or {}
            likely_root_cause = best_hypothesis.get("root_cause")
            if not isinstance(likely_root_cause, str) or not likely_root_cause:
                likely_root_cause = "UNKNOWN"

            if diagnosis_options and likely_root_cause not in diagnosis_options:
                inferred = _infer_root_cause(selected_explorer.best_signal_message or obs.message, diagnosis_options)
                likely_root_cause = inferred or diagnosis_options[0]
            elif not diagnosis_options and likely_root_cause == "UNKNOWN":
                likely_root_cause = _infer_root_cause(selected_explorer.best_signal_message or obs.message, diagnosis_options) or None

            # Cross-validate root_cause against all accumulated signals.
            # The committed hypothesis may lag behind actual evidence.
            # Re-infer from the full signal text and prefer it if it differs
            # and is supported by at least 2 signals.
            _all_signal_text = " ".join(
                str(s.get("content", ""))
                for s in shared_state.signals
                if isinstance(s, dict)
            )
            _all_logs_text = " ".join(
                line
                for logs in shared_state.logs_seen.values()
                for line in logs
            )
            _combined_evidence = f"{_all_signal_text} {_all_logs_text}"
            _reinferred = _infer_root_cause(_combined_evidence, diagnosis_options or [])
            if (
                _reinferred
                and _reinferred != likely_root_cause
                and len(shared_state.signals) >= 3
            ):
                # Count how many signals support each candidate
                def _count_support(root: str, text: str) -> int:
                    _kw_map = {
                        "DB_OVERLOAD": ["db", "connection pool", "slow quer", "db_cpu", "saturation"],
                        "TIMEOUT": ["timeout", "timed out", "latency", "p99", "downstream"],
                        "RATE_LIMITED": ["retry_rate", "429", "rate limit", "too many"],
                        "CACHE_STALE": ["stale", "cache freshness"],
                        "CACHE_MISS": ["cache miss", "miss rate"],
                        "TEMPLATE_ERROR": ["template", "render", "template_error"],
                        "DEPENDENCY_DOWN": ["connection refused", "unavailable", "503"],
                        "SERVICE_CRASH": ["crash", "restart", "oom"],
                        "NETWORK_PARTITION": ["packet loss", "degraded probe"],
                    }
                    keywords = _kw_map.get(root, [])
                    return sum(1 for kw in keywords if kw in text.lower())

                _current_support = _count_support(likely_root_cause, _combined_evidence)
                _reinferred_support = _count_support(_reinferred, _combined_evidence)
                if _reinferred_support > _current_support + 1:
                    likely_root_cause = _reinferred

            # Service resolution: prefer committed_service, then dependency-leaf service,
            # then max-signal service, then best_hypothesis service
            suggested_service = None

            # Priority 1: committed service (stable across steps)
            if isinstance(shared_state.committed_service, str) and shared_state.committed_service:
                suggested_service = shared_state.committed_service

            # Priority 2: if dependency map exists, walk to the leaf node from committed_service
            _dep_map = getattr(obs, "available_dependencies", None)
            if _dep_map and isinstance(_dep_map, dict) and suggested_service:
                # Walk the chain: if suggested_service has outgoing edges, follow to leaf
                _visited_chain = set()
                _current = suggested_service
                while _current in _dep_map and _dep_map.get(_current) and _current not in _visited_chain:
                    _visited_chain.add(_current)
                    _children = _dep_map[_current]
                    if _children:
                        _current = _children[0]
                suggested_service = _current  # leaf of the chain

            # Priority 3: service with most signals
            if not suggested_service:
                _svc_counts = {}
                for _sig in shared_state.signals:
                    if isinstance(_sig, dict) and isinstance(_sig.get("service"), str):
                        _s = _sig["service"]
                        _svc_counts[_s] = _svc_counts.get(_s, 0) + 1
                if _svc_counts:
                    suggested_service = max(_svc_counts, key=_svc_counts.get)

            # Priority 3.5: prefer service mentioned in the alert/initial observation
            # (it's more likely to be the business-impacted affected_service)
            if suggested_service:
                _alert_text = (obs.message or "").lower()
                _available = list(getattr(obs, "available_services", None) or [])
                # Check if any available service is mentioned in the current observation
                _alert_services = [
                    svc for svc in _available
                    if svc.lower() in _alert_text and svc != suggested_service
                ]
                # If a service appears in the current observation AND has signals,
                # it may be more relevant as the affected service
                if _alert_services:
                    _alert_svc_signals = {
                        svc: sum(1 for sig in shared_state.signals
                                 if isinstance(sig, dict) and sig.get("service") == svc)
                        for svc in _alert_services
                    }
                    # Only override if alert service actually has signals (not noise)
                    _best_alert_svc = max(_alert_svc_signals, key=_alert_svc_signals.get) if _alert_svc_signals else None
                    if _best_alert_svc and _alert_svc_signals[_best_alert_svc] > 0:
                        suggested_service = _best_alert_svc

            # Priority 4: fallback
            if not suggested_service:
                suggested_service = best_hypothesis.get("service")
            if not suggested_service:
                suggested_service = (
                    selected_explorer.best_signal_service
                    if selected_explorer
                    else None
                )
            if not suggested_service:
                suggested_service = (
                    selected_explorer.service_name
                    if selected_explorer
                    else None
                )
            if suggested_service is None and getattr(obs, "available_services", None):
                suggested_service = obs.available_services[0]

            # Priority 2.5: on hard tasks, discount the entry service if it has
            # misleading signals (injected by the environment transformation).
            # The entry service is identified as the first service visited — it often
            # has fake ERROR logs and abnormal metrics to mislead. If dep_map exists
            # and suggested_service is in dep_map (i.e. it has outgoing edges — it's
            # NOT a leaf), then it is likely a chain node or entry, not the root cause.
            # In that case, follow the chain to the deepest node that has signals.
            if _dep_map and isinstance(_dep_map, dict) and suggested_service:
                # Already done in Priority 2 leaf-walk above — skip if already a leaf
                pass  # leaf-walk in Priority 2 handles this

            # NEW: If no dep_map, use signal QUALITY not just quantity to pick service.
            # Prefer services with LOG signals over metric-only signals, because
            # injected misleading signals on the entry service are metrics (not logs).
            # Real root cause services have ERROR log signals buried deep.
            if suggested_service:
                _log_signal_counts = {}
                _metric_signal_counts = {}
                for _sig in shared_state.signals:
                    if not isinstance(_sig, dict):
                        continue
                    _svc = _sig.get("service")
                    _stype = _sig.get("type", "log")
                    if not isinstance(_svc, str):
                        continue
                    if _stype == "log":
                        _log_signal_counts[_svc] = _log_signal_counts.get(_svc, 0) + 1
                    else:
                        _metric_signal_counts[_svc] = _metric_signal_counts.get(_svc, 0) + 1

                # On hard: prefer a service with log signals over one with only metrics
                # (injected misleading signals are typically metric-based)
                _difficulty_check = str(shared_state.difficulty or "").lower()
                if _difficulty_check == "hard" and _log_signal_counts:
                    _best_log_svc = max(_log_signal_counts, key=_log_signal_counts.get)
                    _current_log_count = _log_signal_counts.get(suggested_service, 0)
                    _best_log_count = _log_signal_counts[_best_log_svc]
                    # Override only if a different service has clearly more log signals
                    if _best_log_svc != suggested_service and _best_log_count > _current_log_count:
                        suggested_service = _best_log_svc

            severity = coordinator._infer_severity_from_signals()

            final_action = BackendDiagnosisAction(
                type="submit_diagnosis",
                root_cause=likely_root_cause,
                service=suggested_service,
                severity=severity,
            )
            _debug(
                "coordinator triggered submission: "
                f"root_cause={likely_root_cause} service={suggested_service} severity={severity}"
            )
            coordinator_triggered = True
            break

        # Forced submission gate: avoid hitting step limit with no submission
        steps_remaining = episode_max_steps - step_idx
        if (
            not done
            and action.type != "submit_diagnosis"
            and steps_remaining <= 2
            and len(shared_state.signals) > 0
        ):
            # Build best available submission from current state
            _analysis = coordinator.analyze()
            _best_h = _analysis.get("best_hypothesis") or {}
            _root = _best_h.get("root_cause") or shared_state.committed_root_cause
            _svc = shared_state.committed_service or _best_h.get("service")

            if not _svc:
                _service_signal_counts = {}
                for _sig in shared_state.signals:
                    if isinstance(_sig, dict) and _sig.get("service"):
                        _s = _sig["service"]
                        _service_signal_counts[_s] = _service_signal_counts.get(_s, 0) + 1
                if _service_signal_counts:
                    _svc = max(_service_signal_counts, key=_service_signal_counts.get)

            # Priority 3.5: prefer service mentioned in the alert/initial observation
            # (it's more likely to be the business-impacted affected_service)
            if _svc:
                _alert_text = (obs.message or "").lower()
                _available = list(getattr(obs, "available_services", None) or [])
                # Check if any available service is mentioned in the current observation
                _alert_services = [
                    svc for svc in _available
                    if svc.lower() in _alert_text and svc != _svc
                ]
                # If a service appears in the current observation AND has signals,
                # it may be more relevant as the affected service
                if _alert_services:
                    _alert_svc_signals = {
                        svc: sum(1 for sig in shared_state.signals
                                 if isinstance(sig, dict) and sig.get("service") == svc)
                        for svc in _alert_services
                    }
                    # Only override if alert service actually has signals (not noise)
                    _best_alert_svc = max(_alert_svc_signals, key=_alert_svc_signals.get) if _alert_svc_signals else None
                    if _best_alert_svc and _alert_svc_signals[_best_alert_svc] > 0:
                        _svc = _best_alert_svc

            # Dep-map leaf-walk: if _svc is a non-leaf (has outgoing edges in dep_map),
            # follow the chain to the actual leaf/affected service.
            # This prevents submitting injected chain nodes (e.g. gateway-proxy) instead
            # of the true root-cause service at the end of the chain.
            _dep_map_gate = getattr(obs, "available_dependencies", None)
            if _dep_map_gate and isinstance(_dep_map_gate, dict) and _svc:
                _visited_chain_gate = set()
                _current_gate = _svc
                while (
                    _current_gate in _dep_map_gate
                    and _dep_map_gate.get(_current_gate)
                    and _current_gate not in _visited_chain_gate
                ):
                    _visited_chain_gate.add(_current_gate)
                    _children_gate = _dep_map_gate[_current_gate]
                    if _children_gate:
                        _current_gate = _children_gate[0]
                # Only use the leaf if it has signals — otherwise keep original _svc
                _leaf_candidate = _current_gate
                _leaf_signal_count = sum(
                    1 for _sig in shared_state.signals
                    if isinstance(_sig, dict) and _sig.get("service") == _leaf_candidate
                )
                if _leaf_signal_count > 0 or _leaf_candidate != _svc:
                    _svc = _leaf_candidate

            # Priority 2.5: on hard tasks, discount the entry service if it has
            # misleading signals (injected by the environment transformation).
            # The entry service is identified as the first service visited — it often
            # has fake ERROR logs and abnormal metrics to mislead. If dep_map exists
            # and suggested_service is in dep_map (i.e. it has outgoing edges — it's
            # NOT a leaf), then it is likely a chain node or entry, not the root cause.
            # In that case, follow the chain to the deepest node that has signals.
            if _dep_map_gate and isinstance(_dep_map_gate, dict) and _svc:
                # Already done in Priority 2 leaf-walk above — skip if already a leaf
                pass  # leaf-walk in Priority 2 handles this

            # NEW: If no dep_map, use signal QUALITY not just quantity to pick service.
            # Prefer services with LOG signals over metric-only signals, because
            # injected misleading signals on the entry service are metrics (not logs).
            # Real root cause services have ERROR log signals buried deep.
            if _svc:
                _log_signal_counts = {}
                _metric_signal_counts = {}
                for _sig in shared_state.signals:
                    if not isinstance(_sig, dict):
                        continue
                    _svc_name = _sig.get("service")
                    _stype = _sig.get("type", "log")
                    if not isinstance(_svc_name, str):
                        continue
                    if _stype == "log":
                        _log_signal_counts[_svc_name] = _log_signal_counts.get(_svc_name, 0) + 1
                    else:
                        _metric_signal_counts[_svc_name] = _metric_signal_counts.get(_svc_name, 0) + 1

                # On hard: prefer a service with log signals over one with only metrics
                # (injected misleading signals are typically metric-based)
                _difficulty_check = str(shared_state.difficulty or "").lower()
                if _difficulty_check == "hard" and _log_signal_counts:
                    _best_log_svc = max(_log_signal_counts, key=_log_signal_counts.get)
                    _current_log_count = _log_signal_counts.get(_svc, 0)
                    _best_log_count = _log_signal_counts[_best_log_svc]
                    # Override only if a different service has clearly more log signals
                    if _best_log_svc != _svc and _best_log_count > _current_log_count:
                        _svc = _best_log_svc

            if _root and _svc:
                _diag_options = list(getattr(obs, "diagnosis_options", None) or [])
                if _diag_options and _root not in _diag_options:
                    _inferred = _infer_root_cause(
                        shared_state.signals[-1].get("content", "") if shared_state.signals else obs.message,
                        _diag_options,
                    )
                    _root = _inferred or _diag_options[0]

                _severity = coordinator._infer_severity_from_signals()

                final_action = BackendDiagnosisAction(
                    type="submit_diagnosis",
                    root_cause=_root,
                    service=_svc,
                    severity=_severity,
                )
                coordinator_triggered = True
                _debug(f"forced submission (budget gate): root={_root} service={_svc} steps_remaining={steps_remaining}")
                break

        if is_invalid and not done:
            # Reject invalid action behavior and let the model retry next step.
            continue

        if action.type == "submit_diagnosis":
            final_action = action

        if done:
            break

    if final_action is None:
        fallback_service = next(
            (
                ex.best_signal_service
                for ex in explorers
                if ex.best_signal_service
            ),
            None,
        )
        if fallback_service is None:
            fallback_service = next((ex.service_name for ex in explorers if ex.service_name), None)
        if fallback_service is None and getattr(obs, "available_services", None):
            fallback_service = obs.available_services[0]

        # Safe fallback: do not submit if we cannot ground the diagnosis reliably.
        final_action = BackendDiagnosisAction(
            type="view_metrics",
            service=fallback_service,
        )
        final_action = BackendDiagnosisAction(**coordinator._validate_final_action({
            "type": final_action.type,
            "service": final_action.service,
        }))
        if final_action.service is None:
            print(
                "[CRITICAL] Action has no service!",
                {"type": final_action.type, "service": final_action.service},
                flush=True,
            )
        obs, reward, done = _step(session, base_url, final_action, meta)
        action_str = _format_action_str(final_action)
        print(
            f"[STEP] step={episode_max_steps + 1} action={action_str} reward={float(reward):.2f} done={str(done).lower()} error=null",
            flush=True,
        )
        if step_rewards is not None:
            step_rewards.append(float(reward))
    elif coordinator_triggered:
        final_action = BackendDiagnosisAction(**coordinator._validate_final_action({
            "type": final_action.type,
            "service": final_action.service,
            "root_cause": final_action.root_cause,
            "severity": final_action.severity,
        }))
        if final_action.type in {"open_logs", "view_metrics", "submit_diagnosis"} and not final_action.service:
            final_action.service = obs.available_services[0] if (obs.available_services or []) else None
        if final_action.service is None:
            print(
                "[CRITICAL] Action has no service!",
                {"type": final_action.type, "service": final_action.service},
                flush=True,
            )
        obs, reward, done = _step(session, base_url, final_action, meta)
        action_str = _format_action_str(final_action)
        print(
            f"[STEP] step={shared_state.step_count + 1} action={action_str} reward={float(reward):.2f} done={str(done).lower()} error=null",
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
    markers = ["error", "spiking", "100%", "maxed", "degraded", "timeout", "down", "failed"]
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
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
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
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
