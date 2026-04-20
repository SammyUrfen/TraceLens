# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Backend Diagnosis Environment Implementation.

Dataset-driven incident investigation loop: reset → step through log windows → submit diagnosis.
Logs are served in windows of size 3 (latest first), with per-service pointers.
Reward philosophy: reward discovery (first-time signals), not mere actions; final diagnosis dominates.
Difficulty design (dataset guidelines):
- Easy: signal visible in first window; entry_service == affected_service; minimal noise.
- Medium: requires scroll or metrics; signal not immediate; moderate noise.
- Hard: entry_service != affected_service; misleading logs; needs cross-service reasoning.
"""

DIAGNOSIS_TAXONOMY = [
    "DB_OVERLOAD",
    "CACHE_STALE",
    "CACHE_MISS",
    "NETWORK_PARTITION",
    "TEMPLATE_ERROR",
    "DEPLOY_REGRESSION",
    "PAYMENTS_DOWN",
    "SERVICE_CRASH",
    "MEMORY_LEAK",
    "TIMEOUT",
    "RATE_LIMITED",
    "DISK_FULL",
    "CONFIG_ERROR",
    "AUTH_FAILURE",
    "DEPENDENCY_DOWN",
]

import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Set

try:
    from openenv.core.env_server.interfaces import Environment
except Exception:  # pragma: no cover - fallback when openenv is unavailable
    class Environment:  # type: ignore
        pass

try:
    from models import (
        BackendDiagnosisAction,
        BackendDiagnosisObservation,
        BackendDiagnosisState,
    )
except ImportError:  # pragma: no cover - fallback for direct execution
    from models import BackendDiagnosisAction, BackendDiagnosisObservation, BackendDiagnosisState


class BackendDiagnosisEnvironment(Environment):
    """Interactive environment exposing logs in fixed windows."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    LOG_WINDOW: int = 3
    MAX_STEPS: int = 10  # kept for backward compatibility; default budget
    SIGNAL_SCALE: float = 0.05  # keep signal rewards small compared to final reward
    PENALTY_REPEAT: float = -0.02

    def __init__(self, dataset_path: str | Path | None = None):
        dataset_file = Path(dataset_path) if dataset_path else Path(__file__).parent / "incidents.json"
        self._dataset = self._load_dataset(dataset_file)
        self._validate_dataset(self._dataset)
        self._incidents = self._flatten_incidents(self._dataset)
        if not self._incidents:
            raise ValueError("Dataset must contain at least one incident")

        self._state: BackendDiagnosisState | None = None
        self._reset_count = 0
        self._current_incident: Dict[str, object] | None = None
        self.max_steps: int = self.MAX_STEPS

    def reset(self, seed: int | None = None, difficulty: str | None = None) -> BackendDiagnosisObservation:
        """Select a random incident, initialize state, and return the initial observation.

        - Picks one incident from the loaded dataset (all difficulty buckets combined).
        - Initializes state with the incident id, entry_service as current_service, empty log/metric tracking, log pointers, step counter, and done flag.
        - Returns the alert message and available tools for the first step.
        """

        if seed is not None:
            random.seed(seed)


        self._reset_count += 1
        pool = [inc for inc in self._incidents if difficulty is None or inc.get("difficulty") == difficulty]
        if not pool:
            raise ValueError(f"No incidents available for difficulty={difficulty}")
        base_incident = random.choice(pool)

        # Semi-dynamic transformation: deep-copy and apply complexity
        incident = copy.deepcopy(base_incident)
        incident = self._apply_complexity(incident)
        self._current_incident = incident

        entry_service = self._current_incident.get("entry_service")
        self.max_steps = int(self._current_incident.get("max_steps", self.MAX_STEPS) or self.MAX_STEPS)
        self.state = BackendDiagnosisState(
            incident_id=self._current_incident.get("incident_id", "unknown_incident"),
            current_service=entry_service,
            log_pointers={},
            logs_seen={},
            metrics_seen={},
            discovered_signals_count=0,
            steps_taken=0,
            done=False,
            difficulty=self._current_incident.get("difficulty"),
            services_visited=set(),
            max_possible_signals=self._estimate_max_signals(self._current_incident),
            seen_signals=set(),
            last_action=(None, None),
            action_history={},
        )

        return BackendDiagnosisObservation(
            message=self._current_incident.get("alert", ""),
            available_tools=["open_logs", "view_metrics"],
            available_services=self._available_services(),
            diagnosis_options=self._diagnosis_options(),
            reward=0.0,
            available_dependencies=self._get_dependencies(),
        )

    def step(
        self,
        action: BackendDiagnosisAction,
        seed: Optional[int] = None,
        difficulty: Optional[str] = None,
    ) -> BackendDiagnosisObservation:
        """Process one agent action, update state, and return the observation.

        Applies a small step penalty for navigation actions, enforces a max step budget,
        and handles actions: open_logs, scroll_logs, view_metrics, submit_diagnosis.
        Log windows are served in chunks of size LOG_WINDOW per service, with per-service
        pointers that move backwards in time as the agent scrolls.
        Reward shaping: reward only first-time discovery of signals (error lines or abnormal metrics),
        scale signals small (SIGNAL_SCALE), apply step penalty (-0.01) and repeat penalty (-0.02),
        no direct reward for selecting services; exploration is neutral unless repeating.
        """

        if self.state is None or self._current_incident is None:
            # Lazily initialize to support stateless HTTP calls when step is invoked before reset.
            self.reset(seed=seed, difficulty=difficulty)

        if self.state.done:
            obs = BackendDiagnosisObservation(
                message="Episode already finished. Please reset.",
                available_tools=[],
                available_services=self._available_services(),
                diagnosis_options=self._diagnosis_options(),
                reward=0.0,
                done=True,
            )
            obs.signals_discovered = self.state.discovered_signals_count
            obs.services_explored = len(self.state.services_visited)
            obs.progress_score = self._progress_score()
            return obs

        prev_signals = self.state.discovered_signals_count if self.state else 0
        self.state.steps_taken += 1

        if self.state.steps_taken > self.max_steps:
            self.state.done = True
            obs = BackendDiagnosisObservation(
                message="Step limit reached.",
                available_tools=[],
                available_services=self._available_services(),
                diagnosis_options=self._diagnosis_options(),
                reward=0.0,
                done=True,
            )
            obs.signals_discovered = self.state.discovered_signals_count
            obs.services_explored = len(self.state.services_visited)
            obs.progress_score = self._progress_score()
            return obs

        try:
            reward = 0.0
            invalid_action = False

            key = (action.type, action.service or "")
            action_key = f"{key[0]}|{key[1]}"
            count = self.state.action_history.get(action_key, 0) + 1
            self.state.action_history[action_key] = count

            if action.type == "open_logs":
                obs, done, reward_delta = self._handle_open_logs(action)
                reward += reward_delta
            elif action.type == "scroll_logs":
                obs, done, reward_delta = self._handle_scroll_logs()
                reward += reward_delta
            elif action.type == "view_metrics":
                obs, done, reward_delta = self._handle_view_metrics(action)
                reward += reward_delta
            elif action.type == "submit_diagnosis":
                obs, reward, done, _info = self._handle_submit(action)
            else:
                obs = BackendDiagnosisObservation(
                    message="Invalid action",
                    available_tools=["open_logs", "view_metrics"],
                )
                reward = -0.05
                done = False
                invalid_action = True

            if not invalid_action and action.type != "submit_diagnosis":
                new_signal_count = (self.state.discovered_signals_count - prev_signals) if self.state else 0
                new_signals = new_signal_count > 0
                if new_signals:
                    # Upgrade 2: Evidence reward — reward meaningful signal discovery, capped per step
                    reward += min(0.1, 0.02 * new_signal_count)
                else:
                    if count > 1:
                        reward += self.PENALTY_REPEAT * count

            # Phase 2: Light step penalty after step 3 — encourage efficiency
            if not invalid_action and action.type != "submit_diagnosis" and self.state.steps_taken > 3:
                reward -= 0.005

        except Exception as e:
            obs = BackendDiagnosisObservation(
                message=f"Invalid action: {e}",
                available_tools=["open_logs", "view_metrics"],
            )
            reward = -0.1
            done = False
            invalid_action = True

        if self.state is not None:
            self.state.last_action = (action.type, action.service)
        obs.reward = reward
        obs.done = bool(done)
        obs.signals_discovered = self.state.discovered_signals_count
        obs.services_explored = len(self.state.services_visited)
        obs.progress_score = self._progress_score()
        obs.available_services = self._available_services()
        obs.diagnosis_options = self._diagnosis_options()
        obs.available_dependencies = self._get_dependencies()
        return obs

    @staticmethod
    def _estimate_max_signals(incident: Dict[str, object]) -> int:
        """Approximate max signals as total ERROR lines across services."""
        services = incident.get("services", {}) if isinstance(incident, dict) else {}
        total = 0
        for svc_data in services.values():
            logs = svc_data.get("logs", []) if isinstance(svc_data, dict) else []
            total += sum(1 for line in logs if isinstance(line, str) and ("ERROR" in line or "error" in line))
        return total

    def _progress_score(self) -> float:
        if self.state is None:
            return 0.0

        fallback_max = 0
        if self.state.max_possible_signals <= 0 and self._current_incident:
            fallback_max = self._estimate_max_signals(self._current_incident)

        max_signals = self.state.max_possible_signals or fallback_max or 0
        if max_signals <= 0:
            return 0.0

        return self.state.discovered_signals_count / max_signals

    def _available_services(self) -> list[str]:
        if self._current_incident is None:
            return []
        services = self._current_incident.get("services", {})
        if isinstance(services, dict):
            return list(services.keys())
        return []

    def _diagnosis_options(self) -> list[str]:
        if self._current_incident is None:
            return []
        options = self._current_incident.get("diagnosis_options", [])
        return list(options) if isinstance(options, list) else []

    def _register_signals(self, signal_keys: Set[str]) -> int:
        """Track newly discovered signal keys; return count of new ones."""

        if self.state is None:
            return 0

        new_signals = signal_keys - self.state.seen_signals
        if new_signals:
            self.state.seen_signals.update(new_signals)
        return len(new_signals)

    def _handle_open_logs(self, action: BackendDiagnosisAction):
        if not action.service:
            return (
                BackendDiagnosisObservation(
                    message="Invalid action",
                    available_tools=["open_logs", "view_metrics"],
                ),
                False,
                -0.05,
            )

        services = self._current_incident["services"]
        service_data = services.get(action.service)
        if service_data is None:
            return (
                BackendDiagnosisObservation(
                    message="Invalid action",
                    available_tools=["open_logs", "view_metrics"],
                ),
                False,
                -0.05,
            )

        logs = service_data.get("logs", [])

        if action.service not in self.state.log_pointers:
            pointer = max(0, len(logs) - self.LOG_WINDOW)
            self.state.log_pointers[action.service] = pointer
        else:
            pointer = self.state.log_pointers[action.service]

        self.state.current_service = action.service
        self.state.logs_seen[action.service] = True
        self.state.services_visited.add(action.service)

        window = logs[pointer : pointer + self.LOG_WINDOW]

        reward_delta = 0.0
        error_lines = {line for line in window if "ERROR" in line or "error" in line}
        signal_keys = {f"log|{action.service}|{line}" for line in error_lines}
        new_count = self._register_signals(signal_keys)
        if new_count:
            self.state.discovered_signals_count += new_count
            reward_delta += self.SIGNAL_SCALE

        return (
            BackendDiagnosisObservation(
                message="\n".join(window),
                available_tools=["open_logs", "scroll_logs", "view_metrics", "submit_diagnosis"],
            ),
            False,
            reward_delta,
        )

    def _handle_scroll_logs(self):
        if not self.state.current_service:
            return (
                BackendDiagnosisObservation(
                    message="Invalid action",
                    available_tools=["open_logs", "view_metrics"],
                ),
                False,
                -0.05,
            )

        service = self.state.current_service
        service_data = self._current_incident["services"][service]
        logs = service_data.get("logs", [])

        current_pointer = self.state.log_pointers.get(service, len(logs))
        new_pointer = max(0, current_pointer - self.LOG_WINDOW)
        self.state.log_pointers[service] = new_pointer

        window = logs[new_pointer : new_pointer + self.LOG_WINDOW]

        reward_delta = 0.0

        error_lines = {line for line in window if "ERROR" in line or "error" in line}
        signal_keys = {f"log|{service}|{line}" for line in error_lines}
        new_count = self._register_signals(signal_keys)
        if new_count:
            self.state.discovered_signals_count += new_count
            reward_delta += self.SIGNAL_SCALE

        return (
            BackendDiagnosisObservation(
                message="\n".join(window),
                available_tools=["open_logs", "scroll_logs", "submit_diagnosis", "view_metrics"],
            ),
            False,
            reward_delta,
        )

    def _handle_view_metrics(self, action: BackendDiagnosisAction):
        """Return formatted metrics for a service and apply metric-based shaping."""
        if not action.service:
            return (
                BackendDiagnosisObservation(
                    message="Invalid action",
                    available_tools=["open_logs", "view_metrics"],
                ),
                False,
                -0.05,
            )

        services = self._current_incident["services"]
        service_data = services.get(action.service)
        if service_data is None:
            return (
                BackendDiagnosisObservation(
                    message="Invalid action",
                    available_tools=["open_logs", "view_metrics"],
                ),
                False,
                -0.05,
            )

        metrics = service_data.get("metrics", {})
        self.state.metrics_seen[action.service] = True
        self.state.current_service = action.service

        lines = [f"{k}: {v}" for k, v in metrics.items()] if metrics else ["no metrics available"]
        message = "\n".join(lines)

        reward_delta = 0.0
        new_count = 0
        if metrics:
            abnormal_markers = {"high", "spiking", "100%", "maxed"}
            signal_keys = {
                f"metric|{action.service}|{k}|{str(v).lower()}"
                for k, v in metrics.items()
                if str(v).lower() in abnormal_markers
            }
            new_count = self._register_signals(signal_keys)
            if new_count:
                self.state.discovered_signals_count += new_count
                reward_delta += self.SIGNAL_SCALE
        # No metrics means neutral reward (no penalty, no bonus)

        # Signals are rewarded only on first discovery; exploration without new signals is neutral aside from step/repeat penalties.
        return (
            BackendDiagnosisObservation(
                message=message,
                available_tools=["open_logs", "scroll_logs", "submit_diagnosis", "view_metrics"],
            ),
            False,
            reward_delta,
        )

    def grade_episode(self, final_action: BackendDiagnosisAction) -> float:
        """Deterministic grader independent of reward.

        Scores in [0.0, 1.0] based on exact matches to ground_truth fields:
        root_cause (0.6), affected_service (0.3), severity (0.1).
        """

        if self._current_incident is None:
            return 0.0

        gt = self._current_incident.get("ground_truth", {})
        score = 0.0
        root_weight = 0.6
        related_root_causes = {
            "TIMEOUT": ["RATE_LIMITED", "DB_OVERLOAD"],
            "RATE_LIMITED": ["TIMEOUT"],
            "DB_OVERLOAD": ["TIMEOUT"],
        }

        if final_action.root_cause == gt.get("root_cause"):
            score += root_weight
        elif final_action.root_cause in related_root_causes.get(gt.get("root_cause"), []):
            score += root_weight * 0.6
        if final_action.service == gt.get("affected_service"):
            score += 0.3
        if getattr(final_action, "severity", None) == gt.get("severity"):
            score += 0.1

        return score

    @staticmethod
    def validate_hard_incidents(dataset: Dict[str, object]) -> bool:
        """Validate 'hard' incidents for cross-service misleading signals."""

        hard_incidents = dataset.get("hard", []) if isinstance(dataset, dict) else []
        abnormal_markers = {"high", "spiking", "100%", "maxed"}

        for inc in hard_incidents:
            gt = inc.get("ground_truth", {})
            services = inc.get("services", {})
            entry = inc.get("entry_service")
            affected = gt.get("affected_service")

            if entry == affected:
                return False

            entry_misleading = False
            non_entry_misleading = False

            for svc_name, svc_data in services.items():
                logs = svc_data.get("logs", []) if isinstance(svc_data, dict) else []
                metrics = svc_data.get("metrics", {}) if isinstance(svc_data, dict) else {}

                has_signal = any("ERROR" in line or "error" in line for line in logs) or any(
                    str(v).lower() in abnormal_markers for v in metrics.values()
                )

                if svc_name == entry and has_signal:
                    entry_misleading = True
                if svc_name != entry and has_signal:
                    non_entry_misleading = True

            if not (entry_misleading and non_entry_misleading):
                return False

        return True

    def _handle_submit(self, action: BackendDiagnosisAction):
        self.state.done = True

        diagnosis_options: Sequence[str] = self._current_incident.get("diagnosis_options", [])
        invalid_options = [opt for opt in diagnosis_options if opt not in DIAGNOSIS_TAXONOMY]
        allowed_options = [opt for opt in diagnosis_options if opt in DIAGNOSIS_TAXONOMY]
        allowed_taxonomy = allowed_options if allowed_options else DIAGNOSIS_TAXONOMY

        if action.root_cause not in allowed_taxonomy:
            return (
                BackendDiagnosisObservation(
                    message="Invalid diagnosis option submitted.",
                    available_tools=[],
                    done=True,
                ),
                -1.0,
                True,
                {
                    "error": "invalid_root_cause",
                    "invalid_options_filtered": invalid_options,
                    "final_correct": False,
                },
            )

        gt_root_cause = self._current_incident.get("ground_truth", {}).get("root_cause")
        final_correct = action.root_cause == gt_root_cause
        reward = 1.0 if final_correct else 0.0

        evidence_score = self.state.discovered_signals_count if self.state else 0
        if evidence_score == 0:
            reward *= 0.7
        elif self.state and self.state.difficulty == "hard" and evidence_score < 2:
            reward *= 0.85

        # Early-submit penalty: discourage rushing on medium/hard
        if self.state and self.state.steps_taken <= 2 and self.state.difficulty in ("medium", "hard"):
            reward *= 0.6

        # Multi-service discovery bonus: reward structured exploration
        if self.state and len(self.state.services_visited) >= 2:
            reward = min(1.0, reward + 0.1)

        # Upgrade 3: Reasoning depth bonus — reward deep multi-service exploration on correct diagnosis
        if final_correct and self.state and len(self.state.services_visited) >= 3:
            reward = min(1.0, reward + 0.1)

        return (
            BackendDiagnosisObservation(
                message="Diagnosis submitted.",
                available_tools=[],
                done=True,
            ),
            reward,
            True,
            {
                "final_correct": final_correct,
                "invalid_options_filtered": invalid_options,
            },
        )

    @property
    def state(self) -> BackendDiagnosisState:
        """Return internal environment state for debugging and grading purposes."""
        return self._state

    @state.setter
    def state(self, value: BackendDiagnosisState):
        self._state = value

    @staticmethod
    def _load_dataset(path: Path) -> Dict[str, object]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _validate_dataset(dataset: Dict[str, object]) -> None:
        """Validate diagnosis options against taxonomy and sanity check hard incidents."""

        def _validate_options(incidents: List[Dict[str, object]], difficulty: str) -> None:
            for inc in incidents:
                options = inc.get("diagnosis_options", [])
                invalid = [opt for opt in options if opt not in DIAGNOSIS_TAXONOMY]
                if invalid:
                    raise ValueError(f"Invalid diagnosis_options for {difficulty} incident {inc.get('incident_id')}: {invalid}")

        for key in ("easy", "medium", "hard"):
            incidents = dataset.get(key, []) if isinstance(dataset, dict) else []
            if not isinstance(incidents, list):
                continue
            _validate_options(incidents, key)

        if not BackendDiagnosisEnvironment.validate_hard_incidents(dataset):
            raise ValueError("Hard incident validation failed: ensure cross-service misleading signals are present")

    @staticmethod
    def _flatten_incidents(dataset: Dict[str, object]) -> List[Dict[str, object]]:
        combined: List[Dict[str, object]] = []
        for key in ("easy", "medium", "hard"):
            incidents = dataset.get(key, []) if isinstance(dataset, dict) else []
            if isinstance(incidents, list):
                for inc in incidents:
                    inc_copy = dict(inc)
                    inc_copy.setdefault("difficulty", key)
                    combined.append(inc_copy)
        return combined

    # -------------------------------------------------------------------------
    # Semi-dynamic transformation layer
    # -------------------------------------------------------------------------

    # Pool of realistic-sounding noise service names
    _NOISE_SERVICE_POOL: List[str] = [
        "observer-cache", "logging-sidecar", "metrics-collector",
        "health-monitor", "audit-logger", "tracing-proxy",
        "config-watcher", "session-store", "rate-limiter-sidecar",
        "cdn-purger", "event-bus", "schema-registry",
    ]

    # Filler INFO lines used for padding and noise service logs
    _FILLER_LOG_TEMPLATES: List[str] = [
        "INFO {svc} heartbeat ok",
        "INFO {svc} request accepted",
        "INFO {svc} request routed",
        "INFO {svc} cache lookup complete",
        "INFO {svc} dependency call started",
        "INFO {svc} dependency call completed",
        "INFO {svc} response serialized",
        "INFO {svc} response sent",
        "INFO {svc} connection pool healthy",
        "INFO {svc} gc pause within budget",
        "INFO {svc} config reload skipped (no changes)",
        "INFO {svc} upstream latency nominal",
    ]

    def _apply_complexity(self, incident: Dict[str, object]) -> Dict[str, object]:
        """Orchestrate semi-dynamic transformations based on difficulty.

        Easy:   no transformations (original behavior)
        Medium: 1 noise service, shuffle logs, split clues, delay signals (pad 4-6)
        Hard:   2 noise services, shuffle, split, misleading signals, delay (pad 8-12), dependency chain
        """
        difficulty = incident.get("difficulty", "easy")

        if difficulty == "easy":
            return incident

        if difficulty == "medium":
            incident = self._add_noise_services(incident, count=1)
            incident = self._shuffle_log_positions(incident)
            incident = self._split_clues_across_services(incident)
            incident = self._delay_key_signals(incident, padding=random.randint(4, 6))
            # Bump max_steps to give agent time to navigate the added complexity
            incident["max_steps"] = int(incident.get("max_steps", self.MAX_STEPS) or self.MAX_STEPS) + 2
            return incident

        if difficulty == "hard":
            incident = self._add_noise_services(incident, count=2)
            incident = self._shuffle_log_positions(incident)
            incident = self._split_clues_across_services(incident)
            incident = self._add_misleading_signals(incident)
            incident = self._delay_key_signals(incident, padding=random.randint(8, 12))
            incident = self._inject_dependency_chain(incident)
            incident["max_steps"] = int(incident.get("max_steps", self.MAX_STEPS) or self.MAX_STEPS) + 4
            return incident

        return incident

    # -- Transformation 1: Noise Services ------------------------------------

    def _add_noise_services(self, incident: Dict[str, object], count: int) -> Dict[str, object]:
        """Inject `count` fake services with benign logs and normal metrics."""
        services = incident.get("services", {})
        existing_names = set(services.keys())
        available_noise = [n for n in self._NOISE_SERVICE_POOL if n not in existing_names]
        random.shuffle(available_noise)

        for name in available_noise[:count]:
            filler_count = random.randint(8, 14)
            logs = [t.format(svc=name) for t in random.choices(self._FILLER_LOG_TEMPLATES, k=filler_count)]
            logs.insert(0, f"INFO {name} worker started")
            normal_metrics = random.choice([
                {"cpu": "30%", "memory": "45%"},
                {"latency_p99": "normal", "error_rate": "0.01%"},
                {"connections": "12", "queue_depth": "0"},
                {"gc_pause": "2ms", "heap_usage": "40%"},
            ])
            services[name] = {"logs": logs, "metrics": normal_metrics}

        incident["services"] = services
        return incident

    # -- Transformation 2: Shuffle Log Positions -----------------------------

    @staticmethod
    def _shuffle_log_positions(incident: Dict[str, object]) -> Dict[str, object]:
        """Shuffle logs within each service, keeping the first 'worker started' line anchored."""
        services = incident.get("services", {})
        for svc_name, svc_data in services.items():
            if not isinstance(svc_data, dict):
                continue
            logs = svc_data.get("logs", [])
            if len(logs) <= 2:
                continue
            # Keep the first line (worker started) as anchor
            anchor = logs[0]
            rest = logs[1:]
            random.shuffle(rest)
            svc_data["logs"] = [anchor] + rest
        return incident

    # -- Transformation 3: Split Clues Across Services -----------------------

    @staticmethod
    def _split_clues_across_services(incident: Dict[str, object]) -> Dict[str, object]:
        """Take the key ERROR from the affected service and insert a related upstream clue elsewhere."""
        gt = incident.get("ground_truth", {})
        affected = gt.get("affected_service")
        services = incident.get("services", {})

        if not affected or affected not in services:
            return incident

        affected_logs = services[affected].get("logs", []) if isinstance(services.get(affected), dict) else []
        error_lines = [l for l in affected_logs if isinstance(l, str) and ("ERROR" in l or "error" in l)]
        if not error_lines:
            return incident

        # Pick a different service to inject the upstream clue into
        other_services = [s for s in services if s != affected]
        if not other_services:
            return incident

        target_svc = random.choice(other_services)
        upstream_clue = f"ERROR dependency call to {affected} returned failure"
        target_logs = services[target_svc].get("logs", []) if isinstance(services[target_svc], dict) else []
        # Insert the clue somewhere in the middle
        insert_pos = max(1, len(target_logs) // 2)
        target_logs.insert(insert_pos, upstream_clue)
        services[target_svc]["logs"] = target_logs
        return incident

    # -- Transformation 4: Add Misleading Signals ----------------------------

    @staticmethod
    def _add_misleading_signals(incident: Dict[str, object]) -> Dict[str, object]:
        """Add a fake ERROR + abnormal metric to the entry service to act as a false lead.

        Only applies when entry_service != affected_service (typical for hard tasks).
        """
        gt = incident.get("ground_truth", {})
        affected = gt.get("affected_service")
        entry = incident.get("entry_service")
        services = incident.get("services", {})

        if not entry or not affected or entry == affected:
            return incident

        if entry not in services or not isinstance(services[entry], dict):
            return incident

        # Inject a misleading error log
        misleading_errors = [
            "ERROR timeout connecting to upstream dependency",
            "ERROR unexpected latency spike in request handler",
            "ERROR intermittent connection reset observed",
            "ERROR request queue saturation detected",
        ]
        entry_logs = services[entry].get("logs", [])
        entry_logs.append(random.choice(misleading_errors))
        services[entry]["logs"] = entry_logs

        # Inject an abnormal metric to make the false lead convincing
        entry_metrics = services[entry].get("metrics", {})
        misleading_metric = random.choice([
            ("error_rate", "spiking"),
            ("latency_p99", "high"),
            ("cpu_usage", "100%"),
        ])
        entry_metrics[misleading_metric[0]] = misleading_metric[1]
        services[entry]["metrics"] = entry_metrics
        return incident

    # -- Transformation 5: Delay Key Signals ---------------------------------

    def _delay_key_signals(self, incident: Dict[str, object], padding: int) -> Dict[str, object]:
        """Prepend filler INFO lines to the affected service, burying ERRORs deep."""
        gt = incident.get("ground_truth", {})
        affected = gt.get("affected_service")
        services = incident.get("services", {})

        if not affected or affected not in services:
            return incident

        svc_data = services[affected]
        if not isinstance(svc_data, dict):
            return incident

        logs = svc_data.get("logs", [])
        filler = [t.format(svc=affected) for t in random.choices(self._FILLER_LOG_TEMPLATES, k=padding)]
        svc_data["logs"] = filler + logs
        return incident

    # -- Transformation 6: Inject Dependency Chain ---------------------------

    def _inject_dependency_chain(self, incident: Dict[str, object]) -> Dict[str, object]:
        """Build a synthetic dependency chain leading to the affected service.

        Creates upstream services that each log a cascading error pointing
        to their downstream, forcing the agent to traverse the full chain.
        Example chain: gateway-proxy → middleware-router → (affected service)
        """
        gt = incident.get("ground_truth", {})
        affected = gt.get("affected_service")
        services = incident.get("services", {})

        if not affected or affected not in services:
            return incident

        chain_templates = [
            ("gateway-proxy", "middleware-router"),
            ("edge-balancer", "request-dispatcher"),
            ("ingress-controller", "service-mesh"),
        ]
        existing_names = set(services.keys())
        # Pick a chain that doesn't collide with existing service names
        chain = None
        for candidate in chain_templates:
            if candidate[0] not in existing_names and candidate[1] not in existing_names:
                chain = candidate
                break
        if chain is None:
            return incident

        upstream, midstream = chain

        # midstream → points to affected
        mid_logs = [
            f"INFO {midstream} worker started",
        ]
        mid_logs += [t.format(svc=midstream) for t in random.choices(self._FILLER_LOG_TEMPLATES, k=6)]
        mid_logs += [
            f"WARN {midstream} elevated latency on downstream call to {affected}",
            f"ERROR {midstream} downstream {affected} returned error",
        ]
        services[midstream] = {
            "logs": mid_logs,
            "metrics": {"downstream_errors": "spiking", "latency_p99": "high"},
        }

        # upstream → points to midstream
        up_logs = [
            f"INFO {upstream} worker started",
        ]
        up_logs += [t.format(svc=upstream) for t in random.choices(self._FILLER_LOG_TEMPLATES, k=6)]
        up_logs += [
            f"WARN {upstream} dependency {midstream} responding slowly",
            f"ERROR {upstream} cascading failure from {midstream}",
        ]
        services[upstream] = {
            "logs": up_logs,
            "metrics": {"error_rate": "high", "upstream_health": "degraded"},
        }

        # Store the dependency map on the incident for observation exposure
        incident["_dependency_map"] = {
            upstream: [midstream],
            midstream: [affected],
        }

        incident["services"] = services
        return incident

    # -- Dependency graph helper ---------------------------------------------

    def _get_dependencies(self) -> Optional[Dict[str, List[str]]]:
        """Return the dependency map if one was injected, else None."""
        if not self._current_incident:
            return None
        dep_map = self._current_incident.get("_dependency_map")
        if dep_map and isinstance(dep_map, dict):
            return dep_map
        return None
