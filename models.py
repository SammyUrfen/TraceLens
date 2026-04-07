# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Backend Diagnosis Environment.

Minimal models supporting the interactive incident investigation loop.
"""
from typing import Literal, Optional, Set

from pydantic import BaseModel


class BackendDiagnosisAction(BaseModel):
    """Agent action for the backend diagnosis environment."""

    type: Literal["open_logs", "scroll_logs", "submit_diagnosis", "view_metrics"]
    service: Optional[str] = None
    root_cause: Optional[str] = None
    severity: Optional[str] = None


class BackendDiagnosisObservation(BaseModel):
    """Observation returned to the agent after each step."""

    message: str
    available_tools: list[str]
    available_services: list[str] | None = None
    diagnosis_options: list[str] | None = None
    reward: float | None = None
    done: bool = False
    signals_discovered: int | None = None
    services_explored: int | None = None
    progress_score: float | None = None


class BackendDiagnosisReward(BaseModel):
    """Reward payload for schema completeness."""

    value: float
    done: bool


class BackendDiagnosisState(BaseModel):
    """Environment state exposed via the state endpoint."""

    incident_id: str
    current_service: Optional[str]
    log_pointers: dict[str, int]
    logs_seen: dict[str, bool]
    metrics_seen: dict[str, bool]
    discovered_signals_count: int
    steps_taken: int
    done: bool
    difficulty: Optional[str] = None
    services_visited: Set[str] = set()
    max_possible_signals: int = 0
    seen_signals: Set[str] = set()
    last_action: Optional[tuple[Optional[str], Optional[str]]] = (None, None)
    action_history: dict[str, int] = {}
