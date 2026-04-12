from __future__ import annotations
import json
import typing
from typing import Optional, Dict, Any

try:
    from openenv_core.http_env_client import HTTPEnvClient, StepResult
except ImportError:
    try:
        from core.http_env_client import HTTPEnvClient, StepResult
    except ImportError:
        # Provide a Generic stub so type hinting with [Action, Observation] succeeds
        T1 = typing.TypeVar('T1')
        T2 = typing.TypeVar('T2')
        class HTTPEnvClient(typing.Generic[T1, T2]): pass
        StepResult = Any

from .models import Action, Observation, State

class InterviewEnvClient(HTTPEnvClient[Action, Observation]):
    """
    OpenEnv compatible client for the AI Interview Simulator.
    This enables RL algorithms (e.g. TRL's GRPO) to train agents natively via Python.
    """
    def _step_payload(self, action: Action) -> dict:
        """Convert typed action to JSON for HTTP POST to /step"""
        return action.dict()
    
    def _parse_result(self, payload: dict) -> StepResult[Observation]:
        """Parse HTTP JSON response from /step into typed observation and standard OpenEnv result"""
        return StepResult(
            observation=Observation(**payload["observation"]),
            reward=payload["reward"].get("value", 0.0),
            done=payload.get("done", False)
        )
    
    def _parse_state(self, payload: dict) -> State:
        """Parse HTTP JSON response from /state into typed state object"""
        return State(**payload)
