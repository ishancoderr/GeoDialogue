from __future__ import annotations

import logging
from typing import Any, TypedDict
from uuid import uuid4

import requests

from app.schemas.searchSchema import PlannerOutput, SearchRequest

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    request: SearchRequest
    trace_id: str
    planner: PlannerOutput
    endpoint: str
    params: dict[str, str | list[str]]
    payload: dict[str, Any]


class OpenAIPlan(TypedDict, total=False):
    indicator: str
    dataset: str
    geo: str
    time: str


class OpenAIPlanningError(ValueError):
    pass


def new_trace_id() -> str:
    return uuid4().hex[:8]


def trace(trace_id: str, step: str, message: str, *args: Any) -> None:
    logger.info("[%s] %s " + message, trace_id, step, *args)


def http_error_text(exc: requests.HTTPError) -> str:
    response = exc.response
    if response is None:
        return str(exc)
    body = response.text.strip()
    return body[:1000] if body else str(exc)
