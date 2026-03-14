from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., description='Example: "German population in 2022"')


class PlannerOutput(BaseModel):
    indicator: str | None = None
    dataset: str
    filters: dict[str, str | list[str]]
    source: Literal["kb", "catalog", "default"]


class SearchResponse(BaseModel):
    agent_flow: str
    planner: PlannerOutput
    endpoint: str
    params: dict[str, str | list[str]]
    payload: dict[str, Any]

