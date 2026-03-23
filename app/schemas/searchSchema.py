from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., description='Example: "German population in 2022"')


class PlannerOutput(BaseModel):
    indicator: str | None = None
    dataset: str
    geo: str | None = None
    time: str | None = None
    filters: dict[str, str | list[str]] = Field(default_factory=dict)
    source: Literal["openai", "kb", "catalog", "default"]


class MissingnessReport(BaseModel):
    agent: str
    source: str
    partition_available: bool = False
    feature_count: int = 0
    matched_feature_count: int = 0
    missing_types: list[str] = Field(default_factory=list)
    joinable: bool = False
    join_key: str | None = None
    recoverable: bool = False
    confidence: float = 0.0
    note: str = ""


class MissingnessDecision(BaseModel):
    status: str
    recommended_source: str | None = None
    joinable: bool = False
    join_key: str | None = None
    missing_types: list[str] = Field(default_factory=list)
    reports: list[MissingnessReport] = Field(default_factory=list)
    note: str = ""


class SearchResponse(BaseModel):
    agent_flow: str
    planner: PlannerOutput
    endpoint: str
    params: dict[str, str | list[str]]
    payload: dict[str, Any]
    missingness: MissingnessDecision | None = None
