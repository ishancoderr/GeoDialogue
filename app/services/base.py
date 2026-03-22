from __future__ import annotations

import logging
from typing import Any, TypedDict
from uuid import uuid4

from app.schemas.searchSchema import PlannerOutput, SearchRequest, SearchResponse

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


def build_search_chain():
    from langchain_core.runnables import RunnableLambda

    from app.services.eurostatAgent import EurostatRetrieverAgent
    from app.services.plannerAgent import PlannerAgent

    planner_agent = PlannerAgent()
    retriever_agent = EurostatRetrieverAgent()
    return RunnableLambda(planner_agent.invoke) | RunnableLambda(retriever_agent.invoke)


def run_search(request: SearchRequest) -> SearchResponse:
    trace_id = new_trace_id()
    trace(trace_id, "STEP 0", "Search flow started.")
    chain = build_search_chain()
    result: AgentState = chain.invoke({"request": request, "trace_id": trace_id})
    trace(trace_id, "STEP 15", "Search flow completed successfully.")
    return SearchResponse(
        agent_flow="PlannerAgent -> EurostatRetrieverAgent",
        planner=result["planner"],
        endpoint=result["endpoint"],
        params=result["params"],
        payload=result["payload"],
    )
