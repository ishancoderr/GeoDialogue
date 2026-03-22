from __future__ import annotations

import json

import requests
import langgraph.graph as langgraph

from app.services.agents.common import AgentState, OpenAIPlan, OpenAIPlanningError, http_error_text, new_trace_id, trace
from app.services.agents.openaiCommon import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_TIMEOUT, planner_snapshot_json
from app.schemas.searchSchema import SearchRequest, SearchResponse


def _prompt() -> str:
    return (
        "Return valid json only. "
        "Plan a Eurostat request from the user query. "
        "Use only the provided knowledge base. "
        "Return four string fields: indicator, dataset, geo, time. "
        "Use empty strings when unknown. "
        f"Knowledge base: {planner_snapshot_json()}"
    )


def _schema() -> dict[str, object]:
    return {
        "name": "eurostat_query_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "indicator": {"type": "string"},
                "dataset": {"type": "string"},
                "geo": {"type": "string"},
                "time": {"type": "string"},
            },
            "required": ["indicator", "dataset", "geo", "time"],
            "additionalProperties": False,
        },
    }


def _parse_plan(output_text: str) -> OpenAIPlan:
    candidate = json.loads(output_text)
    if not isinstance(candidate, dict):
        raise OpenAIPlanningError("Planner returned a non-object payload.")
    return candidate


def _post_openai(path: str, payload: dict[str, object]) -> requests.Response:
    response = requests.post(
        f"{OPENAI_BASE_URL.rstrip('/')}/{path.lstrip('/')}",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=OPENAI_TIMEOUT,
    )
    response.raise_for_status()
    return response


def _responses_payload(query: str) -> dict[str, object]:
    return {
        "model": OPENAI_MODEL,
        "instructions": _prompt(),
        "input": query,
        "text": {"format": {"type": "json_schema", **_schema()}},
        "store": False,
    }


def _chat_payload(query: str) -> dict[str, object]:
    return {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": _prompt()},
            {"role": "user", "content": query},
        ],
        "response_format": {"type": "json_object"},
    }


def create_plan(query: str, trace_id: str) -> OpenAIPlan:
    if not OPENAI_API_KEY:
        raise OpenAIPlanningError("OPENAI_API_KEY is not set.")

    trace(trace_id, "STEP 3", "LangGraph planner request started for query=%r", query)
    try:
        response = _post_openai("responses", _responses_payload(query))
        output_text = response.json().get("output_text")
        if not isinstance(output_text, str) or not output_text.strip():
            raise OpenAIPlanningError("Responses API returned empty planner output.")
        plan = _parse_plan(output_text)
        trace(trace_id, "STEP 4", "LangGraph planner succeeded via Responses API.")
        return plan
    except requests.HTTPError as exc:
        trace(trace_id, "STEP 4", "LangGraph Responses planner HTTP error, falling back to chat: %s", http_error_text(exc))
    except (requests.RequestException, ValueError, json.JSONDecodeError, OpenAIPlanningError) as exc:
        trace(trace_id, "STEP 4", "LangGraph Responses planner fallback to chat: %s", exc)

    try:
        response = _post_openai("chat/completions", _chat_payload(query))
        payload = response.json()
        choices = payload.get("choices", [])
        if not isinstance(choices, list) or not choices:
            raise OpenAIPlanningError("Chat Completions returned no choices.")
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OpenAIPlanningError("Chat Completions returned empty planner content.")
        plan = _parse_plan(content)
    except requests.HTTPError as exc:
        raise OpenAIPlanningError(f"Chat Completions request failed: {http_error_text(exc)}") from exc
    except requests.RequestException as exc:
        raise OpenAIPlanningError(f"Chat Completions network error: {exc}") from exc
    except (ValueError, json.JSONDecodeError, OpenAIPlanningError) as exc:
        raise OpenAIPlanningError(f"Chat Completions returned invalid planner output: {exc}") from exc
    trace(trace_id, "STEP 5", "LangGraph planner succeeded via Chat Completions API.")
    return plan


def _build_planner_agent():
    from app.services.agents.plannerAgent import PlannerAgent

    return PlannerAgent(
        plan_builder=create_plan,
        start_message="Planner started for query=%r",
        vocab_message="KB vocabulary loaded for validation and normalization.",
        source="openai",
    )


def build_search_graph():
    from app.services.agents.eurostatAgent import EurostatRetrieverAgent

    planner_agent = _build_planner_agent()
    retriever_agent = EurostatRetrieverAgent()
    graph = langgraph.StateGraph(AgentState)
    graph.add_node("planner", planner_agent.invoke)
    graph.add_node("retriever", retriever_agent.invoke)
    graph.add_edge(langgraph.START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", langgraph.END)
    return graph.compile()


def run_search(request: SearchRequest) -> SearchResponse:
    trace_id = new_trace_id()
    trace(trace_id, "STEP 0", "Search flow started.")
    graph = build_search_graph()
    result: AgentState = graph.invoke({"request": request, "trace_id": trace_id})
    trace(trace_id, "STEP 15", "Search flow completed successfully.")
    return SearchResponse(
        agent_flow="LangGraph: PlannerAgent -> EurostatRetrieverAgent",
        planner=result["planner"],
        endpoint=result["endpoint"],
        params=result["params"],
        payload=result["payload"],
    )
