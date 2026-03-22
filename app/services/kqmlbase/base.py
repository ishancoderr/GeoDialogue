from __future__ import annotations

import requests

from app.kqml.kqml import KQMLError, parse_message
from app.services.agents.common import AgentState, OpenAIPlan, OpenAIPlanningError, http_error_text, new_trace_id, trace
from app.services.agents.openaiCommon import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_TIMEOUT, planner_snapshot
from app.schemas.searchSchema import SearchRequest, SearchResponse


def _prompt() -> str:
    return (
        "You are a KQML planning agent for Eurostat search. "
        "Reply with exactly one KQML message and nothing else. "
        "Use the performative tell. "
        "Put the plan in :content using this shape: "
        '(search-plan :indicator "..." :dataset "..." :geo "..." :time "..."). '
        "Use empty strings when a field is unknown. "
        f"Use only this knowledge base: {planner_snapshot()}"
    )


def _parse_plan(output_text: str) -> OpenAIPlan:
    try:
        msg = parse_message(output_text)
    except KQMLError as exc:
        raise OpenAIPlanningError(f"KQML planner returned invalid KQML: {exc}") from exc

    content = msg.slots.get(":content")
    if not isinstance(content, list) or not content:
        raise OpenAIPlanningError("KQML planner returned no :content payload.")
    head = content[0]
    if not isinstance(head, str) or head.lower() != "search-plan":
        raise OpenAIPlanningError("KQML planner returned unsupported :content. Expected (search-plan ...).")

    items = content[1:]
    if len(items) % 2 != 0:
        raise OpenAIPlanningError("KQML planner content must contain key/value pairs.")

    plan: OpenAIPlan = {"indicator": "", "dataset": "", "geo": "", "time": ""}
    for i in range(0, len(items), 2):
        key = items[i]
        value = items[i + 1]
        if not isinstance(key, str):
            continue
        key_name = key.lower()
        if key_name in {":indicator", ":dataset", ":geo", ":time"}:
            plan[key_name[1:]] = value if isinstance(value, str) else ""
    return plan


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
        "store": False,
    }


def _chat_payload(query: str) -> dict[str, object]:
    return {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": _prompt()},
            {"role": "user", "content": query},
        ],
    }


def create_plan(query: str, trace_id: str) -> OpenAIPlan:
    if not OPENAI_API_KEY:
        raise OpenAIPlanningError("OPENAI_API_KEY is not set.")

    trace(trace_id, "STEP 3", "KQML planner request started for query=%r", query)
    try:
        response = _post_openai("responses", _responses_payload(query))
        output_text = response.json().get("output_text")
        if not isinstance(output_text, str) or not output_text.strip():
            raise OpenAIPlanningError("Responses API returned empty KQML planner output.")
        plan = _parse_plan(output_text)
        trace(trace_id, "STEP 4", "KQML planner succeeded via Responses API.")
        return plan
    except requests.HTTPError as exc:
        trace(trace_id, "STEP 4", "KQML Responses planner HTTP error, falling back to chat: %s", http_error_text(exc))
    except (requests.RequestException, ValueError, OpenAIPlanningError) as exc:
        trace(trace_id, "STEP 4", "KQML Responses planner fallback to chat: %s", exc)

    try:
        response = _post_openai("chat/completions", _chat_payload(query))
        payload = response.json()
        choices = payload.get("choices", [])
        if not isinstance(choices, list) or not choices:
            raise OpenAIPlanningError("Chat Completions returned no choices for KQML planner.")
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OpenAIPlanningError("Chat Completions returned empty KQML planner content.")
        plan = _parse_plan(content)
    except requests.HTTPError as exc:
        raise OpenAIPlanningError(f"KQML Chat Completions request failed: {http_error_text(exc)}") from exc
    except requests.RequestException as exc:
        raise OpenAIPlanningError(f"KQML Chat Completions network error: {exc}") from exc
    except (ValueError, OpenAIPlanningError) as exc:
        raise OpenAIPlanningError(f"KQML Chat Completions returned invalid KQML output: {exc}") from exc
    trace(trace_id, "STEP 5", "KQML planner succeeded via Chat Completions API.")
    return plan


def _build_planner_agent():
    from app.services.agents.plannerAgent import PlannerAgent

    return PlannerAgent(
        plan_builder=create_plan,
        start_message="KQML planner started for query=%r",
        vocab_message="KB vocabulary loaded for KQML validation and normalization.",
        source="openai",
    )


def run_kqml_search(request: SearchRequest) -> SearchResponse:
    from app.services.agents.eurostatAgent import EurostatRetrieverAgent

    trace_id = new_trace_id()
    trace(trace_id, "STEP 0", "KQML search flow started.")

    planner_agent = _build_planner_agent()
    retriever_agent = EurostatRetrieverAgent()

    planner_state = planner_agent.invoke({"request": request, "trace_id": trace_id})
    result: AgentState = retriever_agent.invoke(planner_state)

    trace(trace_id, "STEP 15", "KQML search flow completed successfully.")
    return SearchResponse(
        agent_flow="KQML PlannerAgent -> EurostatRetrieverAgent",
        planner=result["planner"],
        endpoint=result["endpoint"],
        params=result["params"],
        payload=result["payload"],
    )
