from __future__ import annotations

import requests

from app.kqml.kqml import KQMLError, Sexp, parse_sexp
from app.services.agents.agent1 import (
    Agent1,
    AgentState,
    OpenAIPlan,
    OpenAIPlanningError,
    PlannerAgent,
    build_api_missingness_report,
    build_missingness_decision,
    http_error_text,
    load_default_partitions,
    new_trace_id,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENAI_TIMEOUT,
    planner_snapshot,
    trace,
)
from app.services.agents.agent2 import Agent2
from app.schemas.searchSchema import SearchRequest, SearchResponse


def _prompt() -> str:
    return (
        "You are a KQML planning agent for Eurostat search. "
        "Reply with exactly one KQML message and nothing else. "
        "Use the performative tell. "
        "Return exactly this outer shape: "
        '(tell :content (search-plan :indicator "..." :dataset "..." :geo "..." :time "...")). '
        "If the user asks about missing data but does not name an indicator, "
        'use indicator "housing_deprivation" and dataset "ilc_lvho02". '
        "Use empty strings when a field is unknown. "
        f"Use only this knowledge base: {planner_snapshot()}"
    )


def _extract_first_list(text: str) -> str:
    start = text.find("(")
    if start < 0:
        return text.strip()

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:].strip()


def _content_from_expr(expr: Sexp) -> list[Sexp] | None:
    if not isinstance(expr, list) or not expr:
        return None

    head = expr[0]
    if isinstance(head, str) and head.lower() == "search-plan":
        return expr

    if isinstance(head, str) and head.lower() == "tell":
        rest = expr[1:]
        for i in range(0, len(rest) - 1, 2):
            key = rest[i]
            value = rest[i + 1]
            if isinstance(key, str) and key.lower() == ":content" and isinstance(value, list):
                return value
        for item in rest:
            if isinstance(item, list) and item and isinstance(item[0], str) and item[0].lower() == "search-plan":
                return item
            nested = _content_from_expr(item)
            if nested is not None:
                return nested
    return None


def _parse_plan(output_text: str) -> OpenAIPlan:
    normalized = _extract_first_list(output_text.strip())
    try:
        expr = parse_sexp(normalized)
    except KQMLError as exc:
        raise OpenAIPlanningError(f"KQML planner returned invalid KQML: {exc}") from exc

    content = _content_from_expr(expr)
    if not isinstance(content, list) or not content:
        raise OpenAIPlanningError(
            f"KQML planner returned no search-plan payload. Raw output: {output_text[:500]}"
        )

    head = content[0]
    if not isinstance(head, str) or head.lower() != "search-plan":
        raise OpenAIPlanningError("KQML planner returned unsupported content. Expected (search-plan ...).")

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
        trace(trace_id, "STEP 5", "KQML Chat Completions raw content=%r", content[:500])
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
    return PlannerAgent(
        plan_builder=create_plan,
        start_message="KQML planner started for query=%r",
        vocab_message="KB vocabulary loaded for KQML validation and normalization.",
        source="openai",
    )


def run_kqml_search(request: SearchRequest) -> SearchResponse:
    trace_id = new_trace_id()
    trace(trace_id, "STEP 0", "KQML search flow started.")

    planner_agent = _build_planner_agent()
    planner_state = planner_agent.invoke({"request": request, "trace_id": trace_id})

    agent1 = Agent1()
    agent2 = Agent2()
    geojson1, geojson2 = load_default_partitions()

    local_report = agent1.build_local_report(planner_state, geojson1, geojson2)
    trace(trace_id, "STEP 13", "Agent1 local missingness report=%s", local_report.model_dump())

    request_message = agent1.build_request_message(request, planner_state, local_report)
    trace(trace_id, "STEP 14", "Agent1 -> Agent2 KQML message=%s", request_message)

    reply_message, api_state = agent2.reply_to_missingness_request(request_message, planner_state, geojson2)
    trace(trace_id, "STEP 15", "Agent2 -> Agent1 KQML message=%s", reply_message)

    remote_report = agent1.report_from_reply(reply_message)
    reports = [local_report, remote_report]
    if api_state is not None:
        reports.append(build_api_missingness_report("agent2", success=True, note="Eurostat returned data for the requested slice."))
        result_state = api_state
    else:
        reports.append(build_api_missingness_report("agent2", success=False, note="Eurostat data was not available."))
        result_state = {
            "request": request,
            "trace_id": trace_id,
            "planner": planner_state["planner"],
            "endpoint": "",
            "params": {},
            "payload": {},
        }

    decision = build_missingness_decision(reports)
    trace(trace_id, "STEP 16", "Final missingness decision=%s", decision.model_dump())
    trace(trace_id, "STEP 17", "KQML search flow completed successfully.")

    return SearchResponse(
        agent_flow="KQML Agent1 -> Agent2 -> EurostatRetrieverAgent",
        planner=result_state["planner"],
        endpoint=result_state["endpoint"],
        params=result_state["params"],
        payload=result_state["payload"],
        missingness=decision,
    )
