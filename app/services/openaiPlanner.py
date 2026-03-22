from __future__ import annotations

import json
import logging
import os

import requests
from dotenv import load_dotenv

from app.services import kb
from app.services.base import OpenAIPlan, OpenAIPlanningError, trace

load_dotenv()

logger = logging.getLogger(__name__)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT_SECONDS", os.getenv("EUROSTAT_TIMEOUT_SECONDS", "30")))


def _http_error_text(exc: requests.HTTPError) -> str:
    response = exc.response
    if response is None:
        return str(exc)
    body = response.text.strip()
    return body[:1000] if body else str(exc)


def _planner_snapshot() -> dict[str, object]:
    endpoint_terms = kb.endpoint_terms()
    return {
        "dataset_map": kb.dataset_map(),
        "country_codes": kb.country_codes(),
        "endpoint_terms": {
            "common_filters": endpoint_terms.get("common_filters", {}),
            "endpoint_examples": endpoint_terms.get("endpoint_examples", {}),
        },
    }


def _prompt() -> str:
    snapshot_json = json.dumps(_planner_snapshot(), ensure_ascii=True, separators=(",", ":"))
    return (
        "Return valid json only. "
        "Plan a Eurostat request from the user query. "
        "Use only the provided knowledge base. "
        "Return four string fields: indicator, dataset, geo, time. "
        "Use empty strings when unknown. "
        f"Knowledge base: {snapshot_json}"
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


def _responses_request(query: str) -> OpenAIPlan:
    response = requests.post(
        f"{OPENAI_BASE_URL.rstrip('/')}/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_MODEL,
            "instructions": _prompt(),
            "input": query,
            "text": {"format": {"type": "json_schema", **_schema()}},
            "store": False,
        },
        timeout=OPENAI_TIMEOUT,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        logger.warning("[responses] OpenAI HTTP %s body=%s", response.status_code, _http_error_text(exc))
        raise
    payload = response.json()
    output_text = payload.get("output_text")
    if not isinstance(output_text, str) or not output_text.strip():
        raise OpenAIPlanningError("Responses API returned empty planner output.")
    candidate = json.loads(output_text)
    if not isinstance(candidate, dict):
        raise OpenAIPlanningError("Responses API returned a non-object planner payload.")
    return candidate


def _chat_request(query: str) -> OpenAIPlan:
    response = requests.post(
        f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": _prompt()},
                {"role": "user", "content": query},
            ],
            "response_format": {"type": "json_object"},
        },
        timeout=OPENAI_TIMEOUT,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        logger.warning("[chat.completions] OpenAI HTTP %s body=%s", response.status_code, _http_error_text(exc))
        raise
    payload = response.json()
    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise OpenAIPlanningError("Chat Completions returned no choices.")
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise OpenAIPlanningError("Chat Completions returned empty content.")
    candidate = json.loads(content)
    if not isinstance(candidate, dict):
        raise OpenAIPlanningError("Chat Completions returned a non-object planner payload.")
    return candidate


def create_plan(query: str, trace_id: str) -> OpenAIPlan:
    if not OPENAI_API_KEY:
        raise OpenAIPlanningError("OPENAI_API_KEY is not set.")

    trace(trace_id, "STEP 3", "OpenAI planner request started for query=%r", query)
    try:
        plan = _responses_request(query)
        trace(trace_id, "STEP 4", "OpenAI planner succeeded via Responses API.")
        return plan
    except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
        logger.warning("[%s] STEP 4 OpenAI Responses planner failed, falling back to Chat Completions: %s", trace_id, exc)
        try:
            plan = _chat_request(query)
        except requests.HTTPError as chat_exc:
            raise OpenAIPlanningError(f"Chat Completions request failed: {_http_error_text(chat_exc)}") from chat_exc
        except requests.RequestException as chat_exc:
            raise OpenAIPlanningError(f"Chat Completions network error: {chat_exc}") from chat_exc
        except (ValueError, json.JSONDecodeError) as chat_exc:
            raise OpenAIPlanningError(f"Chat Completions returned invalid JSON output: {chat_exc}") from chat_exc
        trace(trace_id, "STEP 5", "OpenAI planner succeeded via Chat Completions API.")
        return plan
