from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse

from app.kqml.kqml import KQMLError, KQMLMessage, json_to_sexp, parse_message
from app.schemas.searchSchema import SearchRequest
from app.services.agents.common import OpenAIPlanningError
from app.services.kqmlbase.base import run_kqml_search

router = APIRouter(tags=["kqml"])
logger = logging.getLogger(__name__)

KQML_MAX_PAYLOAD_CHARS = int(os.getenv("KQML_MAX_PAYLOAD_CHARS", "20000"))
KQML_SEARCH_DOCS_SCHEMA = {
    "requestBody": {
        "required": True,
        "content": {
            "text/plain": {
                "schema": {"type": "string"},
                "examples": {
                    "population_2022": {
                        "summary": "Population example",
                        "value": '(ask-one :content (search :query "German population in 2022") :reply-with "r1")',
                    }
                },
            },
            "application/json": {
                "schema": {"type": "object"},
                "examples": {
                    "json_fallback": {
                        "summary": "JSON fallback (converted to KQML)",
                        "value": {"query": "German population in 2022"},
                    }
                },
            },
        },
    }
}

def _http_error_detail(exc: requests.HTTPError) -> str:
    response = exc.response
    if response is None:
        return str(exc)
    body = response.text.strip()
    if body:
        return f"Upstream HTTP {response.status_code}: {body[:500]}"
    return f"Upstream HTTP {response.status_code}: {exc}"


def _extract_query(msg: KQMLMessage) -> str:
    content = msg.slots.get(":content")
    if not isinstance(content, list) or not content:
        raise KQMLError("Missing or invalid :content.")
    head = content[0]
    if not isinstance(head, str) or head.lower() != "search":
        raise KQMLError("Unsupported :content. Expected (search ...).")

    items = content[1:]
    if len(items) % 2 != 0:
        raise KQMLError("Content slots must be key/value pairs.")

    for i in range(0, len(items), 2):
        k = items[i]
        v = items[i + 1]
        if isinstance(k, str) and k.lower() == ":query" and isinstance(v, str) and v.strip():
            return v.strip()
    raise KQMLError("Missing :query in (search ...) content.")


def _truncate_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    if len(raw) <= KQML_MAX_PAYLOAD_CHARS:
        return payload, False
    summary = {
        "truncated": True,
        "max_chars": KQML_MAX_PAYLOAD_CHARS,
        "keys": list(payload.keys()),
    }
    return summary, True


@router.post(
    "/api/v1/kqml/search",
    response_class=PlainTextResponse,
    responses={
        200: {
            "description": "KQML reply",
            "content": {"text/plain": {"schema": {"type": "string"}}},
        }
    },
    openapi_extra=KQML_SEARCH_DOCS_SCHEMA,
)
async def kqml_search(request: Request) -> PlainTextResponse:
    logger.info("HTTP STEP A POST /api/v1/kqml/search received content_type=%s", request.headers.get("content-type"))
    try:
        raw = await request.body()
        content_type = (request.headers.get("content-type") or "").lower()

        query: str
        reply_with: Any = "nil"

        if "application/json" in content_type or raw.lstrip().startswith(b"{"):
            try:
                obj = json.loads(raw.decode("utf-8", errors="strict"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
            if not isinstance(obj, dict) or not isinstance(obj.get("query"), str) or not obj["query"].strip():
                raise HTTPException(status_code=400, detail='JSON must be {"query": "<non-empty string>"}.')
            query = obj["query"].strip()
        else:
            try:
                text = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid UTF-8 body: {exc}") from exc
            msg = parse_message(text)
            query = _extract_query(msg)
            reply_with = msg.slots.get(":reply-with", "nil")

        logger.info("HTTP STEP B POST /api/v1/kqml/search parsed query=%r reply_with=%r", query, reply_with)
        result = run_kqml_search(SearchRequest(query=query))
        payload, truncated = _truncate_payload(result.payload)

        logger.info(
            "HTTP STEP C POST /api/v1/kqml/search completed source=%s dataset=%s endpoint=%s truncated=%s",
            result.planner.source,
            result.planner.dataset,
            result.endpoint,
            truncated,
        )

        reply = KQMLMessage(
            performative="tell",
            slots={
                ":in-reply-to": reply_with,
                ":content": [
                    "search-result",
                    ":indicator",
                    result.planner.indicator or "nil",
                    ":dataset",
                    result.planner.dataset,
                    ":geo",
                    result.planner.geo or "nil",
                    ":time",
                    result.planner.time or "nil",
                    ":filters",
                    json_to_sexp(result.planner.filters),
                    ":endpoint",
                    result.endpoint,
                    ":params",
                    json_to_sexp(result.params),
                    ":payload",
                    json_to_sexp(payload),
                    ":truncated",
                    "t" if truncated else "f",
                ],
            },
        )
        return PlainTextResponse(content=reply.dump())
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else 502
        if status_code < 400 or status_code > 599:
            status_code = 502
        detail = _http_error_detail(exc)
        logger.exception("POST /api/v1/kqml/search upstream HTTP error: %s", detail)
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except requests.RequestException as exc:
        logger.exception("POST /api/v1/kqml/search network error: %s", exc)
        raise HTTPException(status_code=503, detail=f"Network error: {exc}") from exc
    except OpenAIPlanningError as exc:
        logger.exception("POST /api/v1/kqml/search OpenAI planning error: %s", exc)
        raise HTTPException(status_code=502, detail=f"OpenAI planning error: {exc}") from exc
    except ValueError as exc:
        logger.exception("POST /api/v1/kqml/search payload validation error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Payload validation error: {exc}") from exc
    except KQMLError as exc:
        logger.exception("POST /api/v1/kqml/search KQML parse error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
