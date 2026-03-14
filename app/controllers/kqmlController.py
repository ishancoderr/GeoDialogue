from __future__ import annotations

import json
import os
from typing import Any
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import PlainTextResponse

from app.kqml.kqml import KQMLError, KQMLMessage, dump_sexp, json_to_sexp, parse_message
from app.schemas.searchSchema import SearchRequest
from app.services.searchService import run_search

router = APIRouter(tags=["kqml"])

KQML_MAX_PAYLOAD_CHARS = int(os.getenv("KQML_MAX_PAYLOAD_CHARS", "20000"))


def _extract_query(msg: KQMLMessage) -> str:
    content = msg.slots.get(":content")
    if not isinstance(content, list) or not content:
        raise KQMLError("Missing or invalid :content.")
    head = content[0]
    if not isinstance(head, str) or head.lower() != "search":
        raise KQMLError("Unsupported :content. Expected (search ...).")

    # content form: (search :query "..." ...)
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
    # Keep a minimal summary when payload is huge.
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
    openapi_extra={
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
    },
)
async def kqml_search(request: Request, body: Annotated[str, Body(...)] ) -> PlainTextResponse:
    # `body` is present only for Swagger; we use raw bytes to support text/plain + JSON.
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

        result = run_search(SearchRequest(query=query))
        payload, truncated = _truncate_payload(result.payload)

        reply = KQMLMessage(
            performative="tell",
            slots={
                ":in-reply-to": reply_with,
                ":content": [
                    "search-result",
                    ":dataset",
                    result.planner.dataset,
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
    except KQMLError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
