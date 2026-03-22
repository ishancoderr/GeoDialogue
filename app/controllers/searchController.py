from __future__ import annotations

import logging

import requests
from fastapi import APIRouter, HTTPException

from app.schemas.searchSchema import SearchRequest, SearchResponse
from app.services.base import OpenAIPlanningError, run_search

router = APIRouter(tags=["search"])
logger = logging.getLogger(__name__)


def _http_error_detail(exc: requests.HTTPError) -> str:
    response = exc.response
    if response is None:
        return str(exc)
    body = response.text.strip()
    if body:
        return f"Upstream HTTP {response.status_code}: {body[:500]}"
    return f"Upstream HTTP {response.status_code}: {exc}"


@router.post("/api/v1/data/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    logger.info("HTTP STEP A POST /api/v1/data/search received query=%r", request.query)
    try:
        response = run_search(request)
        logger.info(
            "HTTP STEP B POST /api/v1/data/search completed source=%s dataset=%s endpoint=%s params=%s",
            response.planner.source,
            response.planner.dataset,
            response.endpoint,
            response.params,
        )
        return response
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else 502
        if status_code < 400 or status_code > 599:
            status_code = 502
        detail = _http_error_detail(exc)
        logger.exception("POST /api/v1/data/search upstream HTTP error: %s", detail)
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except requests.RequestException as exc:
        logger.exception("POST /api/v1/data/search network error: %s", exc)
        raise HTTPException(status_code=503, detail=f"Network error: {exc}") from exc
    except OpenAIPlanningError as exc:
        logger.exception("POST /api/v1/data/search OpenAI planning error: %s", exc)
        raise HTTPException(status_code=502, detail=f"OpenAI planning error: {exc}") from exc
    except ValueError as exc:
        logger.exception("POST /api/v1/data/search payload validation error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Payload validation error: {exc}") from exc
