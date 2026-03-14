from __future__ import annotations

import requests
from fastapi import APIRouter, HTTPException

from app.schemas.searchSchema import SearchRequest, SearchResponse
from app.services.searchService import run_search

router = APIRouter(tags=["search"])


@router.post("/api/v1/data/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    try:
        return run_search(request)
    except requests.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Eurostat HTTP error: {exc}") from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"Eurostat network error: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Payload validation error: {exc}") from exc
