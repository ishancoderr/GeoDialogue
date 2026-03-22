from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

from app.services.base import AgentState, trace

try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
except ImportError:
    def retry(*args: Any, **kwargs: Any):
        def wrapper(func: Any) -> Any:
            return func
        return wrapper

    def retry_if_exception_type(_types: Any) -> Any:
        return None

    def stop_after_attempt(_attempts: int) -> Any:
        return None

    def wait_exponential(*args: Any, **kwargs: Any) -> Any:
        return None

load_dotenv()

EUROSTAT_BASE_URL = os.getenv("EUROSTAT_BASE_URL", "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data")
DEFAULT_TIMEOUT = int(os.getenv("EUROSTAT_TIMEOUT_SECONDS", "30"))


class EurostatRetrieverAgent:
    @retry(
        retry=retry_if_exception_type((requests.RequestException, ValueError)),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _fetch_payload(self, trace_id: str, dataset: str, filters: dict[str, str | list[str]]) -> tuple[str, dict[str, str | list[str]], dict[str, Any]]:
        endpoint = f"{EUROSTAT_BASE_URL.rstrip('/')}/{dataset}"
        params: dict[str, str | list[str]] = {"lang": "en", "format": "JSON"}
        params.update(filters)

        trace(trace_id, "STEP 13", "Eurostat request started dataset=%s endpoint=%s params=%s", dataset, endpoint, params)
        response = requests.get(endpoint, params=params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Unexpected Eurostat payload type.")
        trace(trace_id, "STEP 14", "Eurostat request finished status=%s payload_keys=%s", response.status_code, list(payload.keys())[:10])
        return endpoint, params, payload

    def invoke(self, state: AgentState) -> AgentState:
        planner = state["planner"]
        trace_id = state["trace_id"]
        endpoint, params, payload = self._fetch_payload(trace_id, planner.dataset, planner.filters)
        return {
            "request": state["request"],
            "trace_id": trace_id,
            "planner": planner,
            "endpoint": endpoint,
            "params": params,
            "payload": payload,
        }
