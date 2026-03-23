from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

from app.kqml.kqml import KQMLMessage, parse_message
from app.services.agents.agent1 import (
    AgentState,
    OpenAIPlanningError,
    build_api_missingness_report,
    build_partition_missingness_report,
    trace,
)

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
        result: AgentState = {
            "request": state["request"],
            "trace_id": trace_id,
            "planner": planner,
            "endpoint": endpoint,
            "params": params,
            "payload": payload,
        }
        if "missingness" in state:
            result["missingness"] = state["missingness"]
        return result


class Agent2:
    def reply_to_missingness_request(
        self,
        request_message: str,
        planner_state: AgentState,
        geojson2: dict[str, object] | None,
    ) -> tuple[str, AgentState | None]:
        retriever_agent = EurostatRetrieverAgent()

        parsed = parse_message(request_message)
        content = parsed.slots.get(":content")
        if not isinstance(content, list) or not content or str(content[0]).lower() != "find-missing-data":
            raise OpenAIPlanningError("Agent2 received an unsupported KQML missingness request.")

        remote_report = build_partition_missingness_report(
            agent_name="agent2",
            source_name="europe_housing_geojson2",
            geojson=geojson2,
            planner=planner_state["planner"],
        )

        api_state: AgentState | None = None
        try:
            api_state = retriever_agent.invoke(planner_state)
            api_report = build_api_missingness_report("agent2", success=True, note="Eurostat returned data for the requested slice.")
            remote_report.recoverable = True
            remote_report.note = f"{remote_report.note} Agent2 can also ask Eurostat for the missing value.".strip()
            status = "found"
        except requests.RequestException as exc:
            api_report = build_api_missingness_report("agent2", success=False, note=f"Eurostat lookup failed: {exc}")
            status = "missing"
        except ValueError as exc:
            api_report = build_api_missingness_report("agent2", success=False, note=f"Eurostat payload could not be parsed: {exc}")
            status = "missing"

        reply = KQMLMessage(
            performative="tell",
            slots={
                ":sender": "agent2",
                ":receiver": "agent1",
                ":in-reply-to": parsed.slots.get(":reply-with", "missingness-r1"),
                ":content": [
                    "missing-data-result",
                    ":status",
                    status,
                    ":source",
                    remote_report.source,
                    ":missing-types",
                    remote_report.missing_types,
                    ":joinable",
                    remote_report.joinable,
                    ":join-key",
                    remote_report.join_key or "",
                    ":recoverable",
                    remote_report.recoverable or api_report.recoverable,
                    ":confidence",
                    remote_report.confidence,
                    ":note",
                    f"{remote_report.note} {api_report.note}".strip(),
                ],
            },
        )
        return reply.dump(), api_state
