from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal, TypedDict
from xml.etree import ElementTree

import requests
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda

try:
    from cachetools import TTLCache
except ImportError:
    class TTLCache(dict):  # type: ignore[no-redef]
        def __init__(self, maxsize: int, ttl: int):
            super().__init__()

try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
except ImportError:
    def retry(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        def wrapper(func: Any) -> Any:
            return func

        return wrapper

    def retry_if_exception_type(_types: Any) -> Any:  # type: ignore[no-redef]
        return None

    def stop_after_attempt(_attempts: int) -> Any:  # type: ignore[no-redef]
        return None

    def wait_exponential(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        return None

from app.schemas.searchSchema import PlannerOutput, SearchRequest, SearchResponse

load_dotenv()

EUROSTAT_BASE_URL = os.getenv(
    "EUROSTAT_BASE_URL",
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data",
)
EUROSTAT_SDMX_BASE_URL = os.getenv(
    "EUROSTAT_SDMX_BASE_URL",
    "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1",
)
DEFAULT_DATASET = os.getenv("EUROSTAT_DATASET", "ilc_lvho02")
DEFAULT_TIMEOUT = int(os.getenv("EUROSTAT_TIMEOUT_SECONDS", "30"))

KB_DIR = Path(__file__).resolve().parent.parent / "kb"
CATALOG_CACHE: TTLCache[str, list[tuple[str, str]]] = TTLCache(maxsize=1, ttl=6 * 60 * 60)


class AgentState(TypedDict):
    request: SearchRequest
    planner: PlannerOutput
    endpoint: str
    params: dict[str, str | list[str]]
    payload: dict[str, Any]


def _load_kb_json(file_name: str) -> dict[str, Any]:
    file_path = KB_DIR / file_name
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid KB format: {file_name}")
    return payload


def _dataset_map() -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in _load_kb_json("dataset_map.json").items()}


def _synonyms_map() -> dict[str, str]:
    return {str(k).lower(): str(v).lower() for k, v in _load_kb_json("synonyms.json").items()}


def _country_codes() -> dict[str, str]:
    return {str(k).lower(): str(v).upper() for k, v in _load_kb_json("country_codes.json").items()}


def _normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_]+", text.lower()))


def _detect_indicator(query: str) -> str | None:
    normalized = _normalize_text(query)
    matches: list[tuple[int, str]] = []
    for phrase, indicator in _synonyms_map().items():
        if phrase in normalized:
            matches.append((len(phrase), indicator))
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


def _parse_filters(query: str) -> dict[str, str | list[str]]:
    q = query.lower()
    filters: dict[str, str | list[str]] = {}

    year_match = re.search(r"\b(19|20)\d{2}\b", q)
    if year_match:
        filters["time"] = year_match.group(0)

    for alias, geo in _country_codes().items():
        if re.search(rf"\b{re.escape(alias)}\b", q):
            filters["geo"] = geo
            break

    for key, value in re.findall(r"\b([a-z_]+)\s*=\s*([A-Za-z0-9_]+)\b", query):
        filters[key] = value

    geo_code_match = re.search(r"\b([A-Z]{2})\b", query)
    if geo_code_match and "geo" not in filters:
        filters["geo"] = geo_code_match.group(1)

    return filters


def _dataflow_catalog() -> list[tuple[str, str]]:
    cached = CATALOG_CACHE.get("all")
    if cached is not None:
        return cached

    url = f"{EUROSTAT_SDMX_BASE_URL.rstrip('/')}/dataflow/ESTAT/all/latest"
    response = requests.get(url, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()

    root = ElementTree.fromstring(response.text)
    ns = {
        "structure": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
        "common": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common",
        "xml": "http://www.w3.org/XML/1998/namespace",
    }
    flows: list[tuple[str, str]] = []
    for node in root.findall(".//structure:Dataflow", ns):
        dataset = node.attrib.get("id", "").lower()
        if not dataset:
            continue
        name_node = node.find("./common:Name[@xml:lang='en']", ns)
        label = (name_node.text or "").strip().lower() if name_node is not None else ""
        flows.append((dataset, label))

    CATALOG_CACHE["all"] = flows
    return flows


def _discover_dataset_from_catalog(query: str) -> str | None:
    tokens = [t for t in re.findall(r"[a-z0-9_]+", query.lower()) if len(t) > 2]
    if not tokens:
        return None
    try:
        flows = _dataflow_catalog()
    except (requests.RequestException, ElementTree.ParseError):
        return None

    best_dataset: str | None = None
    best_score = 0
    for dataset, label in flows:
        haystack = f"{dataset} {label}"
        score = sum(1 for token in tokens if token in haystack)
        if score > best_score:
            best_score = score
            best_dataset = dataset
    return best_dataset if best_score > 0 else None


class PlannerAgent:
    def invoke(self, state: dict[str, Any]) -> AgentState:
        request: SearchRequest = state["request"]
        merged_filters = _parse_filters(request.query)

        indicator = _detect_indicator(request.query)
        dataset: str
        source: Literal["kb", "catalog", "default"]

        if indicator and indicator in _dataset_map():
            dataset = _dataset_map()[indicator]
            source = "kb"
        else:
            discovered = _discover_dataset_from_catalog(request.query)
            if discovered:
                dataset = discovered
                source = "catalog"
            else:
                dataset = DEFAULT_DATASET
                source = "default"

        planner = PlannerOutput(
            indicator=indicator,
            dataset=dataset,
            filters=merged_filters,
            source=source,
        )
        return {"request": request, "planner": planner}


class EurostatRetrieverAgent:
    @retry(
        retry=retry_if_exception_type((requests.RequestException, ValueError)),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _fetch_payload(self, dataset: str, filters: dict[str, str | list[str]]) -> tuple[str, dict[str, str | list[str]], dict[str, Any]]:
        endpoint = f"{EUROSTAT_BASE_URL.rstrip('/')}/{dataset}"
        params: dict[str, str | list[str]] = {"lang": "en", "format": "JSON"}
        params.update(filters)

        response = requests.get(endpoint, params=params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Unexpected Eurostat payload type.")
        return endpoint, params, payload

    def invoke(self, state: AgentState) -> AgentState:
        planner = state["planner"]
        endpoint, params, payload = self._fetch_payload(planner.dataset, planner.filters)
        return {
            "request": state["request"],
            "planner": planner,
            "endpoint": endpoint,
            "params": params,
            "payload": payload,
        }


planner_agent = PlannerAgent()
retriever_agent = EurostatRetrieverAgent()
search_chain = RunnableLambda(planner_agent.invoke) | RunnableLambda(retriever_agent.invoke)


def run_search(request: SearchRequest) -> SearchResponse:
    result: AgentState = search_chain.invoke({"request": request})
    return SearchResponse(
        agent_flow="PlannerAgent -> EurostatRetrieverAgent",
        planner=result["planner"],
        endpoint=result["endpoint"],
        params=result["params"],
        payload=result["payload"],
    )
