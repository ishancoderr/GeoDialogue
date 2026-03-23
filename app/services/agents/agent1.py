from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, TypedDict
from uuid import uuid4

import requests
from dotenv import load_dotenv

from app.kqml.kqml import KQMLMessage, parse_message
from app.schemas.searchSchema import MissingnessDecision, MissingnessReport, PlannerOutput, SearchRequest
from app.services import kb

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT_SECONDS", os.getenv("EUROSTAT_TIMEOUT_SECONDS", "30")))
MISSINGNESS_GEOJSON1_PATH = os.getenv("MISSINGNESS_GEOJSON1_PATH", "").strip()
MISSINGNESS_GEOJSON2_PATH = os.getenv("MISSINGNESS_GEOJSON2_PATH", "").strip()
_JOIN_KEY_CANDIDATES = ["geo_code", "nuts_id", "id", "name", "geo", "country", "country_code"]
_GEO_PROPERTY_CANDIDATES = ["geo", "geo_code", "country", "country_code", "name", "admin", "admin_name", "nuts_id"]
_TIME_PROPERTY_CANDIDATES = ["time", "year", "date", "period"]


class AgentState(TypedDict, total=False):
    request: SearchRequest
    trace_id: str
    planner: PlannerOutput
    endpoint: str
    params: dict[str, str | list[str]]
    payload: dict[str, Any]
    missingness: MissingnessDecision


class OpenAIPlan(TypedDict, total=False):
    indicator: str
    dataset: str
    geo: str
    time: str


class OpenAIPlanningError(ValueError):
    pass


def planner_snapshot() -> dict[str, object]:
    endpoint_terms = kb.endpoint_terms()
    return {
        "dataset_map": kb.dataset_map(),
        "country_codes": kb.country_codes(),
        "endpoint_terms": {
            "common_filters": endpoint_terms.get("common_filters", {}),
            "endpoint_examples": endpoint_terms.get("endpoint_examples", {}),
        },
    }


def planner_snapshot_json() -> str:
    return json.dumps(planner_snapshot(), ensure_ascii=True, separators=(",", ":"))


def new_trace_id() -> str:
    return uuid4().hex[:8]


def trace(trace_id: str, step: str, message: str, *args: Any) -> None:
    logger.info("[%s] %s " + message, trace_id, step, *args)


def http_error_text(exc: requests.HTTPError) -> str:
    response = exc.response
    if response is None:
        return str(exc)
    body = response.text.strip()
    return body[:1000] if body else str(exc)


def load_geojson_partition(path_text: str) -> dict[str, Any] | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_default_partitions() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    return load_geojson_partition(MISSINGNESS_GEOJSON1_PATH), load_geojson_partition(MISSINGNESS_GEOJSON2_PATH)


def _feature_list(geojson: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(geojson, dict):
        return []
    features = geojson.get("features")
    if not isinstance(features, list):
        return []
    return [feature for feature in features if isinstance(feature, dict)]


def _properties(feature: dict[str, Any]) -> dict[str, Any]:
    props = feature.get("properties")
    return props if isinstance(props, dict) else {}


def _normalized_strings(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        out: set[str] = set()
        for item in value:
            out.update(_normalized_strings(item))
        return out
    text = str(value).strip().lower()
    return {text} if text else set()


def _match_geo(properties: dict[str, Any], geo: str | None) -> bool:
    if not geo:
        return True
    wanted = geo.strip().lower()
    for key in _GEO_PROPERTY_CANDIDATES:
        if wanted in _normalized_strings(properties.get(key)):
            return True
    return False


def _match_time(properties: dict[str, Any], time_value: str | None) -> bool:
    if not time_value:
        return True
    wanted = time_value.strip().lower()
    for key in _TIME_PROPERTY_CANDIDATES:
        if wanted in _normalized_strings(properties.get(key)):
            return True
    return False


def _indicator_keys(indicator: str | None) -> set[str]:
    keys = {"population"}
    if indicator:
        keys.add(indicator.strip().lower())
    return keys


def _has_indicator(properties: dict[str, Any], indicator: str | None) -> bool:
    for key in _indicator_keys(indicator):
        if key in properties and properties[key] not in (None, "", []):
            return True
    return False


def _join_key(left_keys: set[str], right_keys: set[str]) -> str | None:
    overlap = left_keys & right_keys
    for key in _JOIN_KEY_CANDIDATES:
        if key in overlap:
            return key
    return next(iter(sorted(overlap)), None)


def build_partition_missingness_report(
    agent_name: str,
    source_name: str,
    geojson: dict[str, Any] | None,
    planner: PlannerOutput,
    peer_geojson: dict[str, Any] | None = None,
) -> MissingnessReport:
    features = _feature_list(geojson)
    if not features:
        return MissingnessReport(
            agent=agent_name,
            source=source_name,
            note=f"{source_name} is unavailable or has no features.",
            missing_types=["spatial_missingness"],
        )

    geo_features = [feature for feature in features if _match_geo(_properties(feature), planner.geo)]
    matched_features = [feature for feature in geo_features if _match_time(_properties(feature), planner.time)]

    property_keys = {key.lower() for feature in features for key in _properties(feature).keys()}
    peer_keys = {key.lower() for feature in _feature_list(peer_geojson) for key in _properties(feature).keys()}
    join_key = _join_key(property_keys, peer_keys) if peer_keys else None
    has_indicator = any(_has_indicator(_properties(feature), planner.indicator) for feature in matched_features)

    missing_types: list[str] = []
    if planner.geo and not geo_features:
        missing_types.append("spatial_missingness")
    if planner.time and geo_features and not matched_features:
        missing_types.append("temporal_missingness")
    if matched_features and not has_indicator:
        missing_types.append("attribute_missingness")
    if missing_types and join_key:
        missing_types.append("joinability_missingness")

    if not missing_types:
        note = f"{source_name} already contains the requested slice."
    elif join_key:
        note = f"{source_name} is missing the requested value but can be linked with key '{join_key}'."
    else:
        note = f"{source_name} is missing the requested value and no join key was detected."

    can_join = bool(join_key and missing_types)
    confidence = 0.9 if not missing_types else 0.75 if can_join else 0.55
    return MissingnessReport(
        agent=agent_name,
        source=source_name,
        partition_available=True,
        feature_count=len(features),
        matched_feature_count=len(matched_features),
        missing_types=missing_types,
        joinable=can_join,
        join_key=join_key if can_join else None,
        recoverable=can_join,
        confidence=confidence,
        note=note,
    )


def build_api_missingness_report(agent_name: str, success: bool, note: str) -> MissingnessReport:
    return MissingnessReport(
        agent=agent_name,
        source="eurostat",
        partition_available=success,
        recoverable=success,
        confidence=0.95 if success else 0.2,
        note=note,
    )


def build_missingness_decision(reports: list[MissingnessReport]) -> MissingnessDecision:
    missing_types = sorted({missing for report in reports for missing in report.missing_types})
    best_joinable = next((report for report in reports if report.joinable and report.join_key), None)
    eurostat_report = next((report for report in reports if report.source == "eurostat" and report.recoverable), None)

    if not missing_types:
        status = "complete"
        recommended_source = None
        joinable = False
        join_key = None
        note = "No missingness was detected for the requested slice."
    elif eurostat_report is not None:
        status = "recoverable"
        recommended_source = "eurostat"
        joinable = True
        join_key = best_joinable.join_key if best_joinable is not None else "geo"
        note = "Agent2 can recover the missing data from Eurostat."
    elif best_joinable is not None:
        status = "joinable"
        recommended_source = best_joinable.source
        joinable = True
        join_key = best_joinable.join_key
        note = f"Agent1 can recover the missing data by linking with {best_joinable.source}."
    else:
        status = "missing_not_joinable"
        recommended_source = None
        joinable = False
        join_key = None
        note = "Missing data was detected, but Agent1 could not find a joinable recovery source."

    return MissingnessDecision(
        status=status,
        recommended_source=recommended_source,
        joinable=joinable,
        join_key=join_key,
        missing_types=missing_types,
        reports=reports,
        note=note,
    )


class PlannerAgent:
    def __init__(
        self,
        plan_builder: Callable[[str, str], OpenAIPlan],
        start_message: str,
        vocab_message: str,
        source: str = "openai",
    ) -> None:
        self._plan_builder = plan_builder
        self._start_message = start_message
        self._vocab_message = vocab_message
        self._source = source

    def _resolve_planner_output(self, model_plan: OpenAIPlan, trace_id: str) -> PlannerOutput:
        indicator = kb.normalize_indicator(model_plan.get("indicator"))
        dataset = str(model_plan.get("dataset", "")).strip().lower() or None
        if dataset and dataset not in kb.dimensions_map():
            dataset = None

        if not indicator and dataset:
            indicator = kb.indicator_from_dataset(dataset)
            trace(trace_id, "STEP 7", "Indicator inferred from validated dataset=%s", indicator)
        elif indicator:
            trace(trace_id, "STEP 7", "Indicator normalized from planner=%s", indicator)

        if not dataset and indicator and indicator in kb.dataset_map():
            dataset = kb.dataset_map()[indicator]
            trace(trace_id, "STEP 8", "Dataset resolved from planner indicator=%s -> %s", indicator, dataset)
        elif dataset:
            trace(trace_id, "STEP 8", "Dataset accepted directly from planner=%s", dataset)

        if not dataset:
            raise OpenAIPlanningError("Planner did not return a valid Eurostat dataset or indicator.")

        trace(trace_id, "STEP 9", "Starting filter merge from planner output only.")
        filters: dict[str, str | list[str]] = {}
        geo = kb.normalize_geo(model_plan.get("geo"))
        time_value = kb.normalize_time(model_plan.get("time"))
        if geo:
            filters["geo"] = geo
        if time_value:
            filters["time"] = time_value

        filters = kb.sanitize_filters(dataset, filters)
        trace(trace_id, "STEP 10", "Filters after sanitization=%s", filters)
        filters = kb.apply_dimension_defaults(dataset, filters)
        trace(trace_id, "STEP 11", "Filters after dataset defaults=%s", filters)

        planner = PlannerOutput(
            indicator=indicator,
            dataset=dataset,
            geo=filters.get("geo") if isinstance(filters.get("geo"), str) else None,
            time=filters.get("time") if isinstance(filters.get("time"), str) else None,
            filters=filters,
            source=self._source,
        )
        trace(
            trace_id,
            "STEP 12",
            "Planner resolved source=%s indicator=%s dataset=%s geo=%s time=%s filters=%s",
            planner.source,
            planner.indicator,
            planner.dataset,
            planner.geo,
            planner.time,
            planner.filters,
        )
        return planner

    def invoke(self, state: dict[str, Any]) -> AgentState:
        request = state["request"]
        trace_id = state["trace_id"]

        trace(trace_id, "STEP 1", self._start_message, request.query)
        trace(trace_id, "STEP 2", self._vocab_message)

        model_plan = self._plan_builder(request.query, trace_id)
        planner = self._resolve_planner_output(model_plan, trace_id)
        return {"request": request, "trace_id": trace_id, "planner": planner}


class Agent1:
    def build_local_report(
        self,
        planner_state: AgentState,
        geojson1: dict[str, object] | None,
        geojson2: dict[str, object] | None,
    ) -> MissingnessReport:
        return build_partition_missingness_report(
            agent_name="agent1",
            source_name="europe_housing_geojson1",
            geojson=geojson1,
            planner=planner_state["planner"],
            peer_geojson=geojson2,
        )

    def build_request_message(
        self,
        request: SearchRequest,
        planner_state: AgentState,
        local_report: MissingnessReport,
    ) -> str:
        planner = planner_state["planner"]
        message = KQMLMessage(
            performative="ask-one",
            slots={
                ":sender": "agent1",
                ":receiver": "agent2",
                ":reply-with": "missingness-r1",
                ":content": [
                    "find-missing-data",
                    ":query",
                    request.query,
                    ":indicator",
                    planner.indicator or "",
                    ":dataset",
                    planner.dataset,
                    ":geo",
                    planner.geo or "",
                    ":time",
                    planner.time or "",
                    ":agent1-missing-types",
                    local_report.missing_types,
                ],
            },
        )
        return message.dump()

    def report_from_reply(self, reply_text: str) -> MissingnessReport:
        parsed = parse_message(reply_text)
        content = parsed.slots.get(":content")
        if not isinstance(content, list) or not content or str(content[0]).lower() != "missing-data-result":
            raise OpenAIPlanningError("Agent1 received an unsupported KQML missingness reply.")

        def content_value(items: list[object], key: str) -> object | None:
            key_lower = key.lower()
            for i in range(0, len(items) - 1, 2):
                candidate = items[i]
                if isinstance(candidate, str) and candidate.lower() == key_lower:
                    return items[i + 1]
            return None

        items = content[1:]
        missing_types = content_value(items, ":missing-types")
        joinable = content_value(items, ":joinable")
        join_key = content_value(items, ":join-key")
        recoverable = content_value(items, ":recoverable")
        confidence = content_value(items, ":confidence")
        note = content_value(items, ":note")
        source = content_value(items, ":source")
        return MissingnessReport(
            agent="agent2",
            source=str(source) if source is not None else "europe_housing_geojson2",
            partition_available=True,
            feature_count=0,
            matched_feature_count=0,
            missing_types=missing_types if isinstance(missing_types, list) else [],
            joinable=bool(joinable),
            join_key=str(join_key).strip() or None if join_key is not None else None,
            recoverable=bool(recoverable),
            confidence=float(confidence) if isinstance(confidence, (int, float)) else 0.0,
            note=str(note) if note is not None else "",
        )
