from __future__ import annotations

from typing import Any

from app.schemas.searchSchema import PlannerOutput
from app.services import kb, openaiPlanner
from app.services.base import AgentState, OpenAIPlanningError, trace


class PlannerAgent:
    def invoke(self, state: dict[str, Any]) -> AgentState:
        request = state["request"]
        trace_id = state["trace_id"] #  this is use for track the logs ..

        trace(trace_id, "STEP 1", "Planner started for query=%r", request.query)
        trace(trace_id, "STEP 2", "KB vocabulary loaded for validation and normalization.")

        model_plan = openaiPlanner.create_plan(request.query, trace_id)

        indicator = kb.normalize_indicator(model_plan.get("indicator"))
        dataset = str(model_plan.get("dataset", "")).strip().lower() or None
        if dataset and dataset not in kb.dimensions_map():
            dataset = None

        if not indicator and dataset:
            indicator = kb.indicator_from_dataset(dataset)
            trace(trace_id, "STEP 7", "Indicator inferred from validated dataset=%s", indicator)
        elif indicator:
            trace(trace_id, "STEP 7", "Indicator normalized from OpenAI=%s", indicator)

        if not dataset and indicator and indicator in kb.dataset_map():
            dataset = kb.dataset_map()[indicator]
            trace(trace_id, "STEP 8", "Dataset resolved from OpenAI indicator=%s -> %s", indicator, dataset)
        elif dataset:
            trace(trace_id, "STEP 8", "Dataset accepted directly from OpenAI=%s", dataset)

        if not dataset:
            raise OpenAIPlanningError("OpenAI planner did not return a valid Eurostat dataset or indicator.")

        trace(trace_id, "STEP 9", "Starting filter merge from OpenAI plan only.")
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
            source="openai",
        )
        trace(trace_id, "STEP 12", "Planner resolved source=%s indicator=%s dataset=%s geo=%s time=%s filters=%s", planner.source, planner.indicator, planner.dataset, planner.geo, planner.time, planner.filters)
        return {"request": request, "trace_id": trace_id, "planner": planner}
