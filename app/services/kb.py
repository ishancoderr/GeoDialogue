from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

KB_DIR = Path(__file__).resolve().parent.parent / "kb"


def _load_json(file_name: str) -> dict[str, Any]:
    path = KB_DIR / file_name
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid KB format: {file_name}")
    return payload


def dataset_map() -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in _load_json("dataset_map.json").items()}


def country_codes() -> dict[str, str]:
    return {str(k).lower(): str(v).upper() for k, v in _load_json("country_codes.json").items()}


def dimensions_map() -> dict[str, dict[str, Any]]:
    raw = _load_json("dimensions_map.json")
    return {str(k).lower(): v for k, v in raw.items() if isinstance(v, dict)}


def endpoint_terms() -> dict[str, Any]:
    return _load_json("endpoint_terms.json")


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_]+", text.lower()))


def normalize_indicator(value: str | None) -> str | None:
    if not value:
        return None
    normalized = normalize_text(value)
    if normalized in dataset_map():
        return normalized
    return None


def normalize_geo(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip()
    if re.fullmatch(r"[A-Za-z]{2}", cleaned):
        return cleaned.upper()
    return country_codes().get(cleaned.lower())


def normalize_time(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip()
    if re.fullmatch(r"(19|20)\d{2}", cleaned):
        return cleaned
    return cleaned if cleaned else None


def indicator_from_dataset(dataset: str | None) -> str | None:
    if not dataset:
        return None
    normalized_dataset = dataset.lower()
    for indicator, mapped_dataset in dataset_map().items():
        if mapped_dataset.lower() == normalized_dataset:
            return indicator
    return None


def sanitize_filters(dataset: str | None, filters: dict[str, Any]) -> dict[str, str | list[str]]:
    allowed = {str(k).lower() for k in endpoint_terms().get("common_filters", {}).keys()}
    if dataset:
        dims = dimensions_map().get(dataset.lower(), {})
        allowed.update(str(k).lower() for k in dims.get("required", []))
        defaults = dims.get("defaults", {})
        if isinstance(defaults, dict):
            allowed.update(str(k).lower() for k in defaults.keys())

    sanitized: dict[str, str | list[str]] = {}
    for key, value in filters.items():
        normalized_key = str(key).strip().lower()
        if normalized_key not in allowed:
            continue
        if isinstance(value, str) and value.strip():
            sanitized[normalized_key] = value.strip()

    geo_value = sanitized.get("geo")
    if isinstance(geo_value, str):
        geo = normalize_geo(geo_value)
        if geo:
            sanitized["geo"] = geo
        else:
            sanitized.pop("geo", None)

    time_value = sanitized.get("time")
    if isinstance(time_value, str):
        normalized_time = normalize_time(time_value)
        if normalized_time:
            sanitized["time"] = normalized_time
        else:
            sanitized.pop("time", None)

    return sanitized


def apply_dimension_defaults(dataset: str | None, filters: dict[str, str | list[str]]) -> dict[str, str | list[str]]:
    enriched = dict(filters)
    if not dataset:
        return enriched
    dims = dimensions_map().get(dataset.lower(), {})
    defaults = dims.get("defaults", {})
    if isinstance(defaults, dict):
        for key, value in defaults.items():
            if key.lower() not in enriched and isinstance(value, str):
                enriched[key.lower()] = value
    return enriched
