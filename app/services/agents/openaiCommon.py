from __future__ import annotations

import json
import os

from dotenv import load_dotenv

from app.services import kb

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT_SECONDS", os.getenv("EUROSTAT_TIMEOUT_SECONDS", "30")))


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
