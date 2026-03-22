# GeoDialogue
This project focuses on designing a multi-agent system for geographic data exchange.

## Ingestion Setup (Enterprise Baseline)

### 1. Create virtual environment

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\Activate.ps1
```

### 2. Configure environment

```bash
Copy-Item .env.example .env
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run ingestion test

```bash
python fetch_eurostat_data.py --log-level INFO
```

Expected output:
- JSON file written to `data/ilc_lvho02_raw.json`
- `Updated:` timestamp
- `Dimensions:` list

## Runtime Architecture

- `fetch_eurostat_data.py`: CLI entry point + runtime config resolution
- `requirements.txt`: locked runtime dependencies
- `.env.example`: configuration contract for deployment environments
- `.gitignore`: excludes secrets, virtual env, generated data

## FastAPI Endpoint

Implemented endpoint:

- `POST /api/v1/data/search`
- `POST /api/v1/kqml/search` (KQML)

Run API:

```bash
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Optional OpenAI planner:

```bash
# in .env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_TIMEOUT_SECONDS=30
```

Test request:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/data/search" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"German population in 2022\"}"
```

KQML request:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/kqml/search" ^
  -H "Content-Type: text/plain" ^
  --data "(ask-one :content (search :query \"German population in 2022\") :reply-with \"r1\")"
```

## Hybrid Knowledge Base (Planner Agent)

`parse_request` now uses a local KB before remote discovery:

- `app/kb/dataset_map.json`: indicator key -> Eurostat dataset code
- `app/kb/country_codes.json`: country aliases -> `geo` code
- `app/kb/dimensions_map.json`: dataset dimension metadata (required/defaults)
- `app/kb/endpoint_terms.json`: Eurostat endpoint parameter vocabulary used to constrain OpenAI output

Routing order:

1. OpenAI planner parses messy natural language into structured Eurostat fields when `OPENAI_API_KEY` is configured
2. Local KB validates indicator, dataset, `geo`, `time`, and allowed endpoint params
3. Eurostat dataflow catalog discovery for unresolved datasets
4. `.env` default dataset fallback

## Search Flow

`POST /api/v1/data/search` runs:

1. `PlannerAgent` (LangChain runnable): optionally calls OpenAI Responses API to normalize messy user text into strict JSON, then validates it against the Eurostat KB and resolves the dataset.
2. `EurostatRetrieverAgent` (LangChain runnable): constructs the Eurostat API URL and fetches raw JSON payload.

Response includes:

- `planner.indicator`: normalized indicator (example `population`)
- `planner.dataset`: selected dataset code (example `demo_pjan`)
- `planner.geo`: normalized country code (example `DE`)
- `planner.time`: normalized time filter (example `2022`)
- `planner.filters`: parsed filters (example `geo=DE`, `time=2022`)
- `endpoint` + `params`: final API call details
- `payload`: raw Eurostat JSON
