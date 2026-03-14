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

- `app/kb/synonyms.json`: natural-language phrase -> indicator key
- `app/kb/dataset_map.json`: indicator key -> Eurostat dataset code
- `app/kb/country_codes.json`: country aliases -> `geo` code
- `app/kb/dimensions_map.json`: dataset dimension metadata (required/defaults)

Routing order:

1. Local KB mapping
2. Eurostat dataflow catalog discovery
3. `.env` default dataset fallback

## Search Flow

`POST /api/v1/data/search` runs:

1. `PlannerAgent` (LangChain runnable): parse intent from natural language, resolve dataset via local KB, fallback to Eurostat dataflow catalog.
2. `EurostatRetrieverAgent` (LangChain runnable): construct Eurostat API URL and fetch raw JSON payload.

Response includes:

- `planner.indicator`: normalized indicator (example `population`)
- `planner.dataset`: selected dataset code (example `demo_pjan`)
- `planner.filters`: parsed filters (example `geo=DE`, `time=2022`)
- `endpoint` + `params`: final API call details
- `payload`: raw Eurostat JSON
