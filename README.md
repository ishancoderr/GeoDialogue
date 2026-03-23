# GeoDialogue

GeoDialogue is a FastAPI prototype for missingness search with two explicit agents:
- `Agent1` checks its own local GeoJSON partition first.
- `Agent2` checks a second GeoJSON partition and can also query Eurostat.

The current prototype exposes one endpoint:
- `POST /api/v1/kqml/search`

## Idea

The system is built around missingness search rather than simple API retrieval.

Example user need:
`I am searching my missing data for Bulgaria 2007`

What happens:
1. OpenAI turns the user text into a structured plan.
2. `Agent1` checks `europe_housing_geojson1.geojson`.
3. If the value is missing, `Agent1` sends a KQML message to `Agent2`.
4. `Agent2` checks `europe_housing_geojson2.geojson`.
5. If needed, `Agent2` queries Eurostat.
6. The API returns a final KQML response with a missingness decision.

## Flow Chart

```text
User Query
   |
   v
POST /api/v1/kqml/search
   |
   v
OpenAI Planner
   |
   v
Validated Search Plan
   |
   v
Agent1 checks geojson1
   |
   +--> found locally ---------> Final KQML reply
   |
   +--> missing
           |
           v
      KQML: find-missing-data
           |
           v
      Agent2 checks geojson2
           |
           +--> found ----------> Final KQML reply
           |
           +--> still missing
                   |
                   v
             Eurostat lookup
                   |
                   v
             Final KQML reply
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create `.env` from `.env.example` and set at least these values:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_TIMEOUT_SECONDS=30
EUROSTAT_TIMEOUT_SECONDS=30
LOG_LEVEL=INFO
MISSINGNESS_GEOJSON1_PATH=data/missingness/europe_housing_geojson1.geojson
MISSINGNESS_GEOJSON2_PATH=data/missingness/europe_housing_geojson2.geojson
```

Run the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## How To Call The Endpoint

Route:

```text
POST /api/v1/kqml/search
```

### 1. Call with real KQML

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/kqml/search" ^
  -H "Content-Type: text/plain" ^
  --data '(ask-one :content (search :query "I am searching my missing data for Bulgaria 2007") :reply-with "r1")'
```

### 2. Call with JSON fallback

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/kqml/search" ^
  -H "Content-Type: application/json" ^
  -d '{"query":"I am searching my missing data for Bulgaria 2007"}'
```

### 3. Call from Python

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/api/v1/kqml/search",
    data='(ask-one :content (search :query "I am searching my missing data for Bulgaria 2007") :reply-with "r1")',
    headers={"Content-Type": "text/plain"},
    timeout=30,
)

print(response.status_code)
print(response.text)
```

## Example KQML Response

```lisp
(tell
  :in-reply-to "r1"
  :content (
    search-result
    :indicator "housing_deprivation"
    :dataset "ilc_lvho02"
    :geo "BG"
    :time "2007"
    :filters (dict :geo "BG" :time "2007")
    :endpoint "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ilc_lvho02"
    :params (dict :lang "en" :format "JSON" :geo "BG" :time "2007")
    :payload (dict)
    :truncated "f"
    :missingness (dict :status "recoverable" :recommended_source "eurostat")
  )
)
```

## Human-Readable Agent-to-Agent KQML

Agent1 to Agent2:

```lisp
(ask-one
  :sender "agent1"
  :receiver "agent2"
  :reply-with "missingness-r1"
  :content (
    find-missing-data
    :query "I am searching my missing data for Bulgaria 2007"
    :indicator "housing_deprivation"
    :dataset "ilc_lvho02"
    :geo "BG"
    :time "2007"
    :agent1-missing-types (attribute_missingness)
  )
)
```

Agent2 to Agent1:

```lisp
(tell
  :sender "agent2"
  :receiver "agent1"
  :in-reply-to "missingness-r1"
  :content (
    missing-data-result
    :status "found"
    :source "europe_housing_geojson2"
    :missing-types (attribute_missingness joinability_missingness)
    :joinable t
    :join-key "geo"
    :recoverable t
    :confidence 0.75
    :note "europe_housing_geojson2 is missing the requested value but can be linked with key 'geo'. Agent2 can also ask Eurostat for the missing value."
  )
)
```

More detail is in [KQML_COMMUNICATION.md](KQML_COMMUNICATION.md).

## Project Structure

```text
app/
  controllers/
    kqmlController.py
  services/
    kb.py
    agents/
      agent1.py
      agent2.py
    kqmlbase/
      base.py
data/
  missingness/
    europe_housing_geojson1.geojson
    europe_housing_geojson2.geojson
```

## What Each Part Does

- `app/services/agents/agent1.py`: Agent1, planner class, OpenAI settings, trace helpers, partition loading, and missingness decision logic.
- `app/services/agents/agent2.py`: Agent2 and the Eurostat retriever logic.
- `app/services/kqmlbase/base.py`: KQML orchestration and planner parsing.
- `app/services/kb.py`: dataset, country, dimension, and endpoint validation helpers.

## Notes

- OpenAI is required for planning. If OpenAI fails, the search does not continue.
- The current prototype focuses on the KQML endpoint and Agent1-to-Agent2 communication.
- The demo GeoJSON files are intentionally small and human-readable.
