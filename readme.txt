GeoDialogue focuses on missingness search with two explicit agents.

Agent1 checks data/missingness/europe_housing_geojson1.geojson first.
Agent2 checks data/missingness/europe_housing_geojson2.geojson next and can also query Eurostat.

Endpoint:
- POST /api/v1/kqml/search

Quick flow:
User query -> OpenAI planner -> Agent1 -> Agent2 -> Eurostat -> final KQML reply

Example call:
curl -X POST "http://127.0.0.1:8000/api/v1/kqml/search" -H "Content-Type: application/json" -d '{"query":"I am searching my missing data for Bulgaria 2007"}'

Eurostat API reference:
https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/api-getting-started/api
