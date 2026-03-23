# KQML Communication

This document explains the KQML-only workflow used in GeoDialogue.

## Endpoint

```text
POST /api/v1/kqml/search
```

Main files:
- `app/controllers/kqmlController.py`
- `app/services/kqmlbase/base.py`
- `app/services/agents/agent1.py`
- `app/services/agents/agent2.py`

## High-Level Flow

```text
Client -> /api/v1/kqml/search -> OpenAI planner -> Agent1 -> Agent2 -> Eurostat -> final KQML reply
```

More detailed flow:

```text
Client request
   |
   v
KQML controller
   |
   v
OpenAI returns search-plan
   |
   v
Agent1 checks geojson1
   |
   +--> complete locally
   |       |
   |       v
   |   final search-result
   |
   +--> missing
           |
           v
      Agent1 sends find-missing-data
           |
           v
      Agent2 checks geojson2
           |
           +--> found in partition 2
           |       |
           |       v
           |   final search-result
           |
           +--> not found
                   |
                   v
             Agent2 queries Eurostat
                   |
                   v
             final search-result
```

## How The Client Calls It

### KQML request

```lisp
(ask-one
  :content (search :query "I am searching my missing data for Bulgaria 2007")
  :reply-with "r1"
)
```

### JSON fallback

```json
{"query":"I am searching my missing data for Bulgaria 2007"}
```

## Planner Output Shape

OpenAI must return one KQML message in this outer form:

```lisp
(tell
  :content (
    search-plan
    :indicator "housing_deprivation"
    :dataset "ilc_lvho02"
    :geo "BG"
    :time "2007"
  )
)
```

That output is then validated against the local KB before the agents continue.

## Agent1 To Agent2 Message

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

## Agent2 Reply

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

## Final Response To Client

```lisp
(tell
  :in-reply-to "r1"
  :content (
    search-result
    :indicator "housing_deprivation"
    :dataset "ilc_lvho02"
    :geo "BG"
    :time "2007"
    :missingness (
      dict
      :status "recoverable"
      :recommended_source "eurostat"
      :joinable t
      :join_key "geo"
    )
  )
)
```

## Why This Matters

This design makes the thesis story clear:
- `Agent1` is the local checker and coordinator.
- `Agent2` is the recovery agent.
- KQML is used for readable inter-agent communication.
- Eurostat is the external recovery source when the local partitions are incomplete.
