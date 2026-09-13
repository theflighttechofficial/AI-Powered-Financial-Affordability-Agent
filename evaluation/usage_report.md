# Token Usage and Cost Report

## Summary

This run made real model calls for image amount extraction (`src/image_extraction.py`, local Ollama) and message fact extraction (`src/message_facts.py`, Groq primary with a local Ollama fallback on failure). All other computation -- the 90-day balance forecast, affordability decision, and payment-plan ranking -- is deterministic and makes no model calls.

| Metric | Value |
|---|---|
| Total model calls | 155 |
| Total input tokens | 125,342 |
| Total output tokens | 36,165 |
| Total tokens | 161,507 |

## By model

| Model | Calls | Input tokens | Output tokens | Est. cost |
|---|---|---|---|---|
| openai/gpt-oss-120b | 155 | 125,342 | 36,165 | $0.0405 |

**Estimated total cost: $0.0405**
**Average cost per request in this run: see per-request note below.**

Note: image and message extraction calls are cached on disk (`.image_extraction_cache.json`, `.message_facts_cache.json`) keyed by image path / message content, so the counts above reflect a full cold run against this dataset -- a second run against the same dataset makes zero additional calls.

Per-request average: this dataset has 16 blank-amount events (image calls) and up to ~215 messages (message-fact calls), shared across all 250 requests in requests.csv, so the per-request marginal cost is far below the per-call cost -- divide the total cost above by 250 for an average, or note that most requests trigger zero fresh calls since they reuse already-cached user history.
