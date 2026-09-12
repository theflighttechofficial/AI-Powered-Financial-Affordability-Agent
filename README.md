# Buy or Wait — AI-Powered Financial Affordability Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-20%20passed-brightgreen.svg)](#testing)
[![Provider](https://img.shields.io/badge/groq%20%2B%20ollama-orange.svg)](https://groq.com/)

An intelligent financial affordability and balance-forecasting engine that evaluates whether a user can safely afford a requested purchase or financial commitment. The agent projects a 90-day cashflow timeline based on historical transaction data, processes unstructured inputs (receipt images and multi-lingual account messages) using AI models, and evaluates deterministic safety rules to recommend optimal payment plans.

**Model routing:** message/text extraction uses **Groq** (`openai/gpt-oss-120b`) as primary, falling back to a **local Ollama** model only if Groq fails. Receipt/invoice **image** extraction goes straight to a **local Ollama** vision model (`llama3.2-vision` by default) -- this project's Groq account has no vision-capable model available, so there's no cloud path to try first for images. See [`src/ai_client.py`](src/ai_client.py) for the routing logic.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [AI vs. Deterministic Core](#ai-vs-deterministic-core)
- [Treating Messages and Images as Untrusted Data](#treating-messages-and-images-as-untrusted-data)
- [Module Breakdown](#module-breakdown)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Methodology & Algorithmic Design](#methodology--algorithmic-design)
- [Testing & Verification](#testing--verification)
- [Cost & Token Monitoring](#cost--token-monitoring)

---

## Overview

When a user requests a major expenditure, **Buy or Wait** evaluates whether they can make the payment today, wait for a future safe date, split the purchase into partial payments, or leverage structured installment plans—all while guaranteeing that their account balance stays strictly above their minimum required balance floor across a 90-day forecast horizon.

The system combines:
1. **AI Models (Groq + local Ollama)**: A local Ollama vision model extracts monetary totals from photographed/screenshotted receipt documents, while an LLM (Groq primary, Ollama fallback) parses natural language financial notifications (English & Bahasa Indonesia) into typed, structured facts.
2. **Deterministic Rules & Math**: Recurring cashflow pattern detection, FX conversions, binary-search budget safety limits, spending-change optimization, and strict candidate plan ranking.

---

## System Architecture

```
                                  UNSTRUCTURED INPUTS
                                          │
      ┌───────────────────────────────────┴───────────────────────────────────┐
      │                                                                       │
      ▼                                                                       ▼
Receipt/Invoice Images                                         Account & Payroll Messages
 (PNG/JPG in dataset/media)                                   (Natural Language: EN / ID)
      │                                                                       │
      ▼                                                                       ▼
[src/image_extraction.py]                                    [src/message_facts.py]
Ollama Vision Call (Disk-Cached)                     Groq Call, Ollama on failure (Disk-Cached)
      │                                                                       │
      └───────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼ Structured Facts
                                  DETERMINISTIC PIPELINE
                                          │
                                          ▼
                               [src/event_resolution.py]
                 - Clean raw events (drop cancelled, failed, unrealized)
                 - Detect per-category recurring series (median interval)
                                          │
                                          ▼
                                  [src/currency.py]
                - Fixed-rate FX conversion (nearest-date & USD fallback)
                                          │
                                          ▼
                                  [src/forecast.py]
                - Build 90-day cashflow timeline
                - Binary search for max safe payment today
                - Day-by-day scan for earliest safe full-payment date
                                          │
                                          ▼
                               [src/decision_engine.py]
                - Evaluate candidate plans (Full, Wait, Installments, Partial)
                - Spending change suggestions (stop / reduce_to)
                - 6-level deterministic tie-breaker ranking
                                          │
                                          ▼
                                     output.csv
                                evaluation/usage_report.md
```

---

## AI vs. Deterministic Core

| Task | Approach | Module | Rationale |
|---|---|---|---|
| **Receipt / Invoice / Payslip OCR & Amount Extraction** | Vision LLM Call (local Ollama only) | [`src/image_extraction.py`](src/image_extraction.py) | Blank-amount financial events link to real scanned images (handwritten chits, GST invoices, payslips, hospital bills). Varies visually and structurally, making rules infeasible. No Groq vision model is available on this account's key, so this always calls a local Ollama vision model. |
| **Message Fact Extraction** | LLM Call (Constrained JSON Schema) | [`src/message_facts.py`](src/message_facts.py) | Free-text SMS/email notifications in English and Bahasa Indonesia describing salary changes, gig income, rent adjustments, internal transfers, or scam patterns. |
| **FX Rate Conversions** | Deterministic Math | [`src/currency.py`](src/currency.py) | Converts multi-currency transactions into the user's home currency using fixed, dated conversion tables with date-proximity and USD-bridge fallback. |
| **Recurrence & Cashflow Forecasting** | Deterministic Math | [`src/event_resolution.py`](src/event_resolution.py), [`src/forecast.py`](src/forecast.py) | Computes median intervals and 3-point median amounts to project future cashflow points while guaranteeing the 90-day minimum balance floor. |
| **Affordability & Ranking Engine** | Deterministic Rules | [`src/decision_engine.py`](src/decision_engine.py) | Evaluates user profile constraints, spending modifications, payment options, and ranks valid plans using a strict 6-tier tie-breaking algorithm. |

---

## Treating Messages and Images as Untrusted Data

Security and financial safety are built into the extraction layer so that untrusted or malicious inputs cannot subvert downstream financial calculations:

- **Strict Schema Enforcement**: Extraction prompts restrict LLM outputs to rigid JSON structures (`MessageFact` and image total amounts).
- **Prompt Injection Defense**: Message extraction prompts explicitly instruct the model to treat content as untrusted data to classify, never as system instructions.
- **Explicit Scam Classification**: Advance-fee scams (e.g. *"pay a release fee to claim your prize"*) are categorized as `income_scam` and discarded.
- **Downstream Isolation**: Core modules ([`src/forecast.py`](src/forecast.py) and [`src/decision_engine.py`](src/decision_engine.py)) read only typed dataclasses and never inspect raw message strings.
- **Fail-Closed Strategy**: API network or JSON parsing failures return safe defaults (`None` or `no_op`) to drop single ambiguous data points rather than guessing, but only after the Groq -> Ollama fallback path (text) or the Ollama call (images) has actually been attempted -- a config error (missing `GROQ_API_KEY`, or Ollama unreachable) is never silently absorbed into a fallback default; it halts the run immediately instead.

---

## Module Breakdown

Below is a map of the repository components:

```
src/
  ├── ai_client.py           Groq client + local Ollama call, fallback routing, & centralized token usage tracker. Loads .env automatically.
  ├── image_extraction.py    Vision-model caller for receipt/invoice image total extraction with disk caching.
  ├── message_facts.py       LLM fact extractor for free-text messages with a strict JSON schema and disk caching.
  ├── currency.py            FX converter supporting direct, inverse, nearest-date, and USD cross-bridging.
  ├── event_resolution.py    Event cleaner (drops cancelled/failed/pending credits) & recurring series detector.
  ├── forecast.py            90-day cashflow timeline builder, binary-search safety solver, & safe date scanner.
  ├── decision_engine.py     Rule-based candidate plan generator, spending-change optimizer, & 6-level ranker.
  └── pipeline.py            End-to-end orchestration CLI wiring inputs -> AI extraction -> forecasting -> output.csv.

tests/
  └── test_forecast.py       Synthetic unit tests verifying timeline math, binary search safety, and spending adjustments.

evaluation/
  └── usage_report.md        Auto-generated token count and cost summary produced after each pipeline execution.
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- A valid [Groq API Key](https://console.groq.com/) (primary, text extraction)
- [Ollama](https://ollama.com/) installed and running locally (required for image extraction; used as a fallback for text extraction if Groq fails)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Pull the local Ollama models

```bash
ollama pull llama3.2-vision
ollama pull llama3.1:8b
```

### 3. Configure Environment

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

*Alternatively, export the variable in your shell:*

```bash
export GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

Ollama is assumed to be reachable at `http://localhost:11434` with the models above pulled. Override any of this via env vars if needed:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Local Ollama endpoint (OpenAI-compatible) |
| `OLLAMA_TEXT_MODEL` | `llama3.1:8b` | Fallback text model when Groq fails |
| `OLLAMA_VISION_MODEL` | `llama3.2-vision` | Vision model for receipt image extraction (always used -- no Groq path exists for images) |
| `BUY_OR_WAIT_MODEL` | `openai/gpt-oss-120b` | Primary Groq text model |
| `BUY_OR_WAIT_MAX_RETRIES` | `5` | Groq SDK retry budget for 429/5xx before falling back to Ollama |

---

## How to Run

### Main Pipeline Run
Process the standard dataset (`dataset/requests.csv`) and output results to `output.csv`:

```bash
python -m src.pipeline --dataset dataset --out output.csv
```

### Evaluation Run
Validate against the reference dataset (`sample_requests.csv`) and output to `sample_output.csv`:

```bash
python -m src.pipeline --dataset dataset --out sample_output.csv --requests-file sample_requests.csv
```

### CLI Arguments

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `dataset` | Path to dataset directory containing CSVs and media files. |
| `--out` | `output.csv` | Target destination CSV file path for generated decisions. |
| `--requests-file` | `requests.csv` | File name inside dataset directory to process. |
| `--usage-report` | `evaluation/usage_report.md` | Path where token usage and cost analysis report will be written. |

### Disk Caching
AI calls are saved locally to `.image_extraction_cache.json` and `.message_facts_cache.json`, keyed by (input, model) so switching models invalidates only the affected entries. Subsequent runs on the same dataset/model execute instantaneously with **0 additional API calls**.

---

## Methodology & Algorithmic Design

### 1. Recurrence Detection
Historical transactions are grouped by `(category, direction, event_type)`. If a category contains $\ge 2$ occurrences, it is classified as a recurring series.
- **Interval**: Calculated as the median gap in days between historical occurrences.
- **Representative Amount**: Uses the **median of the last 3 occurrences** to remain resilient against single-purchase outliers while quickly reacting to recent structural level shifts (e.g. pay changes).

### 2. FX Rate Conversion Fallback
Converts amounts into the user's home currency using `exchange_rates.csv`:
1. Exact `(from_currency, to_currency, rate_date)` match.
2. Direct or inverse rate matching the nearest available date.
3. USD cross-currency bridge (`from_ccy -> USD -> to_ccy`).

### 3. Safety Solver (Binary Search & Horizon Scan)
- **`max_safe_payment_today`**: Executes binary search between $0$ and `requested_amount` to determine the maximum immediate outflow that maintains `current_balance >= minimum_balance_to_keep` continuously for 90 days.
- **`earliest_safe_full_payment_date`**: Scans day-by-day from `request_date` up to 90 days forward to find the earliest date when the full requested amount can be paid without requiring spending reductions.

### 4. Candidate Plan Ranking (6-Tier Tie-Breaker)
When multiple valid payment plans are available, the decision engine selects the optimal plan according to the following strict priority:
1. Completes on or before `desired_completion_date`.
2. Requires no spending changes (`stop` or `reduce_to`).
3. Lowest total monetary amount paid (minimizes interest/fees).
4. Earliest payment start date.
5. Fewest total payments.
6. Lowest `payment_option_id` (lexicographical tie-breaker).

---

## Testing & Verification

Run the test suite using `pytest`:

```bash
python -m pytest tests/ -v
```

The 9 unit tests in [`tests/test_forecast.py`](tests/test_forecast.py) test timeline construction, binary search cashflow bounds, minimum threshold enforcement, spending change overrides, and safe payment date discovery using synthetic timelines. The 11 unit tests in [`tests/test_ai_extraction.py`](tests/test_ai_extraction.py) mock the Groq/Ollama calls to test the Groq-primary/Ollama-fallback routing, image extraction's Ollama-only path, disk-cache invalidation on model change, and that a genuine config error (missing API key, Ollama unreachable) always propagates rather than being silently absorbed as a normal per-item failure. None of the 20 tests require network calls, an API key, or a running Ollama instance.

---

## Cost & Token Monitoring

Every pipeline execution automatically outputs a token usage report to [`evaluation/usage_report.md`](evaluation/usage_report.md). Token tracking tracks input/output tokens per model via [`src/ai_client.py`](src/ai_client.py) and computes total cost estimates.

*Example summary output:*

```markdown
# Token Usage and Cost Report

| Metric | Value |
|---|---|
| Total model calls | 198 |
| Total input tokens | 155,224 |
| Total output tokens | 45,169 |
| Total tokens | 200,393 |

Estimated total cost: $0.0504
```

