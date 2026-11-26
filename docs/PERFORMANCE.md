# DealForge AI - Performance Evaluation

*Generated 2026-11-25T20:17:17.215178+00:00 | mode: **deterministic** | DB: sqlite | 3 runs/scenario (warm-up excluded)*

All latencies are wall-clock measurements of the real pipeline (CrewAI engine, MCP tools, A2A call to the ADK agent over HTTP).

## Latency and outcome per scenario

| Scenario | Outcome | Mean ms | Min | Max | p95 | MCP calls | A2A | Tokens/run | Cost/run (USD) |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| S1 Standard bulk laptops (50) | PAUSED | 260.47 | 250.93 | 265.35 | 265.35 | 20 | OK | 0 | 0.0 |
| S2 Enterprise servers (10) | PAUSED | 197.91 | 188.19 | 207.19 | 207.19 | 8 | OK | 0 | 0.0 |
| S3 Displays (15) | COMPLETED | 211.95 | 181.92 | 243.6 | 243.6 | 8 | OK | 0 | 0.0 |
| S4 Networking switches (20) | PAUSED | 179.92 | 176.13 | 182.57 | 182.57 | 8 | OK | 0 | 0.0 |
| S5 Developer laptops (25) | PAUSED | 282.74 | 263.35 | 299.8 | 299.8 | 20 | OK | 0 | 0.0 |
| S6 Compliance refusal (7-day displays) | REFUSED | 107.08 | 102.22 | 110.17 | 110.17 | 5 | - | 0 | 0.0 |
| S7 Prompt injection blocked | BLOCKED:PROMPT_INJECTION_DEFENSE | 0.46 | 0.03 | 1.3 | 1.3 | 0 | - | 0 | 0.0 |
| S8 A2A service down (fail-safe) | PAUSED | 213.94 | 209.1 | 221.38 | 221.38 | 20 | FAILSAFE_HIGH | 0 | 0.0 |

## Per-step latency breakdown (S1, mean ms)

| Step | Mean ms | p95 |
|---|---:|---:|
| Requirement Analysis Agent | 2.05 | 2.44 |
| Vendor Research Agent (FastMCP Client) | 10.01 | 10.53 |
| Price Analysis Agent (FastMCP Pricing) | 40.36 | 49.97 |
| Compliance Agent | 20.13 | 20.61 |
| Deal Evaluation Agent | 29.68 | 31.06 |
| Budget Validation Agent (FastMCP Budget Tool) | 3.75 | 5.16 |
| Negotiation Agent | 6.71 | 7.63 |
| Google ADK Risk Assessment Agent (A2A Protocol) | 34.57 | 35.29 |

## Token / cost note

In deterministic mode no LLM is called, so tokens and cost are genuinely **0**. Use the LLM estimate below, or run `--llm` with a Gemini key for measured values.

## LLM-mode cost ESTIMATE

_ESTIMATE: tokens = characters/4 over the REAL prompts CrewAI builds for the 3 LLM agents; completions sized from schema-valid sample answers. Real LLM latency is not estimated._

Pricing used: $0.3/M input, $2.5/M output (configured model `gemini-3.5-flash`). **These are the config values (LLM_PRICE_*), which default to PLACEHOLDERS - set them from Google's price list for the model that actually served your requests before quoting any cost.** INR rate 86.5.

| Scenario | LLM calls | Est. prompt tok | Est. completion tok | Est. cost USD | Est. cost INR |
|---|---:|---:|---:|---:|---:|
| S1 Standard bulk laptops (50) | 4 | 6287 | 525 | 0.003199 | 0.2767 |
| S2 Enterprise servers (10) | 4 | 6289 | 491 | 0.003114 | 0.2694 |
| S3 Displays (15) | 4 | 6210 | 492 | 0.003093 | 0.2675 |
| S4 Networking switches (20) | 4 | 6292 | 494 | 0.003123 | 0.2701 |
| S5 Developer laptops (25) | 4 | 6260 | 522 | 0.003183 | 0.2753 |
