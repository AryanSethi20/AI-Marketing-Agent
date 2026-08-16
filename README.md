# AI Marketing Agent 🎯

A modular, production-grade CLI agent that discovers **live web trends** for a
keyword, selects the **top 3 topics most relevant to a target company's brand**,
generates a **complete marketing campaign plan** for each, and renders a clean,
modern **3-page PDF report** — exactly one topic per page.

Runs on **local Ollama models**, **OpenAI**, or **free API providers
(Groq / OpenRouter)** — switched entirely via environment variables, with
**zero hardcoded secrets**.

---

## Architecture Overview

```
                ┌──────────────────────────────────────────────┐
                │                  main.py (CLI)                │
                └───────┬───────────┬───────────┬──────────────┘
                        │           │           │
        ┌───────────────▼──┐  ┌─────▼───────┐  ┌▼────────────────┐
        │   TrendAgent     │  │ Evaluator   │  │  CampaignAgent  │
        │ live search →    │─▶│ Agent       │─▶│ 1 plan / topic  │
        │ trend extraction │  │ top-3 picks │  │ (Pydantic)      │
        └───────┬──────────┘  └─────┬───────┘  └────────┬────────┘
                │                   │                    │
   ┌────────────▼─────────┐   ┌─────▼──────────┐   ┌─────▼──────────────┐
   │ tools/search_tool.py │   │ llm_factory.py │   │ utils/pdf_generator│
   │ DuckDuckGo (keyless) │   │ Ollama/OpenAI/ │   │ ReportLab, 1 page  │
   │ news + web queries   │   │ Groq/OpenRouter│   │ per topic          │
   └──────────────────────┘   └────────────────┘   └────────────────────┘
```

**Design principles**

| Concern | Approach |
|---|---|
| Model flexibility | Single OpenAI-SDK client; every provider (incl. Ollama at `http://localhost:11434/v1`) is OpenAI-compatible, so routing is pure configuration (`config.py`). |
| Determinism | All agent outputs are **schema-constrained** via Pydantic models; the JSON schema is injected into the system prompt. |
| Prompt resilience | Native `response_format=json_object` where supported, markdown-fence stripping, brace-slicing recovery, and **validation-error feedback retries** (the model sees its own error and corrects itself). |
| Fault tolerance | Search retries with backoff; headline-based trend fallback; evaluator guarantees **exactly 3** topics (pads/truncates); campaign skeleton fallback keeps the report complete. |
| Secrets | Loaded only from environment / `.env`. Nothing hardcoded. |
| Layout guarantee | Explicit `PageBreak()` between topics → Topic *N* is always on page *N*. |

---

## Project Structure

```
marketing_agent/
├── config.py               # Typed env configuration (pydantic-settings)
├── llm_factory.py          # Unified LLM client + structured-output engine
├── tools/
│   └── search_tool.py      # DuckDuckGo live search (no API key)
├── agents/
│   ├── trend_agent.py      # Keyword → 5–10 structured trends
│   ├── evaluator_agent.py  # Trends → top 3 brand-relevant topics
│   └── campaign_agent.py   # Topic → full campaign plan (Pydantic)
├── utils/
│   └── pdf_generator.py    # ReportLab report, 1 page per topic
├── main.py                 # CLI entrypoint (argparse)
├── requirements.txt        # Pinned dependencies
├── .env.example            # Configuration template
└── README.md
```

---

## 1. System Prerequisites

- **Python 3.10+** (developed and tested on 3.12)
- `pip` and (recommended) a virtual environment
- Internet access (DuckDuckGo trend search)
- **Optional:** [Ollama](https://ollama.com) for fully local, free inference

## 2. Installation

```bash
git clone <your-repo-url> marketing_agent
cd marketing_agent

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env               # then edit .env
```

## 3. Choose Your Model Backend

### Option A — Local Ollama (free, private, default)

```bash
# 1. Install Ollama: https://ollama.com/download
# 2. Pull a model (pick one based on your hardware):
ollama pull qwen2.5:32b     # best quality (needs ~20 GB RAM/VRAM)
ollama pull llama3.2        # lightweight, runs on most laptops

# 3. Make sure the server is running (usually automatic):
ollama serve                # exposes http://localhost:11434
```

`.env`:
```env
MODEL_PROVIDER=ollama
MODEL_NAME=qwen2.5:32b       # or llama3.2
OLLAMA_BASE_URL=http://localhost:11434/v1
```

> Ollama needs no API key — the client sends a harmless placeholder.

### Option B — OpenAI

`.env`:
```env
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4o-mini
OPENAI_API_KEY=sk-...your key...
```

### Option C — Groq (generous free tier)

`.env`:
```env
MODEL_PROVIDER=groq
MODEL_NAME=llama-3.3-70b-versatile
OPENAI_API_KEY=gsk_...your Groq key...
```

### Option D — OpenRouter (free community models)

`.env`:
```env
MODEL_PROVIDER=openrouter
MODEL_NAME=meta-llama/llama-3.3-70b-instruct:free
OPENAI_API_KEY=sk-or-...your OpenRouter key...
```

### Option E — Any custom OpenAI-compatible gateway

`.env`:
```env
MODEL_PROVIDER=custom
CUSTOM_BASE_URL=https://my-gateway.example.com/v1
MODEL_NAME=my-model
OPENAI_API_KEY=...if required...
```

## 4. Usage

```bash
python main.py --keyword "gaming" --company "Mastercard" --output "campaign.pdf"
```

More examples:

```bash
# Fintech trends for Mastercard, default output name (campaign_report.pdf)
python main.py --keyword "fintech" --company "Mastercard"

# Food trends for a CPG brand, verbose logging
python main.py --keyword "food" --company "Nestlé" --output food_report.pdf --verbose

# Help
python main.py --help
```

**CLI flags**

| Flag | Required | Default | Description |
|---|---|---|---|
| `--keyword` | ✅ | — | Domain to research (e.g. `"food"`, `"fintech"`) |
| `--company` | ✅ | — | Company used for brand-fit evaluation |
| `--output` | ❌ | `campaign_report.pdf` | Output PDF path |
| `--verbose` | ❌ | off | Debug logging incl. HTTP clients |

## 5. What the Report Contains

Each of the 3 pages presents one topic with:

- **Topic header** — trend name, relevance score, creative campaign title
- **Why this fits the brand** — evaluator rationale
- **Target Audience** — segments, demographics, psychographics
- **Core Message** — single-minded proposition
- **Channel Strategy** — 3–6 channels with concrete activation tactics (table)
- **Key Performance Indicators** — 3–6 measurable, quantified KPIs (table)
- **Brand Synergy** — how the campaign reinforces brand, strategy, and products

Styling: dark-slate banner, warm-orange accent, Helvetica typography, soft
rules and tinted table headers — with a hard `PageBreak()` ensuring exactly
one topic per page.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `MODEL_PROVIDER=... requires OPENAI_API_KEY` | Add your key to `.env` (or switch to `ollama`). |
| `Connection refused` on `localhost:11434` | Start Ollama: `ollama serve`, and confirm the model is pulled. |
| DuckDuckGo rate-limit warnings | The tool retries with backoff automatically; re-run after a minute if it persists. |
| Empty/odd JSON from small local models | The corrective-retry loop usually recovers; otherwise try a larger model (e.g. `qwen2.5:32b`) or lower `LLM_TEMPERATURE`. |
| Slow local generation | Use `llama3.2` for speed, or Groq for fast free hosted inference. |

## 7. Security Notes

- No secrets in code — configuration is environment-only (`.env` is
  git-ignored by convention; only `.env.example` ships).
- Ollama runs fully local: no data leaves your machine except DuckDuckGo
  search queries.
