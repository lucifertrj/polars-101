# Sales Data Analytics Agent

A Haystack tool-calling agent that answers questions about sales data using Polars,
plus a Streamlit UI with a chat interface, a prebuilt report, and a data explorer.

## How to run

### 1. Install dependencies

This project uses [uv](https://docs.astral.sh/uv/). Dependencies are declared in `pyproject.toml`.

```bash
uv sync
```

### 2. Set your OpenAI API key

Create a `.env` file in the project root (loaded automatically via `python-dotenv`):

```
OPENAI_API_KEY=sk-...
```

### 3. Run

**CLI — single question:**

```bash
uv run python app.py "What is the total revenue by category?"
```

**CLI — interactive mode:**

```bash
uv run python app.py
```

**Streamlit UI:**

```bash
uv run streamlit run ui.py
```

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | Agent factory (`create_agent`) and CLI entrypoint. |
| `tool.py` | The `execute_polars_code` tool — runs model-generated Polars code in a locked-down subprocess — plus `get_csv_context`. |
| `analytics.py` | Deterministic Polars helpers (KPIs, revenue-by, top products, monthly trend, rep leaderboard) that power the Report page — no LLM involved. |
| `ui.py` | Streamlit entrypoint: the Chat page, backed by the agent. |
| `nav.py` | Shared sidebar navigation across pages. |
| `pages/` | Additional Streamlit pages: `Report` (prebuilt dashboard) and `Data Explorer`. |
| `data/sales_data.csv` | The sales dataset. |
| `flow.md` | Architecture diagram and loop description. |

The **Chat** page uses the LLM agent to answer free-form questions; the **Report**
and **Data Explorer** pages use `analytics.py` directly for fast, deterministic
numbers with no model calls.
