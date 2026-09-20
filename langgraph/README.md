# Local Business Intelligence Assistant

A local, secure, LangGraph-powered business intelligence assistant for retail and sales analysis. The application lets a user ask natural-language questions about sales performance, inventory, customer sentiment, pricing, and operational trends, then converts those questions into safe SQLite queries, executes them locally, summarizes the results, and visualizes the output with Plotly.

This project is designed for local experimentation, internal BI workflows, and low-friction business analysis without requiring an external MCP server or an enterprise data platform.

## Why this project exists

Most BI tools require either:

- a complex enterprise stack,
- a managed cloud warehouse,
- or a heavy dashboard platform.

This project keeps the workflow lightweight:

- local SQLite data source,
- LangGraph orchestration,
- safe read-only SQL generation,
- Streamlit UI for interaction,
- optional tracing with LangSmith,
- business-friendly summaries and charts.

It is especially useful for:

- small teams working from local data,
- prototypes and demos,
- privacy-conscious analytics,
- analyst-friendly conversational Q&A over business data.

---

## Key features

- Natural-language business question handling
- Route selection for general, business, diagnostic, and report queries
- Safe SQL generation with strict validation
- Read-only SQLite execution
- DataFrame-based result inspection
- Executive summary generation for business interpretation
- Plotly chart generation for useful visual insights
- Thread-based conversational flow in the Streamlit UI
- Optional tracing and observability with LangSmith
- No external MCP dependency

---

## Architecture

This project uses a simple but robust architecture:

1. User enters a question in Streamlit
2. LangGraph routes the prompt based on intent
3. Schema context is loaded from the local database
4. The LLM generates a safe SQL query
5. The SQL is validated before execution
6. The query runs against SQLite in read-only mode
7. Results are summarized into business insight
8. A relevant Plotly chart is generated if useful
9. Final response and optional chart are returned to the UI

### Core components

- Streamlit app: user interface and local session management
- LangGraph workflow: orchestration and routing
- SQLite data layer: schema inspection and safe execution
- Gemini model: SQL generation, summarization, and chart creation
- Plotly: chart rendering
- LangSmith: optional tracing and monitoring

---

## Repository structure

```text
local_bi_langgraph/
├── app.py                  # Streamlit app
├── bi_workflow.py          # LangGraph workflow and BI logic
├── data_access.py          # DB schema and query safety
├── pyproject.toml          # Python dependencies
├── README.md               # Project documentation
├── .env.example            # Environment template
├── sweden_data.db          # Local SQLite database
└── ...
```

The project expects the local SQLite database to be available in the sibling project folder, typically:

```text
./sweden_data.db
```

---

## Tech stack

- Python 3.11+
- Streamlit
- LangGraph
- Google Gemini via langchain-google-genai
- SQLite
- Pandas
- Plotly
- LangSmith

---

## Prerequisites

Before running the app, ensure you have:

- Python 3.11 installed
- Access to the local SQLite database
- A valid Google Gemini API key
- Optional: LangSmith API key for tracing

---

## Quick start

### 1. Clone or open the project

```bash
cd local_bi_langgraph
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -e .
```

### 4. Configure environment variables

Create a `.env` file:

```bash
cp .env.example .env
```

Then fill in the required values:

```env
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL=gemini-2.5-flash
LANGSMITH_API_KEY=your_langsmith_key_if_needed
LANGSMITH_PROJECT=local-bi-assistant
LANGSMITH_TRACING=false
```

### 5. Run the app

```bash
streamlit run app.py
```

The UI should open locally in your browser on the Streamlit default port.

---

## Supported question types

The workflow is optimized for business questions such as:

- What was total revenue last month?
- Why did sales drop in Q2?
- Which products are running low on stock?
- Which categories had the strongest revenue growth?
- Prepare a monthly business report with key actions.
- What are the latest customer sentiment trends?
- Compare competitor pricing for top products.

The app routes based on intent:

- general
- business
- diagnostic
- report

---

## Data safety and governance

This project is intentionally designed with strict data safety controls.

### Safety model

- Query execution is limited to read-only SQL
- Validation rejects unsafe operations
- Only SELECT and WITH statements are allowed
- No UPDATE, DELETE, INSERT, DROP, ALTER, or DDL actions
- Schema-aware generation reduces the risk of invalid or accidental joins
- Results are not written back to the source database

### Why this matters

This prevents:

- destructive DB operations,
- broad unbounded queries,
- accidental schema changes,
- accidental cross-table misuse.

The SQL validator is a core part of the application and is not treated as optional.

---

## Database context

This app is designed to work with a local SQLite retail dataset. It reads schema metadata and uses the actual database tables and columns to generate queries in-context.

### Example data areas supported

- sales
- revenue
- inventory / stock
- customer feedback
- competitor pricing
- product or category performance
- business trend analysis

The workflow is tuned for retail-analysis use cases and expects business semantics such as:

- revenue in SEK
- low stock based on reorder thresholds
- recent dates and trend comparisons
- category/product-level interpretation

---

## Conversation and session behavior

The UI stores conversation history in the Streamlit session state and includes a thread identifier for near-term request continuity.

Current behavior:

- user-specific history is kept within the local browser session,
- new conversation resets the thread and clears chat history,
- the app is designed for local, session-scoped continuity rather than remote durable memory.

This is suitable for local prototype and internal tool use. If you want production-scale conversation persistence, the next step would be:

- a persistent database-backed memory layer,
- durable checkpoints,
- user authentication,
- per-user storage and retention policies.

---

## Observability and tracing

Optional tracing is supported via LangSmith.

### Enable tracing

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=local-bi-assistant
```

### Why tracing is useful

- debug SQL generation mistakes,
- understand unhelpful or unstable prompts,
- monitor workflow routing,
- inspect LLM calls and tool usage,
- audit business analysis behavior in development and QA.

> LangSmith free-tier usage is possible, but a valid write-enabled key and project access are required. If tracing is not configured correctly, the app still works locally without it.

---

## How the workflow works

The workflow lives in the LangGraph orchestration layer and follows this logical flow:

1. Route the input
2. Load database schema
3. Generate a candidate SQL query
4. Validate query safety
5. Execute query as a safe read-only tool
6. Parse results into a DataFrame
7. Generate business insight summary
8. Generate a chart if the result is visualizable
9. Return final response

This is implemented in the BI workflow and is intentionally explicit rather than highly abstracted.

---

## Local development notes

### Running in development

```bash
poetry install
poetry run streamlit run app.py
```

The project can also be run directly with a standard Python virtual environment if preferred.

### Common checks

```bash
python -m compileall .
```

This can be used to catch syntax issues before running the app.

---

## Production considerations

This is a solid prototype and local internal tool, but for production deployment you should consider:

- persistent conversation storage
- user authentication and authorization
- role-based access controls
- query throttling
- structured audit logs
- monitoring dashboards
- stricter prompt safety layers
- business validation for generated SQL
- deployment behind a secure app server
- governance around model cost and output quality

---

## Limitations

- The app is designed around a local SQLite dataset and retail business patterns.
- The SQL generation layer depends on the database schema being well-structured.
- The chart generation is intentionally safe but not always perfect for every dataset shape.
- Business summaries are useful, but LLM-generated interpretation should be reviewed in production-critical workflows.
- The current conversational memory is local/session-scoped rather than durable enterprise-grade memory.

---

## Recommended next steps

If you want to move this toward a more production-grade state, the strongest next additions are:

1. Durable conversation memory with a database-backed checkpoint layer
2. User auth and tenant isolation
3. Structured logs and metrics
4. Prompt and tool governance controls
5. Query result caching
6. Advanced chart quality checks
7. UI improvements and analytics dashboard packaging

---

## License

This project currently does not declare a license in the repository metadata. If you plan to publish it publicly, add an appropriate open-source license such as MIT or Apache 2.0.

---

## Summary

The Local Business Intelligence Assistant is a practical, low-friction, local-first BI experience built on LangGraph, Streamlit, SQLite, and Gemini. It demonstrates how an LLM can safely translate natural-language business questions into validated read-only SQL, analyze results, and present the outcome in a business-friendly format.

It is best suited for:

- local analytics workflows,
- prototyping and demos,
- internal business intelligence usage,
- privacy-conscious analysis with no heavy cloud dependency.

If you want, I can also turn this into:

- a more executive-style GitHub README,
- a more technical engineering README,
- or a one-page product pitch README with screenshots and feature callouts.
