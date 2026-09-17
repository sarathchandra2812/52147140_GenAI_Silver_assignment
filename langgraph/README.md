# Local BI Assistant with LangGraph

This project is a complete rebuild of the business-intelligence assistant using LangGraph orchestration and a Streamlit interface.

## What it does

- Connects to the existing SQLite database used by the original BI project.
- Reuses the same source workbook when needed for schema/context refresh.
- Uses LangGraph native `MessagesState` and checkpoint memory for conversations.
- Accepts natural-language business questions and converts them into safe read-only SQL.
- Retrieves data from SQLite, summarizes findings, and generates a small Plotly chart when useful.
- Keeps conversations alive in a per-user thread using LangGraph memory.

## Reused data sources

The app checks for the original project files in the sibling folder:

- ../local_bi_analyst/sweden_data.db
- ../local_bi_analyst/smb_intelligence_real_world_sweden.xlsx

If the database exists, it is used directly. If not, the app will show a clear message.

## Setup

```bash
cd local_bi_langgraph
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
streamlit run app.py
```

## Environment

- `GOOGLE_API_KEY` is required for model responses.
- `GOOGLE_MODEL` optionally selects the Gemini model and defaults to `gemini-2.5-flash`.
- LangGraph owns graph state, routing, message history, and checkpoint memory.
- No Ollama or LangChain integration is used directly by this project.

## Notes

- The project intentionally does not modify anything in the original `local_bi_analyst` code.
- It uses the same business files and database rather than creating a second duplicate dataset.
