from __future__ import annotations

import logging
import os
import typing
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langsmith import traceable

from bi_workflow import create_bi_graph

logger = logging.getLogger("local_bi_langgraph.ui")

load_dotenv()

BASE_DIR = Path(__file__).resolve()
DB_PATH = BASE_DIR / "sweden_data.db"

PROMPT_GROUPS = {
    "Easy": [
        "What was our total revenue last month?",
        "Which products are currently low on stock based on their reorder levels?",
        "What is the average customer rating across all our products?",
        "How many total units of inventory do we currently have across all retailers?",
        "List all competitors who have priced our products below 500 SEK.",
    ],
    "Medium": [
        "Show me a trend of daily revenue over the last 30 days.",
        "What are our top 5 best-selling product categories by total units sold?",
        "Generate a monthly business performance report.",
        "Which retailer generated the highest total revenue last quarter?",
        "Show me the average competitor price for each product category.",
    ],
    "Complex": [
        "Which specific products drove the highest total revenue last month but are currently flagged as low on stock?",
        "Is there a correlation between the average customer rating and the total units sold for each product over the last 90 days?",
        "Are there any products with an average customer rating below 3.0 where our average selling price is higher than the average competitor price?",
        "Why did our total revenue drop this week compared to last week? Break down the performance differences by product category.",
        "How does our daily total revenue compare to the average daily competitor price changes for our top 3 best-selling products over the last month?",
    ],
}

st.set_page_config(page_title="Local BI Assistant", page_icon="📊", layout="wide")


@st.cache_resource
def get_workflow():
    logger.info("Initializing cached LangGraph workflow")
    return create_bi_graph()


def _trace_inputs(inputs: dict[str, typing.Any]) -> dict[str, typing.Any]:
    return {
        "prompt": inputs.get("prompt", ""),
        "thread_id": inputs.get("thread_id", ""),
    }


def _trace_outputs(output: dict[str, typing.Any]) -> dict[str, typing.Any]:
    output = output or {}
    dataframe = output.get("dataframe", pd.DataFrame())
    return {
        "route": output.get("route", ""),
        "sql": output.get("sql", ""),
        "response_chars": len(output.get("response", "")),
        "row_count": len(dataframe) if isinstance(dataframe, pd.DataFrame) else 0,
        "chart_created": output.get("chart") is not None,
    }


@traceable(
    name="streamlit_langgraph_request",
    run_type="chain",
    process_inputs=_trace_inputs,
    process_outputs=_trace_outputs,
)
def invoke_workflow_with_trace(
    workflow: typing.Any,
    state: dict[str, typing.Any],
    prompt: str,
    thread_id: str,
) -> dict[str, typing.Any]:
    return workflow.invoke(
        state,
        config={
            "configurable": {"thread_id": thread_id},
            "metadata": {"thread_id": thread_id, "prompt": prompt},
            "tags": ["streamlit", "local-bi", "langgraph"],
        },
    )


def _init_session() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"thread-{uuid.uuid4().hex[:8]}"


def _render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("dataframe") is not None and not message["dataframe"].empty:
            with st.expander("Query Result", expanded=False):
                st.dataframe(message["dataframe"], width="stretch")
        if message.get("sql"):
            with st.expander("SQL", expanded=False):
                st.code(message["sql"], language="sql")
        if message.get("chart") is not None:
            st.plotly_chart(message["chart"], width="stretch")


def main() -> None:
    _init_session()
    logger.info("Streamlit rerun started thread_id=%s", st.session_state.thread_id)
    st.title("Local Business Intelligence Assistant")
    st.caption("LangGraph + Streamlit + Local SQLite data")

    with st.sidebar:
        st.header("System")
        st.write(f"DB: {DB_PATH.name}")
        st.write("Status: Ready" if DB_PATH.exists() else "Database missing")
        st.write("Tracing: " + ("Enabled" if os.getenv("LANGSMITH_TRACING", "").lower() == "true" else "Disabled"))
        st.caption(f"Thread: `{st.session_state.thread_id}`")
        if st.button("New conversation", width="stretch"):
            st.session_state.thread_id = f"thread-{uuid.uuid4().hex[:8]}"
            st.session_state.history = []
            logger.info("Started new conversation thread_id=%s", st.session_state.thread_id)
            st.rerun()
        st.markdown("---")
        st.subheader("Prompt library")
        st.caption("Choose a prompt to run it in the chat.")
        for difficulty, prompts in PROMPT_GROUPS.items():
            with st.expander(f"{difficulty} prompts", expanded=difficulty == "Easy"):
                for index, prompt_text in enumerate(prompts):
                    if st.button(
                        prompt_text,
                        key=f"sample_{difficulty.lower()}_{index}",
                        width="stretch",
                    ):
                        st.session_state.pending_prompt = prompt_text

    for message in st.session_state.history:
        _render_message(message)

    prompt = st.chat_input("Ask about sales, inventory, customer sentiment, or competitor pricing...")
    if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if not prompt:
        return

    logger.info("User submitted prompt=%r", prompt)
    user_message = {"role": "user", "content": prompt}
    st.session_state.history.append(user_message)
    _render_message(user_message)

    workflow = get_workflow()
    with st.chat_message("assistant"), st.spinner("Analyzing your data..."):
        try:
                state = {
                    "messages": st.session_state.history,
                    "question": prompt,
                    "route": "",
                    "schema": "",
                    "sql": "",
                    "dataframe": pd.DataFrame(),
                    "chart": None,
                    "summary": "",
                    "response": "",
                    "report_type": "",
                    "sql_error": "",
                    "sql_retry_count": 0,
                }
                result = invoke_workflow_with_trace(workflow, state, prompt, st.session_state.thread_id)
                logger.info("Workflow completed route=%s", result.get("route", ""))
                final_answer = result.get("response", "I could not produce a response.")
                final_data = result.get("dataframe", pd.DataFrame())
                final_sql = result.get("sql", "")
                final_chart = result.get("chart")
                assistant_message = {
                    "role": "assistant",
                    "content": final_answer,
                    "sql": final_sql,
                    "dataframe": final_data,
                    "chart": final_chart,
                }
                st.session_state.history.append(assistant_message)
                _render_message(assistant_message)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.error("Workflow failed prompt=%r error=%s", prompt, exc)
            error_message = {
                "role": "assistant",
                "content": f"I couldn't complete that request: {exc}",
                "sql": "",
                "dataframe": pd.DataFrame(),
                "chart": None,
            }
            st.session_state.history.append(error_message)
            _render_message(error_message)


if __name__ == "__main__":
    main()
