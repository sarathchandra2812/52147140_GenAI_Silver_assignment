from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
import sys
import threading
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

BASE_DIR = Path(__file__).resolve().parent
MCP_SERVER_PATH = BASE_DIR / "mcp_bi_server.py"
# DB_PATH = BASE_DIR / "smb_intelligence.db"
DB_PATH = BASE_DIR / "sweden_data.db"
# MODEL_NAME = "qwen2.5-coder:3b"
# MODEL_NAME = "qwen3:4b"
# MODEL_NAME = "gemini-3.6-flash"
# MODEL_NAME = "gemini-3.5-flash-lite"
MODEL_NAME = "gemini-3.1-flash-lite"
QUERY_ERROR_PREFIX = "QUERY_ERROR:"
SCHEMA_ERROR_PREFIX = "SCHEMA_ERROR:"
MAX_INSIGHT_ROWS = 100
CAPABILITIES_MESSAGE = "I'm a Business Intelligence assistant for this store's data. Ask me about sales and revenue, " "inventory or restocking needs, customer ratings and sentiment, or competitor pricing \u2014 " "or use the Weekly/Monthly report buttons in the sidebar."

T = TypeVar("T")
load_dotenv()  # This loads variables from your .env file into the environment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(funcName)s:%(lineno)d - %(message)s")
logger = logging.getLogger("local_bi_assistant")

st.set_page_config(page_title="Local BI Assistant", page_icon="📊", layout="wide")


@st.cache_resource
def get_llm() -> ChatGoogleGenerativeAI:
    """Keep the local Ollama client warm across Streamlit reruns for fast responses."""
    logger.info("Initializing Model client (model=%s)", MODEL_NAME)
    # return ChatOllama(model=MODEL_NAME, temperature=0, keep_alive="30m")
    # return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)


def _tool_text(result: Any) -> str:
    """Extract textual content from an MCP CallToolResult defensively."""
    if getattr(result, "isError", False):
        logger.error("MCP tool call reported an execution error: %s", result)
        raise RuntimeError("The MCP tool reported an execution error.")

    text_parts = [block.text for block in getattr(result, "content", []) if isinstance(getattr(block, "text", None), str)]
    if text_parts:
        return "\n".join(text_parts).strip()

    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return json.dumps(structured)

    logger.error("MCP tool call returned no readable content: %s", result)
    raise RuntimeError("The MCP tool returned no readable content.")


def _message_text(response: Any) -> str:
    """Normalize a LangChain response's content into plain text."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()

    return str(content).strip()


def _normalize_sql(model_output: str) -> str:
    """
    Normalize and validate Gemini's SQL response.

    The MCP server performs the final read-only validation, but we reject
    obviously malformed model output here before sending it to MCP.
    """
    output = model_output.strip()

    # Remove common markdown fencing.
    output = re.sub(r"^```(?:sql)?\s*", "", output, flags=re.IGNORECASE)
    output = re.sub(
        r"\s*```$",
        "",
        output,
        flags=re.IGNORECASE,
    ).strip()

    if not output:
        raise ValueError("The model returned an empty SQL response.")

    # Remove a trailing semicolon only.
    output = output.rstrip(";").strip()

    # SQL must begin directly with SELECT or WITH.
    if not re.match(
        r"^(SELECT|WITH)\b",
        output,
        flags=re.IGNORECASE,
    ):
        logger.error(
            "Model returned non-SQL output: %s",
            output,
        )
        raise ValueError("The model did not return a SELECT/WITH query.")

    # Reject multiple statements.
    cleaned = re.sub(
        r"/\*.*?\*/",
        " ",
        output,
        flags=re.DOTALL,
    )
    cleaned = re.sub(
        r"--[^\n]*",
        " ",
        cleaned,
    )

    statements = [statement.strip() for statement in cleaned.split(";") if statement.strip()]

    if len(statements) != 1:
        logger.error(
            "Model returned multiple SQL statements: %s",
            output,
        )
        raise ValueError("Only one SQL statement is allowed.")

    return output


# def _build_chart(dataframe: pd.DataFrame) -> Any | None:
#     """Create a simple Plotly chart from the query result without executing model-generated code."""
#     if dataframe.empty or len(dataframe.columns) < 2:
#         return None

#     columns = set(dataframe.columns)

#     # Time-series result
#     date_columns = [column for column in dataframe.columns if "date" in column.lower()]

#     numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()

#     if date_columns and numeric_columns:
#         x_column = date_columns[0]
#         y_column = numeric_columns[0]

#         chart_df = dataframe.copy()
#         chart_df[x_column] = pd.to_datetime(
#             chart_df[x_column],
#             errors="coerce",
#         )
#         chart_df = chart_df.dropna(subset=[x_column])

#         if not chart_df.empty:
#             return px.line(
#                 chart_df,
#                 x=x_column,
#                 y=y_column,
#                 title=f"{y_column} over time",
#             )

#     # Categorical + numeric result
#     categorical_columns = dataframe.select_dtypes(exclude="number").columns.tolist()

#     if categorical_columns and numeric_columns:
#         x_column = categorical_columns[0]
#         y_column = numeric_columns[0]

#         chart_df = dataframe.copy()

#         # Prevent unreadable charts with very large categorical results.
#         if len(chart_df) > 20:
#             chart_df = chart_df.nlargest(
#                 20,
#                 y_column,
#             )

#         return px.bar(
#             chart_df,
#             x=x_column,
#             y=y_column,
#             title=f"{y_column} by {x_column}",
#         )

#     # Two numeric columns
#     if len(numeric_columns) >= 2:
#         return px.scatter(
#             dataframe,
#             x=numeric_columns[0],
#             y=numeric_columns[1],
#             title=f"{numeric_columns[1]} vs {numeric_columns[0]}",
#         )

#     return None


def _prepare_dataframe_for_insight(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Limit the amount of query data sent to Gemini for insight generation.

    Keep the first rows for small results and cap large results to avoid
    unnecessary prompt/token usage.
    """
    if dataframe.empty:
        return dataframe

    if len(dataframe) <= MAX_INSIGHT_ROWS:
        return dataframe

    logger.info(
        "Limiting insight input from %d rows to %d rows",
        len(dataframe),
        MAX_INSIGHT_ROWS,
    )

    return dataframe.head(MAX_INSIGHT_ROWS)


def _unwrap_exception(exc: BaseException) -> BaseException:
    """Unwrap anyio TaskGroup exception groups down to the underlying error."""
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc


async def _fetch_schema(session: ClientSession, cached_schema: str | None) -> str:
    """Fetch schema, using an optional caller-provided cache value."""
    if isinstance(cached_schema, str) and cached_schema:
        logger.info("Using cached database schema")
        return cached_schema

    logger.info("Fetching database schema from MCP server")
    schema_result = await session.call_tool("get_database_schema", {})
    schema = _tool_text(schema_result)
    if schema.startswith(SCHEMA_ERROR_PREFIX):
        logger.error("Schema retrieval failed: %s", schema)
        raise RuntimeError(schema)

    logger.info("Schema fetched and cached successfully")
    return schema


async def _get_latest_data_date(session: ClientSession) -> str:
    """Return the latest available sales date from the database."""
    result = await session.call_tool(
        "execute_read_query",
        {
            "query": """
                SELECT MAX(date) AS latest_date
                FROM sales_data
            """.strip()
        },
    )

    result_text = _tool_text(result)

    if result_text.startswith(QUERY_ERROR_PREFIX):
        raise RuntimeError(result_text)

    records = json.loads(result_text)

    if not records or not records[0].get("latest_date"):
        raise RuntimeError("Could not determine the latest available data date.")

    return str(records[0]["latest_date"])[:10]


async def _generate_capabilities_response(
    user_prompt: str,
    schema: str,
    llm: ChatGoogleGenerativeAI,
) -> str:
    """Generate a concise, schema-aware response for non-data questions."""
    prompt = f"""You are a friendly Business Intelligence assistant.

User message: {user_prompt}

Available database schema:
{schema}

You MUST reply with EXACTLY this structure. Do not leave any bullet points out.

Hi! I can analyze your business data to answer questions about sales, pricing, inventory, and customer feedback.

Try one of these questions:
* (Write one specific question about sales or revenue using the schema)
* (Write one specific question about inventory or reorder levels using the schema)
* (Write one specific question about customer ratings or competitor pricing using the schema)

Do not write SQL. Keep the complete response under 100 words.
"""
    try:
        logger.info("Requesting capabilities response from model...")
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        message = _message_text(response)
        logger.info("Model responded to capabilities request. Empty response: %s", not bool(message.strip()))
    except (TimeoutError, OSError, RuntimeError, ValueError) as exc:
        logger.error("Dynamic capabilities response failed: %s", exc or type(exc).__name__)
        return CAPABILITIES_MESSAGE

    if not message:
        logger.info("Dynamic capabilities response was empty; using fallback")
        return CAPABILITIES_MESSAGE

    logger.info("Dynamic capabilities response generated successfully")
    return message


def _is_non_bi_message(user_prompt: str) -> bool:
    """
    Detect obvious greetings/capability questions locally.

    This avoids spending a Gemini request on questions that do not require
    database analysis.
    """
    text = user_prompt.strip().lower()

    if not text:
        return True

    greetings = {
        "hi",
        "hello",
        "hey",
        "hello there",
        "hi there",
        "good morning",
        "good afternoon",
        "good evening",
    }

    if text in greetings:
        return True

    capability_patterns = (
        r"\bwhat can you do\b",
        r"\bwhat do you do\b",
        r"\bhow can you help\b",
        r"\bwhat can you help me with\b",
        r"\bwhat are your capabilities\b",
        r"\bwhat kind of questions can i ask\b",
        r"\bhelp me\b$",
        r"\bhi there\b$",
    )

    return any(re.search(pattern, text) for pattern in capability_patterns)


async def run_bi_agent(
    user_prompt: str,
    llm: ChatGoogleGenerativeAI,
    cached_schema: str | None,
) -> tuple[pd.DataFrame, str, str, Any | None, str]:
    """Run schema retrieval, Text-to-SQL, execution, and insight generation over MCP stdio."""
    try:
        return await _run_bi_agent(user_prompt, llm, cached_schema)
    except BaseException as exc:  # unwrap TaskGroup wrapper for a clear message
        unwrapped = _unwrap_exception(exc)
        logger.error("BI agent failed for prompt %r: %s", user_prompt, unwrapped or type(unwrapped).__name__)
        raise unwrapped from exc


async def _run_bi_agent(
    user_prompt: str,
    llm: ChatGoogleGenerativeAI,
    cached_schema: str | None,
) -> tuple[pd.DataFrame, str, str, Any | None, str]:

    server_parameters = StdioServerParameters(
        # sys.executable guarantees the MCP subprocess uses this exact environment.
        command=sys.executable,
        args=[str(MCP_SERVER_PATH)],
        cwd=str(BASE_DIR),
    )

    async with (
        stdio_client(server_parameters) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        logger.info("MCP session initialized for prompt: %s", user_prompt)

        schema = await _fetch_schema(session, cached_schema)
        latest_data_date = await _get_latest_data_date(session)

        logger.info("Latest available dataset date: %s", latest_data_date)

        if _is_non_bi_message(user_prompt):
            logger.info("Detected non-BI message locally: %r", user_prompt)

            message = await _generate_capabilities_response(user_prompt, schema, llm)

            return (pd.DataFrame(), "", message, None, schema)

        sql_prompt = f"""
            You are a BI SQL expert using SQLite.

            SCHEMA:
            {schema}

            QUESTION:
            {user_prompt}

            LATEST DATA DATE:
            {latest_data_date}

            Rules:
            - Output ONLY one valid SQLite SELECT/WITH query.
            - Use only tables and columns present in the schema.
            - Use total_revenue_sek for revenue and units_sold for units.
            - Low stock means current_stock_units < reorder_level_units.
            - Do not assume date + retailer + product_name is unique.
            - Do not create many-to-many joins.
            - Aggregate customer_feedback before joining it to sales aggregates.
            - Aggregate competitor_pricing before joining it to sales aggregates.
            - Do not blindly join tables only on product_name.
            - When a question combines multiple tables with different grains, aggregate each table to the required business grain BEFORE joining.
            - Use the actual date columns and interpret relative periods from {latest_data_date}.
            - If the user names a retailer, filter/use that retailer.
            - Never assume a retailer is "our business" unless the user explicitly says which one.
            - For "why", "drop", "increase", or "change" questions, compare the relevant period with the previous comparable period and return a useful breakdown such as category, product, units sold, or revenue.
            - Do not invent columns, metrics, or business facts.
            - If the requested answer is a sales/revenue metric, do not join another table unless that table is actually required to answer the question.

            Return SQL only.
        """.strip()

        logger.info("Requesting SQL generation from model...")
        sql_response = await llm.ainvoke([HumanMessage(content=sql_prompt)])
        raw_sql_output = _message_text(sql_response)
        logger.info("Model responded to SQL generation request. Empty response: %s", not bool(raw_sql_output.strip()))

        generated_sql = _normalize_sql(raw_sql_output)
        logger.info("Generated SQL: %s", generated_sql)

        query_result = await session.call_tool("execute_read_query", {"query": generated_sql})
        result_text = _tool_text(query_result)

        if result_text.startswith(QUERY_ERROR_PREFIX):
            # Small local models occasionally invent columns; give the model one chance to self-correct.
            logger.info("Query failed, attempting self-correction: %s", result_text)
            correction_prompt = f"""
                You are fixing a failed SQLite BI query, all prices are in SEK.

                SCHEMA:
                {schema}

                QUESTION:
                {user_prompt}

                LATEST DATA DATE:
                {latest_data_date}

                FAILED QUERY:
                {generated_sql}

                DATABASE ERROR:
                {result_text}

                Fix the query so that it correctly answers the user's question.

                Rules:
                - Output ONLY one SQLite SELECT/WITH query.
                - Use only tables and columns present in the schema.
                - Fix the actual SQL error; do not change the business meaning.
                - Revenue = SUM(total_revenue_sek).
                - Units = SUM(units_sold).
                - Do not assume date + retailer + product_name is unique.
                - Do not create many-to-many joins.
                - Aggregate customer_feedback before joining it to sales aggregates.
                - Aggregate competitor_pricing before joining it to sales aggregates.
                - Do not blindly join tables only on product_name.
                - If multiple tables have different grains, aggregate them to the required grain before joining.
                - Do not join another table unless it is required to answer the question.
                - Relative dates must use the latest available data date: {latest_data_date}.
                - Do not assume a retailer is "our business".
                - If the user names a retailer, use that retailer.
                - Do not invent columns or metrics.
                - Do not output markdown, explanations, comments, or JSON.

                Return ONLY the corrected SQL query.
            """.strip()
            
            logger.info("Requesting SQL self-correction from model...")
            correction_response = await llm.ainvoke([HumanMessage(content=correction_prompt)])
            correction_text = _message_text(correction_response)
            logger.info("Model responded to SQL self-correction request. Empty response: %s", not bool(correction_text.strip()))
            
            generated_sql = _normalize_sql(correction_text)
            logger.info("Self-corrected SQL: %s", generated_sql)

            query_result = await session.call_tool("execute_read_query", {"query": generated_sql})
            result_text = _tool_text(query_result)

        if result_text.startswith(QUERY_ERROR_PREFIX):
            logger.error("Query failed after self-correction attempt: %s", result_text)
            raise RuntimeError(result_text)

    records = json.loads(result_text)
    if not isinstance(records, list):
        raise TypeError("The MCP server returned JSON in an unexpected shape.")
    dataframe = pd.DataFrame.from_records(records)
    logger.info("Query returned %d row(s)", len(dataframe))

    if dataframe.empty:
        summary = "The query ran successfully but returned no rows for this question."
        logger.info("Query returned no rows for prompt: %s", user_prompt)
    else:

        insight_dataframe = _prepare_dataframe_for_insight(dataframe)

        insight_prompt = f"""
            You are a business analyst explaining database results to a small business owner, all prices are in SEK.

            QUESTION:
            {user_prompt}

            QUERY RESULT:
            {insight_dataframe.to_markdown(index=False)}

            RULES:
            - Answer the user's question directly using only the query result.
            - Use only numbers, dates, products, categories, and facts shown in the result.
            - Do not invent, estimate, or guess missing values.
            - Do not recalculate metrics unless the required values are explicitly present.
            - Do not claim causation when the data only shows correlation.
            - Keep the response concise, specific, and actionable.
            - Do not mention SQL, databases, prompts, or the model.

            OUTPUT FORMAT:

            **📊 Summary**
            Briefly explain what the data shows and answer the question.

            **🔎 Key Insight**
            State the most important business finding supported by the data.

            **💡 Recommendation**
            Give one practical action based only on the available evidence.
        """.strip()

        logger.info("Requesting business insight from model...")
        insight_response = await llm.ainvoke([HumanMessage(content=insight_prompt)])
        summary = _message_text(insight_response) or "No additional insight was generated."
        logger.info("Model responded to business insight request. Empty response: %s", summary == "No additional insight was generated.")

    plot_fig: Any | None = None
    if len(dataframe) > 1 and len(dataframe.columns) >= 2:
        data_csv = dataframe.head(50).to_csv(index=False)
        plot_prompt = f"""
            You are a data visualization expert using Plotly Express, all prices are in SEK.

            DATA COLUMNS:
            {list(dataframe.columns)}

            Use column names exactly as provided in DATA COLUMNS.

            DATA PREVIEW:
            {data_csv}

            Create the single most useful business chart for this data.

            RULES:
            1. Output ONLY ONE Python expression assigned directly to `fig`.
            2. Do not create intermediate variables such as `df_grouped`.
            3. Do not use multiple statements.
            4. Do not use imports, loops, functions, file access, or external data.
            5. The DataFrame is already available as `df`.
            6. Plotly Express is already imported as `px`.
            7. Use only columns that exist in the DATA COLUMNS.
            8. Choose:
            - date + numeric → line chart
            - category/product + numeric → bar chart
            - two numeric variables → scatter chart
            - small part-to-whole data → pie chart
            - give the plot labels meaningful
            9. Use the most relevant business metric for the y-axis.
            10. Do not include unrelated numeric columns.
            11. For rankings, show the top 10–20 rows using `df.nlargest(...)` when appropriate.
            12. Sort the chart meaningfully.
            13. Use a clear business-friendly title.
            14. Do not modify `df`.
            15. IF the data contains multiple categories or groups (e.g., multiple products or categories over time), you MUST use the `color` parameter in Plotly Express to differentiate them and generate a legend (e.g., `color='category'`).
            16. If no useful chart is possible, output exactly:
                fig = None

            Examples:

            fig = px.bar(
                df.nlargest(10, 'total_revenue'),
                x='product_name',
                y='total_revenue',
                title='Top 10 Products by Revenue'
            )

            fig = px.line(
                df,
                x='date',
                y='total_revenue',
                title='Revenue Over Time'
            )
        """.strip()
        
        logger.info("Data is multi-row/column. Requesting Plotly code from model...")
        try:
            plot_response = await llm.ainvoke([HumanMessage(content=plot_prompt)])
            raw_code = _message_text(plot_response).strip()
            logger.info("Model responded to Plotly request. Empty response: %s", not bool(raw_code))
            
            clean_code = raw_code.replace("```python", "").replace("```", "").strip()

            tree = ast.parse(clean_code)
            fig_value: ast.AST | None = None
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "fig":
                            fig_value = node.value
                            break
                if fig_value is not None:
                    break

            if fig_value is None:
                logger.info("Plot code missing required fig assignment")
                plot_fig = None
            else:
                expression = ast.Expression(fig_value)
                compiled_expression = compile(expression, "<plotly_generated>", "eval")
                plot_fig = eval(compiled_expression, {"__builtins__": {}}, {"df": dataframe, "px": px})
            if plot_fig is None:
                logger.info("Plot generation returned code without a fig assignment")
        except (SyntaxError, NameError, TypeError, ValueError, AttributeError) as exc:
            logger.error("Plot generation failed or was bypassed: %s", exc or type(exc).__name__)
            plot_fig = None
    else:
        logger.info("Single scalar or single-row result detected. Skipping plot generation.")

    logger.info("BI agent completed successfully for prompt: %s", user_prompt)
    return dataframe, generated_sql, summary, plot_fig, schema


class _BackgroundLoop:
    """Owns one event loop for the app's lifetime.

    ChatGoogleGenerativeAI lazily binds its internal async HTTP client to whichever loop is
    running on first use. Since get_llm() is cached across Streamlit reruns,
    spinning up a fresh asyncio.run() loop per request left that client bound
    to a loop that had already been closed, raising "Event loop is closed" on
    the next call. Running everything on one persistent loop avoids that.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="bi-agent-loop", daemon=True)
        self._thread.start()
        logger.info("Started persistent background event loop thread")

    def run(self, factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
        future = asyncio.run_coroutine_threadsafe(factory(), self._loop)
        return future.result()


@st.cache_resource
def _get_background_loop() -> _BackgroundLoop:
    return _BackgroundLoop()


def _run_async(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run async MCP/LLM work on the app's persistent background loop."""
    return _get_background_loop().run(factory)


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        plot_fig = message.get("plot_fig")
        if plot_fig is not None:
            st.plotly_chart(plot_fig, width="stretch")

        dataframe = message.get("dataframe")
        sql = message.get("sql")
        if sql or (isinstance(dataframe, pd.DataFrame) and not dataframe.empty):
            with st.expander("View SQL & raw data", expanded=False):
                if sql:
                    st.code(sql, language="sql")
                if isinstance(dataframe, pd.DataFrame):
                    st.dataframe(dataframe, width="stretch")


def main() -> None:
    logger.info("Streamlit app rerun started")
    llm = get_llm()
    st.title("Local Business Intelligence Assistant")
    st.caption(f"MCP stdio + SQLite + Gemini ({MODEL_NAME}) + deterministic BI analytics")

    with st.sidebar:
        st.header("System")
        st.write(f"Database: `{DB_PATH.name}`")
        st.write("Status: " + ("Ready" if DB_PATH.exists() else "Missing - run database_builder.py"))
        schema_cache = st.session_state.get("schema_cache")
        if isinstance(schema_cache, str):
            with st.expander("View database schema", expanded=False):
                st.code(schema_cache, language="sql")

        st.header("Automated Reports & Queries")
        st.write("Trigger an AI-generated business report or insight:")
        
        reports = {
            "Weekly Report": "Generate a weekly business report summarizing total revenue, units sold, the top performing category, and low stock items for the past 7 days.",
            "Monthly Report": "Generate a monthly business report summarizing total revenue, units sold, the top performing category, and low stock items for the past 30 days.",
        }
        
        basic_queries = {
            "Inventory Check": "Which products are currently low on stock based on their reorder levels?",
            "Revenue Trend": "Show me a trend of daily revenue over the last 30 days.",
            "Top Categories": "Compare the total revenue across different product categories.",
        }

        advanced_queries = {
            "Pricing vs Feedback": "Are there any products with an average rating below 3.0 that we are pricing higher than our competitors?",
            "Category Deep Dive": "Show me the total revenue, average customer rating, and average competitor price for each product category.",
            "WoW Growth": "Calculate the week-over-week growth rate for total revenue in SEK.",
            "Price vs Sales Correlation": "Show me a comparison of the average price versus total units sold for all products.",
            "Retailer Performance": "Which retailer sold the most units in the last 60 days, and what was their best-selling product?",
            "Performance Drop Analysis": "Why did our revenue drop this week compared to last week? Compare the performance of our top 3 categories."
        }

        st.subheader("Reports")
        for label, prompt_text in reports.items():
            if st.button(label, use_container_width=True, key=f"btn_{label}"):
                st.session_state.sidebar_prompt = prompt_text
                
        with st.expander("Basic Queries", expanded=True):
            for label, prompt_text in basic_queries.items():
                if st.button(label, use_container_width=True, key=f"btn_{label}"):
                    st.session_state.sidebar_prompt = prompt_text

        with st.expander("Advanced Analytics", expanded=False):
            for label, prompt_text in advanced_queries.items():
                if st.button(label, use_container_width=True, key=f"btn_{label}"):
                    st.session_state.sidebar_prompt = prompt_text

    _init_state()
    for message in st.session_state.messages:
        _render_message(message)

    prompt = st.chat_input("Ask about sales, inventory, feedback, or competitor pricing...")

    # Catch the prompt if a sidebar button was clicked
    if st.session_state.get("sidebar_prompt"):
        prompt = st.session_state.sidebar_prompt
        st.session_state.sidebar_prompt = None

    if not prompt:
        return

    logger.info("User submitted prompt: %s", prompt)
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    _render_message(user_message)

    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            schema_cache = st.session_state.get("schema_cache")
            dataframe, generated_sql, summary, plot_fig, schema = _run_async(lambda: run_bi_agent(prompt, llm, schema_cache if isinstance(schema_cache, str) else None))
            st.session_state["schema_cache"] = schema
            st.markdown(summary)
            if plot_fig is not None:
                st.plotly_chart(plot_fig, width="stretch")
            if generated_sql or not dataframe.empty:
                with st.expander("View SQL & raw data", expanded=False):
                    if generated_sql:
                        st.code(generated_sql, language="sql")
                    if not dataframe.empty:
                        st.dataframe(dataframe, width="stretch")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": summary,
                    "sql": generated_sql,
                    "dataframe": dataframe,
                    "plot_fig": plot_fig,
                }
            )
        except Exception as exc:  # noqa: BLE001
            error_message = f"I could not complete that request: {exc or type(exc).__name__}"
            logger.error("Chat request failed for prompt %r: %s", prompt, exc or type(exc).__name__)
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})


if __name__ == "__main__":
    main()