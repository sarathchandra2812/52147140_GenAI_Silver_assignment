from __future__ import annotations

import ast
import json
import logging
import os
import re
import typing

import pandas as pd
import plotly.express as px
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from data_access import (
    _get_latest_data_date,
    execute_read_query,
    get_database_schema,
    validate_select_query,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(funcName)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger("local_bi_langgraph.workflow")


class BIState(MessagesState):
    question: str
    route: str
    schema: str
    sql: str
    dataframe: pd.DataFrame
    chart: typing.Any
    summary: str
    response: str
    report_type: str
    sql_error: str
    sql_retry_count: int


@tool
def execute_business_query(query: str) -> str:
    """Execute one safe, read-only SQLite BI query and return JSON records."""
    logger.info("BI tool called query=%s", query)
    dataframe = execute_read_query(query)
    return dataframe.to_json(orient="records", date_format="iso")


TOOLS = [execute_business_query]


def get_llm():
    model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("Model initialization blocked: GOOGLE_API_KEY is not configured")
        raise RuntimeError("GOOGLE_API_KEY is required for model responses.")
    logger.info("Initializing ChatGoogleGenerativeAI model=%s", model_name)
    return ChatGoogleGenerativeAI(model=model_name, temperature=0)


def _extract_text(response: typing.Any) -> str:
    content = response.content if hasattr(response, "content") else response
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
    output = model_output.strip()
    output = re.sub(r"^```(?:sql)?\s*", "", output, flags=re.IGNORECASE)
    output = re.sub(r"\s*```$", "", output, flags=re.IGNORECASE).strip()
    if not output:
        logger.error("Model returned empty SQL")
        raise ValueError("The model returned an empty SQL response.")
    output = output.rstrip(";").strip()
    if not re.match(r"^(SELECT|WITH)\b", output, flags=re.IGNORECASE):
        logger.error("Model returned non-read-only SQL: %s", output)
        raise ValueError("The model did not return a SELECT/WITH query.")
    validated = validate_select_query(output)
    logger.info("SQL normalized and validated")
    return validated


def _tool_query_from_state(state: BIState) -> str:
    messages = state.get("messages", [])
    for message in reversed(messages):
        tool_calls = getattr(message, "tool_calls", [])
        if tool_calls:
            return str(tool_calls[-1].get("args", {}).get("query", ""))
    raise RuntimeError("The SQL generation node did not produce a tool query.")


def _looks_like_business_question(text: str) -> bool:
    lower = text.lower().strip()
    if not lower:
        return False
    if lower in {"hi", "hello", "hey", "help", "what can you do?", "what can you help me with?"}:
        return False
    keywords = [
        "sales",
        "revenue",
        "inventory",
        "stock",
        "customer",
        "feedback",
        "rating",
        "pricing",
        "competitor",
        "retailer",
        "product",
        "report",
        "weekly",
        "monthly",
        "month",
        "week",
        "category",
        "sentiment",
        "drop",
        "decline",
        "trend",
        "why",
        "opportunity",
        "recommendation",
        "problem",
        "risk",
    ]
    return any(keyword in lower for keyword in keywords)


def _is_report_request(text: str) -> bool:
    lower = text.lower().strip()
    report_keywords = [
        "weekly report",
        "monthly report",
        "business report",
        "executive summary",
        "summary report",
        "performance report",
        "report for",
    ]
    return any(keyword in lower for keyword in report_keywords)


def _is_diagnostic_request(text: str) -> bool:
    lower = text.lower().strip()
    diagnostic_keywords = [
        "why did",
        "why is",
        "drop",
        "decline",
        "problem",
        "risk",
        "opportunity",
        "recommendation",
        "root cause",
        "what caused",
    ]
    return any(keyword in lower for keyword in diagnostic_keywords)


def _build_fallback_chart(dataframe: pd.DataFrame) -> typing.Any | None:
    if dataframe.empty or len(dataframe.columns) < 2:
        return None

    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    date_columns = [column for column in dataframe.columns if "date" in column.lower()]

    if date_columns and numeric_columns:
        chart_df = dataframe.copy()
        chart_df[date_columns[0]] = pd.to_datetime(chart_df[date_columns[0]], errors="coerce")
        chart_df = chart_df.dropna(subset=[date_columns[0]])
        if not chart_df.empty:
            return px.line(chart_df, x=date_columns[0], y=numeric_columns[0], title=f"{numeric_columns[0]} over time")

    if numeric_columns:
        category_columns = [column for column in dataframe.columns if column not in numeric_columns][:1]
        if category_columns:
            plot_df = dataframe.copy()
            if len(plot_df) > 20:
                plot_df = plot_df.nlargest(20, numeric_columns[0])
            return px.bar(plot_df, x=category_columns[0], y=numeric_columns[0], title=f"{numeric_columns[0]} by {category_columns[0]}")

    if len(numeric_columns) >= 2:
        return px.scatter(dataframe, x=numeric_columns[0], y=numeric_columns[1], title=f"{numeric_columns[1]} vs {numeric_columns[0]}")

    return None


def _build_llm_chart(dataframe: pd.DataFrame) -> typing.Any | None:
    if dataframe.empty or len(dataframe.columns) < 2:
        logger.info("Skipping LLM chart generation: result has insufficient rows or columns")
        return None

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
)""".strip()

    logger.info("Requesting Plotly code from model rows=%d columns=%d", len(dataframe), len(dataframe.columns))
    try:
        response = get_llm().invoke([
            SystemMessage(content="Return only safe Plotly code assigned to fig."),
            HumanMessage(content=plot_prompt),
        ])
        raw_code = _extract_text(response).strip()
        clean_code = raw_code.replace("```python", "").replace("```", "").strip()
        tree = ast.parse(clean_code)
        fig_value: ast.AST | None = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "fig"
                for target in node.targets
            ):
                fig_value = node.value
                break

        if fig_value is None:
            logger.error("Plot model response did not assign fig")
            return None

        compiled_expression = compile(ast.Expression(fig_value), "<plotly_generated>", "eval")
        figure = eval(compiled_expression, {"__builtins__": {}}, {"df": dataframe, "px": px})
        logger.info("LLM Plotly chart generated successfully chart_created=%s", figure is not None)
        return figure
    except (SyntaxError, NameError, TypeError, ValueError, AttributeError) as exc:
        logger.error("LLM Plotly generation failed: %s", exc)
        return None


def route_input(state: BIState) -> BIState:
    messages = state.get("messages", [])
    last_user_message = ""
    for message in reversed(messages):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "type", "")
        if role in {"user", "human"}:
            last_user_message = (
                message.get("content", "")
                if isinstance(message, dict)
                else getattr(message, "content", "")
            )
            break
    is_business = _looks_like_business_question(last_user_message)
    if not is_business:
        route = "general"
    elif _is_report_request(last_user_message):
        route = "report"
    elif _is_diagnostic_request(last_user_message):
        route = "diagnostic"
    else:
        route = "business"

    logger.info("Question routed route=%s question=%r", route, last_user_message)

    return {
        **state,
        "question": last_user_message,
        "route": route,
        "report_type": "standard",
        "response": "I can help with sales, inventory, customer sentiment, competitor pricing, and business reports. Ask me something like: 'What was revenue last month?' or 'Which products are low on stock?'" if route == "general" else "",
    }


def fetch_schema(state: BIState) -> BIState:
    logger.info("Fetching database schema")
    schema = get_database_schema()
    logger.info("Database schema fetched schema_chars=%d", len(schema))
    return {**state, "schema": schema}


def schema_node(state: BIState) -> BIState:
    return fetch_schema(state)


def generate_sql_node(state: BIState) -> BIState:
    model = get_llm().bind_tools(TOOLS)
    latest_data_date = _get_latest_data_date()
    prompt = f"""
You are the BI agent for a local retail business dataset.

Database schema:
{state['schema']}

LATEST DATA DATE:
{latest_data_date}


User request:
{state['question']}

Rules:
- Use the execute_business_query tool to answer the question.
- The tool accepts only one safe read-only SELECT/WITH query.
- Use only tables and columns present in the schema.
- Use the latest date {latest_data_date}.
- Use total_revenue_sek for revenue and units_sold for units.
- Low stock means current_stock_units < reorder_level_units.
- Do not assume date + retailer + product_name is unique.
- Do not create many-to-many joins.
- Aggregate customer_feedback before joining it to sales aggregates.
- Aggregate competitor_pricing before joining it to sales aggregates.
- Do not blindly join tables only on product_name.
- When a question combines multiple tables with different grains, aggregate each table to the required business grain BEFORE joining.
- If the user names a retailer, filter/use that retailer.
- Never assume a retailer is "our business" unless the user explicitly says which one.
- For "why", "drop", "increase", or "change" questions, compare the relevant period with the previous comparable period and return a useful breakdown such as category, product, units sold, or revenue.
- Do not invent columns, metrics, or business facts.
- If the requested answer is a sales/revenue metric, do not join another table unless that table is actually required to answer the question.
- Previous SQL validation error, if any: {state.get('sql_error', '')}
"""
    response = model.invoke([
        SystemMessage(content="You are a safe, precise BI tool-calling agent."),
        HumanMessage(content=prompt),
    ])
    logger.info("BI agent model response tool_calls=%d", len(getattr(response, "tool_calls", [])))
    return {
        **state,
        "messages": [response],
        "sql_error": "",
        "sql_retry_count": state.get("sql_retry_count", 0),
    }


def validate_sql_node(state: BIState) -> BIState:
    try:
        sql = validate_select_query(_tool_query_from_state(state))
        logger.info("Generated SQL passed validation")
        return {**state, "sql": sql, "sql_error": ""}
    except (RuntimeError, ValueError) as exc:
        retry_count = state.get("sql_retry_count", 0) + 1
        logger.error("Generated SQL failed validation retry=%d error=%s", retry_count, exc)
        return {
            **state,
            "sql_error": str(exc),
            "sql_retry_count": retry_count,
        }


def dataframe_node(state: BIState) -> BIState:
    tool_message = next(
        (message for message in reversed(state.get("messages", [])) if isinstance(message, ToolMessage)),
        None,
    )
    if tool_message is None:
        raise RuntimeError("The MCP tool did not return a result.")
    records = json.loads(str(tool_message.content))
    dataframe = pd.DataFrame.from_records(records)
    logger.info("Dataframe node parsed rows=%d columns=%d", len(dataframe), len(dataframe.columns))
    return {**state, "dataframe": dataframe}


def insights_node(state: BIState) -> BIState:
    dataframe = state.get("dataframe")
    if dataframe is None or dataframe.empty:
        logger.info("Query returned no rows")
        summary = "I checked the data and there were no matching rows for that request. Try widening the date range or reframing the question."
        return {"summary": summary}

    preview = dataframe.head(20).to_dict(orient="records")
    model = get_llm()
    route = state.get("route", "business")
    report_hint = "Generate a concise, executive-style answer with the following sections: 1) Key finding, 2) Why it matters, 3) Recommended action."
    if route == "diagnostic":
        report_hint = "Generate a diagnostic insight with the following sections: 1) Likely cause, 2) Business impact, 3) Recommended action, 4) Risk level."
    elif route == "report":
        report_hint = "Generate a short performance report with the following sections: 1) Executive summary, 2) KPI highlights, 3) Risks or issues, 4) Action plan."

    prompt = f"""
User question: {state['question']}

Result preview:
{json.dumps(preview, default=str)}

{report_hint}

Keep the answer practical and useful for a small business owner. Mention if the result is an aggregate or a detail view. all prices are in SEK.
"""
    logger.info("Generating business summary route=%s", route)
    response = model.invoke([
        SystemMessage(content="You are a helpful retail business analyst and consultant."),
        HumanMessage(content=prompt),
    ])
    summary = _extract_text(response)
    logger.info("Business summary generated summary_chars=%d", len(summary))
    return {"summary": summary}


def plotting_node(state: BIState) -> BIState:
    dataframe = state.get("dataframe")
    if dataframe is None or dataframe.empty:
        return {"chart": None}
    chart = _build_llm_chart(dataframe)
    if chart is None:
        chart = _build_fallback_chart(dataframe)
        logger.info("Using fallback chart chart_created=%s", chart is not None)
    return {"chart": chart}


def finalize_response(state: BIState) -> BIState:
    route = state.get("route", "general")
    if state.get("sql_error") and not state.get("dataframe"):
        final_text = f"I could not validate the generated query after {state.get('sql_retry_count', 0)} attempt(s): {state['sql_error']}"
    elif route == "general":
        final_text = state.get("response", "I can help with business questions about sales, stock, ratings, and pricing.")
    else:
        final_text = state.get("summary", "I ran the analysis and here is the result.")

    if route in {"business", "diagnostic", "report"} and isinstance(final_text, str):
        final_text = final_text.strip()
        # if "Recommended action" not in final_text and "Recommended actions" not in final_text:
        #     final_text = final_text + "\n\nRecommended action: focus on the most important metric, validate the root cause with one more slice of data, and prioritize the highest-impact corrective step."

    logger.info("Response finalized route=%s response_chars=%d", route, len(final_text))

    return {
        **state,
        "messages": [{"role": "assistant", "content": final_text}],
        "response": final_text,
        "dataframe": state.get("dataframe", pd.DataFrame()),
    }


def create_bi_graph() -> StateGraph:
    logger.info("Creating LangGraph workflow with checkpoint memory")
    workflow = StateGraph(BIState)
    workflow.add_node("route_input", route_input)
    workflow.add_node("schema", schema_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("validate_sql", validate_sql_node)
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.add_node("dataframe", dataframe_node)
    workflow.add_node("insights", insights_node)
    workflow.add_node("plotting", plotting_node)
    workflow.add_node("finalize_response", finalize_response)

    workflow.set_entry_point("route_input")
    workflow.add_conditional_edges(
        "route_input",
        lambda state: state.get("route", "business"),
        {
            "general": "finalize_response",
            "business": "schema",
            "diagnostic": "schema",
            "report": "schema",
        },
    )
    workflow.add_edge("schema", "generate_sql")
    workflow.add_edge("generate_sql", "validate_sql")
    workflow.add_conditional_edges(
        "validate_sql",
        lambda state: "generate_sql" if state.get("sql_error") and state.get("sql_retry_count", 0) < 2 else "tools",
        {"generate_sql": "generate_sql", "tools": "tools"},
    )
    workflow.add_edge("tools", "dataframe")
    workflow.add_edge("dataframe", "insights")
    workflow.add_edge("dataframe", "plotting")
    workflow.add_edge("insights", "finalize_response")
    workflow.add_edge("plotting", "finalize_response")

    graph = workflow.compile(
        checkpointer=MemorySaver(serde=JsonPlusSerializer(pickle_fallback=True))
    )
    logger.info("LangGraph workflow compiled successfully")

    # try:
    #     from pathlib import Path
    #     image_path = Path(__file__).resolve().parent / "langgraph_workflow.png"
    #     image_bytes = graph.get_graph().draw_mermaid_png()
    #     image_path.write_bytes(image_bytes)
    #     logger.info("Saved workflow image to %s", image_path)
    # except Exception as exc:  # pragma: no cover - best effort image export
    #     logger.warning("Could not save workflow image: %s", exc)

    return graph
