from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "sweden_data.db"
READ_ONLY_PREFIXES = ("select", "with")

FORBIDDEN_SQL_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "attach",
    "detach",
    "pragma",
    "vacuum",
    "reindex",
    "replace",
)

APPLICATION_TABLES = (
    "sales_data",
    "inventory_status",
    "customer_feedback",
    "competitor_pricing",
)

mcp = FastMCP("Smart_BI_Server")


def _open_read_only_connection() -> sqlite3.Connection:
    """Open SQLite in OS-enforced read-only mode, with query_only as defense in depth."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Run database_builder.py first.")

    connection = sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON;")
    return connection


def _is_read_only_query(query: str) -> bool:
    """
    Validate that the query is exactly one read-only SELECT/WITH statement.

    SQLite is still opened in read-only mode and query_only is enabled,
    so this validation acts as an additional defense layer.
    """
    if not query or not query.strip():
        return False

    # Remove comments before validation.
    cleaned = re.sub( r"/\*.*?\*/", " ", query, flags=re.DOTALL)
    cleaned = re.sub( r"--[^\n]*", " ", cleaned).strip()

    # Reject multiple SQL statements.
    statements = [
        statement.strip()
        for statement in cleaned.split(";")
        if statement.strip()
    ]

    if len(statements) != 1:
        return False

    statement = statements[0]
    normalized = statement.lower().strip()

    # Only SELECT and WITH are allowed.
    if not normalized.startswith(READ_ONLY_PREFIXES):
        return False

    # Reject write/admin SQL keywords.
    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if re.search(
            rf"\b{re.escape(keyword)}\b",
            normalized,
        ):
            return False

    return True


@mcp.tool()
def get_database_schema() -> str:
    """
    Return database schema together with business/BI metadata.

    The metadata helps the LLM understand table grain, metric meaning,
    and safe join patterns instead of relying only on column names.
    """
    try:
        with _open_read_only_connection() as connection:
            placeholders = ",".join("?" for _ in APPLICATION_TABLES)

            rows = connection.execute(
                f"""
                SELECT name, sql
                FROM sqlite_master
                WHERE type = 'table'
                  AND name IN ({placeholders})
                  AND sql IS NOT NULL
                ORDER BY name;
                """,
                APPLICATION_TABLES,
            ).fetchall()

            blocks: list[str] = []

            for table_name, create_sql in rows:
                if not create_sql:
                    continue

                sample = pd.read_sql_query(
                    f"SELECT * FROM {table_name} LIMIT 3",
                    connection,
                )

                sample_json = sample.to_json(
                    orient="records",
                    date_format="iso",
                )

                blocks.append(
                    f"{str(create_sql).strip()};\n"
                    f"Sample rows from {table_name}: {sample_json}"
                )

    except (
        sqlite3.Error,
        FileNotFoundError,
        OSError,
        pd.errors.DatabaseError,
    ) as exc:
        return f"SCHEMA_ERROR: {exc}"

    if not blocks:
        return "SCHEMA_ERROR: No application tables were found."

    semantic_metadata = """
        BI RULES

        sales_data:
        - Grain: transaction-level.
        - Revenue = SUM(total_revenue_sek).
        - Do not assume product_name + date + retailer is unique.

        inventory_status:
        - Grain: inventory snapshot.
        - Low stock = current_stock_units < reorder_level_units.

        customer_feedback:
        - Grain: one review per row.
        - Multiple reviews can exist for the same product/date/retailer.
        - Aggregate feedback before joining to sales aggregates.

        competitor_pricing:
        - Grain: one competitor price observation.
        - Multiple competitors can exist for the same product/date.
        - Aggregate competitor_pricing before joining to sales aggregates.

        Retailer:
        - Never assume a retailer is "our business".
        - If the user names a retailer, use that retailer explicitly.

        Dates:
        - Relative periods must use the latest date available in the database,
        not the computer's current date.

        For cross-domain analysis:
        - Prevent joins from multiplying sales rows.
        - Aggregate each table to the required grain before joining.
        - Do not claim causation from correlation alone.
    """.strip()

    return semantic_metadata + "\n\n" + "\n\n".join(blocks)


@mcp.tool()
def execute_read_query(query: str) -> str:
    """Execute one read-only SQLite query and return the results as JSON records."""
    if not query or not query.strip():
        return "QUERY_ERROR: Query cannot be empty."

    if not _is_read_only_query(query):
        return "QUERY_ERROR: Only read-only SELECT or WITH queries are allowed."

    try:
        with _open_read_only_connection() as connection:
            dataframe = pd.read_sql_query(query, connection)
    except (sqlite3.Error, pd.errors.DatabaseError, ValueError, OSError) as exc:
        return f"QUERY_ERROR: {exc}"

    return dataframe.to_json(orient="records", date_format="iso")



if __name__ == "__main__":
    mcp.run(transport="stdio")
