from __future__ import annotations

import logging
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

import pandas as pd

logger = logging.getLogger("local_bi_langgraph.data_access")

ROOT_DIR = Path(__file__).resolve().parent
LEGACY_DIR = ROOT_DIR.parent / "local_bi_analyst"
DB_PATH = LEGACY_DIR / "sweden_data.db"
XLSX_PATH = LEGACY_DIR / "smb_intelligence_real_world_sweden.xlsx"

APPLICATION_TABLES = (
    "sales_data",
    "inventory_status",
    "customer_feedback",
    "competitor_pricing",
)

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


def get_db_path() -> Path:
    if DB_PATH.exists():
        logger.info("Using database path=%s", DB_PATH)
        return DB_PATH
    if (ROOT_DIR / "sweden_data.db").exists():
        logger.info("Using fallback database path=%s", ROOT_DIR / "sweden_data.db")
        return ROOT_DIR / "sweden_data.db"
    logger.error("No SQLite database found")
    raise FileNotFoundError(
        "No database found. Please keep the original local_bi_analyst database in place or add a compatible SQLite database."
    )


def get_excel_path() -> Path:
    if XLSX_PATH.exists():
        return XLSX_PATH
    if (ROOT_DIR / "smb_intelligence_real_world_sweden.xlsx").exists():
        return ROOT_DIR / "smb_intelligence_real_world_sweden.xlsx"
    raise FileNotFoundError("No source Excel workbook found.")


def _open_read_only_connection() -> sqlite3.Connection:
    path = get_db_path()
    logger.info("Opening read-only SQLite connection")
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON;")
    return connection


def validate_select_query(query: str) -> str:
    if not query or not query.strip():
        logger.error("Rejected empty SQL query")
        raise ValueError("The SQL query cannot be empty.")

    cleaned = re.sub(r"/\*.*?\*/", " ", query, flags=re.DOTALL)
    cleaned = re.sub(r"--[^\n]*", " ", cleaned)
    statements = [statement.strip() for statement in cleaned.split(";") if statement.strip()]

    if len(statements) != 1:
        logger.error("Rejected SQL with multiple statements")
        raise ValueError("Only one SQL statement is allowed.")

    statement = statements[0]
    normalized = statement.lower().strip()

    if not normalized.startswith(READ_ONLY_PREFIXES):
        logger.error("Rejected non-SELECT/WITH SQL")
        raise ValueError("Only SELECT or WITH queries are allowed.")

    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", normalized):
            logger.error("Rejected forbidden SQL keyword=%s", keyword)
            raise ValueError(f"Unsafe SQL keyword detected: {keyword}")

    return statement.strip().rstrip(";")


@lru_cache(maxsize=1)
def get_database_schema() -> str:
    logger.info("Loading database schema")
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
                sample = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 3", connection)
                blocks.append(
                    f"{str(create_sql).strip()};\nSample rows from {table_name}: {sample.to_json(orient='records', date_format='iso')}"
                )
    except (sqlite3.Error, FileNotFoundError, OSError, pd.errors.DatabaseError) as exc:
        logger.error("Database schema load failed: %s", exc)
        return f"SCHEMA_ERROR: {exc}"

    if not blocks:
        logger.error("No application tables found")
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
        - Never assume a retailer is 'our business'.
        - If the user names a retailer, use that retailer explicitly.

        Dates:
        - Relative periods must use the latest date available in the database, not the computer's current date.

        For cross-domain analysis:
        - Prevent joins from multiplying sales rows.
        - Aggregate each table to the required grain before joining.
        - Do not claim causation from correlation alone.
    """.strip()

    return semantic_metadata + "\n\n" + "\n\n".join(blocks)


def _get_latest_data_date() -> str:
    """Return the latest available sales date from the database."""
    query = "SELECT MAX(date) AS latest_date FROM sales_data"
    with _open_read_only_connection() as connection:
        records = connection.execute(query).fetchall()
    logger.info(f"Latest date: {records[0][0]}")
    latest_date = records[0][0] if records else None
    return latest_date


def execute_read_query(query: str) -> pd.DataFrame:
    validated = validate_select_query(query)
    logger.info("Executing validated read-only query")
    with _open_read_only_connection() as connection:
        dataframe = pd.read_sql_query(validated, connection)
    logger.info("Read-only query returned rows=%d columns=%d", len(dataframe), len(dataframe.columns))
    return dataframe


def get_data_summary() -> dict[str, object]:
    logger.info("Loading database summary")
    with _open_read_only_connection() as connection:
        row_counts = {}
        for table_name in APPLICATION_TABLES:
            row_counts[table_name] = int(
                connection.execute(f"SELECT COUNT(*) FROM \"{table_name}\"").fetchone()[0]
            )
    summary = {"database": str(get_db_path()), "tables": row_counts}
    logger.info("Database summary loaded tables=%s", sorted(row_counts))
    return summary
