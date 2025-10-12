"""Generic table loader for seeding cfg.* rows from structured files."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import yaml
from sqlalchemy import text

from freshbot.db import get_engine

from .registry_loader import resolve_env_tokens

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    nullable: bool
    has_default: bool


@dataclass(frozen=True)
class TableSchema:
    schema: str
    table: str
    columns: Dict[str, ColumnInfo]

    @property
    def required_columns(self) -> List[str]:
        return [
            name
            for name, info in self.columns.items()
            if not info.nullable and not info.has_default
        ]


def parse_table_name(identifier: str) -> tuple[str, str]:
    if "." in identifier:
        schema, table = identifier.split(".", 1)
        return schema.strip(), table.strip()
    return "public", identifier.strip()


def fetch_table_schema(table_name: str) -> TableSchema:
    schema_name, table = parse_table_name(table_name)
    engine = get_engine()
    sql = text(
        """
        SELECT column_name, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name = :table
        ORDER BY ordinal_position
        """
    )
    with engine.connect() as connection:
        rows = connection.execute(sql, {"schema": schema_name, "table": table}).mappings().all()
    if not rows:
        raise RuntimeError(f"Table '{schema_name}.{table}' does not exist or is inaccessible")
    columns: Dict[str, ColumnInfo] = {}
    for row in rows:
        name = row["column_name"]
        nullable = str(row["is_nullable"]).upper() == "YES"
        has_default = row["column_default"] is not None
        columns[name] = ColumnInfo(name=name, nullable=nullable, has_default=has_default)
    return TableSchema(schema=schema_name, table=table, columns=columns)


def load_rows(path: Path) -> List[Mapping[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return []
    if isinstance(payload, MutableMapping):
        if "rows" in payload:
            payload = payload["rows"]
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise TypeError("Table loader expects a list of row mappings")
    records: List[Mapping[str, Any]] = []
    for row in payload:
        if not isinstance(row, MutableMapping):
            raise TypeError("Table loader rows must be mappings")
        resolved = resolve_env_tokens(dict(row))
        records.append(resolved)
    return records


def validate_rows(schema: TableSchema, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows provided for loading")
    for idx, row in enumerate(rows):
        keys = set(row.keys())
        unknown = keys.difference(schema.columns.keys())
        if unknown:
            raise ValueError(
                f"Row {idx} contains unknown columns for {schema.schema}.{schema.table}: {sorted(unknown)}"
            )
        missing = [col for col in schema.required_columns if col not in keys]
        if missing:
            raise ValueError(
                f"Row {idx} missing required columns for {schema.schema}.{schema.table}: {missing}"
            )


def build_statement(
    schema: TableSchema,
    row: Mapping[str, Any],
    *,
    conflict_columns: Sequence[str] | None,
) -> tuple[str, Dict[str, Any]]:
    columns = list(row.keys())
    column_sql = ", ".join(f"\"{col}\"" for col in columns)
    values_sql = ", ".join(f":{col}" for col in columns)
    base = f"INSERT INTO \"{schema.schema}\".\"{schema.table}\" ({column_sql}) VALUES ({values_sql})"
    params = {col: row[col] for col in columns}
    if conflict_columns:
        update_columns = [col for col in columns if col not in conflict_columns]
        if update_columns:
            update_sql = ", ".join(
                f"\"{col}\" = EXCLUDED.\"{col}\"" for col in update_columns
            )
            statement = f"{base} ON CONFLICT ({', '.join(conflict_columns)}) DO UPDATE SET {update_sql}"
        else:
            statement = f"{base} ON CONFLICT ({', '.join(conflict_columns)}) DO NOTHING"
    else:
        statement = base
    return statement, params


def apply_rows(
    schema: TableSchema,
    rows: Sequence[Mapping[str, Any]],
    *,
    conflict_columns: Sequence[str] | None,
    dry_run: bool,
) -> None:
    engine = get_engine()
    with engine.begin() as connection:
        for row in rows:
            statement, params = build_statement(schema, row, conflict_columns=conflict_columns)
            LOGGER.debug("Executing %s with %s", statement, params)
            if not dry_run:
                connection.execute(text(statement), params)
    if dry_run:
        LOGGER.info(
            "DRY RUN completed for %s.%s (%s rows)",
            schema.schema,
            schema.table,
            len(rows),
        )
    else:
        LOGGER.info(
            "Inserted %s rows into %s.%s",
            len(rows),
            schema.schema,
            schema.table,
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load structured rows into a database table.")
    parser.add_argument("--table", required=True, help="Target table in schema.table format (defaults to public schema).")
    parser.add_argument("--rows-file", required=True, type=Path, help="YAML or JSON file describing rows to load.")
    parser.add_argument("--apply", action="store_true", help="Execute inserts instead of performing a dry run.")
    parser.add_argument(
        "--conflict",
        action="append",
        dest="conflict_columns",
        help="Column name to use for ON CONFLICT clauses (may be provided multiple times).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set log verbosity.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
        force=True,
    )

    schema = fetch_table_schema(args.table)
    rows = load_rows(args.rows_file)
    if not rows:
        LOGGER.warning("No rows found in %s", args.rows_file)
        return
    validate_rows(schema, rows)
    conflict_columns = tuple(args.conflict_columns) if args.conflict_columns else None
    apply_rows(
        schema,
        rows,
        conflict_columns=conflict_columns,
        dry_run=not args.apply,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
