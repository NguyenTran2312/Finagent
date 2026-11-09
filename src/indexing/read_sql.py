from typing import Any

import psycopg
from psycopg import sql

from config.settings import postgres_connection_params


def inspect_database(database: str | None = None) -> None:
    connection_params: dict[str, Any] = postgres_connection_params()
    if database:
        connection_params["dbname"] = database

    with psycopg.connect(**connection_params) as connection:
        tables = connection.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = current_schema()
                AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
        ).fetchall()

        print(f"Database: {connection_params['dbname']}")
        print(f"Total tables: {len(tables)}")

        for (table_name,) in tables:
            row_count = connection.execute(
                sql.SQL("SELECT COUNT(*) FROM {}").format(
                    sql.Identifier(table_name)
                )
            ).fetchone()[0]
            columns = connection.execute(
                """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = current_schema() AND table_name = %s
                ORDER BY ordinal_position
                """,
                (table_name,),
            ).fetchall()
            primary_key_columns = {
                row[0]
                for row in connection.execute(
                    """
                    SELECT key_column.column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS key_column
                      ON key_column.constraint_catalog = tc.constraint_catalog
                     AND key_column.constraint_schema = tc.constraint_schema
                     AND key_column.constraint_name = tc.constraint_name
                     AND key_column.table_schema = tc.table_schema
                     AND key_column.table_name = tc.table_name
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                      AND tc.table_schema = current_schema()
                      AND tc.table_name = %s
                    """,
                    (table_name,),
                ).fetchall()
            }
            print(f"\n{table_name}: {row_count} rows")
            print("  Columns:")
            for name, data_type, nullable, default in columns:
                details = [data_type]
                if name in primary_key_columns:
                    details.append("PRIMARY KEY")
                if default is not None:
                    details.append(f"DEFAULT {default}")
                if nullable == "NO":
                    details.append("NOT NULL")
                print(f"    {name}: {', '.join(details)}")

            foreign_keys = connection.execute(
                """
                SELECT
                    source_column.attname,
                    target_table.relname,
                    target_column.attname
                FROM pg_constraint AS constraint_info
                JOIN pg_class AS source_table
                  ON source_table.oid = constraint_info.conrelid
                JOIN pg_namespace AS source_schema
                  ON source_schema.oid = source_table.relnamespace
                JOIN pg_class AS target_table
                  ON target_table.oid = constraint_info.confrelid
                JOIN LATERAL unnest(constraint_info.conkey)
                    WITH ORDINALITY AS source_key(attnum, position)
                  ON TRUE
                JOIN LATERAL unnest(constraint_info.confkey)
                    WITH ORDINALITY AS target_key(attnum, position)
                  ON target_key.position = source_key.position
                JOIN pg_attribute AS source_column
                  ON source_column.attrelid = constraint_info.conrelid
                 AND source_column.attnum = source_key.attnum
                JOIN pg_attribute AS target_column
                  ON target_column.attrelid = constraint_info.confrelid
                 AND target_column.attnum = target_key.attnum
                WHERE constraint_info.contype = 'f'
                  AND source_schema.nspname = current_schema()
                  AND source_table.relname = %s
                ORDER BY source_key.position
                """,
                (table_name,),
            ).fetchall()
            if foreign_keys:
                print("  Foreign keys:")
                for source_column, referenced_table, target_column in foreign_keys:
                    print(
                        f"    {source_column} -> "
                        f"{referenced_table}.{target_column}"
                    )
