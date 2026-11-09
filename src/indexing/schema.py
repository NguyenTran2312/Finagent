SCHEMA = """
CREATE TABLE IF NOT EXISTS corpus_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    report_year INTEGER NOT NULL,
    scope TEXT NOT NULL,
    scope_original TEXT NOT NULL DEFAULT '',
    scope_variant INTEGER,
    source_path TEXT NOT NULL UNIQUE,
    source_bytes INTEGER NOT NULL CHECK (source_bytes >= 0),
    source_text TEXT,
    preprocessing_version TEXT
);

CREATE TABLE IF NOT EXISTS pages (
    page_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    page_index INTEGER NOT NULL CHECK (page_index >= 0),
    page_number INTEGER NOT NULL,
    start_char INTEGER NOT NULL CHECK (start_char >= 0),
    end_char INTEGER NOT NULL CHECK (end_char >= start_char),
    text TEXT NOT NULL,

    UNIQUE(doc_id, page_index),
    UNIQUE(page_id, doc_id),
    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
);

CREATE TABLE IF NOT EXISTS tables (
    table_key TEXT PRIMARY KEY,

    doc_id TEXT NOT NULL,
    table_id INTEGER NOT NULL,
    page INTEGER NOT NULL CHECK (page >= 0),
    page_id TEXT NOT NULL,
    context TEXT,
    folded_text TEXT NOT NULL,
    n_rows INTEGER NOT NULL CHECK (n_rows >= 0),
    n_cols INTEGER NOT NULL CHECK (n_cols >= 0),
    statement_kind TEXT,
    unit_scale REAL NOT NULL DEFAULT 1.0,
    rows_json TEXT NOT NULL,
    table_unit_json TEXT NOT NULL,
    cell_units_json TEXT NOT NULL,
    header_rows_json TEXT NOT NULL,

    UNIQUE(doc_id, table_id),
    UNIQUE(table_key, doc_id),
    FOREIGN KEY(doc_id) REFERENCES documents(doc_id),
    FOREIGN KEY(page_id, doc_id) REFERENCES pages(page_id, doc_id)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,

    doc_id TEXT NOT NULL,
    chunk_type TEXT NOT NULL CHECK (chunk_type IN ('page', 'table')),
    page_id TEXT,
    table_key TEXT,
    start_char INTEGER CHECK (start_char >= 0),
    end_char INTEGER CHECK (end_char >= start_char),
    text TEXT NOT NULL,
    search_text TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',

    CHECK (
        (
            chunk_type = 'page'
            AND page_id IS NOT NULL
            AND table_key IS NULL
            AND start_char IS NOT NULL
            AND end_char IS NOT NULL
        )
        OR
        (chunk_type = 'table' AND page_id IS NOT NULL AND table_key IS NOT NULL)
    ),
    FOREIGN KEY(doc_id) REFERENCES documents(doc_id),
    FOREIGN KEY(page_id, doc_id) REFERENCES pages(page_id, doc_id),
    FOREIGN KEY(table_key, doc_id) REFERENCES tables(table_key, doc_id)
);

CREATE INDEX IF NOT EXISTS idx_documents_ticker_year_scope
ON documents(ticker, report_year, scope);
CREATE INDEX IF NOT EXISTS idx_tables_statement_kind
ON tables(statement_kind);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_type
ON chunks(doc_id, chunk_type);
"""