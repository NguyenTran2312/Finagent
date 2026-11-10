import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt


CONFIG_PATH = Path(__file__).resolve().with_name("config.yml")
PROJECT_ROOT = CONFIG_PATH.parent.parent


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    return config


CONFIG = load_config()
DATABASE_CONFIG: dict[str, Any] = CONFIG["database"]
EMBEDDING_CONFIG: dict[str, Any] = CONFIG["embedding"]
CHROMA_CONFIG: dict[str, Any] = CONFIG["chroma"]
DENSE_MODEL_NAME: str = EMBEDDING_CONFIG["model_name"]
DENSE_BATCH_SIZE: int = EMBEDDING_CONFIG["batch_size"]
DENSE_MAX_SEQ_LENGTH: int = EMBEDDING_CONFIG["max_seq_length"]
CHROMA_CACHE_ROOT: Path = (PROJECT_ROOT / CHROMA_CONFIG["cache_root"]).resolve()
CHROMA_INDEX_VERSION: str = CHROMA_CONFIG["index_version"]

RRF_CONSTANT: int = 60
STRONG_CANDIDATE_DEPTH: int = 20
RERANK_MODEL_NAME: str = "BAAI/bge-reranker-base"
RERANK_BATCH_SIZE: int = 32


def postgres_connection_params() -> dict[str, Any]:
    """Resolve PostgreSQL connection settings without storing passwords in YAML."""
    password_env = DATABASE_CONFIG.get("password_env", "PGPASSWORD")
    params: dict[str, Any] = {
        "host": os.environ.get("PGHOST", DATABASE_CONFIG["host"]),
        "port": int(os.environ.get("PGPORT", DATABASE_CONFIG["port"])),
        "dbname": os.environ.get("PGDATABASE", DATABASE_CONFIG["name"]),
        "user": os.environ.get("PGUSER", DATABASE_CONFIG["user"]),
        "connect_timeout": int(DATABASE_CONFIG.get("connect_timeout", 10)),
    }
    password = os.environ.get(password_env)
    if password:
        params["password"] = password
    return params


class ParaphraseSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_subqueries: StrictInt = Field(default=8, ge=1)
    max_history_turns: StrictInt = Field(default=5, ge=0)
    max_input_chars: StrictInt = Field(default=8000, ge=1)
    default_report_scope: Literal["consolidated", "separate"] = "consolidated"
    notes_fallback: StrictBool = True


def load_paraphrase_settings(**overrides: Any) -> ParaphraseSettings:
    """Resolve paraphrase settings at call time: defaults < environment < overrides."""
    values: dict[str, Any] = {}
    for field in ("max_subqueries", "max_history_turns", "max_input_chars"):
        name = f"FINAGENT_PARAPHRASE_{field.upper()}"
        if name in os.environ:
            try:
                values[field] = int(os.environ[name].strip())
            except ValueError as error:
                raise ValueError(f"{name} must be an integer: {os.environ[name]!r}") from error
    if "FINAGENT_PARAPHRASE_DEFAULT_REPORT_SCOPE" in os.environ:
        values["default_report_scope"] = os.environ["FINAGENT_PARAPHRASE_DEFAULT_REPORT_SCOPE"].strip()
    name = "FINAGENT_PARAPHRASE_NOTES_FALLBACK"
    if name in os.environ:
        lowered = os.environ[name].strip().lower()
        if lowered not in ("true", "false", "1", "0"):
            raise ValueError(f"{name} must be true or false: {os.environ[name]!r}")
        values["notes_fallback"] = lowered in ("true", "1")
    values.update(overrides)
    return ParaphraseSettings(**values)
