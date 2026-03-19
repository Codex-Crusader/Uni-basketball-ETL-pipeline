"""
app/storage.py
──────────────
All persistence logic in one place: local JSON and Snowflake.

Rule: nothing outside this module reads or writes data files directly.
Everything goes through load_data() / save_to_json() / append_to_json().

Also owns _sanitize() because every JSON serialisation path runs through here.
"""

import json
import os

from app.config import (
    DATA_CFG, SF_CFG, LOCAL_FILE,
)
from app.logger import get_logger

log = get_logger(__name__)


# ── HELPERS ──────────────────────────────────────────────────────────────────

def _sanitize(obj):
    # thank Claude..... Pun intended....
    # I thank you claude for helping me fix this error i was getting
    """Recursively replace float NaN/Inf with None so json.dumps produces valid JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
        return None
    return obj


# ── SNOWFLAKE  (provision kept, disabled for now, ran out of accounts) ────────

def _sf_conn():
    if not SF_CFG.get("enabled"):
        log.debug("[Snowflake] Disabled in config.")
        return None
    try:
        import snowflake.connector  # noqa: PLC0415
        conn = snowflake.connector.connect(
            user     = SF_CFG.get("user")     or os.environ.get("SNOWFLAKE_USER", ""),
            password = SF_CFG.get("password") or os.environ.get("SNOWFLAKE_PASSWORD", ""),
            account  = SF_CFG["account"],
            warehouse= SF_CFG["warehouse"],
            database = SF_CFG["database"],
            schema   = SF_CFG["schema"],
        )
        log.info("[Snowflake] Connected.")
        return conn
    except ImportError:  # so much safety, am I not fabulous enough for you, Snowflake?
        log.warning("[Snowflake] snowflake-connector-python not installed.")
        return None
    except Exception as e:
        log.error("[Snowflake] Connection error: %s", e)
        return None


def _sf_create_table(conn):
    cur  = conn.cursor()
    cols = ", ".join([f"{f} FLOAT" for f in DATA_CFG["features"]])
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {SF_CFG['table']} (
            game_id VARCHAR(80), home_team VARCHAR(100), away_team VARCHAR(100),
            {cols}, outcome INT
        )
    """)
    cur.close()


def save_to_snowflake(data: list):
    conn = _sf_conn()
    if not conn: return
    _sf_create_table(conn)
    cur = conn.cursor()
    cur.execute(f"DELETE FROM {SF_CFG['table']}")
    _sf_insert_batch(cur, data)
    conn.commit(); cur.close(); conn.close()
    log.info("[Snowflake] Saved %d records.", len(data))


def load_from_snowflake() -> list:
    conn = _sf_conn()
    if not conn: return []
    cur  = conn.cursor()
    cur.execute(f"SELECT * FROM {SF_CFG['table']}")
    cols = [c[0].lower() for c in cur.description]
    data = [dict(zip(cols, row)) for row in cur]
    cur.close(); conn.close()
    log.info("[Snowflake] Loaded %d records.", len(data))
    return data


def append_to_snowflake(new_data: list):
    conn = _sf_conn()
    if not conn: return
    _sf_create_table(conn)
    cur = conn.cursor()
    _sf_insert_batch(cur, new_data)
    conn.commit(); cur.close(); conn.close()
    log.info("[Snowflake] Appended %d records.", len(new_data))


def _sf_insert_batch(cur, data: list):
    col_features = DATA_CFG["features"]
    ph   = ", ".join(["%s"] * (3 + len(col_features) + 1))
    cols = "game_id, home_team, away_team, " + ", ".join(col_features) + ", outcome"
    sql  = f"INSERT INTO {SF_CFG['table']} ({cols}) VALUES ({ph})"
    for g in data:
        vals = [g.get("game_id"), g.get("home_team",""), g.get("away_team","")]
        vals += [g.get(feat, 0) for feat in col_features]
        vals.append(g["outcome"])
        cur.execute(sql, vals)
    # I know I could do this with a single bulk insert and avoid the loop,
    # but keeping it simple and compatible with the free Snowflake tier.
    # Avoiding SQLAlchemy or Pandas to keep dependencies minimal. Sue me, Snowflake.


# ── LOCAL JSON ────────────────────────────────────────────────────────────────

def save_to_json(data: list):
    # Are ya winning JSON?
    LOCAL_FILE.parent.mkdir(exist_ok=True)
    with open(LOCAL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log.info("[Storage] Saved %d records → %s", len(data), LOCAL_FILE)


def load_from_json() -> list:
    if not LOCAL_FILE.exists():
        return []
    with open(LOCAL_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data


def append_to_json(new_data: list) -> int:
    existing     = load_from_json()
    existing_ids = {g.get("game_id") for g in existing}
    new_unique   = [g for g in new_data if g.get("game_id") not in existing_ids]
    combined     = existing + new_unique
    save_to_json(combined)
    log.info("[Storage] +%d new games (total: %d)", len(new_unique), len(combined))
    return len(new_unique)


def load_data(storage: str = "local") -> list:
    return load_from_snowflake() if storage == "snowflake" else load_from_json()