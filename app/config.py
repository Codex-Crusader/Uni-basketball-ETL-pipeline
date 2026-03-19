"""
app/config.py
─────────────
Single source of truth for every config value and path constant.
Everything else imports from here — nobody reads config.yaml directly.

Keeping this in one place means changing the config file path or adding
a new section only ever requires touching this file.
"""

from pathlib import Path
import yaml


def load_config(path: str = "config.yaml") -> dict:
    # encoding="utf-8" is required — config contains unicode box-drawing
    # characters in comments that Windows cp1252 cannot decode. learned this
    # the hard way. do not remove the encoding argument.
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG         = load_config()
APP_CFG     = CFG["app"]
HT_CFG      = CFG["home_team"]
DATA_CFG    = CFG["data"]
API_CFG     = CFG["ncaa_api"]
SF_CFG      = CFG["snowflake"]
MODEL_CFG   = CFG["models"]
AL_CFG      = CFG["auto_learn"]
ROSTER_CFG  = CFG.get("roster", {})
ROLLING_CFG = CFG.get("rolling", {})

# Path constants — built once, imported everywhere
DATA_DIR      = Path(DATA_CFG["dir"])
LOCAL_FILE    = Path(DATA_CFG["local_file"])
MODELS_DIR    = Path(MODEL_CFG["dir"])
REGISTRY_FILE = Path(MODEL_CFG["registry_file"])
LEARN_LOG     = Path(AL_CFG["learning_log_file"])
ROSTER_DIR    = Path(ROSTER_CFG.get("cache_dir",    "data/rosters"))
TEAM_ID_CACHE = Path(ROSTER_CFG.get("team_id_cache","data/team_ids.json"))
# Social Credit = path(John_Xina.get("basketball")

# Ensure required directories exist at import time.
# Anything that writes files can assume these exist.
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
ROSTER_DIR.mkdir(exist_ok=True)