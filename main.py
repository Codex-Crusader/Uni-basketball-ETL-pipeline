# import dumpyard.... god that's a lot of imports
import json, os, time, pickle, hashlib, argparse, traceback, warnings
import threading
import numpy as np
import requests

from datetime import datetime, timedelta
from pathlib import Path

import yaml
from flask import Flask, request, jsonify, send_file

from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

warnings.filterwarnings("ignore")


# CONFIG

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
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

DATA_DIR      = Path(DATA_CFG["dir"])
LOCAL_FILE    = Path(DATA_CFG["local_file"])
MODELS_DIR    = Path(MODEL_CFG["dir"])
REGISTRY_FILE = Path(MODEL_CFG["registry_file"])
LEARN_LOG     = Path(AL_CFG["learning_log_file"])
ROSTER_DIR    = Path(ROSTER_CFG.get("cache_dir", "data/rosters"))
TEAM_ID_CACHE = Path(ROSTER_CFG.get("team_id_cache", "data/team_ids.json"))
# Social Credit = path(John_Xina.get("basketball")

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
ROSTER_DIR.mkdir(exist_ok=True)


# SNOWFLAKE  (provision kept, disabled for now, ran out of accounts)

def _sf_conn():
    if not SF_CFG.get("enabled"):
        print("[Snowflake] Disabled in config.")
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
        print("[Snowflake] Connected.")
        return conn
    except ImportError:  # so much safety, am I not fabulous enough for you, Snowflake?
        print("[Snowflake] snowflake-connector-python not installed.")
        return None
    except Exception as e:
        print(f"[Snowflake] Connection error: {e}")
        return None

def _sf_create_table(conn):
    cur = conn.cursor()
    cols = ", ".join([f"{f} FLOAT" for f in DATA_CFG["features"]])
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {SF_CFG['table']} (
            game_id VARCHAR(80), home_team VARCHAR(100), away_team VARCHAR(100),
            {cols}, outcome INT
        )
    """)
    cur.close()

def save_to_snowflake(data):
    conn = _sf_conn()
    if not conn: return
    _sf_create_table(conn)
    cur = conn.cursor()
    cur.execute(f"DELETE FROM {SF_CFG['table']}")
    _sf_insert_batch(cur, data)
    conn.commit(); cur.close(); conn.close()
    print(f"[Snowflake] Saved {len(data)} records.")

def load_from_snowflake():
    conn = _sf_conn()
    if not conn: return []
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {SF_CFG['table']}")
    cols = [c[0].lower() for c in cur.description]
    data = [dict(zip(cols, row)) for row in cur]
    cur.close(); conn.close()
    print(f"[Snowflake] Loaded {len(data)} records.")
    return data

def append_to_snowflake(new_data):
    conn = _sf_conn()
    if not conn: return
    _sf_create_table(conn)
    cur = conn.cursor()
    _sf_insert_batch(cur, new_data)
    conn.commit(); cur.close(); conn.close()
    print(f"[Snowflake] Appended {len(new_data)} records.")

def _sf_insert_batch(cur, data):
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



# ESPN DATA FETCHER  (game box scores)
# All hail, Free ESPN API, the gift that keeps on giving (and giving, and giving)
# no auth, no keys, no limits (well, some limits), just pure unadulterated data access.
# I will offer incense prayers to the ESPN gods

class ESPNFetcher:
    BASE = API_CFG["espn"]["base_url"]

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (research)"})
        self.delay = API_CFG.get("rate_limit_delay", 0.4)

    def _get(self, url, params=None):
        try:
            r = self.session.get(url, params=params, timeout=10)
            r.raise_for_status()
            time.sleep(self.delay)
            return r.json()
        except Exception as e:
            print(f"  [ESPN] {url}: {e}")
            return None

    def get_game_ids(self, start_date: str, end_date: str) -> list:
        ids = []
        current = datetime.strptime(start_date, "%Y%m%d")
        end     = datetime.strptime(end_date,   "%Y%m%d")
        page_size = API_CFG["espn"].get("page_size", 25)
        while current <= end:
            data = self._get(
                f"{self.BASE}{API_CFG['espn']['scoreboard_path']}",
                params={"dates": current.strftime("%Y%m%d"), "limit": page_size}
            )
            if data:
                for event in data.get("events", []):
                    ids.append(event["id"])
            current += timedelta(days=1)
        return ids

    def get_box_score(self, event_id: str):
        data = self._get(
            f"{self.BASE}{API_CFG['espn']['summary_path']}",
            params={"event": event_id}
        )
        if not data:
            return None
        try:
            box       = data.get("boxscore", {})
            box_teams = box.get("teams", [])
            if len(box_teams) < 2:
                return None

            home_d = away_d = None
            for t in box_teams:
                if   t.get("homeAway") == "home": home_d = t
                elif t.get("homeAway") == "away": away_d = t
            if not home_d or not away_d:
                return None

            def stats(td):
                return {s["name"]: s.get("displayValue","0") for s in td.get("statistics",[])}

            hs, as_ = stats(home_d), stats(away_d)

            def flt(d, k, fb=0.0):
                try: return float(str(d.get(k, fb)).replace("%",""))
                except (ValueError, TypeError, AttributeError): return float(fb)

            comps = data.get("header",{}).get("competitions",[{}])
            if not comps: return None
            home_score = away_score = 0
            for ct in comps[0].get("competitors",[]):
                score = int(ct.get("score", 0))
                if ct.get("homeAway") == "home": home_score = score
                elif ct.get("homeAway") == "away": away_score = score
            if home_score == 0 and away_score == 0:
                return None

            def norm_pct(v):
                return v / 100 if v > 1 else v

            return {
                "game_id":        f"ESPN_{event_id}",
                "home_team":      home_d.get("team",{}).get("displayName","Home"),
                "away_team":      away_d.get("team",{}).get("displayName","Away"),
                "home_score":     home_score,
                "away_score":     away_score,
                "home_ppg":       float(home_score),
                "away_ppg":       float(away_score),
                "home_fg_pct":    round(norm_pct(flt(hs, "fieldGoalPct")), 4),
                "away_fg_pct":    round(norm_pct(flt(as_, "fieldGoalPct")), 4),
                "home_rebounds":  flt(hs, "totalRebounds"),
                "away_rebounds":  flt(as_, "totalRebounds"),
                "home_assists":   flt(hs, "assists"),
                "away_assists":   flt(as_, "assists"),
                "home_turnovers": flt(hs, "turnovers"),
                "away_turnovers": flt(as_, "turnovers"),
                "home_steals":    flt(hs, "steals"),
                "away_steals":    flt(as_, "steals"),
                "home_blocks":    flt(hs, "blocks"),
                "away_blocks":    flt(as_, "blocks"),
                "outcome":        1 if home_score > away_score else 0,
                "source":         "espn",
                "fetched_at":     datetime.now().isoformat(),
            } # holy cow, ESPN, I love you but this is a nightmare to parse. it is free so no complaints
        except Exception as e:
            print(f"  [ESPN] Parse error {event_id}: {e}")
            return None


class CustomAPIFetcher:  # Let's play fetch
    def __init__(self):
        custom = API_CFG.get("custom", {})
        self.base_url  = custom.get("base_url", "")
        self.api_key   = custom.get("api_key", "") or os.environ.get("NCAA_API_KEY", "")
        self.endpoint  = custom.get("games_endpoint", "/games")
        self.field_map = custom.get("field_map", {})

    def fetch(self, season, max_games):
        if not self.base_url:
            print("[CustomAPI] base_url not set."); return []
        try:
            r = requests.get(
                self.base_url + self.endpoint,
                params={API_CFG["custom"].get("season_param","season"): season,
                        "limit": max_games, "api_key": self.api_key},
                timeout=15
            )
            r.raise_for_status()
            return [self._map(g) for g in r.json() if self._map(g)]
        except Exception as e:
            print(f"[CustomAPI] {e}"); return []

    def _map(self, raw):
        fm = self.field_map
        try:
            return {
                "game_id":    raw.get(fm.get("game_id","id"),""),
                "home_team":  raw.get(fm.get("home_team","home_team"),""),
                "away_team":  raw.get(fm.get("away_team","away_team"),""),
                "outcome":    1 if raw.get(fm.get("home_score","home_score"),0) >
                                   raw.get(fm.get("away_score","away_score"),0) else 0,
                "source":     "custom",
                "fetched_at": datetime.now().isoformat(),
            }
        except (KeyError, TypeError, AttributeError):
            return None 


def fetch_ncaa_data(max_games=None): # I do not want to play fetch anymore
    max_games = max_games or API_CFG.get("max_games", 500)
    provider  = API_CFG.get("provider", "espn")
    if provider == "custom":
        return CustomAPIFetcher().fetch(API_CFG.get("season", 2024), max_games)
    fetcher  = ESPNFetcher()
    season   = API_CFG.get("season", 2024)
    start, end = f"{season}1101", f"{season+1}0430"
    print(f"[ESPN] Fetching IDs {start}→{end}...")
    game_ids = fetcher.get_game_ids(start, end)
    print(f"[ESPN] {len(game_ids)} events. Fetching box scores...")
    games, errors = [], 0
    for i, gid in enumerate(game_ids[:max_games]):
        g = fetcher.get_box_score(gid)
        if g: games.append(g)
        else: errors += 1
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{min(len(game_ids),max_games)} valid={len(games)} skipped={errors}")
    print(f"[ESPN] Done. Valid={len(games)}, Skipped={errors}")
    return games # writing graceful code is a nightmare these days, this is MY SLOP!



# ESPN ROSTER FETCHER  (player rosters + season stats)

class RosterFetcher:
    """
    Fetches NCAA team rosters and individual player season stats from ESPN.
    Results are cached per-team in data/rosters/<team_id>.json to avoid
    hammering the API on every request.

    ESPN endpoints used:
      GET /teams?limit=500          → all teams + IDs
      GET /teams/{id}/roster        → player list
      GET /athletes/{id}/statistics → player season averages

    All URLs built from config.yaml → ncaa_api.espn
    """
    BASE        = API_CFG["espn"]["base_url"]
    TEAMS_URL   = API_CFG["espn"]["base_url"] + API_CFG["espn"].get("teams_path", "/teams")
    ATHLETE_URL = API_CFG["espn"]["base_url"] + API_CFG["espn"].get("athletes_path", "/athletes")

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (research)"})
        self.delay   = API_CFG.get("rate_limit_delay", 0.4)
        self.ttl_h   = ROSTER_CFG.get("cache_ttl_hours", 24)

    def _get(self, url, params=None):
        try:
            r = self.session.get(url, params=params, timeout=10)
            r.raise_for_status()
            time.sleep(self.delay)
            return r.json()
        except Exception as e:
            print(f"  [Roster] {url}: {e}")
            return None

    # team ID lookup

    @staticmethod
    def _load_team_id_cache() -> dict:
        if TEAM_ID_CACHE.exists():
            with open(TEAM_ID_CACHE) as f:
                return json.load(f)
        return {}

    @staticmethod
    def _save_team_id_cache(cache: dict):
        with open(TEAM_ID_CACHE, "w") as f:
            json.dump(cache, f, indent=2)

    def get_team_id(self, team_name: str) -> str | None:
        """Return ESPN team ID for a given display name. Caches to disk."""
        cache = self._load_team_id_cache()

        # Exact match first
        if team_name in cache:
            return cache[team_name]

        # Case-insensitive fuzzy match in cache
        for k, v in cache.items():
            if team_name.lower() in k.lower() or k.lower() in team_name.lower():
                return v #backup

        # Cache miss  fetch full team list from ESPN
        print(f"[Roster] Looking up ESPN ID for '{team_name}'...")
        data = self._get(self.TEAMS_URL, params={"limit": 1000})
        if not data:
            return None # backup's backup

        # ESPN wraps teams in sports[0].leagues[0].teams
        try:
            teams_raw = (data.get("sports",[{}])[0]
                             .get("leagues",[{}])[0]
                             .get("teams", []))
        except (IndexError, KeyError):
            teams_raw = []

        # Build full cache from response
        for entry in teams_raw:
            t = entry.get("team", {})
            name = t.get("displayName", "")
            tid  = t.get("id", "")
            if name and tid:
                cache[name] = tid #backup prep

        self._save_team_id_cache(cache)

        # Try again after refresh
        if team_name in cache:
            return cache[team_name]
        for k, v in cache.items():
            if team_name.lower() in k.lower() or k.lower() in team_name.lower():
                return v #nuclear option

        print(f"[Roster] Could not find ESPN ID for '{team_name}'.")
        return None

    # roster fetch (utter boilerplate DO NOTT TOUCH!!!)
    
    @staticmethod
    def _cache_path(team_id: str) -> Path:
        return ROSTER_DIR / f"{team_id}.json"

    def _cache_valid(self, team_id: str) -> bool:
        p = self._cache_path(team_id)
        if not p.exists():
            return False
        cached = json.loads(p.read_text())
        fetched = datetime.fromisoformat(cached.get("fetched_at", "2000-01-01"))
        return (datetime.now() - fetched).total_seconds() < self.ttl_h * 3600

    def _load_cached(self, team_id: str) -> dict | None:
        p = self._cache_path(team_id)
        if p.exists():
            return json.loads(p.read_text())
        return None

    def _save_cached(self, team_id: str, data: dict):
        self._cache_path(team_id).write_text(json.dumps(data, indent=2))

    def get_roster(self, team_id: str) -> list[dict]:
        """
        Player list from /teams/{id}/roster.
        ESPN embeds basic stats directly in the athlete objects — we extract
        those first. Only falls back to the per-player /statistics endpoint
        for players where the embedded stats are all zero.
        """
        data = self._get(f"{self.TEAMS_URL}/{team_id}/roster")
        if not data:
            return []
        players = []
        for athlete in data.get("athletes", []):
            player = {
                "id":       athlete.get("id", ""),
                "name":     athlete.get("displayName", "Unknown"),
                "position": athlete.get("position", {}).get("abbreviation", ""),
                "jersey":   athlete.get("jersey", ""),
                # "hotel":  trivago.get("suiii"),
            }
            # Try embedded stats first to avoids 404s on /statistics endpoint
            embedded = self._parse_embedded_stats(athlete)
            player.update(embedded)
            players.append(player)
        return players

    def get_player_stats(self, player_id: str) -> dict:
        """
        Season averages for a player. ESPN's /athletes/{id}/statistics endpoint
        returns 404 for many players — we catch that and return empty stats
        rather than crashing. Stats are better pulled from the roster embed
        via get_roster_with_stats() but this is kept as a fallback.
        """
        data = self._get(f"{self.ATHLETE_URL}/{player_id}/statistics")
        if not data:
            return _empty_player_stats()

        raw = {}
        for cat in data.get("splits", {}).get("categories", []):
            cat_name = cat.get("name", "").lower()
            if cat_name in ("avg", "pergame", "average", "averages"):
                for stat in cat.get("stats", []):
                    raw[stat.get("name","").lower()] = stat.get("value", 0) or 0
                break
        if not raw:
            for stat in (data.get("statistics", {})
                             .get("splits", {})
                             .get("categories", [{}])[0]
                             .get("stats", [])):
                raw[stat.get("name","").lower()] = stat.get("value", 0) or 0

        def g(keys, default=0.0):
            for k in keys:
                v = raw.get(k)
                if v is not None:
                    try: return round(float(v), 4)
                    except (ValueError, TypeError): pass
            return float(default)

        ppg    = g(["points","pointspergame","pts"])
        rpg    = g(["rebounds","reboundspergame","reb","totalrebounds"])
        apg    = g(["assists","assistspergame","ast"])
        spg    = g(["steals","stealspergame","stl"])
        bpg    = g(["blocks","blockspergame","blk"])
        tov    = g(["turnovers","turnoverspergame","to"])
        fg_pct = g(["fieldgoalpercentage","fieldgoalpct","fg%"])
        fgm    = g(["fieldgoalsmade","fgm"])
        fga    = g(["fieldgoalsattempted","fga"])
        # yeah, I forgot what these are for.
        if fg_pct > 1:
            fg_pct = round(fg_pct / 100, 4)

        return {
            "ppg": ppg, "rpg": rpg, "apg": apg,
            "spg": spg, "bpg": bpg, "tov": tov,
            "fg_pct": fg_pct if fg_pct > 0 else 0.45,
            "fgm": fgm, "fga": fga,
        }

    def _parse_embedded_stats(self, athlete: dict) -> dict:
        """
        Pull stats from the player object already embedded in the roster response.
        ESPN includes a 'statistics' array directly on each athlete — this is
        much more reliable than the separate /statistics endpoint.
        """
        raw = {}
        for stat_group in athlete.get("statistics", []):
            for stat in stat_group.get("stats", []):
                raw[stat.get("name","").lower()] = stat.get("value", 0) or 0
        # Also try 'displayStats' which some roster responses use
        for stat in athlete.get("displayStats", []):
            raw[stat.get("name","").lower()] = stat.get("value", 0) or 0

        if not raw:
            return _empty_player_stats()

        def g(keys, default=0.0):
            for k in keys:
                v = raw.get(k)
                if v is not None:
                    try: return round(float(v), 4)
                    except (ValueError, TypeError): pass
            return float(default)

        ppg    = g(["points","pointspergame","pts","avgpoints"])
        rpg    = g(["rebounds","reboundspergame","reb","totalrebounds","avgrebounds"])
        apg    = g(["assists","assistspergame","ast","avgassists"])
        spg    = g(["steals","stealspergame","stl","avgsteals"])
        bpg    = g(["blocks","blockspergame","blk","avgblocks"])
        tov    = g(["turnovers","turnoverspergame","to","avgturnovers"])
        fg_pct = g(["fieldgoalpercentage","fieldgoalpct","fg%","fgpct"])
        fgm    = g(["fieldgoalsmade","fgm"])
        fga    = g(["fieldgoalsattempted","fga"])
        # Yeah... I definitely forgot what these are for.

        if fg_pct > 1:
            fg_pct = round(fg_pct / 100, 4)

        return {
            "ppg": ppg, "rpg": rpg, "apg": apg,
            "spg": spg, "bpg": bpg, "tov": tov,
            "fg_pct": fg_pct if fg_pct > 0 else 0.45,
            "fgm": fgm, "fga": fga,
        }

    def fetch_team(self, team_name: str, force: bool = False) -> dict | None:
        """
        Full pipeline: name → ESPN ID → roster → player stats per player.
        Results cached. Pass force=True to bypass cache.
        """
        team_id = self.get_team_id(team_name)
        if not team_id:
            return None

        if not force and self._cache_valid(team_id):
            print(f"[Roster] Cache hit for {team_name} (ID {team_id})")
            return self._load_cached(team_id)

        print(f"[Roster] Fetching roster for {team_name} (ID {team_id})...")
        players = self.get_roster(team_id)
        if not players:
            print(f"[Roster] No players returned for {team_name}.")
            return None

        # Count how many players already have stats from the embedded response
        has_embedded = sum(1 for p in players if p.get("ppg", 0) > 0)
        print(f"[Roster] {len(players)} players. {has_embedded} have embedded stats.")

        # Write initial progress — all names visible immediately, stats fill in after
        _roster_progress[team_name] = {
            "status": "loading",
            "players": list(players),  # copy — names shown right away
            "done": has_embedded,
            "total": len(players),
        }

        # Only call /statistics for players with no embedded stats
        needs_fetch = [p for p in players if p.get("ppg", 0) == 0 and p.get("id")]
        if needs_fetch:
            print(f"[Roster] Fetching stats for {len(needs_fetch)} players missing data...")
            for i, player in enumerate(needs_fetch):
                fetched = self.get_player_stats(player["id"])
                if any(v > 0 for k, v in fetched.items() if k != "fg_pct"):
                    player.update(fetched)
                # Update progress after each player so dashboard re-renders
                _roster_progress[team_name] = {
                    "status": "loading",
                    "players": list(players),
                    "done": has_embedded + i + 1,
                    "total": len(players),
                }
                if (i+1) % 5 == 0:
                    print(f"  {i+1}/{len(needs_fetch)} done")

        result = {
            "team_name":  team_name,
            "team_id":    team_id,
            "players":    players,
            "fetched_at": datetime.now().isoformat(),
        }
        self._save_cached(team_id, result)
        # Mark progress as complete
        _roster_progress[team_name] = {
            "status": "ready", "players": players,
            "done": len(players), "total": len(players),
        }
        print(f"[Roster] Done. {len(players)} players cached for {team_name}.")
        return result

    def fetch_team_async(self, team_name: str, force: bool = False):
        """
        Non-blocking version of fetch_team. Starts a background thread,
        writes incremental progress to _roster_progress[team_name] as
        each player is processed. The dashboard polls /roster/progress/<team>
        every second and renders players as they appear.
        """
        # If already cached and not forcing, mark ready immediately
        team_id = self.get_team_id(team_name)
        if team_id and not force and self._cache_valid(team_id):
            cached = self._load_cached(team_id)
            if cached:
                _roster_progress[team_name] = {
                    "status": "ready",
                    "players": cached["players"],
                    "done": len(cached["players"]),
                    "total": len(cached["players"]),
                }
                return  # already done, no thread needed

        # Mark as loading so the dashboard shows a spinner
        _roster_progress[team_name] = {
            "status": "loading", "players": [], "done": 0, "total": 0
        }

        def _run():
            try:
                result = self.fetch_team(team_name, force=force)
                if result is None:
                    _roster_progress[team_name] = {
                        "status": "error", "players": [], "done": 0, "total": 0,
                        "message": f"Could not fetch roster for '{team_name}'.",
                    }
            except Exception as e:
                _roster_progress[team_name] = {
                    "status": "error", "players": [], "done": 0, "total": 0,
                    "message": str(e),
                }

        threading.Thread(target=_run, daemon=True).start()
        # Linus torvalds... I pray to you.... I beseech thee...


def _empty_player_stats() -> dict:
    return {"ppg":0.0,"rpg":0.0,"apg":0.0,"spg":0.0,"bpg":0.0,
            "tov":0.0,"fg_pct":0.45,"fgm":0.0,"fga":0.0}


def compute_stats_from_roster(players: list[dict], side: str) -> dict:
    """
    Aggregate individual player stats into team-level feature values.
    side: "home" or "away" — sets the feature name prefix.

    ppg  → sum of players' ppg  (each player contributes their scoring)
    fg_pct → FGA-weighted average across players
    rebounds, assists, steals, blocks, turnovers → sum of per-player averages
    """
    if not players:
        return {}

    prefix = side + "_"

    total_ppg  = sum(p.get("ppg", 0) for p in players)
    total_rpg  = sum(p.get("rpg", 0) for p in players)
    total_apg  = sum(p.get("apg", 0) for p in players)
    total_spg  = sum(p.get("spg", 0) for p in players)
    total_bpg  = sum(p.get("bpg", 0) for p in players)
    total_tov  = sum(p.get("tov", 0) for p in players)
    # why

    # FG% weighted by field goal attempts — more accurate than simple average
    total_fgm = sum(p.get("fgm", 0) for p in players)
    total_fga = sum(p.get("fga", 0) for p in players)
    if total_fga > 0:
        fg_pct = total_fgm / total_fga
    else:
        # Fall back to simple average of individual FG%
        fg_pcts = [p.get("fg_pct", 0.45) for p in players]
        fg_pct  = sum(fg_pcts) / len(fg_pcts)

    return { # oh that's why
        f"{prefix}ppg":       round(total_ppg, 2),
        f"{prefix}fg_pct":    round(fg_pct, 4),
        f"{prefix}rebounds":  round(total_rpg, 2),
        f"{prefix}assists":   round(total_apg, 2),
        f"{prefix}turnovers": round(total_tov, 2),
        f"{prefix}steals":    round(total_spg, 2),
        f"{prefix}blocks":    round(total_bpg, 2),
    }



# LOCAL STORAGE

def save_to_json(data):  # Are ya winning JSON?
    LOCAL_FILE.parent.mkdir(exist_ok=True)
    with open(LOCAL_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Storage] Saved {len(data)} records → {LOCAL_FILE}")

def load_from_json():
    if not LOCAL_FILE.exists():
        return []
    with open(LOCAL_FILE) as f:
        data = json.load(f)
    return data

def append_to_json(new_data):
    existing     = load_from_json()
    existing_ids = {g.get("game_id") for g in existing}
    new_unique   = [g for g in new_data if g.get("game_id") not in existing_ids]
    combined     = existing + new_unique
    save_to_json(combined)
    print(f"[Storage] +{len(new_unique)} new games (total: {len(combined)})")
    return len(new_unique)

def load_data(storage="local"):
    return load_from_snowflake() if storage == "snowflake" else load_from_json()


# TEAM STATS season averages OR rolling window


def build_team_stats(data: list, window: int = None) -> dict:
    """
    Compute per-team feature averages from game records.

    window: if set, only the last N games per team are used (sorted by
    fetched_at descending). None = full season average.

    Mirroring logic: when a team was away, their stats are under away_*
    columns. We read them and store under home_* keys so every team has
    consistent home_* feature names regardless of which side they played on.
    """
    cfg_features = DATA_CFG["features"]

    # Group all games by team with their side and timestamp
    team_games: dict[str, list] = {}  # team -> [(game_dict, side, timestamp)]

    for g in data:
        ht = g.get("home_team", "").strip()
        at = g.get("away_team", "").strip()
        if not ht or not at:
            continue
        ts = g.get("fetched_at", "")
        if ht not in team_games: team_games[ht] = []
        if at not in team_games: team_games[at] = []
        team_games[ht].append((g, "home", ts))
        team_games[at].append((g, "away", ts))

    result = {}

    for team, games_with_side in team_games.items():
        if not team:
            continue

        # Apply rolling window — sort newest-first, slice
        sorted_games = sorted(games_with_side, key=lambda x: x[2], reverse=True)
        windowed     = sorted_games[:window] if window else sorted_games
        if not windowed:
            continue

        accum = {feat: [] for feat in cfg_features}

        for g, side, _ in windowed:
            for feat in cfg_features:
                if feat.startswith("home_"):
                    if side == "home":
                        accum[feat].append(float(g.get(feat, 0)))
                    else:  # team was away — mirror
                        mirror = "away_" + feat[5:]
                        accum[feat].append(float(g.get(mirror, 0)))
                elif feat.startswith("away_"):
                    if side == "away":
                        accum[feat].append(float(g.get(feat, 0)))
                    else:  # team was home — mirror
                        mirror = "home_" + feat[5:]
                        accum[feat].append(float(g.get(mirror, 0)))
                        # This logic assumes that for every "home_X" there is a corresponding
                        # "away_X" that represents the same stat for the other team.
                        # This way we can aggregate stats regardless of home/away side.
                        # I hate myself

        games_used = len(windowed)
        result[team] = {
            feat: round(sum(v) / len(v), 4) if v else 0.0
            for feat, v in accum.items()
        }
        result[team]["games_played"]   = len(games_with_side)  # total, not windowed
        result[team]["games_in_window"] = games_used
        result[team]["wins"] = sum(
            1 for g_data, side, _ in games_with_side
            if (side == "home" and g_data.get("outcome") == 1) or
               (side == "away" and g_data.get("outcome") == 0)
        )

    return result


def get_home_team_stats(data: list, window: int = None):
    ts   = build_team_stats(data, window=window)
    name = HT_CFG["name"]
    if name in ts:
        return {"name": name, "stats": ts[name]}
    for k in ts:
        if name.lower() in k.lower() or k.lower() in name.lower():
            return {"name": k, "stats": ts[k]}
    return None


# MODEL REGISTRY

def _load_registry() -> dict: #this is a modeling registry not a modeling agency
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    return {"versions": [], "active_version": None}

def _save_registry(reg):
    REGISTRY_FILE.parent.mkdir(exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(reg, f, indent=2)

def register_model(model_name, model_obj, metrics, feature_names, training_size) -> str:
    reg        = _load_registry()
    existing   = [int(v["version"].lstrip("v")) for v in reg["versions"]]
    ver_num    = (max(existing) + 1) if existing else 1
    version    = f"v{ver_num}"
    model_hash = hashlib.md5(pickle.dumps(model_obj)).hexdigest()[:8]
    filename   = f"{model_name.lower().replace(' ','_')}_{version}_{model_hash}.pkl"

    model_path = MODELS_DIR / filename
    model_path.write_bytes(pickle.dumps({"model": model_obj, "feature_names": feature_names}))
    # all my lovely models are supposed to be well documented
    entry = {
        "version": version, "model_name": model_name, "filename": filename,
        "metrics": metrics, "feature_names": feature_names,
        "training_size": training_size, "trained_at": datetime.now().isoformat(),
        "hash": model_hash,
    }
    reg["versions"].append(entry)
    reg["active_version"] = version

    keep = MODEL_CFG.get("keep_top_n", 10)
    if len(reg["versions"]) > keep:
        for old in reg["versions"][:-keep]:
            p = MODELS_DIR / old["filename"]
            if p.exists(): p.unlink()
        reg["versions"] = reg["versions"][-keep:]

    _save_registry(reg)
    print(f"[Registry] {model_name} → {version}")
    return version

def load_active_model():
    reg = _load_registry()
    if not reg["active_version"] or not reg["versions"]:
        return None, None
    entry = next((v for v in reg["versions"] if v["version"] == reg["active_version"]), None)
    if not entry:
        return None, None
    path = MODELS_DIR / entry["filename"]
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], entry

def set_active_version(version) -> bool:
    reg = _load_registry()
    if any(v["version"] == version for v in reg["versions"]):
        reg["active_version"] = version
        _save_registry(reg)
        return True
    return False


# LEARNING LOG
# got to log the EE hours somehow....

def _load_log() -> list:
    if LEARN_LOG.exists():
        with open(LEARN_LOG) as f:
            return json.load(f)
    return []

def _append_log(entry: dict):
    log = _load_log()
    log.append(entry)
    LEARN_LOG.parent.mkdir(exist_ok=True)
    with open(LEARN_LOG, "w") as f:
        json.dump(log, f, indent=2)



# FEATURE PREP + MODEL DEFINITIONS

def prepare_data(data):
    cfg_features = DATA_CFG["features"]
    label        = DATA_CFG["label"]
    valid        = [g for g in data if all(feat in g for feat in cfg_features) and label in g]
    if not valid:
        raise ValueError("No valid records with required features.")
    X = np.array([[g[feat] for feat in cfg_features] for g in valid], dtype=float)
    y = np.array([int(g[label]) for g in valid])
    return X, y, cfg_features

# call me gustav for I am cooking here.
def build_models() -> dict:
    enabled = MODEL_CFG.get("enabled", [])
    mc      = MODEL_CFG
    pipe    = {}

    if "gradient_boosting" in enabled:
        c = mc.get("gradient_boosting", {})
        pipe["Gradient Boosting"] = Pipeline([("s", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=c.get("n_estimators",200), learning_rate=c.get("learning_rate",0.05),
                max_depth=c.get("max_depth",4), subsample=c.get("subsample",0.8),
                min_samples_split=c.get("min_samples_split",5), random_state=c.get("random_state",42)))])
    if "random_forest" in enabled:
        c = mc.get("random_forest", {})
        pipe["Random Forest"] = Pipeline([("s", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=c.get("n_estimators",200), max_depth=c.get("max_depth",12),
                min_samples_split=c.get("min_samples_split",5), min_samples_leaf=c.get("min_samples_leaf",2),
                random_state=c.get("random_state",42)))])
    if "extra_trees" in enabled:
        c = mc.get("extra_trees", {})
        pipe["Extra Trees"] = Pipeline([("s", StandardScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators=c.get("n_estimators",200), max_depth=c.get("max_depth",12),
                min_samples_split=c.get("min_samples_split",5), random_state=c.get("random_state",42)))])
    if "svm" in enabled:
        c = mc.get("svm", {})
        pipe["SVM (RBF)"] = Pipeline([("s", StandardScaler()),
            ("clf", SVC(kernel=c.get("kernel","rbf"), C=c.get("C",1.0), gamma=c.get("gamma","scale"),
                        probability=True, random_state=c.get("random_state",42)))])
    if "mlp" in enabled:
        c = mc.get("mlp", {})
        pipe["Neural Network (MLP)"] = Pipeline([("s", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=tuple(c.get("hidden_layer_sizes",[128,64,32])),
                                  activation=c.get("activation","relu"), max_iter=c.get("max_iter",500),
                                  early_stopping=True, validation_fraction=0.1,
                                  random_state=c.get("random_state",42)))])
    if "xgboost" in enabled:
        try:
            from xgboost import XGBClassifier  # noqa: PLC0415
            c = mc.get("xgboost", {})
            pipe["XGBoost"] = Pipeline([("s", StandardScaler()),
                ("clf", XGBClassifier(
                    n_estimators=c.get("n_estimators",200), learning_rate=c.get("learning_rate",0.05),
                    max_depth=c.get("max_depth",4), subsample=c.get("subsample",0.8),
                    colsample_bytree=c.get("colsample_bytree",0.8),
                    eval_metric="logloss", random_state=c.get("random_state",42), verbosity=0))])
        except ImportError:
            print("[Models] XGBoost not installed — skipping.")  # noqa: PLC0415
    return pipe # I do not know what I am cooking here

def compute_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    m = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    if hasattr(model, "predict_proba"):
        m["roc_auc"] = round(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]), 4)
    else:
        m["roc_auc"] = m["accuracy"]
    return m

def get_feature_importances(model, feature_names):
    clf = model.named_steps.get("clf")
    if clf is None: return None
    if hasattr(clf, "feature_importances_"):
        imps = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imps = np.abs(clf.coef_[0]) if clf.coef_.ndim > 1 else np.abs(clf.coef_)
    else:
        return None
    return dict(zip(feature_names, [round(float(v), 6) for v in imps]))


# TRAINING
def train_and_evaluate(storage="local", triggered_by="manual"):
    print(f"\n{'='*70}\nTRAINING ({triggered_by})\n{'='*70}")
    data = load_data(storage)
    if len(data) < DATA_CFG.get("min_games_required", 50):
        print(f"[Train] Not enough data ({len(data)}). Skipping.")
        return None

    X, y, feature_names = prepare_data(data)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=DATA_CFG["test_size"],
        random_state=DATA_CFG["random_state"], stratify=y
    )
    print(f"Dataset: {len(data)} games  Train:{len(X_tr)}  Test:{len(X_te)}")
    print(f"Home win rate: {y.mean():.2%}\n")

    models    = build_models()
    results   = {}
    sel_metric = MODEL_CFG.get("selection_metric", "roc_auc")

    for name, model in models.items():
        print(f"▶ {name}...")
        model.fit(X_tr, y_tr)
        m  = compute_metrics(model, X_te, y_te)
        fi = get_feature_importances(model, feature_names)
        if fi: m["feature_importances"] = fi
        cv = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
        m["cv_roc_auc_mean"] = round(float(cv.mean()), 4)
        m["cv_roc_auc_std"]  = round(float(cv.std()), 4)
        results[name] = {"model": model, "metrics": m}
        print(f"  Acc:{m['accuracy']:.4f} F1:{m['f1']:.4f} "
              f"AUC:{m['roc_auc']:.4f} CV-AUC:{m['cv_roc_auc_mean']:.4f}±{m['cv_roc_auc_std']:.4f}")
        #........................ bruh
    best_name = max(results, key=lambda k: results[k]["metrics"].get(sel_metric, 0))
    best      = results[best_name]
    print(f"\n{'='*70}\nBEST: {best_name}  {sel_metric}={best['metrics'].get(sel_metric,'?')}\n{'='*70}\n")

    if triggered_by != "manual":
        _, active_entry = load_active_model()
        threshold = AL_CFG.get("promote_threshold", 0.002)
        if active_entry:
            current_auc = active_entry["metrics"].get("roc_auc", 0)
            new_auc     = best["metrics"].get("roc_auc", 0)
            if new_auc < current_auc + threshold:
                msg = (f"New AUC {new_auc:.4f} vs current {current_auc:.4f} "
                       f"(threshold +{threshold}). Skipping promotion.")
                print(f"[AutoLearn] {msg}")
                _append_log({
                    "timestamp": datetime.now().isoformat(), "triggered_by": triggered_by,
                    "result": "skipped", "reason": msg, "best_model": best_name,
                    "new_auc": new_auc, "current_auc": current_auc, "dataset_size": len(data),
                }) 
                return None # don't promote, but still log the attempt

    version = register_model(
        model_name=best_name, model_obj=best["model"],
        metrics=best["metrics"], feature_names=feature_names, training_size=len(X_tr),
    )

    comparison = {
        n: {k: v for k, v in r["metrics"].items() if k != "feature_importances"}
        for n, r in results.items()
    }
    snap = {
        "version": version, "trained_at": datetime.now().isoformat(),
        "best_model": best_name, "selection_metric": sel_metric,
        "results": comparison,
        "feature_importances": {n: r["metrics"].get("feature_importances",{}) for n,r in results.items()},
        "triggered_by": triggered_by, "dataset_size": len(data),
    }
    comp_file = MODELS_DIR / f"comparison_{version}.json"
    with open(comp_file, "w") as f: json.dump(_sanitize(snap), f, indent=2)
    import shutil
    shutil.copy(comp_file, MODELS_DIR / "latest_comparison.json")

    _append_log({
        "timestamp": datetime.now().isoformat(), "triggered_by": triggered_by,
        "result": "promoted", "version": version, "model_name": best_name,
        "roc_auc": best["metrics"].get("roc_auc",0), "f1": best["metrics"]["f1"],
        "accuracy": best["metrics"]["accuracy"], "dataset_size": len(data),
    })
    print(f"[Train] Registered & promoted → {version}")
    return {"version": version, "model_name": best_name, "metrics": best["metrics"]}



# AUTO-LEARNING SCHEDULER

# No clanker of mine will go into this world without the ability to learn from it's mistakes.
class AutoLearnScheduler:
    def __init__(self, storage="local"):
        self.storage          = storage
        self.fetch_interval   = AL_CFG.get("fetch_interval_hours", 6)   * 3600
        self.retrain_interval = AL_CFG.get("retrain_interval_hours", 24) * 3600
        self.min_new_games    = AL_CFG.get("min_new_games_to_retrain", 15)
        self._thread          = None
        self._stop            = threading.Event()
        self._last_fetch      = 0.0
        self._last_retrain    = 0.0
        self._status          = "idle"

    def start(self):
        if not AL_CFG.get("enabled", True):
            print("[AutoLearn] Disabled in config.")
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[AutoLearn] Scheduler started — "
              f"fetch every {self.fetch_interval//3600}h, "
              f"retrain every {self.retrain_interval//3600}h.")

    def stop(self):
        self._stop.set()

    def _loop(self):
        time.sleep(60)
        while not self._stop.is_set():
            now = time.time()
            try:
                if now - self._last_fetch >= self.fetch_interval:
                    self._status = "fetching"
                    print("[AutoLearn] Fetching new games...")
                    new_games = fetch_ncaa_data()
                    added = append_to_json(new_games) if new_games else 0
                    self._last_fetch = time.time()
                    print(f"[AutoLearn] {added} new games added.")
                    if added >= self.min_new_games:
                        self._status = "training"
                        print(f"[AutoLearn] {added} new games → retraining...")
                        train_and_evaluate(self.storage, triggered_by="new_data")
                        self._last_retrain = time.time()
                elif now - self._last_retrain >= self.retrain_interval:
                    self._status = "training"
                    print("[AutoLearn] Scheduled retrain...")
                    train_and_evaluate(self.storage, triggered_by="scheduler")
                    self._last_retrain = time.time()
            except Exception as e:
                print(f"[AutoLearn] Error: {e}")
                traceback.print_exc()
            finally:
                self._status = "idle"
            for _ in range(60):
                if self._stop.is_set(): break
                time.sleep(60)

    def get_state(self) -> dict:
        def countdown(last, interval):
            rem = max(0, int(last + interval - time.time()))
            h, m = divmod(rem // 60, 60)
            return f"{h}h {m}m" if h else f"{m}m"
        return {
            "enabled":            AL_CFG.get("enabled", True),
            "status":             self._status,
            "fetch_interval_h":   self.fetch_interval   // 3600,
            "retrain_interval_h": self.retrain_interval // 3600,
            "min_new_games":      self.min_new_games,
            "promote_threshold":  AL_CFG.get("promote_threshold", 0.002),
            "next_fetch_in":      countdown(self._last_fetch,   self.fetch_interval),
            "next_retrain_in":    countdown(self._last_retrain, self.retrain_interval),
            "last_fetch":    datetime.fromtimestamp(self._last_fetch).isoformat()   if self._last_fetch   else None,
            "last_retrain":  datetime.fromtimestamp(self._last_retrain).isoformat() if self._last_retrain else None,
        }


_scheduler = AutoLearnScheduler()

_roster_progress: dict = {}


# FLASK HELPERS

def _sanitize(obj): # thank Claude..... Pun intended....
    #I thank you claude for helping me fix this error i was getting
    """Recursively replace float NaN/Inf with None so json.dumps produces valid JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
        return None
    return obj


app = Flask(__name__)


# FLASK ROUTES — CORE

@app.route("/")
def serve_dashboard():
    return send_file("dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Predict from raw feature values (stats mode)."""
    model, entry = load_active_model()
    if model is None:
        return jsonify({"error": "No trained model. Run --train first."}), 400
    try:
        payload       = request.json
        feature_names = entry.get("feature_names", DATA_CFG["features"])
        X             = np.array([[float(payload[f]) for f in feature_names]])
        pred          = int(model.predict(X)[0])
        conf          = float(max(model.predict_proba(X)[0])) if hasattr(model, "predict_proba") else None
        return jsonify({
            "prediction":       "Home Win" if pred == 1 else "Away Win",
            "prediction_value": pred,
            "confidence":       conf,
            "model_name":       entry["model_name"],
            "version":          entry["version"],
        })
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/from_roster", methods=["POST"])
def predict_from_roster():
    """
    Predict from selected player lists (roster mode).

    Request body:
    {
      "home_players": [ {ppg, rpg, apg, spg, bpg, tov, fg_pct, fgm, fga}, ... ],
      "away_players": [ {...}, ... ]
    }

    Aggregates individual stats into team-level features, then runs through
    the same model as stats mode. Also returns computed_stats so the dashboard
    can show what numbers were actually used.
    """
    model, entry = load_active_model()
    if model is None:
        return jsonify({"error": "No trained model. Run --train first."}), 400
    try:
        payload      = request.json
        home_players = payload.get("home_players", [])
        away_players = payload.get("away_players", [])

        if not home_players:
            return jsonify({"error": "No home players selected."}), 400
        if not away_players:
            return jsonify({"error": "No away players selected."}), 400

        home_stats = compute_stats_from_roster(home_players, "home")
        away_stats = compute_stats_from_roster(away_players, "away")
        combined   = {**home_stats, **away_stats}

        feature_names = entry.get("feature_names", DATA_CFG["features"])

        # Fill any missing features with 0  shouldn't happen but just in case, graceful as F#@K
        X = np.array([[float(combined.get(f, 0)) for f in feature_names]])

        pred = int(model.predict(X)[0])
        conf = float(max(model.predict_proba(X)[0])) if hasattr(model, "predict_proba") else None

        return jsonify({
            "prediction":       "Home Win" if pred == 1 else "Away Win",
            "prediction_value": pred,
            "confidence":       conf,
            "model_name":       entry["model_name"],
            "version":          entry["version"],
            "computed_stats":   combined,   # so dashboard can show what was used
            "home_count":       len(home_players),
            "away_count":       len(away_players),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/analytics")
def analytics():
    try:
        data = load_from_json()
        if not data:
            return jsonify({"error": "No data."}), 400
        total      = len(data)
        home_wins  = sum(1 for g in data if g.get("outcome") == 1)
        cfg_feats  = DATA_CFG["features"]
        hw_games   = [g for g in data if g.get("outcome") == 1]
        aw_games   = [g for g in data if g.get("outcome") == 0]
        def avg(games, field):
            v = [g[field] for g in games if field in g]; return round(sum(v)/len(v),4) if v else 0
        comp_file = MODELS_DIR / "latest_comparison.json"
        mc = fi = {}
        if comp_file.exists():
            with open(comp_file) as f: c = json.load(f)
            mc = c.get("results",{}); fi = c.get("feature_importances",{})
        sources = {}
        for g in data: sources[g.get("source","unknown")] = sources.get(g.get("source","unknown"),0)+1
        return jsonify(_sanitize({
            "total_games": total, "home_wins": home_wins, "away_wins": total-home_wins,
            "home_win_rate": round(home_wins/total,4) if total else 0,
            "feature_stats": {
                "home_win": {feat: avg(hw_games, feat) for feat in cfg_feats},
                "away_win": {feat: avg(aw_games, feat) for feat in cfg_feats},
            },
            "model_comparison": mc, "feature_importances": fi, "data_sources": sources,
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model_info")
def model_info():
    _, entry = load_active_model()
    if entry is None: return jsonify({"error": "No active model."}), 400
    return jsonify(entry)


@app.route("/registry")
def registry():
    return jsonify(_load_registry())


@app.route("/registry/activate/<version>", methods=["POST"])
def activate_version(version):
    if set_active_version(version):
        return jsonify({"status": "ok", "active_version": version})
    return jsonify({"error": f"Version {version} not found."}), 404


@app.route("/debug")
def debug():
    data = load_from_json()
    comp_file = MODELS_DIR / "latest_comparison.json"
    _, entry  = load_active_model()
    return jsonify({
        "data_file":         str(LOCAL_FILE),
        "data_file_exists":  LOCAL_FILE.exists(),
        "game_count":        len(data),
        "comparison_exists": comp_file.exists(),
        "active_model":      entry.get("model_name") if entry else None,
        "active_version":    entry.get("version") if entry else None,
        "registry_exists":   REGISTRY_FILE.exists(),
        "roster_dir":        str(ROSTER_DIR),
        "team_id_cache":     str(TEAM_ID_CACHE),
        "cwd":               os.getcwd(),
    })


@app.route("/features")
def features():
    return jsonify({
        "features": DATA_CFG["features"],
        "rolling_windows": ROLLING_CFG.get("available_windows", [5, 10, 15, 20]),
        "default_window":  ROLLING_CFG.get("default_window", 12),
    })



# FLASK ROUTES, TEAMS & ROLLING AVERAGES

@app.route("/teams")
def teams(): # i know this is a bit of a mess but it was the quickest way to get rolling stats without re-processing everything on the dashboard side. 
    # I promise I'll refactor this into a proper API layer later.
    data   = load_from_json()
    if not data: return jsonify({"error": "No data."}), 400
    window = int(request.args["window"]) if "window" in request.args else None
    ts     = build_team_stats(data, window=window)
    teams_list = sorted(
        [{"name": n, **s} for n, s in ts.items() if s.get("games_played", 0) >= 3],
        key=lambda x: x["name"]
    )
    return jsonify({"teams": teams_list, "count": len(teams_list), "window": window})


@app.route("/team_stats/<path:team_name>")
def team_stats(team_name):
    data   = load_from_json()
    if not data: return jsonify({"error": "No data."}), 400
    window = int(request.args["window"]) if "window" in request.args else None
    ts     = build_team_stats(data, window=window)
    if team_name in ts:
        return jsonify({"name": team_name, "stats": ts[team_name], "window": window})
    matches = [k for k in ts if team_name.lower() in k.lower()]
    if matches:
        return jsonify({"name": matches[0], "stats": ts[matches[0]], "window": window})
    return jsonify({"error": f"Team '{team_name}' not found."}), 404
    # basically to dumb it down for the dashboard search box, which does a simple substring match against team names. 
    # This way it can handle minor typos or variations without needing an exact match.
    # I know I am awesome


@app.route("/home_team")
def home_team_endpoint():
    data   = load_from_json()
    window = int(request.args["window"]) if "window" in request.args else None
    cfg    = {"name": HT_CFG["name"], "court": HT_CFG.get("court_name",""), "espn_id": HT_CFG.get("espn_id","")}
    ht     = get_home_team_stats(data, window=window) if data else None
    return jsonify({"config": cfg, "stats": ht, "window": window})


# FLASK ROUTES, ROSTERS

# This is a bit more complex due to the asynchronous fetching and caching of rosters.
# very last moment inspiration here
@app.route("/roster/<path:team_name>")
def get_roster_route(team_name):
    """
    Kicks off a background fetch and returns immediately with status "loading".
    Dashboard polls /roster/progress/<team> every second to get players as they appear.
    Pass ?force=1 to bypass cache.
    """
    force   = request.args.get("force", "0") == "1"
    fetcher = RosterFetcher()
    fetcher.fetch_team_async(team_name, force=force)
    # Return current progress state immediately (may already be "ready" if cached)
    prog = _roster_progress.get(team_name, {"status": "loading", "players": [], "done": 0, "total": 0})
    return jsonify(prog)


@app.route("/roster/progress/<path:team_name>")
def roster_progress(team_name):
    """
    Returns current fetch progress for a team.
    Dashboard polls this every second while status == "loading".
    { status: "loading"|"ready"|"error", players: [...], done: N, total: M }
    """
    prog = _roster_progress.get(team_name)
    if prog is None:
        return jsonify({"status": "not_started", "players": [], "done": 0, "total": 0})
    return jsonify(prog)


@app.route("/roster/refresh/<path:team_name>", methods=["POST"])
def refresh_roster(team_name):
    """Force a fresh ESPN fetch for a team's roster."""
    fetcher = RosterFetcher()
    fetcher.fetch_team_async(team_name, force=True)
    return jsonify({"status": "started", "team": team_name})


# FLASK ROUTES, AUTO-LEARN

# keep the user informed of what's going on in the mysterious training dungeon
@app.route("/autolearn/status")
def autolearn_status():
    return jsonify(_scheduler.get_state())
# you would not think with how pretty the dashboard looks 
# the code would be a "pretty princess too".... well shoot call this code buster and dip it in the oil


@app.route("/autolearn/trigger", methods=["POST"])
def autolearn_trigger():
    def _run(): train_and_evaluate("local", triggered_by="manual_trigger")
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/learning_log")
def learning_log():
    n   = request.args.get("n", 50, type=int)
    log = _load_log()
    return jsonify({"log": log[-n:], "total": len(log)})



# CLI

def main():
    parser = argparse.ArgumentParser(description="Basketball Predictor v2.2")
    parser.add_argument("--fetch",               action="store_true", help="Fetch real NCAA games from ESPN")
    parser.add_argument("--fetch-rosters",        action="store_true", help="Pre-fetch rosters for all teams in dataset")
    parser.add_argument("--generate-synthetic",   action="store_true", help="Generate synthetic fallback data")
    parser.add_argument("--train",                action="store_true", help="Train all models, register best")
    parser.add_argument("--serve",                action="store_true", help="Start web server + auto-learn scheduler")
    parser.add_argument("--storage", choices=["local","snowflake"], default="local")
    parser.add_argument("--activate", metavar="VERSION")
    parser.add_argument("--list-models",          action="store_true")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    # I love you 
    
    if args.list_models:
        reg = _load_registry()
        if not reg["versions"]: print("No registered models."); return
        print(f"\n{'Ver':<6} {'Model':<28} {'AUC':<8} {'F1':<8} Trained At")
        print("-"*70)
        for v in reg["versions"]:
            m      = v["metrics"]
            active = " ◀ ACTIVE" if v["version"] == reg["active_version"] else ""
            print(f"{v['version']:<6} {v['model_name']:<28} "
                  f"{m.get('roc_auc',0):.4f}   {m.get('f1',0):.4f}   "
                  f"{v['trained_at'][:19]}{active}")
        return

    if args.activate:
        ok = set_active_version(args.activate)
        print(f"Active → {args.activate}" if ok else f"Version {args.activate} not found.")
        return

    if args.fetch:
        games = fetch_ncaa_data()
        if games:
            append_to_snowflake(games) if args.storage == "snowflake" else append_to_json(games)
        return

    if args.fetch_rosters:
        data = load_from_json()
        if not data:
            print("[Roster] No game data found. Run --fetch first.")
            return
        # Collect unique team names
        teams_seen = set()
        for g in data:
            teams_seen.add(g.get("home_team","").strip())
            teams_seen.add(g.get("away_team","").strip())
        teams_seen.discard("")
        print(f"[Roster] Pre-fetching rosters for {len(teams_seen)} teams...")
        fetcher = RosterFetcher()
        ok = fail = 0
        for name in sorted(teams_seen):
            result = fetcher.fetch_team(name)
            if result: ok += 1
            else:       fail += 1
        print(f"[Roster] Done. Success: {ok}, Failed: {fail}")
        return

    if args.generate_synthetic:
        data = _generate_synthetic(500)
        save_to_snowflake(data) if args.storage == "snowflake" else save_to_json(data)
        return

    if args.train:
        train_and_evaluate(args.storage, triggered_by="manual")
        return

    if args.serve:
        print(f"\n{'='*70}")
        print(f"  {APP_CFG['name']}  v{APP_CFG['version']}")
        print(f"  Dashboard  → http://localhost:{APP_CFG['port']}")
        print(f"  Home team  : {HT_CFG['name']}")
        print(f"  Auto-learn : {'ON' if AL_CFG.get('enabled') else 'OFF'}")
        print(f"{'='*70}\n")
        _scheduler.storage = args.storage
        _scheduler.start()
        app.run(debug=APP_CFG.get("debug", False),
                port=APP_CFG.get("port", 5000),
                host=APP_CFG.get("host", "0.0.0.0"),
                use_reloader=False)
        return

    parser.print_help()


# SYNTHETIC FALLBACK

# you fools, you thought depriving me of my API access would stop me?
# You have activated my trap card.
# I summon from my deck: THE SYNTHETIC DATA GENERATOR!
def _generate_synthetic(num_games=500):
    rng       = np.random.default_rng(42)
    home_name = HT_CFG["name"]
    pool      = [home_name] + [f"Team_{i}" for i in range(1, 65)]
    data      = []
    for i in range(num_games):
        ht = pool[i % len(pool)]
        at = pool[(i*7+3) % len(pool)]
        if ht == at: at = pool[(i*7+4) % len(pool)]

        hp,ap   = rng.uniform(60,95,2)
        hfg,afg = rng.uniform(0.38,0.55,2)
        hrb,arb = rng.uniform(28,48,2)
        ha,aa   = rng.uniform(10,22,2)
        ht_,at_ = rng.uniform(8,17,2)
        hst,ast = rng.uniform(4,10,2)
        hbl,abl = rng.uniform(2,8,2)

        h_str = hp*0.3+hfg*80+hrb*0.5+ha*0.8-ht_*0.6+hst*0.4+3
        a_str = ap*0.3+afg*80+arb*0.5+aa*0.8-at_*0.6+ast*0.4
        outcome = (1 if h_str>a_str else 0) if rng.random()>0.15 else (0 if h_str>a_str else 1)

        data.append({
            "game_id": f"SYN_{i+1:05d}", "home_team": ht, "away_team": at,
            "home_ppg":round(float(hp),2),       "away_ppg":round(float(ap),2),
            "home_fg_pct":round(float(hfg),4),   "away_fg_pct":round(float(afg),4),
            "home_rebounds":round(float(hrb),2), "away_rebounds":round(float(arb),2),
            "home_assists":round(float(ha),2),   "away_assists":round(float(aa),2),
            "home_turnovers":round(float(ht_),2),"away_turnovers":round(float(at_),2),
            "home_steals":round(float(hst),2),   "away_steals":round(float(ast),2),
            "home_blocks":round(float(hbl),2),   "away_blocks":round(float(abl),2),
            "outcome":int(outcome), "source":"synthetic",
        })
    print(f"[Synthetic] Generated {len(data)} games.")
    return data # very linear data


if __name__ == "__main__":
    main()

# Current status: "I HAVE AN IDEA!!! (croods reference)" level.
# live data is done.
# much better than before, but now only some improvements remaining.
# this.... this is good.....

# coffee log 22 -> 23
