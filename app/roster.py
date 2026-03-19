"""
app/roster.py
─────────────
ESPN roster fetching, per-player stat aggregation, and the shared
_roster_progress dict that api.py polls for live updates.

ESPN endpoints used:
  GET /teams?limit=1000         → all teams + IDs
  GET /teams/{id}/roster        → player list (with embedded stats where available)
  GET /athletes/{id}/statistics → player season averages (fallback, 404-prone)

Results are cached per-team in data/rosters/<team_id>.json with a TTL
configured in config.yaml → roster.cache_ttl_hours.
"""

import json
import threading
import numpy as np
from datetime import datetime
from pathlib import Path

from app.config import API_CFG, ROSTER_CFG, ROSTER_DIR, TEAM_ID_CACHE
from app.logger import get_logger

log = get_logger(__name__)

# Shared progress state — api.py imports and reads this dict.
# RosterFetcher writes into it as players are fetched so the dashboard
# can poll /roster/progress/<team> and render players as they appear.
_roster_progress: dict = {}


def _empty_player_stats() -> dict:
    return {
        "ppg": 0.0, "rpg": 0.0, "apg": 0.0, "spg": 0.0, "bpg": 0.0,
        "tov": 0.0, "fg_pct": 0.45, "fgm": 0.0, "fga": 0.0,
    }


def compute_stats_from_roster(players: list[dict], side: str) -> dict:
    """
    Aggregate individual player stats into team-level feature values.
    side: "home" or "away" — sets the feature name prefix.

    Player stats from the roster are season averages — not in-game scores.
    So these are already pre-game values by nature. No leakage here.
    The model feature names match exactly what the training data contains.

    Model features (fed to prediction):
      fg_pct, rebounds, assists, turnovers, steals, blocks, ast_to_tov

    Insight features (display only — dashboard context, never to model):
      ppg total, eff_score, scoring_dominance, depth_signal,
      role_balance, efficiency_spread
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

    total_fgm = sum(p.get("fgm", 0) for p in players)
    total_fga = sum(p.get("fga", 0) for p in players)
    if total_fga > 0:
        fg_pct = total_fgm / total_fga
    else:
        # Fall back to simple average of individual FG%
        fg_pcts = [p.get("fg_pct", 0.45) for p in players]
        fg_pct  = sum(fg_pcts) / len(fg_pcts)

    ast_to_tov = round(total_apg / (total_tov if total_tov > 0 else 0.1), 3)
    eff_score  = round(total_ppg * fg_pct, 2)  # insight only

    ppg_values = [p.get("ppg", 0) for p in players if p.get("ppg", 0) > 0]

    # Scoring dominance — high std dev = one player carrying the team
    scoring_dominance = round(float(np.std(ppg_values)), 3) if len(ppg_values) > 1 else 0.0

    # Depth signal — count of meaningful contributors (≥5 ppg)
    depth_signal = sum(1 for p in players if p.get("ppg", 0) >= 5.0)

    # Role balance — top scorer's share of total ppg
    role_balance = (
        round(max(ppg_values) / (total_ppg if total_ppg > 0 else 0.1), 3)
        if ppg_values else 0.0
    )

    # Efficiency spread — std dev of FG% across players who actually attempt shots
    fg_pct_values     = [p.get("fg_pct", 0.45) for p in players if p.get("fga", 0) > 0]
    efficiency_spread = round(float(np.std(fg_pct_values)), 4) if len(fg_pct_values) > 1 else 0.0

    return { # oh that's why
        # model features — must match config.yaml features list
        f"{prefix}fg_pct":     round(fg_pct, 4),
        f"{prefix}rebounds":   round(total_rpg, 2),
        f"{prefix}assists":    round(total_apg, 2),
        f"{prefix}turnovers":  round(total_tov, 2),
        f"{prefix}steals":     round(total_spg, 2),
        f"{prefix}blocks":     round(total_bpg, 2),
        f"{prefix}ast_to_tov": ast_to_tov,
        # insight features — dashboard display only, never fed to model
        f"insight_{side}_ppg":               total_ppg,
        f"insight_{side}_eff_score":         eff_score,
        f"insight_{side}_scoring_dominance": scoring_dominance,
        f"insight_{side}_depth_signal":      float(depth_signal),
        f"insight_{side}_role_balance":      role_balance,
        f"insight_{side}_efficiency_spread": efficiency_spread,
    }


class RosterFetcher:
    """
    Fetches NCAA team rosters and individual player season stats from ESPN.
    Results are cached per-team in data/rosters/<team_id>.json to avoid
    hammering the API on every request.
    """

    TEAMS_URL   = API_CFG["espn"]["base_url"] + API_CFG["espn"].get("teams_path",    "/teams")
    ATHLETE_URL = API_CFG["espn"]["base_url"] + API_CFG["espn"].get("athletes_path", "/athletes")

    def __init__(self):
        import requests
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (research)"})
        self.delay = API_CFG.get("rate_limit_delay", 0.4)
        self.ttl_h = ROSTER_CFG.get("cache_ttl_hours", 24)

    def _get(self, url, params=None):
        import time
        try:
            r = self.session.get(url, params=params, timeout=10)
            r.raise_for_status()
            time.sleep(self.delay)
            return r.json()
        except Exception as e:
            log.debug("  [Roster] %s: %s", url, e)
            return None

    # ── team ID lookup ──────────────────────────────────────────────────────

    @staticmethod
    def _load_team_id_cache() -> dict:
        if TEAM_ID_CACHE.exists():
            with open(TEAM_ID_CACHE, encoding="utf-8") as f:
                return json.load(f)
        return {}

    @staticmethod
    def _save_team_id_cache(cache: dict):
        with open(TEAM_ID_CACHE, "w", encoding="utf-8") as f:
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
                return v  # backup

        # Cache miss — fetch full team list from ESPN
        log.info("[Roster] Looking up ESPN ID for '%s'...", team_name)
        data = self._get(self.TEAMS_URL, params={"limit": 1000})
        if not data:
            return None  # backup's backup

        # ESPN wraps teams in sports[0].leagues[0].teams
        try:
            teams_raw = (
                data.get("sports", [{}])[0]
                    .get("leagues", [{}])[0]
                    .get("teams", [])
            )
        except (IndexError, KeyError):
            teams_raw = []

        for entry in teams_raw:
            t = entry.get("team", {})
            name = t.get("displayName", "")
            tid  = t.get("id", "")
            if name and tid:
                cache[name] = tid  # backup prep

        self._save_team_id_cache(cache)

        if team_name in cache:
            return cache[team_name]
        for k, v in cache.items():
            if team_name.lower() in k.lower() or k.lower() in team_name.lower():
                return v  # nuclear option

        log.warning("[Roster] Could not find ESPN ID for '%s'.", team_name)
        return None

    # ── roster fetch (utter boilerplate DO NOTT TOUCH!!!) ──────────────────

    @staticmethod
    def _cache_path(team_id: str) -> Path:
        return ROSTER_DIR / f"{team_id}.json"

    def _cache_valid(self, team_id: str) -> bool:
        p = self._cache_path(team_id)
        if not p.exists():
            return False
        cached  = json.loads(p.read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(cached.get("fetched_at", "2000-01-01"))
        return (datetime.now() - fetched).total_seconds() < self.ttl_h * 3600

    def _load_cached(self, team_id: str) -> dict | None:
        p = self._cache_path(team_id)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def _save_cached(self, team_id: str, data: dict):
        self._cache_path(team_id).write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def get_roster(self, team_id: str) -> list[dict]:
        """
        Player list from /teams/{id}/roster.
        Tries embedded stats first to avoid 404s on /statistics endpoint.
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
            embedded = RosterFetcher._parse_embedded_stats(athlete)
            player.update(embedded)
            players.append(player)
        return players

    def get_player_stats(self, player_id: str) -> dict:
        """
        Season averages for a player. Returns empty stats gracefully on 404.
        Better pulled via embedded stats in get_roster() but kept as fallback.
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
            for stat in (
                data.get("statistics", {})
                    .get("splits", {})
                    .get("categories", [{}])[0]
                    .get("stats", [])
            ):
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

    @staticmethod
    def _parse_embedded_stats(athlete: dict) -> dict:
        """
        Pull stats from the player object already embedded in the roster response.
        ESPN includes a 'statistics' array directly on each athlete — this is
        much more reliable than the separate /statistics endpoint.
        """
        raw = {}
        for stat_group in athlete.get("statistics", []):
            for stat in stat_group.get("stats", []):
                raw[stat.get("name","").lower()] = stat.get("value", 0) or 0
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
        """Full pipeline: name → ESPN ID → roster → player stats. Results cached."""
        team_id = self.get_team_id(team_name)
        if not team_id:
            return None

        if not force and self._cache_valid(team_id):
            log.debug("[Roster] Cache hit for %s (ID %s)", team_name, team_id)
            return self._load_cached(team_id)

        log.info("[Roster] Fetching roster for %s (ID %s)...", team_name, team_id)
        players = self.get_roster(team_id)
        if not players:
            log.warning("[Roster] No players returned for %s.", team_name)
            return None

        has_embedded = sum(1 for p in players if p.get("ppg", 0) > 0)
        log.info("[Roster] %d players. %d have embedded stats.", len(players), has_embedded)

        # Write initial progress — names visible immediately, stats fill in after
        _roster_progress[team_name] = {
            "status": "loading",
            "players": list(players),
            "done":    has_embedded,
            "total":   len(players),
        }

        needs_fetch = [p for p in players if p.get("ppg", 0) == 0 and p.get("id")]
        if needs_fetch:
            log.info("[Roster] Fetching stats for %d players missing data...", len(needs_fetch))
            for i, player in enumerate(needs_fetch):
                fetched = self.get_player_stats(player["id"])
                if any(v > 0 for k, v in fetched.items() if k != "fg_pct"):
                    player.update(fetched)
                _roster_progress[team_name] = {
                    "status":  "loading",
                    "players": list(players),
                    "done":    has_embedded + i + 1,
                    "total":   len(players),
                }
                if (i+1) % 5 == 0:
                    log.debug("  %d/%d done", i+1, len(needs_fetch))

        result = {
            "team_name":  team_name,
            "team_id":    team_id,
            "players":    players,
            "fetched_at": datetime.now().isoformat(),
        }
        self._save_cached(team_id, result)
        _roster_progress[team_name] = {
            "status":  "ready",
            "players": players,
            "done":    len(players),
            "total":   len(players),
        }
        log.info("[Roster] Done. %d players cached for %s.", len(players), team_name)
        return result

    def fetch_team_async(self, team_name: str, force: bool = False):
        """
        Non-blocking version of fetch_team. Starts a background thread,
        writes incremental progress to _roster_progress[team_name].
        The dashboard polls /roster/progress/<team> every second.
        """
        team_id = self.get_team_id(team_name)
        if team_id and not force and self._cache_valid(team_id):
            cached = self._load_cached(team_id)
            if cached:
                _roster_progress[team_name] = {
                    "status":  "ready",
                    "players": cached["players"],
                    "done":    len(cached["players"]),
                    "total":   len(cached["players"]),
                }
                return  # already done, no thread needed

        _roster_progress[team_name] = {
            "status": "loading", "players": [], "done": 0, "total": 0,
        }

        def _run():
            try:
                result = self.fetch_team(team_name, force=force)
                if result is None:
                    _roster_progress[team_name] = {
                        "status":  "error",
                        "players": [], "done": 0, "total": 0,
                        "message": f"Could not fetch roster for '{team_name}'.",
                    }
            except Exception as e:
                _roster_progress[team_name] = {
                    "status":  "error",
                    "players": [], "done": 0, "total": 0,
                    "message": str(e),
                }

        threading.Thread(target=_run, daemon=True).start()
        # Linus torvalds... I pray to you.... I beseech thee...