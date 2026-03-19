"""
app/fetcher.py
──────────────
ESPN box-score fetcher and multi-season fetch pipeline.

All hail, Free ESPN API, the gift that keeps on giving (and giving, and giving)
no auth, no keys, no limits (well, some limits), just pure unadulterated data access.
I will offer incense prayers to the ESPN gods

v2.5: get_game_ids now returns (id, date) tuples so game_date lands on each
record — required for correct chronological ordering in the enrich step.
"""

import time
import os
from datetime import datetime, timedelta

import requests

from app.config import DATA_CFG, API_CFG
from app.enrichment import enrich_with_pregame_averages
from app.logger import get_logger

log = get_logger(__name__)


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
            log.debug("  [ESPN] %s: %s", url, e)
            return None

    def get_game_ids(self, start_date: str, end_date: str) -> list:
        """
        v2.5: returns list of (game_id, game_date) tuples instead of just IDs.
        game_date is the actual date the game was played (YYYY-MM-DD string),
        used later to sort games chronologically for the rolling average
        enrichment step. Without real dates, enrichment order is wrong.
        """
        results  = []
        current  = datetime.strptime(start_date, "%Y%m%d")
        end      = datetime.strptime(end_date,   "%Y%m%d")
        page_size = API_CFG["espn"].get("page_size", 25)
        while current <= end:
            date_str = current.strftime("%Y%m%d")
            data = self._get(
                f"{self.BASE}{API_CFG['espn']['scoreboard_path']}",
                params={"dates": date_str, "limit": page_size},
            )
            if data:
                for event in data.get("events", []):
                    # ESPN event date is ISO format e.g. "2024-01-15T19:00Z"
                    raw_date  = event.get("date", "")
                    game_date = raw_date[:10] if raw_date else current.strftime("%Y-%m-%d")
                    results.append((event["id"], game_date))
            current += timedelta(days=1)
        return results

    def get_box_score(self, event_id: str, game_date: str = ""):
        data = self._get(
            f"{self.BASE}{API_CFG['espn']['summary_path']}",
            params={"event": event_id},
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

            home_fg  = norm_pct(flt(hs, "fieldGoalPct"))
            away_fg  = norm_pct(flt(as_, "fieldGoalPct"))
            home_ast = flt(hs, "assists")
            away_ast = flt(as_, "assists")
            home_tov = flt(hs, "turnovers")
            away_tov = flt(as_, "turnovers")

            home_ast_to_tov = round(home_ast / (home_tov if home_tov > 0 else 0.1), 3)
            away_ast_to_tov = round(away_ast / (away_tov if away_tov > 0 else 0.1), 3)

            # NOTE v2.5: these in-game stats are stored as-is here.
            # enrich_with_pregame_averages() will REPLACE the feature fields
            # (home_fg_pct, home_rebounds, etc.) with pre-game rolling averages
            # and move the originals to home_game_fg_pct etc. for analytics.
            # ppg and eff_score are analytics-only regardless — not in features.
            return {
                "game_id":         f"ESPN_{event_id}",
                "game_date":       game_date,
                "home_team":       home_d.get("team",{}).get("displayName","Home"),
                "away_team":       away_d.get("team",{}).get("displayName","Away"),
                "home_score":      home_score,
                "away_score":      away_score,
                "home_ppg":        float(home_score),   # analytics only
                "away_ppg":        float(away_score),   # analytics only
                "home_fg_pct":     round(home_fg, 4),
                "away_fg_pct":     round(away_fg, 4),
                "home_rebounds":   flt(hs, "totalRebounds"),
                "away_rebounds":   flt(as_, "totalRebounds"),
                "home_assists":    home_ast,
                "away_assists":    away_ast,
                "home_turnovers":  home_tov,
                "away_turnovers":  away_tov,
                "home_steals":     flt(hs, "steals"),
                "away_steals":     flt(as_, "steals"),
                "home_blocks":     flt(hs, "blocks"),
                "away_blocks":     flt(as_, "blocks"),
                "home_ast_to_tov": home_ast_to_tov,
                "away_ast_to_tov": away_ast_to_tov,
                "home_eff_score":  round(float(home_score) * home_fg, 2),  # analytics only
                "away_eff_score":  round(float(away_score) * away_fg, 2),  # analytics only
                "outcome":         1 if home_score > away_score else 0,
                "source":          "espn",
                "fetched_at":      datetime.now().isoformat(),
                "pregame_enriched": False,  # set to True after enrich step
            } # holy cow, ESPN, I love you but this is a nightmare to parse. it is free so no complaints
        except Exception as e:
            log.debug("  [ESPN] Parse error %s: %s", event_id, e)
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
            log.warning("[CustomAPI] base_url not set.")
            return []
        try:
            import requests as _req
            r = _req.get(
                self.base_url + self.endpoint,
                params={
                    API_CFG["custom"].get("season_param","season"): season,
                    "limit":   max_games,
                    "api_key": self.api_key,
                },
                timeout=15,
            )
            r.raise_for_status()
            return [self._map(g) for g in r.json() if self._map(g)]
        except Exception as e:
            log.error("[CustomAPI] %s", e)
            return []

    def _map(self, raw):
        fm = self.field_map
        try:
            return {
                "game_id":    raw.get(fm.get("game_id",   "id"),         ""),
                "home_team":  raw.get(fm.get("home_team", "home_team"),  ""),
                "away_team":  raw.get(fm.get("away_team", "away_team"),  ""),
                "outcome":    1 if raw.get(fm.get("home_score","home_score"),0) >
                                   raw.get(fm.get("away_score","away_score"),0) else 0,
                "source":     "custom",
                "fetched_at": datetime.now().isoformat(),
            }
        except (KeyError, TypeError, AttributeError):
            return None


def fetch_ncaa_data(max_games: int = None) -> list:
    """
    Multi-season fetch. Loops over all seasons in config and appends until
    max_games is hit. Each season contributes ~1,500-2,500 valid box scores.
    Falls back to single-season behaviour if 'seasons' key is absent.
    I do not want to play fetch anymore — but 500 games was embarrassing.
    """
    max_games = max_games or API_CFG.get("max_games", 3000)
    provider  = API_CFG.get("provider", "espn")

    if provider == "custom":
        return CustomAPIFetcher().fetch(API_CFG.get("season", 2024), max_games)

    fetcher = ESPNFetcher()
    seasons = API_CFG.get("seasons", [API_CFG.get("season", 2024)])

    all_games: list = []
    seen_ids:  set  = set()

    for season in seasons:
        if len(all_games) >= max_games:
            log.info("[ESPN] Cap of %d reached before season %d. Stopping.", max_games, season)
            break

        start, end = f"{season}1101", f"{season+1}0430"
        log.info("[ESPN] Season %d-%s  Fetching IDs %s→%s...",
                 season, str(season+1)[2:], start, end)

        game_id_dates = fetcher.get_game_ids(start, end)
        log.info("[ESPN] %d events found this season.", len(game_id_dates))

        season_valid = season_errors = 0
        cap_remaining = max_games - len(all_games)

        for i, (gid, game_date) in enumerate(game_id_dates[:cap_remaining]):
            if gid in seen_ids:
                continue
            g = fetcher.get_box_score(gid, game_date=game_date)
            if g:
                all_games.append(g)
                seen_ids.add(gid)
                season_valid += 1
            else:
                season_errors += 1
            if (i+1) % 50 == 0:
                log.info("  %d/%d  valid=%d  skipped=%d  total=%d",
                         i+1, min(len(game_id_dates), cap_remaining),
                         season_valid, season_errors, len(all_games))

        log.info("[ESPN] Season %d done. Valid=%d  Skipped=%d  Running total=%d",
                 season, season_valid, season_errors, len(all_games))

    log.info("[ESPN] All seasons done. Total raw games=%d", len(all_games))

    # Enrich with pre-game rolling averages before returning.
    # This is the key step — feature fields now contain what we knew BEFORE
    # each game, not what happened DURING it.
    window    = DATA_CFG.get("pregame_window",    10)
    min_games = DATA_CFG.get("pregame_min_games",  1)
    log.info("[ESPN] Enriching with pre-game rolling averages (window=%d)...", window)
    enriched = enrich_with_pregame_averages(all_games, window=window, min_games=min_games)
    log.info("[ESPN] Enrichment done. %d training-ready games.", len(enriched))
    return enriched  # writing graceful code is a nightmare these days, this is MY SLOP!