"""
app/enrichment.py
─────────────────
Pre-game rolling average enrichment — the core fix for the leakage problem.

Previously: home_fg_pct = 0.52 = what the team shot IN this game.
            The model learned "teams that shot well won" — circular.

Now: home_fg_pct = 0.47 = what the team averaged in their LAST 10 games.
     The model learns "teams that have been shooting well recently tend to win"
     which is a genuine pre-game prediction.

Original in-game stats are preserved under home_game_* for analytics.
"""

from collections import defaultdict

from app.config import DATA_CFG
from app.logger import get_logger

log = get_logger(__name__)

# The raw stat keys that get replaced by rolling averages.
# ast_to_tov is derived so it gets recomputed from rolling avg ast + tov.
_ENRICH_FEATS = [
    "fg_pct", "rebounds", "assists", "turnovers",
    "steals", "blocks",
]


def _extract_team_game_stats(game: dict, side: str) -> dict:
    """Pull one team's in-game performance from a record."""
    stats = {feat: game.get(f"{side}_{feat}", 0.0) for feat in _ENRICH_FEATS}
    # Store ast and tov separately too so we can recompute ast_to_tov correctly
    stats["assists"]   = game.get(f"{side}_assists",   0.0)
    stats["turnovers"] = game.get(f"{side}_turnovers", 0.0)
    return stats


def _rolling_avg(history: list, feat: str, window: int) -> float:
    """Average of the last N values for a given stat from team history."""
    recent = history[-window:]
    vals   = [g[feat] for g in recent if g.get(feat) is not None]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _sort_key_for_game(g: dict) -> str:
    """
    Chronological sort key for a game record.
    Uses game_date if available (YYYY-MM-DD). Falls back to numeric
    ESPN game ID — ESPN IDs are sequential so higher = more recent.
    This fallback handles existing records fetched before v2.5.
    """
    d = g.get("game_date", "")
    if d:
        return d
    gid = g.get("game_id", "")
    try:
        return str(int(gid.replace("ESPN_", "").replace("SYN_", "0"))).zfill(12)
    except (ValueError, AttributeError):
        return gid


def enrich_with_pregame_averages(
    games:     list,
    window:    int = 10,
    min_games: int = 1,
) -> list:
    """
    Replace in-game feature values with pre-game rolling averages.

    For each game record (sorted chronologically):
      - Look up each team's last `window` games from history
      - Compute rolling averages of efficiency stats
      - Overwrite the feature fields with those averages
      - Preserve original in-game values under home_game_* / away_game_*
      - Recompute ast_to_tov from rolling avg assists ÷ rolling avg turnovers

    Games where either team has fewer than `min_games` of history are
    excluded from the output (cold-start problem — no prior data to average).
    These are typically the first game of each team's season.

    Returns only enriched games. Games already flagged pregame_enriched=True
    pass through unchanged so re-running --enrich is safe.
    """
    window    = window    or DATA_CFG.get("pregame_window",    10)
    min_games = min_games or DATA_CFG.get("pregame_min_games",  1)

    # Separate already-enriched games — pass them through without reprocessing
    already_enriched = [g for g in games if g.get("pregame_enriched") is True]
    to_enrich        = [g for g in games if g.get("pregame_enriched") is not True]

    if not to_enrich:
        return already_enriched

    # Sort chronologically so history builds in the right order
    sorted_games = sorted(to_enrich, key=_sort_key_for_game)

    team_history: dict = defaultdict(list)  # team → list of per-game stat dicts
    enriched_new: list = []

    for game in sorted_games:
        home = game.get("home_team", "").strip()
        away = game.get("away_team", "").strip()
        if not home or not away:
            continue

        home_hist = team_history[home]
        away_hist = team_history[away]

        # Skip if either team has too little history to average meaningfully
        if len(home_hist) < min_games or len(away_hist) < min_games:
            # Still record this game in history so future games can use it
            team_history[home].append(_extract_team_game_stats(game, "home"))
            team_history[away].append(_extract_team_game_stats(game, "away"))
            continue

        enriched_game = dict(game)

        # Save original in-game stats before overwriting
        for feat in _ENRICH_FEATS:
            enriched_game[f"home_game_{feat}"] = game.get(f"home_{feat}", 0.0)
            enriched_game[f"away_game_{feat}"] = game.get(f"away_{feat}", 0.0)
        # Save original ast_to_tov too
        enriched_game["home_game_ast_to_tov"] = game.get("home_ast_to_tov", 0.0)
        enriched_game["away_game_ast_to_tov"] = game.get("away_ast_to_tov", 0.0)

        # Replace feature fields with pre-game rolling averages
        for feat in _ENRICH_FEATS:
            enriched_game[f"home_{feat}"] = _rolling_avg(home_hist, feat, window)
            enriched_game[f"away_{feat}"] = _rolling_avg(away_hist, feat, window)

        # Recompute ast_to_tov from rolling avg of raw assists and turnovers.
        # Averaging per-game ratios is less accurate than computing the ratio
        # of the averages — so we do it this way. Small detail, right call.
        for side in ("home", "away"):
            hist = team_history[away if side == "away" else home]
            avg_ast = _rolling_avg(hist, "assists",   window)
            avg_tov = _rolling_avg(hist, "turnovers", window)
            enriched_game[f"{side}_ast_to_tov"] = round(
                avg_ast / (avg_tov if avg_tov > 0 else 0.1), 3
            )

        enriched_game["pregame_enriched"]    = True
        enriched_game["pregame_window_used"] = min(len(home_hist), len(away_hist), window)
        enriched_new.append(enriched_game)

        # Add this game to history AFTER processing — never use a game as
        # its own pre-game feature. That would be the original leakage again.
        team_history[home].append(_extract_team_game_stats(game, "home"))
        team_history[away].append(_extract_team_game_stats(game, "away"))

    total_enriched = len(already_enriched) + len(enriched_new)
    cold_start     = len(to_enrich) - len(enriched_new)
    log.info(
        "[Enrich] %d already enriched + %d newly enriched = %d total. "
        "%d cold-start games excluded.",
        len(already_enriched), len(enriched_new), total_enriched, cold_start,
    )
    return already_enriched + enriched_new