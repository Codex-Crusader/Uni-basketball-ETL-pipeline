"""
app/preprocessing.py
────────────────────
Data quality validation, feature preparation, and team-stats aggregation.

_validate_training_data()  — leakage detection, class balance, sample ratio
_adaptive_depth()          — scale tree depth to dataset size (anti-overfit)
prepare_data()             — build X / y from enriched game records
build_team_stats()         — per-team rolling averages from game records
get_home_team_stats()      — convenience wrapper for the configured home team
"""

import numpy as np

from app.config import DATA_CFG, HT_CFG
from app.logger import get_logger

log = get_logger(__name__)


# ── VALIDATION ────────────────────────────────────────────────────────────────

def _validate_training_data(
    X:             np.ndarray,
    y:             np.ndarray,
    feature_names: list,
) -> dict:
    """
    Run sanity checks on the training matrix before fitting any model.

    Checks:
      1. Leakage — feature correlation with outcome above threshold
         (score-derived features correlate ~0.9+, clean features ~0.1-0.4)
      2. Near-zero variance — constant feature, useless to model
      3. Class balance — home win rate outside 40-70% biases all predictions
      4. Sample-to-feature ratio — too few samples per feature → overfit
    """
    report    = {"warnings": [], "info": []}
    n, p      = X.shape
    threshold = DATA_CFG.get("leakage_correlation_threshold", 0.70)

    log.info("[Validate] Samples: %d  Features: %d  Home win rate: %.2f%%",
             n, p, y.mean() * 100)

    # 1. Leakage detection
    for i, name in enumerate(feature_names):
        corr = float(np.corrcoef(X[:, i], y)[0, 1])
        if abs(corr) > threshold:
            msg = (f"LEAKAGE WARNING: '{name}' correlates {corr:+.3f} with outcome. "
                   f"Score-derived feature suspected. Check config features list.")
            log.warning("[Validate] ⚠  %s", msg)
            report["warnings"].append(msg)
        elif abs(corr) < 0.01:
            msg = f"LOW SIGNAL: '{name}' correlates only {corr:+.4f} — minimal predictive value."
            log.debug("[Validate] ℹ  %s", msg)
            report["info"].append(msg)

    # 2. Near-zero variance
    for i, name in enumerate(feature_names):
        std = float(np.std(X[:, i]))
        if std < 0.001:
            msg = f"ZERO VARIANCE: '{name}' std={std:.6f} — constant feature."
            log.warning("[Validate] ⚠  %s", msg)
            report["warnings"].append(msg)

    # 3. Class balance
    home_win_rate = float(y.mean())
    if home_win_rate < 0.40 or home_win_rate > 0.70:
        msg = (f"CLASS IMBALANCE: home win rate = {home_win_rate:.2%}. "
               f"Expected 40-70% for NCAA.")
        log.warning("[Validate] ⚠  %s", msg)
        report["warnings"].append(msg)
    else:
        log.info("[Validate] ✓  Class balance OK (%.2f%% home wins)", home_win_rate * 100)

    # 4. Sample-to-feature ratio
    ratio = n / p
    if ratio < 20:
        msg = (f"LOW SAMPLE RATIO: {n} samples / {p} features = {ratio:.1f}x. "
               f"Recommend ≥20x. Fetch more data.")
        log.warning("[Validate] ⚠  %s", msg)
        report["warnings"].append(msg)
    else:
        log.info("[Validate] ✓  Sample ratio OK (%.1fx)", ratio)

    log.info("[Validate] %d warning(s)", len(report["warnings"]))
    return report


# ── ADAPTIVE DEPTH ────────────────────────────────────────────────────────────

def _adaptive_depth(base_depth: int, n_samples: int, n_features: int) -> int:
    """
    Scale tree max_depth down for small datasets to prevent overfitting.

    Rule of thumb: each leaf needs ~10 × n_features samples to split
    reliably. At depth D there are up to 2^D leaves, so:
        max_safe = log2(n_samples / (10 * n_features))

    Returns at least 3 so the model can still learn non-trivial patterns.

    n=2300, p=14 → depth 4   (current dataset)
    n=5000, p=14 → depth 5
    n=10000,p=14 → depth 6
    """
    if n_samples < 1 or n_features < 1:
        return 3
    max_safe = max(3, int(np.log2(n_samples / max(10 * n_features, 1))))
    return min(base_depth, max_safe)


# ── FEATURE PREPARATION ───────────────────────────────────────────────────────

def prepare_data(data: list):
    """
    Build the feature matrix X and label vector y.

    v2.5: only games with pregame_enriched=True are used for training.
    Un-enriched games (in-game stats, not pre-game averages) would
    re-introduce leakage. Falls back to all games with a warning if
    none are enriched yet — handles existing data before --enrich is run.
    """
    cfg_features = DATA_CFG["features"]
    label        = DATA_CFG["label"]

    enriched = [g for g in data if g.get("pregame_enriched") is True]
    if enriched:
        valid = [
            g for g in enriched
            if all(feat in g for feat in cfg_features) and label in g
        ]
        if len(valid) < 50:
            log.warning(
                "[PrepData] Only %d enriched games. Run --enrich to fix this.", len(valid)
            )
            valid = [
                g for g in data
                if all(feat in g for feat in cfg_features) and label in g
            ]
        else:
            enrichment_rate = len(enriched) / len(data) * 100
            log.info("[PrepData] Using %d enriched games (%.0f%% of dataset)",
                     len(valid), enrichment_rate)
    else:
        log.warning("[PrepData] No enriched games found. Run: python main.py --enrich")
        log.warning("[PrepData] Training on in-game stats — predictions will be less accurate.")
        valid = [
            g for g in data
            if all(feat in g for feat in cfg_features) and label in g
        ]

    if not valid:
        raise ValueError("No valid records with required features.")

    X = np.array([[g[feat] for feat in cfg_features] for g in valid], dtype=float)
    y = np.array([int(g[label]) for g in valid])
    return X, y, cfg_features


# ── TEAM STATS ────────────────────────────────────────────────────────────────

def build_team_stats(data: list, window: int = None) -> dict:
    """
    Compute per-team feature averages from game records.

    window: if set, only the last N games per team are used (sorted by
    game_date descending). None = full dataset average.

    Mirroring logic: when a team was away, their stats are under away_*
    columns. We read them and store under home_* keys so every team has
    consistent home_* feature names regardless of which side they played on.

    v2.5: game records now store pre-game rolling averages as feature values,
    so these averages are already meaningful form estimates.
    """
    cfg_features = DATA_CFG["features"]
    team_games: dict[str, list] = {}

    for g in data:
        ht = g.get("home_team", "").strip()
        at = g.get("away_team", "").strip()
        if not ht or not at:
            continue
        ts = g.get("game_date") or g.get("fetched_at", "")
        for team, side in ((ht, "home"), (at, "away")):
            if team not in team_games:
                team_games[team] = []
            team_games[team].append((g, side, ts))

    result = {}

    for team, games_with_side in team_games.items():
        if not team:
            continue

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
                    else:
                        mirror = "away_" + feat[5:]
                        accum[feat].append(float(g.get(mirror, 0)))
                elif feat.startswith("away_"):
                    if side == "away":
                        accum[feat].append(float(g.get(feat, 0)))
                    else:
                        mirror = "home_" + feat[5:]
                        accum[feat].append(float(g.get(mirror, 0)))
                        # This logic assumes that for every "home_X" there is a corresponding
                        # "away_X" that represents the same stat for the other team.
                        # This way we can aggregate stats regardless of home/away side.
                        # I hate myself

        result[team] = {
            feat: round(sum(v) / len(v), 4) if v else 0.0
            for feat, v in accum.items()
        }
        result[team]["games_played"]    = len(games_with_side)
        result[team]["games_in_window"] = len(windowed)
        result[team]["wins"] = sum(
            1 for g_data, side, _ in games_with_side
            if (side == "home" and g_data.get("outcome") == 1) or
               (side == "away" and g_data.get("outcome") == 0)
        )

    return result


def get_home_team_stats(data: list, window: int = None):
    """Return stats for the configured home team, with fuzzy name matching."""
    ts   = build_team_stats(data, window=window)
    name = HT_CFG["name"]
    if name in ts:
        return {"name": name, "stats": ts[name]}
    for k in ts:
        if name.lower() in k.lower() or k.lower() in name.lower():
            return {"name": k, "stats": ts[k]}
    return None