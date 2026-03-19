# Changelog

---

## v2.5.1 — Modular Refactor and Logging

**Released:** March 2026

### Architecture
- Split 2000-line `main.py` into `app/` package with 10 modules: `config`, `logger`, `storage`, `enrichment`, `fetcher`, `roster`, `preprocessing`, `models`, `scheduler`, `api`
- `main.py` is now the CLI entry point only (~200 lines)
- No circular imports — dependency chain verified
- `dashboard.html` served via absolute path resolution, resolves correctly regardless of working directory

### Logging
- All `print()` replaced with Python `logging` module
- `RotatingFileHandler`: `data/app.log`, 10 MB per file, 2 backups (30 MB ceiling)
- Console: INFO and above. File: DEBUG and above
- Module-level loggers: `bball.app.fetcher`, `bball.app.models`, etc.

### Hyperparameters
- `min_samples_leaf` restored from 4 to 2 for RF, ET, GBM (was over-regularized)
- `min_samples_split` restored from 10 to 5
- MLP architecture restored from `[64, 32]` to `[128, 64, 32]`
- SVM `C` raised from 1.0 to 2.0
- `n_estimators` raised from 200 to 300 for all ensemble models
- XGBoost `min_child_weight=3` added

### Bug Fixes
- `XGBClassifier = None` added in `except ImportError` block to satisfy type checker
- Unused `team_history_syn` variable removed from `_generate_synthetic()`
- Unused `load_from_json` import removed from `preprocessing.py`
- Unused `MODEL_CFG` import removed from `api.py`

---

## v2.5.0 — Pre-Game Rolling Averages (Leakage Fix)

**Released:** March 2026

### The Problem
Feature fields stored in-game box score statistics. `home_fg_pct = 0.52` meant what the team shot during the game — circular, not predictive. AUC of 0.9666 was a red flag, not a achievement.

### The Fix
`enrich_with_pregame_averages()` replaces feature fields with rolling averages from the team's prior games, computed in chronological order. `home_fg_pct = 0.47` now means what they averaged over their last 10 games going into this game — knowable before tipoff.

### Changes
- `enrich_with_pregame_averages()` pipeline added to `fetcher.py`
- Original in-game stats preserved under `home_game_*` / `away_game_*` for analytics
- `pregame_enriched` flag on every record; training filters to True-only
- `game_date` extracted from ESPN event metadata for correct chronological ordering
- `--enrich` CLI command back-fills existing `games.json` without re-fetching
- `prepare_data()` warns if no enriched records found
- AUC sanity check added: warns above 0.80 (possible leakage), below 0.52 (chance level)
- `data.pregame_window: 10` and `data.pregame_min_games: 1` added to config

### Multi-Season Fetch
- `seasons: [2022, 2023, 2024]` in config
- `max_games: 3000` cap
- `get_game_ids()` now returns `(game_id, game_date)` tuples
- ~2900 real games across 3 seasons

### Features Removed from Model Vector
- `home_ppg`, `away_ppg` (actual game score — leakage)
- `home_eff_score`, `away_eff_score` (score-derived — leakage)
- Feature count: 18 to 14

### Model Results After Fix
- AUC: 0.9666 to ~0.74 — correct, not a regression
- Best model: Gradient Boosting 74.4% AUC
- Honest pre-game prediction range: 0.60-0.75

---

## v2.4.0 — Regularization and Validation

**Released:** February 2026

- `_validate_training_data()` added: leakage detection, zero-variance check, class balance check, sample ratio check
- `_adaptive_depth()`: scales tree depth to dataset size
- `build_models()` takes `n_samples`, `n_features` for adaptive depth
- Regularization tightened: `min_samples_leaf=4`, `min_samples_split=10`, MLP `[64,32]`
- `analytics()` fix: `mc = fi = {}` split to separate assignments
- `_generate_synthetic()` outcome driven by efficiency metrics, not scores

---

## v2.3.0 — Multi-Season Fetch and New Features

**Released:** January 2026

- Multi-season fetch: seasons list in config, loops until `max_games` cap
- New derived features: `home_ast_to_tov`, `away_ast_to_tov`
- `_parse_embedded_stats` made `@staticmethod`
- `_generate_synthetic()` bumped to 5000 games
- `--max-games` CLI override argument added
- `compute_stats_from_roster()` insight features added (display only)

---

## v2.2.0 — Roster System

**Released:** December 2025

- `RosterFetcher` class: team ID lookup, roster fetch, per-player stats
- Async roster fetch with progress polling (`_roster_progress` shared state)
- `/predict/from_roster` endpoint with FGA-weighted aggregation
- `--fetch-rosters` CLI flag
- Rolling form window: `build_team_stats(window=N)`, `?window=N` query params
- Six-tab dashboard: Predict, Overview, Model Comparison, Features, Registry, Auto-Learn
