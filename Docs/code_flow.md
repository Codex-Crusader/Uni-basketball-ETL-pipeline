# Code Flow Documentation

Complete architecture and data flow documentation for the NCAA basketball predictor system, v2.5.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Components](#architecture-components)
3. [Command Execution Flows](#command-execution-flows)
4. [Auto-Learning Pipeline](#auto-learning-pipeline)
5. [Data Flow](#data-flow)
6. [Storage Architecture](#storage-architecture)
7. [Model Training Pipeline](#model-training-pipeline)
8. [Prediction Serving](#prediction-serving)
9. [Roster System](#roster-system)
10. [Dashboard Integration](#dashboard-integration)

---

## System Overview

The basketball predictor is built as a config-driven, self-improving ML pipeline with a modular package structure, a background auto-learn scheduler, and a pre-game data enrichment step that eliminates the leakage present in earlier versions.

```
basketball_ver3/
    main.py              CLI entry point only (~200 lines)
    config.yaml          All settings
    dashboard.html       Single-page frontend
    app/                 All business logic
        config.py        Config loading and path constants
        logger.py        Logging setup (rotating file + console)
        storage.py       JSON and Snowflake I/O
        enrichment.py    Pre-game rolling average pipeline
        fetcher.py       ESPN multi-season game fetcher
        roster.py        ESPN roster fetcher and player aggregation
        preprocessing.py Validation, feature prep, team stats
        models.py        Registry, build, train, evaluate
        scheduler.py     AutoLearnScheduler background thread
        api.py           Flask application and all routes
```

### Design Principles

**Config-Driven Architecture:** All settings live in `config.yaml`. No value is hardcoded in Python. Feature lists, model hyperparameters, thresholds, team name, intervals, file paths, and API endpoints are all read from config at startup via `app/config.py`.

**Modular with no circular imports:** Dependencies flow in one direction only (see dependency chain in README). Each module has exactly one responsibility.

**Pre-game features only:** As of v2.5, the feature vector contains rolling averages of prior games, not in-game statistics. The enrichment pipeline runs between fetch and training. Models trained on in-game stats are inadmissible — they describe outcomes rather than predicting them.

**Promote-only model updates:** A new model only replaces the active model if its ROC-AUC exceeds the current model by at least `promote_threshold` (default 0.002). The model can only improve over time.

**Structured logging:** Every module uses `logging.getLogger("bball.<module>")`. Log output goes to both console (INFO+) and `data/app.log` (DEBUG+, rotating at 10 MB, 2 backups).

---

## Architecture Components

### Layer 0: Config Loading (`app/config.py`)

Runs at module import time. All other modules import constants from here — nobody reads `config.yaml` directly.

```python
CFG         = load_config()
APP_CFG     = CFG["app"]
HT_CFG      = CFG["home_team"]
DATA_CFG    = CFG["data"]            # features list, paths, split sizes, pregame window
API_CFG     = CFG["ncaa_api"]        # ESPN endpoints, seasons list, rate limit
SF_CFG      = CFG["snowflake"]       # credentials via env vars
MODEL_CFG   = CFG["models"]          # enabled models and hyperparams
AL_CFG      = CFG["auto_learn"]      # intervals, thresholds
ROSTER_CFG  = CFG.get("roster", {})
ROLLING_CFG = CFG.get("rolling", {})
```

Directories (`DATA_DIR`, `MODELS_DIR`, `ROSTER_DIR`) are created at import time. Any module that writes files can assume they exist.

---

### Layer 1: Logging (`app/logger.py`)

Must be initialised before any other module is imported. `main.py` calls `setup_logging()` as its very first action.

```python
# main.py
from app.logger import setup_logging
setup_logging()
# now import everything else
```

Child loggers are obtained by name in each module:

```python
from app.logger import get_logger
log = get_logger(__name__)
# produces logger named bball.app.fetcher, bball.app.models, etc.
```

The `_configured` guard makes `setup_logging()` idempotent — safe to call multiple times.

---

### Layer 2: Data Ingestion (`app/fetcher.py`)

#### ESPN Game Data (`ESPNFetcher`)

Real NCAA data, no API key required. v2.5 adds multi-season support and `game_date` extraction.

```
get_game_ids(start_date, end_date)
    Iterates day-by-day over the date range
    GET /scoreboard?dates=YYYYMMDD
    Returns list of (event_id, game_date) tuples
    game_date extracted from event["date"] ISO string

get_box_score(event_id, game_date)
    GET /summary?event=ID
    Parse boxscore.teams for home/away stats
    Parse header.competitions[0].competitors for scores
    Normalize FG% strings ("45.5" -> 0.455)
    Returns game dict with 14 raw features, game_date, and pregame_enriched=False
```

The returned record stores in-game statistics at this stage. The enrichment step replaces the feature fields with rolling averages before the record is used for training.

**Multi-season fetch** (`fetch_ncaa_data`):

```
For each season in config seasons list [2022, 2023, 2024]:
    If running total >= max_games: stop
    Fetch all game IDs for the season (Nov 1 to Apr 30)
    For each ID, fetch box score
    Deduplicate across seasons by game_id

After all seasons:
    Call enrich_with_pregame_averages() on the full batch
    Return only enriched records
```

#### Custom API (`CustomAPIFetcher`)

Stub for user-provided APIs. Set `provider: custom` in config, fill `base_url` and `field_map`. No code changes needed.

---

### Layer 3: Pre-Game Enrichment (`app/enrichment.py`)

This is the core change in v2.5. It converts raw in-game box scores into genuinely predictive pre-game features.

```
enrich_with_pregame_averages(games, window=10, min_games=1)

    1. Separate already-enriched records (pass through unchanged)
    2. Sort remaining records chronologically by game_date
       (falls back to ESPN ID numeric ordering for records without game_date)

    3. For each game in chronological order:

        Look up team_history[home_team] and team_history[away_team]

        If either team has fewer than min_games of history:
            Record this game in their history (for future games)
            Skip this game — excluded from output (cold-start)

        Else:
            Save original in-game stats under home_game_* / away_game_*
            Replace home_fg_pct, home_rebounds, etc. with rolling averages
            Recompute ast_to_tov from rolling avg assists / rolling avg turnovers
            Set pregame_enriched = True
            Set pregame_window_used = min(home_hist_len, away_hist_len, window)
            Add to output

        Add this game to team history AFTER processing
        (a game must never be its own pre-game feature)

    Return already_enriched + newly enriched records
```

The result: `home_fg_pct = 0.47` means "what this team averaged over their last 10 games going into tonight" — knowable before tipoff. Before the fix, it meant "what they shot in this game" — which is the outcome, not a predictor.

Original in-game stats (`home_game_fg_pct`, `away_game_fg_pct`, etc.) remain in the record for analytics display only. They are never in the training feature vector.

---

### Layer 4: Storage (`app/storage.py`)

Common interface for both backends. The rest of the codebase calls `load_data(storage)` and never knows which backend is active.

**Local JSON:**
```python
save_to_json(data)      # full overwrite
load_from_json()        # full read
append_to_json(data)    # load + deduplicate by game_id + save
load_data("local")      # routes to load_from_json
```

**Snowflake:**
```python
_sf_conn()              # reads credentials from env vars
save_to_snowflake(data) # DELETE + INSERT
append_to_snowflake(data)
load_data("snowflake")  # routes to load_from_snowflake
```

`_sanitize(obj)` recursively replaces `float('nan')` and `float('inf')` with `None`. Required because XGBoost cross-validation occasionally produces NaN scores, and Python's `json.dumps` writes NaN literally — invalid JSON that crashes the browser.

---

### Layer 5: Team Stats Engine (`app/preprocessing.py`)

`build_team_stats(data, window=None)` powers the predict form's auto-fill.

For every game in the dataset, each team's pre-game feature values are accumulated regardless of whether they were home or away that game. Mirroring logic: when a team was away, their `away_*` values are stored under `home_*` keys so every team ends up with consistent `home_*` feature names.

With rolling window: only the most recent N games per team are used (sorted by `game_date` descending).

Since v2.5, the values being averaged are already pre-game rolling averages (from enrichment). Averaging them again across games produces a stable season-form estimate.

---

### Layer 6: Model Training (`app/models.py`)

`train_and_evaluate(storage, triggered_by)` runs the full pipeline:

```
1. load_data(storage)
2. prepare_data(data)
       Filter: only records with pregame_enriched=True
       (falls back to all records with a warning if none are enriched)
       X shape: (n_enriched, 14)
       y shape: (n_enriched,)
3. _validate_training_data(X, y, feature_names)
       Leakage check: correlation > 0.70 with outcome -> WARNING
       Zero-variance check
       Class balance check: home win rate outside 40-70% -> WARNING
       Sample-to-feature ratio check: < 20x -> WARNING
4. train_test_split(stratify=y, test_size=0.2)
5. build_models(n_samples, n_features)
       Adaptive depth: cap tree depth at log2(n/10p)
       All wrapped in StandardScaler -> estimator Pipeline
6. For each model:
       fit on train set
       compute_metrics on test set
       get_feature_importances
       cross_val_score (5-fold, scoring="roc_auc")
7. best = max by selection_metric (default: roc_auc)
8. AUC sanity check:
       > 0.80: warn (are records enriched?)
       < 0.52: warn (barely above chance)
       0.52-0.80: expected honest prediction range
9. Promote gate (triggered_by != "manual"):
       if new_auc < current_auc + promote_threshold: log "skipped", return None
10. register_model(best) -> versioned .pkl
11. Save comparison JSON
12. _append_log(result)
```

`stratify=y` ensures the ~69% home win rate is preserved in both train and test sets.

---

### Layer 7: Model Registry (`app/models.py`)

Registry file: `models/registry.json`

```json
{
  "active_version": "v3",
  "versions": [
    {
      "version": "v1",
      "model_name": "Gradient Boosting",
      "filename": "gradient_boosting_v1_a3f2c1d4.pkl",
      "metrics": { "roc_auc": 0.7441, "f1": 0.8161, ... },
      "feature_names": ["home_fg_pct", "away_fg_pct", ...],
      "training_size": 2328,
      "trained_at": "2026-03-19T14:22:11",
      "hash": "a3f2c1d4"
    }
  ]
}
```

Each `.pkl` stores `{"model": pipeline_obj, "feature_names": list}`. Feature names are stored with the model to prevent silent mismatches if the feature list changes between versions.

Pruning: when `len(versions) > keep_top_n`, oldest `.pkl` files are deleted from disk.

---

### Layer 8: Flask Application (`app/api.py`)

The Flask `app` object and `_scheduler` instance live here and are imported by `main.py`. The dashboard is served from the project root:

```python
from pathlib import Path

@app.route("/")
def serve_dashboard():
    return send_file(Path(__file__).parent.parent / "dashboard.html")
```

This resolves correctly regardless of working directory because it is relative to `__file__` (the `api.py` file), not to `os.getcwd()`.

---

## Command Execution Flows

### `python main.py --fetch`

```
main()
    fetch_ncaa_data(max_games=3000)
        ESPNFetcher.get_game_ids(start, end)  [per season]
            Returns (game_id, game_date) tuples
        For each (gid, game_date):
            ESPNFetcher.get_box_score(gid, game_date)
                Store game_date on record
                pregame_enriched = False
        enrich_with_pregame_averages(all_games, window=10)
            Sort by game_date
            Build rolling history per team
            Replace feature fields with rolling averages
            Set pregame_enriched = True
        Return enriched games
    append_to_json(games)
        Deduplicate by game_id
        save_to_json(existing + new_unique)
```

### `python main.py --enrich`

Applies the enrichment pipeline to an existing `games.json` that was fetched before v2.5 (when game records stored in-game stats). This avoids a full re-fetch.

```
main()
    load_from_json()
    enrich_with_pregame_averages(data, window=10)
        Already-enriched records pass through unchanged
        New records: sort by ESPN ID (sequential = chronological proxy)
        Build rolling history, replace feature fields
    save_to_json(enriched)
```

After `--enrich`, run `--train`. The existing `games.json` is usable; no re-fetch is needed.

### `python main.py --train`

```
main()
    train_and_evaluate("local", triggered_by="manual")
        load_from_json()
        prepare_data(data)
            Filter to pregame_enriched=True records (~2800 of 2900)
            X: (2800, 14)  y: (2800,)
        _validate_training_data(X, y, features)
        train_test_split (stratify, 80/20)
        build_models(n_samples=2240, n_features=14)
            _adaptive_depth: base_depth=4 -> ceiling=4 at this dataset size
        For each of 5 models:
            fit, metrics, importances, 5-fold CV
        Best: Gradient Boosting AUC 0.7441
        register_model -> v1 .pkl
        Write latest_comparison.json
        _append_log(promoted)
```

### `python main.py --serve`

```
main()
    setup_logging()        [already called at top of main.py]
    _scheduler.start()
        daemon thread: AutoLearnScheduler._loop()
            sleep(60) [let Flask start first]
            [see Auto-Learning Pipeline]
    app.run(debug=False, port=5000, use_reloader=False)
```

`use_reloader=False` prevents the scheduler from starting twice during Flask's debug reload cycle.

---

## Auto-Learning Pipeline

`AutoLearnScheduler` runs as a daemon thread. It manages two independent intervals:

```
_loop():
    sleep(60)

    while not stopped:
        now = time.time()

        if now - last_fetch >= fetch_interval (6h):
            status = "fetching"
            new_games = fetch_ncaa_data()
                [includes enrichment — new records arrive pre-enriched]
            added = append_to_json(new_games)
            last_fetch = now

            if added >= min_new_games (15):
                status = "training"
                train_and_evaluate(triggered_by="new_data")
                    [promote gate active — must beat current AUC + 0.002]
                last_retrain = now

        elif now - last_retrain >= retrain_interval (24h):
            status = "training"
            train_and_evaluate(triggered_by="scheduler")
            last_retrain = now

        status = "idle"
        sleep in 60-second chunks (checking stop event each time)
```

The promote gate:

```
new_auc = candidate_model.metrics["roc_auc"]
current_auc = active_model.metrics["roc_auc"]

if new_auc >= current_auc + promote_threshold:
    register_model()  -> new active version
    log: result="promoted"
else:
    log: result="skipped", reason="...", new_auc=..., current_auc=...
    return None  (no version change)
```

The active model's AUC is monotonically non-decreasing over time. The model can only improve or stay the same.

---

## Data Flow

### End-to-End Stats-Mode Prediction

```
User opens http://localhost:5000
    Flask: send_file(Path(__file__).parent.parent / "dashboard.html")
        [resolves to project root regardless of working directory]

Browser JS init():
    fetch("/features")          feature list + rolling window options
    fetch("/home_team")         Duke stats from season averages
    fetch("/teams")             all teams for opponent dropdown
    fetch("/model_info")        active version and metrics
    fetch("/analytics")         stats + comparison + feature importances
    fetch("/registry")          all versions for Registry tab
    fetch("/autolearn/status")  scheduler state
    fetch("/learning_log")      training history

User selects away team
    onAwayTeamChange()
        fetch("/team_stats/Kansas Jayhawks?window=12")
            build_team_stats(data, window=12)
                [uses pre-game rolling averages stored in game records]
            Auto-fill away stat fields

User clicks Predict
    POST /predict { home_fg_pct: 0.472, away_fg_pct: 0.461, ... }
        load_active_model()
            registry.json -> active_version -> load .pkl
            payload = {model: Pipeline, feature_names: [...]}
        X = np.array([[payload[f] for f in feature_names]])
        pred = model.predict(X)[0]
            [Pipeline: StandardScaler.transform -> clf.predict]
        conf = max(model.predict_proba(X)[0])
        return {prediction, confidence, model_name, version}
```

### End-to-End Roster-Mode Prediction

```
User switches to Roster mode
    fetch("/roster/Duke Blue Devils")
        RosterFetcher.fetch_team_async("Duke Blue Devils")
            If cached and valid:
                Write _roster_progress["Duke Blue Devils"] = {status: "ready", ...}
                Return (no thread needed)
            Else:
                Set _roster_progress = {status: "loading", ...}
                Start background thread: fetch_team(team_name)
                    get_team_id -> ESPN team lookup or cache
                    get_roster -> /teams/{id}/roster
                        _parse_embedded_stats per athlete
                    For players missing stats:
                        get_player_stats(player_id) -> /athletes/{id}/statistics
                        Graceful 404 handling
                    Save to data/rosters/<team_id>.json
                    Write final progress: {status: "ready", players: [...]}
        Return current _roster_progress state immediately

Dashboard polls GET /roster/progress/Duke Blue Devils every 1 second
    Return _roster_progress.get(team_name)
    Browser renders player cards as they arrive, shows progress bar

User selects players and clicks Predict (Roster mode)
    POST /predict/from_roster
    {
        "home_players": [{ppg, rpg, apg, spg, bpg, tov, fg_pct, fgm, fga}, ...],
        "away_players": [{...}, ...]
    }
        compute_stats_from_roster(home_players, "home")
            Sum ppg, rpg, apg, spg, bpg, tov
            FGA-weighted fg_pct (total_fgm / total_fga)
            Derive ast_to_tov from summed totals
            Separate model features from insight_ features
        compute_stats_from_roster(away_players, "away")
        X = np.array([[combined.get(f, 0) for f in feature_names]])
        pred, conf [same Pipeline as stats mode]
        return {prediction, confidence, computed_stats, insights,
                home_count, away_count, model_name, version}
```

---

## Storage Architecture

### Local JSON (`data/games.json`)

Record structure (v2.5):

```json
{
  "game_id":         "ESPN_401703521",
  "game_date":       "2024-01-15",
  "home_team":       "Duke Blue Devils",
  "away_team":       "Kansas Jayhawks",
  "home_score":      84,
  "away_score":      72,
  "home_ppg":        84.0,
  "away_ppg":        72.0,
  "home_fg_pct":     0.4670,
  "away_fg_pct":     0.4480,
  "home_rebounds":   36.4,
  "away_rebounds":   32.1,
  "home_assists":    15.8,
  "away_assists":    13.2,
  "home_turnovers":  11.9,
  "away_turnovers":  13.7,
  "home_steals":     7.1,
  "away_steals":     6.4,
  "home_blocks":     4.2,
  "away_blocks":     3.1,
  "home_ast_to_tov": 1.328,
  "away_ast_to_tov": 0.964,
  "home_game_fg_pct":     0.5210,
  "away_game_fg_pct":     0.3854,
  "home_eff_score":  35.77,
  "away_eff_score":  27.75,
  "outcome":         1,
  "source":          "espn",
  "fetched_at":      "2026-03-19T14:00:00",
  "pregame_enriched": true,
  "pregame_window_used": 10
}
```

The feature fields (`home_fg_pct`, `home_rebounds`, etc.) contain **pre-game rolling averages**. The original in-game stats are preserved under `home_game_*` keys for analytics. `home_ppg` / `away_ppg` and `home_eff_score` / `away_eff_score` are stored but are not in the training feature vector.

### Roster Cache (`data/rosters/<team_id>.json`)

```json
{
  "team_name":  "Duke Blue Devils",
  "team_id":    "150",
  "players":    [ { "id": "...", "name": "...", "ppg": 16.4, ... } ],
  "fetched_at": "2026-03-19T14:00:00"
}
```

TTL: 24 hours. `_cache_valid()` compares `fetched_at` to now.

---

## Model Training Pipeline Detail

### Why ROC-AUC as Selection Metric

With ~69% home wins across three seasons, a model that predicts "Home Win" for every game achieves 69% accuracy but AUC = 0.5. ROC-AUC measures whether the model's probability estimates correctly rank home wins above away wins — it penalises models that achieve accuracy purely through class imbalance. It is threshold-independent and the correct metric here.

### Why Pre-Game Features Matter

The v2.4 validation showed home_fg_pct correlating +0.81 with outcome. That is nearly as high as using the score itself. In-game shooting percentage is not a pre-game predictor — it is a consequence of winning. The fix shifts every feature to the rolling average from prior games, where correlations with outcome drop to 0.10-0.25, which is the range of genuine predictive signal.

### Why Adaptive Depth

At n=2300 training samples and p=14 features, `log2(2300 / (10 x 14)) = 4.04`. Setting tree depth to 10 in config is overridden to 4 at runtime. This prevents trees from memorising the training set. As more data is collected, the ceiling rises automatically — no config change needed.

### Why Pipeline

SVM and MLP are scale-sensitive. The Pipeline ensures `StandardScaler` is fit only on `X_train` and applied consistently to `X_test` and to inference inputs. The entire Pipeline is pickled as one object, so loading a model version automatically gets the correct scaler.

---

## API Reference

| Method | Endpoint | Returns |
|--------|----------|---------|
| GET | `/` | `dashboard.html` from project root |
| POST | `/predict` | prediction, confidence, version (stats mode) |
| POST | `/predict/from_roster` | prediction, confidence, computed_stats, insights, player counts |
| GET | `/analytics` | game stats, model comparison, feature importances, enrichment rate |
| GET | `/model_info` | active model registry entry |
| GET | `/registry` | full registry JSON |
| POST | `/registry/activate/<v>` | `{status: ok}` |
| GET | `/features` | feature list + rolling window options from config |
| GET | `/teams?window=N` | all teams with season or rolling averages |
| GET | `/team_stats/<n>?window=N` | single team stats (fuzzy match), optional window |
| GET | `/home_team?window=N` | configured home team and stats |
| GET | `/roster/<team_name>` | kick off async roster fetch; returns current progress state |
| GET | `/roster/progress/<team_name>` | poll progress: status, players so far, done/total |
| POST | `/roster/refresh/<team_name>` | force fresh ESPN fetch (bypass 24h cache) |
| GET | `/autolearn/status` | scheduler state and countdowns |
| POST | `/autolearn/trigger` | starts background retrain |
| GET | `/learning_log?n=50` | last N log entries |
| GET | `/debug` | health check: paths, game count, enrichment rate, active model |

---

## Dashboard Integration

### Tab Rendering Strategy

Charts are not drawn on page load. They are drawn fresh each time a tab becomes visible via `switchTab()`. This solves the Chart.js 0x0 canvas problem: `display:none` tabs have no dimensions at render time.

`requestAnimationFrame` defers execution by one paint cycle, ensuring the browser has applied `display:block` before Chart.js measures the canvas.

`loadAnalytics()` fetches data and updates DOM stat cards immediately. It does not draw charts. This separation means a slow analytics fetch never blocks tab switching.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Config-driven via YAML | No secrets or tunables in Python source |
| Pre-game rolling averages as features | Eliminates leakage — model trains on knowable pre-game data |
| game_date on every record | Correct chronological ordering for enrichment |
| enrichment as a separate pipeline step | Can back-fill existing data without re-fetching |
| ROC-AUC for model selection | Robust to class imbalance (~69% home wins) |
| Stratified train-test split | Preserves class ratio in both sets |
| Pipeline (Scaler + clf) | Scaler trained on train set only; no leakage |
| Adaptive tree depth | Prevents overfitting; scales automatically with dataset size |
| `Path(__file__).parent.parent / "dashboard.html"` | Resolves correctly from `app/api.py` to project root |
| `use_reloader=False` | Prevents scheduler starting twice in Flask debug mode |
| `_sanitize()` before JSON | XGBoost CV can produce NaN; invalid JSON crashes browser |
| Lazy chart rendering | Chart.js cannot render into 0x0 hidden canvases |
| Module-level `_roster_progress` | Shared state between Flask and background roster threads |
| FGA-weighted fg_pct aggregation | More accurate than simple average when players have unequal shot volume |
| `window=None` default in `build_team_stats` | Full season average unless caller explicitly requests rolling |
| RotatingFileHandler 10 MB x 2 | Bounded disk usage; always have recent history |
| setup_logging() called before all other imports | Ensures all module-level loggers attach to the configured handler |
