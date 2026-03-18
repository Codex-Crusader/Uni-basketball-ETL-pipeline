# 🔄 Code Flow Documentation

> *Complete architecture and data flow documentation for the NCAA basketball predictor system*

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

## 🏛️ System Overview

The basketball predictor is built as a **config-driven, self-improving ML pipeline** with seven operational modes and a background auto-learn scheduler.

```
┌─────────────────────────────────────────────────────────────┐
│                     main.py (~1000 lines)                    │
│              Config-Driven Single Script Architecture        │
└────────────────────┬────────────────────────────────────────┘
                     │
              config.yaml (all settings)
                     │
          Command Line Arguments Parser
                     │
    ┌────────┬────────┼────────┬────────┬────────┬────────┐
    │        │        │        │        │        │        │
    ▼        ▼        ▼        ▼        ▼        ▼        ▼
[--fetch] [--fetch [--gen  [--train] [--serve] [--list [--activate
         -rosters] synth]                      models]   VERSION]
```

### Design Philosophy

**Config-Driven Architecture:** All settings live in `config.yaml`. No value is hardcoded in Python — feature lists, model hyperparameters, thresholds, team name, intervals, file paths, and API endpoints are all read from config at startup.

**Single Script + Background Thread:** All logic resides in `main.py`. When `--serve` is used, a daemon `AutoLearnScheduler` thread runs alongside Flask, continuously fetching data and retraining without user intervention.

**Promote-Only Model Updates:** A new model only replaces the active model if its ROC-AUC exceeds the current model by at least `promote_threshold` (default 0.002). The model can only improve, never regress.

---

## 📊 Architecture Components

### Layer 0: Config Loading

**Function:** `load_config(path)` — runs at module import time

```python
CFG         = load_config()
APP_CFG     = CFG["app"]
HT_CFG      = CFG["home_team"]       # Duke Blue Devils, court, ESPN ID
DATA_CFG    = CFG["data"]            # features list, paths, split sizes
API_CFG     = CFG["ncaa_api"]        # ESPN endpoints, season, rate limit
SF_CFG      = CFG["snowflake"]       # credentials via env vars
MODEL_CFG   = CFG["models"]          # enabled models + hyperparams
AL_CFG      = CFG["auto_learn"]      # intervals, thresholds
ROSTER_CFG  = CFG.get("roster", {})  # cache dir, TTL, team_id cache path
ROLLING_CFG = CFG.get("rolling", {}) # available windows, default window
```

Every subsequent function reads from these module-level dicts. Changing any setting requires only a `config.yaml` edit.

---

### Layer 1: Command Line Interface

**Function:** `main()`

**Commands:**

```bash
--fetch                # Fetch real NCAA games from ESPN API
--fetch-rosters        # Pre-fetch and cache rosters for all teams in dataset
--generate-synthetic   # Generate synthetic fallback data (500 games)
--train                # Train all models, register best by ROC-AUC
--serve                # Start Flask + auto-learn scheduler
--list-models          # Print version table to console
--activate VERSION     # Set active model (e.g. --activate v3)
--storage local|snowflake  # Storage backend (default: local)
```

---

### Layer 2: Data Ingestion

#### ESPN Game Data (`ESPNFetcher`)

Real NCAA data, no API key required.

```
get_game_ids(start, end)
  │
  └─ Iterates day-by-day from Nov 1 to Apr 30 of configured season
     └─ GET /scoreboard?dates=YYYYMMDD → list of event IDs

get_box_score(event_id)
  │
  ├─ GET /summary?event=ID
  ├─ Parses boxscore.teams → home_d, away_d
  ├─ Extracts stats dict: {fieldGoalPct, totalRebounds, assists, ...}
  ├─ Reads header.competitions[0].competitors for scores
  └─ Returns game dict with 14 features + outcome + metadata
```

**Rate limiting:** `time.sleep(delay)` between every request (configurable in `config.yaml`).

**Field normalization:** FG% from ESPN arrives as `"45.5"` (percent string). `norm_pct()` divides by 100 if value > 1.

#### Custom API (`CustomAPIFetcher`)

Stub for user-provided APIs. Set `provider: custom` in `config.yaml`, fill `base_url` and `field_map`. No code changes needed.

#### Synthetic Fallback (`_generate_synthetic`)

Used when ESPN is unavailable. Generates realistic game records using uniform distributions and a weighted strength formula with home court advantage (+3). Outcome has 15% noise to avoid perfect separability.

---

### Layer 3: Storage Abstraction

**Local (default):**

```python
save_to_json(data)      # full overwrite
load_from_json()        # full read
append_to_json(data)    # load → deduplicate by game_id → save
```

Deduplication prevents the same ESPN game from being counted twice across fetch runs.

**Snowflake (optional):**

```python
_sf_conn()              # reads credentials from env vars
_sf_create_table(conn)  # CREATE TABLE IF NOT EXISTS
save_to_snowflake(data) # DELETE + bulk INSERT
append_to_snowflake(data) # INSERT only
```

Snowflake is disabled by default (`enabled: false` in config). Enable by setting `enabled: true` and providing `SNOWFLAKE_USER` / `SNOWFLAKE_PASSWORD` environment variables.

**Common interface:** `load_data(storage)` routes to either backend transparently. Training code never knows which backend is active.

---

### Layer 4: Team Stats Engine

**Function:** `build_team_stats(data, window=None)`

This is what powers the predict form's auto-fill. For every game in the dataset, each team's stats are accumulated regardless of whether they were home or away.

**Rolling window support:** If `window` is set, only the most recent N games per team are used (sorted by `fetched_at` descending). `None` means full season average. The `/teams` and `/team_stats/<name>` endpoints accept a `?window=N` query param that is passed directly through to this function.

```
For each game:
  Home team → accumulate home_* columns under their home_* feature keys
  Home team → accumulate away_* columns (mirrored) under their home_* keys
  Away team → same logic, mirrored

Per team:
  Sort games newest-first → slice to window (or keep all)
  Average all accumulated values → season/window average per feature
  Count games_played (total, not windowed), games_in_window, wins
```

This means Duke's `home_ppg` stat represents their average points scored per game — not just when they were the home side. The feature names remain `home_*` / `away_*` because that's what the model was trained on; they represent "the team filling the home slot" vs "the team filling the away slot" in a prediction.

**`get_home_team_stats(data, window=None)`:** Fuzzy-matches the configured home team name and returns their stats dict. Used by the `/home_team` endpoint to pre-fill the prediction form.

---

### Layer 5: Model Training

**Function:** `train_and_evaluate(storage, triggered_by)`

**Models built by `build_models()`:**

All wrapped in `StandardScaler → estimator` Pipeline. This means scaling is part of the model object itself — no separate scaler needs to be saved or loaded.

| Key in config | Class | Notes |
|---------------|-------|-------|
| `gradient_boosting` | `GradientBoostingClassifier` | Sequential trees |
| `random_forest` | `RandomForestClassifier` | Parallel ensemble |
| `extra_trees` | `ExtraTreesClassifier` | Randomized splits |
| `svm` | `SVC(probability=True)` | RBF kernel |
| `mlp` | `MLPClassifier` | 128→64→32, early stopping |
| `xgboost` | `XGBClassifier` | Optional — graceful `ImportError` skip |

**Training sequence:**

```
1. load_data(storage)
2. prepare_data(data) → X, y, feature_names
3. train_test_split(stratify=y, test_size=0.2)
4. For each model:
   a. model.fit(X_train, y_train)
   b. compute_metrics(model, X_test, y_test)
      → accuracy, precision, recall, f1, roc_auc, confusion_matrix
   c. get_feature_importances(model, feature_names)
      → reads clf.feature_importances_ or abs(clf.coef_)
   d. cross_val_score(model, X, y, cv=5, scoring="roc_auc")
      → cv_roc_auc_mean, cv_roc_auc_std
5. best = max by config selection_metric (default: roc_auc)
6. Promote gate (if triggered_by != "manual"):
   if new_auc < current_auc + promote_threshold → skip, log, return None
7. register_model(best)
8. Save comparison_vN.json + latest_comparison.json
9. _append_log(result)
```

**`stratify=y`** in the train-test split ensures the home win ratio is preserved in both sets — important because real NCAA data has ~73% home win rate.

---

### Layer 6: Model Registry

**Functions:** `register_model`, `load_active_model`, `set_active_version`

**Registry file:** `models/registry.json`

```json
{
  "active_version": "v3",
  "versions": [
    {
      "version": "v1",
      "model_name": "Extra Trees",
      "filename": "extra_trees_v1_a3f2c1d4.pkl",
      "metrics": { "roc_auc": 0.9766, "f1": 0.9474, ... },
      "feature_names": ["home_ppg", "away_ppg", ...],
      "training_size": 400,
      "trained_at": "2025-03-15T14:22:11",
      "hash": "a3f2c1d4"
    },
    ...
  ]
}
```

**Version numbering:** Sequential integer, prefixed `v`. Each version gets a short MD5 hash of the serialized model object for integrity.

**`.pkl` format:** Each file contains `{"model": pipeline_obj, "feature_names": list}`. Storing feature names with the model prevents silent mismatches if the feature list changes between versions.

**Pruning:** When `len(versions) > keep_top_n` (default 10), oldest `.pkl` files are deleted from disk and their entries removed from the registry.

**Rollback:** `set_active_version("v2")` simply updates `active_version` in the registry. The next `/predict` call loads that version's `.pkl`.

---

### Layer 7: Learning Log

**File:** `data/learning_log.json`

Every training run appends one entry — whether promoted or skipped:

```json
{
  "timestamp": "2025-03-15T20:00:01",
  "triggered_by": "new_data",
  "result": "promoted",
  "version": "v3",
  "model_name": "Gradient Boosting",
  "roc_auc": 0.9891,
  "f1": 0.96,
  "dataset_size": 500
}
```

```json
{
  "triggered_by": "scheduler",
  "result": "skipped",
  "reason": "New AUC 0.9812 vs current 0.9891 (threshold +0.002). Skipping.",
  "new_auc": 0.9812,
  "current_auc": 0.9891
}
```

This log is what the Dashboard's **Auto-Learn tab** reads.

---

## 🚀 Command Execution Flows

### Flow 1: `python main.py --fetch`

```
main()
  │
  └─ fetch_ncaa_data(max_games=500)
      │
      ├─ ESPNFetcher.get_game_ids("20241101", "20250430")
      │   └─ ~180 days × 25 games/day = thousands of IDs
      │
      └─ For each game_id (up to max_games):
          │
          ├─ ESPNFetcher.get_box_score(gid)
          │   ├─ GET /summary?event=ID
          │   ├─ Parse box_teams → home_d, away_d
          │   ├─ Extract stats (FG%, rebounds, assists, turnovers, steals, blocks)
          │   ├─ Normalize pct fields
          │   ├─ Read scores → determine outcome
          │   └─ Return game dict (14 features + metadata)
          │
          └─ append_to_json(games)
              ├─ load_from_json() → existing games
              ├─ Filter: keep only game_ids not already stored
              └─ save_to_json(existing + new_unique)
```

---

### Flow 2: `python main.py --fetch-rosters`

```
main()
  │
  ├─ load_from_json() → existing game records
  │
  ├─ Collect unique team names from home_team + away_team fields
  │   └─ discard empty strings
  │
  └─ RosterFetcher.fetch_team(name) for each team  ← blocking, not async
      │
      ├─ get_team_id(team_name)
      │   ├─ Load data/team_ids.json cache
      │   ├─ Exact match → return ID
      │   ├─ Case-insensitive fuzzy match → return ID
      │   ├─ Cache miss → GET /teams?limit=1000
      │   │   └─ Parse sports[0].leagues[0].teams → build full cache
      │   └─ Save updated cache → try exact + fuzzy again
      │
      ├─ _cache_valid(team_id)
      │   └─ Check data/rosters/<team_id>.json exists + fetched_at < 24h ago
      │
      ├─ get_roster(team_id)
      │   ├─ GET /teams/{id}/roster
      │   ├─ For each athlete: extract id, name, position, jersey
      │   └─ _parse_embedded_stats(athlete)
      │       ├─ Read athlete.statistics[] and athlete.displayStats[]
      │       └─ Extract ppg, rpg, apg, spg, bpg, tov, fg_pct, fgm, fga
      │
      ├─ For players where ppg == 0 (no embedded stats):
      │   └─ get_player_stats(player_id)
      │       ├─ GET /athletes/{id}/statistics
      │       ├─ Find "avg"/"pergame" category in splits
      │       └─ Extract same stat fields (graceful 404 handling)
      │
      └─ Save result → data/rosters/<team_id>.json
          { team_name, team_id, players: [...], fetched_at }
```

---

### Flow 3: `python main.py --train`

```
main()
  │
  └─ train_and_evaluate("local", triggered_by="manual")
      │
      ├─ load_from_json() → 500 game dicts
      │
      ├─ prepare_data(data)
      │   ├─ Filter: only records with all 14 features present
      │   ├─ X = np.array([[g[feat] for feat in cfg_features] for g in valid])
      │   │   shape: (500, 14)
      │   └─ y = np.array([g["outcome"] for g in valid])
      │       shape: (500,)   ~73% ones (home wins)
      │
      ├─ train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
      │   X_train: (400, 14),  X_test: (100, 14)
      │
      ├─ build_models() → 5 (or 6) Pipeline objects
      │
      ├─ For each model:
      │   ├─ Pipeline.fit(X_train, y_train)
      │   │   (StandardScaler fits on X_train, transforms, then clf.fit)
      │   ├─ compute_metrics(model, X_test, y_test)
      │   ├─ get_feature_importances(model, feature_names)
      │   └─ cross_val_score(model, X, y, cv=5, scoring="roc_auc")
      │
      ├─ best = max by roc_auc
      │
      ├─ register_model(best_name, best_pipeline, metrics, features, 400)
      │   ├─ Increment version number
      │   ├─ Hash model bytes (MD5)
      │   ├─ model_path.write_bytes(pickle.dumps({model, feature_names}))
      │   ├─ Append entry to registry.json
      │   └─ Prune old versions if > keep_top_n
      │
      ├─ _sanitize(snap) → json.dump → comparison_vN.json + latest_comparison.json
      │   (_sanitize replaces NaN/Inf with None for valid JSON)
      │
      └─ _append_log(promoted entry)
```

---

### Flow 4: `python main.py --serve`

```
main()
  │
  ├─ _scheduler.start()
  │   └─ daemon thread: AutoLearnScheduler._loop()
  │       └─ time.sleep(60)  # wait for Flask to start
  │           └─ [see Auto-Learning Pipeline section]
  │
  └─ app.run(debug=False, port=5000, use_reloader=False)
      │        ↑
      │   use_reloader=False prevents the scheduler from
      │   starting twice in Flask's debug reload process
      │
      └─ Serving at http://localhost:5000
```

---

## 🤖 Auto-Learning Pipeline

**Class:** `AutoLearnScheduler` — runs as a daemon thread.

```
_loop():
  sleep(60)  ← let Flask initialize

  while not stopped:
    now = time.time()

    if now - last_fetch >= fetch_interval (6h):
      status = "fetching"
      new_games = fetch_ncaa_data()
      added = append_to_json(new_games)
      last_fetch = now

      if added >= min_new_games (15):
        status = "training"
        train_and_evaluate(triggered_by="new_data")
        last_retrain = now

    elif now - last_retrain >= retrain_interval (24h):
      status = "training"
      train_and_evaluate(triggered_by="scheduler")
      last_retrain = now

    status = "idle"
    sleep in 60s chunks (checking stop event each time)
```

**Promote gate inside `train_and_evaluate` when `triggered_by != "manual"`:**

```
new_auc = best_model.metrics["roc_auc"]
current_auc = active_entry.metrics["roc_auc"]

if new_auc >= current_auc + promote_threshold:
    register_model() → new active version
    log: { result: "promoted" }
else:
    log: { result: "skipped", reason: "..." }
    return None  ← no version change
```

This means the active model is immutable unless something genuinely better emerges.

---

## 🔄 Data Flow

### End-to-End Stats-Mode Prediction Flow

```
User opens http://localhost:5000
  │
  └─ Flask: send_file("dashboard.html")
      │
      └─ Browser executes JS init():
          │
          ├─ fetch("/features")         → feature list + rolling window options
          ├─ fetch("/home_team")        → Duke stats from season averages
          ├─ fetch("/teams")            → all teams dropdown
          ├─ fetch("/model_info")       → active version + metrics
          ├─ fetch("/analytics")        → stats + comparison + feature importances
          ├─ fetch("/registry")         → all versions for Registry tab
          ├─ fetch("/autolearn/status") → scheduler state
          └─ fetch("/learning_log")     → training history

User selects away team from dropdown
  │
  └─ onAwayTeamChange()
      └─ fetch("/team_stats/Kansas Jayhawks?window=12")
          └─ build_team_stats(data, window=12) → last 12 games only
              └─ Auto-fill away stat fields (green border)

User clicks "Predict"
  │
  └─ POST /predict  { home_ppg: 84.3, away_ppg: 72.1, ... }
      │
      ├─ load_active_model()
      │   └─ registry.json → active_version → load .pkl
      │       └─ payload = { model: Pipeline, feature_names: [...] }
      │
      ├─ X = np.array([[payload[f] for f in feature_names]])
      │
      ├─ pred = model.predict(X)[0]
      │   (Pipeline: StandardScaler.transform → clf.predict)
      │
      ├─ conf = max(model.predict_proba(X)[0])
      │
      └─ return { prediction, confidence, model_name, version }
```

### End-to-End Roster-Mode Prediction Flow

```
User switches to Roster mode in dashboard
  │
  └─ fetch("/roster/Duke Blue Devils")
      │
      ├─ RosterFetcher.fetch_team_async("Duke Blue Devils")
      │   ├─ If cached and valid → write _roster_progress["Duke Blue Devils"]
      │   │   { status: "ready", players: [...], done: N, total: N }
      │   │   return (no thread needed)
      │   │
      │   └─ Else → set _roster_progress = { status: "loading", ... }
      │       └─ threading.Thread(target=_run, daemon=True).start()
      │           └─ fetch_team(team_name) [see --fetch-rosters flow]
      │               └─ writes incremental progress to _roster_progress
      │                  after each player stat fetch
      │
      └─ Return current _roster_progress state immediately

Dashboard polls GET /roster/progress/Duke Blue Devils every 1 second
  │
  └─ return _roster_progress.get(team_name)
      { status: "loading"|"ready"|"error", players: [...], done: N, total: M }
      └─ Browser renders player cards as they appear, shows progress bar

User selects players and clicks "Predict (Roster)"
  │
  └─ POST /predict/from_roster
      {
        "home_players": [ {ppg, rpg, apg, spg, bpg, tov, fg_pct, fgm, fga}, ... ],
        "away_players": [ {...}, ... ]
      }
      │
      ├─ compute_stats_from_roster(home_players, "home")
      │   ├─ Sum ppg, rpg, apg, spg, bpg, tov across all players
      │   ├─ FGA-weighted average for fg_pct
      │   │   (total_fgm / total_fga; falls back to simple avg if fga = 0)
      │   └─ Returns { home_ppg, home_fg_pct, home_rebounds, ... }
      │
      ├─ compute_stats_from_roster(away_players, "away") → { away_* }
      │
      ├─ combined = { **home_stats, **away_stats }
      │
      ├─ X = np.array([[combined.get(f, 0) for f in feature_names]])
      │   (missing features filled with 0 rather than crashing)
      │
      ├─ pred, conf — same pipeline as stats mode
      │
      └─ return { prediction, confidence, computed_stats, home_count, away_count, ... }
          └─ computed_stats shown in dashboard so user sees what numbers were used
```

---

## 💾 Storage Architecture

### Local JSON

**File:** `data/games.json`

**Record structure:**
```json
{
  "game_id": "ESPN_401703521",
  "home_team": "Duke Blue Devils",
  "away_team": "Kansas Jayhawks",
  "home_score": 84,
  "away_score": 72,
  "home_ppg": 84.0,
  "away_ppg": 72.0,
  "home_fg_pct": 0.4921,
  "away_fg_pct": 0.3854,
  "home_rebounds": 39.0,
  "away_rebounds": 31.0,
  "home_assists": 17.0,
  "away_assists": 11.0,
  "home_turnovers": 11.0,
  "away_turnovers": 14.0,
  "home_steals": 8.0,
  "away_steals": 6.0,
  "home_blocks": 5.0,
  "away_blocks": 3.0,
  "outcome": 1,
  "source": "espn",
  "fetched_at": "2025-03-15T14:00:00"
}
```

**Deduplication:** `append_to_json` builds a set of existing `game_id` values and filters new data against it before writing. Same game can never appear twice regardless of how many times `--fetch` is run.

### Roster Cache

**Directory:** `data/rosters/<team_id>.json`

**Record structure:**
```json
{
  "team_name": "Duke Blue Devils",
  "team_id": "150",
  "players": [
    {
      "id": "4432783",
      "name": "Kyle Filipowski",
      "position": "C",
      "jersey": "30",
      "ppg": 16.4,
      "rpg": 8.1,
      "apg": 2.3,
      "spg": 0.8,
      "bpg": 1.4,
      "tov": 2.2,
      "fg_pct": 0.517,
      "fgm": 5.9,
      "fga": 11.4
    }
  ],
  "fetched_at": "2025-03-15T14:00:00"
}
```

**TTL:** 24 hours by default (`roster.cache_ttl_hours` in config). `_cache_valid()` compares `fetched_at` to now.

**Team ID cache:** `data/team_ids.json` maps display name → ESPN team ID. Built on first lookup and reused on all subsequent calls. Fuzzy matching (`team_name.lower() in k.lower()`) handles minor name variations.

### Snowflake (Optional)

Schema auto-created from `DATA_CFG["features"]` list — adding a feature to config automatically adds it to the CREATE TABLE statement. Credentials read exclusively from environment variables.

---

## 🎯 Model Training Pipeline Detail

### Why ROC-AUC as Selection Metric

ROC-AUC measures the model's ability to rank home wins above away wins across all decision thresholds — it's threshold-independent. With ~73% home win rate in real NCAA data, accuracy alone would reward a model that just predicts "Home Win" every time. ROC-AUC penalizes this. Configurable to `f1`, `accuracy`, etc. in `config.yaml`.

### Why Stratified Split

The real ESPN data has ~73% home wins. Without `stratify=y`, a random 20% test set might have 80% or 65% home wins by chance — making evaluation unstable. Stratification guarantees the test set mirrors the full dataset's class distribution.

### Why Pipeline (Scaler + Estimator)

SVM and MLP are sensitive to feature scale — `home_ppg` (range 60-95) and `home_fg_pct` (range 0.38-0.55) are on completely different scales. Wrapping in a Pipeline means the scaler is trained only on `X_train`, then applied to `X_test` — no data leakage. The entire Pipeline is pickled as one object, so loading a model version automatically gets the correct scaler.

### NaN Sanitization

XGBoost's cross-validation occasionally returns `NaN` when it cannot compute a CV score (e.g. label imbalance in a fold). Python's `json.dumps` writes `NaN` literally — which is invalid JSON spec and causes `JSON.parse` to throw in the browser. `_sanitize(obj)` recursively replaces any `float('nan')` or `float('inf')` with `None` before serialization.

---

## 🏃 Roster System

**Class:** `RosterFetcher`

**ESPN endpoints used:**

| Endpoint | Purpose |
|----------|---------|
| `GET /teams?limit=1000` | All teams — builds team name → ID cache |
| `GET /teams/{id}/roster` | Player list with embedded stats |
| `GET /athletes/{id}/statistics` | Per-player season averages (fallback only) |

**Stat extraction priority:**

```
1. athlete.statistics[] + athlete.displayStats[]  ← embedded in roster response
   (preferred — avoids separate API calls)

2. GET /athletes/{id}/statistics                  ← only for players with ppg == 0
   (fallback — ESPN returns 404 for many players; caught gracefully)

3. _empty_player_stats()                          ← all zeros / fg_pct = 0.45 default
   (last resort — player still appears in list, just with no stats)
```

**Async pattern:**

`fetch_team_async(team_name)` starts a background thread and returns immediately. The thread writes incremental progress to the module-level `_roster_progress` dict as each player is processed:

```python
_roster_progress: dict = {}   # module-level, shared between Flask threads

# Progress states:
{ "status": "loading", "players": [...so_far...], "done": 12, "total": 15 }
{ "status": "ready",   "players": [...all...],    "done": 15, "total": 15 }
{ "status": "error",   "players": [],             "done": 0,  "total": 0,
  "message": "Could not fetch roster for 'Team X'." }
```

Player names appear in the dashboard immediately (from the roster endpoint before stat fetches begin), and stats fill in as the thread progresses. The dashboard polls `/roster/progress/<team>` every second until `status == "ready"`.

**Aggregation (`compute_stats_from_roster`):**

```
team_ppg       = sum(player.ppg for each player)
team_rebounds  = sum(player.rpg for each player)
team_assists   = sum(player.apg for each player)
team_steals    = sum(player.spg for each player)
team_blocks    = sum(player.bpg for each player)
team_turnovers = sum(player.tov for each player)

team_fg_pct:
  if sum(fga) > 0:  total_fgm / total_fga   ← FGA-weighted (more accurate)
  else:             mean(player.fg_pct)       ← simple average fallback
```

Result dict uses `{side}_*` prefix (`home_*` or `away_*`) to match the model's expected feature names exactly.

---

## 🌐 API Reference

| Method | Endpoint | Returns |
|--------|----------|---------|
| GET | `/` | `dashboard.html` |
| POST | `/predict` | prediction, confidence, version (stats mode) |
| POST | `/predict/from_roster` | prediction, confidence, computed_stats, player counts |
| GET | `/analytics` | games stats, model comparison, feature importances |
| GET | `/model_info` | active model entry from registry |
| GET | `/registry` | full registry JSON |
| POST | `/registry/activate/<v>` | `{status: ok}` |
| GET | `/features` | feature list + rolling window options from config |
| GET | `/teams?window=N` | all teams with season or rolling averages |
| GET | `/team_stats/<name>?window=N` | single team stats (fuzzy match), optional window |
| GET | `/home_team?window=N` | configured home team + stats |
| GET | `/roster/<team_name>` | kick off async roster fetch; returns immediate progress state |
| GET | `/roster/progress/<team_name>` | poll fetch progress: status, players so far, done/total |
| POST | `/roster/refresh/<team_name>` | force fresh ESPN fetch (bypass 24h cache) |
| GET | `/autolearn/status` | scheduler state + countdowns |
| POST | `/autolearn/trigger` | starts background retrain |
| GET | `/learning_log?n=50` | last N log entries |
| GET | `/debug` | health check (paths, counts, active model, roster dirs) |

---

## 🎨 Dashboard Integration

### Tab Rendering Strategy

Charts are not drawn on page load — they are drawn fresh each time a tab becomes visible via `switchTab()`. This solves the Chart.js 0×0 canvas problem: hidden `display:none` tabs have no dimensions at render time.

```
switchTab(name)
  │
  ├─ Remove .active from all tabs and panes
  ├─ Add .active to selected pane
  │
  └─ requestAnimationFrame(() => {
        if name == "overview":
            drawOutcomeChart()   // uses analyticsData
            drawRadar()          // uses model_comparison
            drawProgressChart()  // uses registryData
        elif name == "comparison":
            drawComparisonCharts()   // grouped bar + multi-radar
            buildComparisonTable()   // metric bars table
        elif name == "features":
            drawFeatureChart()   // home_win vs away_win averages
            buildFiSelector()    // dropdown + importance bar chart
    })
```

`requestAnimationFrame` defers execution by one paint cycle, ensuring the browser has applied `display:block` before Chart.js measures the canvas.

### Data Separation

`loadAnalytics()` fetches data and updates stat cards (plain DOM — always safe). It does **not** draw charts. Charts are drawn by `switchTab()` only when their canvas is visible and has real pixel dimensions. This separation means a slow analytics fetch never blocks tab switching.

---

## 🔧 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Config-driven via YAML | No secrets or tunables in Python source |
| ROC-AUC for model selection | Robust to class imbalance (~73% home wins) |
| Stratified train-test split | Preserves class ratio in both sets |
| Pipeline (Scaler + clf) | Scaler trained on train set only; no leakage |
| Model versioning with hash | Detects model file corruption |
| Promote threshold | Prevents model regression from noise |
| `use_reloader=False` | Prevents scheduler starting twice in Flask debug |
| `_sanitize()` before JSON | XGBoost CV can return NaN; invalid JSON crashes browser |
| Lazy chart rendering | Chart.js cannot render into 0×0 hidden canvases |
| game_id deduplication | Safe to run `--fetch` multiple times |
| Roster embedded stats first | Avoids per-player `/statistics` calls; fewer 404s |
| Async roster fetch + progress | Player names visible immediately; stats stream in |
| Module-level `_roster_progress` | Shared state between Flask request thread and roster thread |
| FGA-weighted fg_pct aggregation | More accurate than simple average when players have unequal shot volume |
| `window=None` default in `build_team_stats` | Full season average unless caller explicitly requests rolling |

---

*This document provides a complete technical reference for understanding how the system operates at every level, from config loading to chart rendering.*
