# Variable List

Complete reference for every feature, config key, API field, and runtime variable in the system. Version 2.5.1.

---

## Table of Contents

1. [Game Features (Model Input)](#1-game-features-model-input)
2. [Target Variable](#2-target-variable)
3. [Game Metadata Fields](#3-game-metadata-fields)
4. [Analytics-Only Fields](#4-analytics-only-fields)
5. [Team Stats Fields (Dashboard)](#5-team-stats-fields-dashboard)
6. [Player Stats Fields (Roster)](#6-player-stats-fields-roster)
7. [Roster Cache Fields](#7-roster-cache-fields)
8. [Model Metrics](#8-model-metrics)
9. [Registry Entry Fields](#9-registry-entry-fields)
10. [Learning Log Entry Fields](#10-learning-log-entry-fields)
11. [Config Keys (config.yaml)](#11-config-keys-configyaml)
12. [Environment Variables](#12-environment-variables)
13. [API Response Fields](#13-api-response-fields)
14. [CLI Arguments](#14-cli-arguments)
15. [Runtime State Variables](#15-runtime-state-variables)

---

## 1. Game Features (Model Input)

These 14 fields form the feature matrix X passed to every model. Order is fixed by `config.yaml -> data.features`. All are continuous floating-point values.

**Critical v2.5 change:** These fields now store pre-game rolling averages, not in-game box score statistics. For any given game record, `home_fg_pct = 0.47` means "this team averaged 47% FG over their last 10 games going into this game" — information that was available before tipoff. In v2.4 and earlier it meant "what they shot during this game" — which is circular.

The window size is configurable (`data.pregame_window`, default 10). Records where either team has fewer prior games than `data.pregame_min_games` are excluded from training.

| Variable | Type | Description | Typical Range |
|----------|------|-------------|---------------|
| `home_fg_pct` | float | Rolling avg field goal percentage for the team in the home slot | 0.40 – 0.52 |
| `away_fg_pct` | float | Rolling avg field goal percentage for the team in the away slot | 0.40 – 0.52 |
| `home_rebounds` | float | Rolling avg total rebounds per game | 30 – 42 |
| `away_rebounds` | float | Rolling avg total rebounds per game | 30 – 42 |
| `home_assists` | float | Rolling avg assists per game | 11 – 18 |
| `away_assists` | float | Rolling avg assists per game | 11 – 18 |
| `home_turnovers` | float | Rolling avg turnovers committed per game | 10 – 16 |
| `away_turnovers` | float | Rolling avg turnovers committed per game | 10 – 16 |
| `home_steals` | float | Rolling avg steals per game | 4 – 10 |
| `away_steals` | float | Rolling avg steals per game | 4 – 10 |
| `home_blocks` | float | Rolling avg blocks per game | 2 – 8 |
| `away_blocks` | float | Rolling avg blocks per game | 2 – 8 |
| `home_ast_to_tov` | float | Rolling avg assists divided by rolling avg turnovers | 0.6 – 2.0 |
| `away_ast_to_tov` | float | Rolling avg assists divided by rolling avg turnovers | 0.6 – 2.0 |

**Note on ast_to_tov:** This is computed as `rolling_avg(assists) / rolling_avg(turnovers)`, not as an average of the per-game ratio. The former is more numerically stable and avoids outliers from single games with 1-2 turnovers.

**Feature count:** 14. In v2.4 the count was effectively 18 (including `home_ppg`, `away_ppg`, `home_eff_score`, `away_eff_score`). Those four were removed because they are score-derived and therefore leakage.

---

## 2. Target Variable

| Variable | Type | Values | Description |
|----------|------|--------|-------------|
| `outcome` | int | `1` or `0` | 1 = Home Win, 0 = Away Win |

Derived in `get_box_score()`:
```python
outcome = 1 if home_score > away_score else 0
```

Games where both scores are 0 are discarded (game not yet played or ESPN parse failure).

**Class distribution (3 seasons, ~2900 games):** approximately 69% Home Win, 31% Away Win. Slightly less imbalanced than single-season data because more varied matchups are included.

---

## 3. Game Metadata Fields

Stored in `data/games.json` alongside features. Not used as model inputs.

| Field | Type | Description |
|-------|------|-------------|
| `game_id` | string | Unique identifier. Format: `ESPN_<event_id>` or `SYN_<n>` for synthetic |
| `game_date` | string | Date the game was played, YYYY-MM-DD. Extracted from ESPN event.date ISO string. Required for correct chronological ordering in the enrichment step. |
| `home_team` | string | Full team display name from ESPN (e.g. "Duke Blue Devils") |
| `away_team` | string | Full team display name from ESPN |
| `home_score` | int | Final score of the home team |
| `away_score` | int | Final score of the away team |
| `source` | string | "espn", "custom", or "synthetic" |
| `fetched_at` | string | ISO 8601 timestamp of when the record was fetched |
| `pregame_enriched` | bool | True if the feature fields contain rolling averages. False if they still contain in-game stats. Training code filters to `pregame_enriched=True` only. |
| `pregame_window_used` | int | The effective window used for this record (= min(home_history_len, away_history_len, configured_window)). Present only when pregame_enriched=True. |

---

## 4. Analytics-Only Fields

These fields are stored in `games.json` but are **not** in the training feature vector. They are used only for analytics display in the dashboard.

| Field | Type | Description | Why not a feature |
|-------|------|-------------|-------------------|
| `home_ppg` | float | Points scored by the home team in this game | This is the game score. Using the score as a feature to predict the outcome is circular. |
| `away_ppg` | float | Points scored by the away team in this game | Same reason. |
| `home_eff_score` | float | home_score x home_fg_pct. Stored at game level. | Derived from the score, therefore also leakage. |
| `away_eff_score` | float | away_score x away_fg_pct. | Same reason. |
| `home_game_fg_pct` | float | Original in-game FG% before enrichment replaced it | Present only on enriched records. The corresponding feature field now holds the rolling average. |
| `away_game_fg_pct` | float | Original in-game FG% before enrichment | Same. |
| `home_game_rebounds` | float | Original in-game rebounds before enrichment | Same. |
| `away_game_rebounds` | float | Original in-game rebounds before enrichment | Same. |
| `home_game_assists` | float | Original in-game assists | Same. |
| `away_game_assists` | float | Same. | Same. |
| `home_game_turnovers` | float | Original in-game turnovers | Same. |
| `away_game_turnovers` | float | Same. | Same. |
| `home_game_steals` | float | Original in-game steals | Same. |
| `away_game_steals` | float | Same. | Same. |
| `home_game_blocks` | float | Original in-game blocks | Same. |
| `away_game_blocks` | float | Same. | Same. |
| `home_game_ast_to_tov` | float | Original in-game ast/tov ratio | Same. |
| `away_game_ast_to_tov` | float | Same. | Same. |

---

## 5. Team Stats Fields (Dashboard)

Returned by `/teams` and `/team_stats/<name>`. Computed by `build_team_stats(data, window)` in `app/preprocessing.py`.

Both endpoints accept `?window=N`. When set, only the team's most recent N games are used. When omitted, all games are used (full dataset average).

Since game records now store pre-game rolling averages as feature values, these aggregations compound one layer of averaging on top of another. This is intentional — the result is a stable season-form estimate suitable for pre-filling the prediction form.

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Team display name |
| `home_fg_pct` | float | Average of the pre-game rolling avg FG% across all games (regardless of home/away side) |
| `away_fg_pct` | float | Average of opponent's rolling avg FG% across all games |
| `home_rebounds` | float | Average of the pre-game rolling avg rebounds per game |
| `away_rebounds` | float | Average of opponent's rolling avg rebounds |
| `home_assists` | float | Average of the pre-game rolling avg assists |
| `away_assists` | float | Opponent average |
| `home_turnovers` | float | Average of the pre-game rolling avg turnovers |
| `away_turnovers` | float | Opponent average |
| `home_steals` | float | Average of the pre-game rolling avg steals |
| `away_steals` | float | Opponent average |
| `home_blocks` | float | Average of the pre-game rolling avg blocks |
| `away_blocks` | float | Opponent average |
| `home_ast_to_tov` | float | Average of pre-game rolling avg ast/tov ratio |
| `away_ast_to_tov` | float | Opponent average |
| `games_played` | int | Total games this team appears in (not windowed) |
| `games_in_window` | int | Games used for the averages above (= games_played when window is None) |
| `wins` | int | Total wins across all games (not windowed) |

**Why home_* / away_* naming:** The feature names match the model's training columns. When a team fills the home slot in a prediction, their season averages go into the `home_*` feature fields. The naming is positional (home slot vs away slot), not directional (home venue vs away venue).

---

## 6. Player Stats Fields (Roster)

Individual player records. Also the format expected by `/predict/from_roster`.

| Field | Type | Description | Typical Range |
|-------|------|-------------|---------------|
| `id` | string | ESPN athlete ID | — |
| `name` | string | Player full display name | — |
| `position` | string | Position abbreviation (C, G, F) | — |
| `jersey` | string | Jersey number as string | — |
| `ppg` | float | Points per game (season average) | 0 – 35 |
| `rpg` | float | Rebounds per game | 0 – 15 |
| `apg` | float | Assists per game | 0 – 12 |
| `spg` | float | Steals per game | 0 – 4 |
| `bpg` | float | Blocks per game | 0 – 4 |
| `tov` | float | Turnovers per game | 0 – 6 |
| `fg_pct` | float | Field goal percentage as decimal. Defaults to 0.45 if unavailable. | 0.20 – 0.75 |
| `fgm` | float | Field goals made per game | 0 – 15 |
| `fga` | float | Field goals attempted per game | 0 – 25 |

**Stat extraction priority:**
1. Embedded stats in the roster API response (`athlete.statistics[]` and `athlete.displayStats[]`)
2. Per-player `/athletes/{id}/statistics` endpoint (called only for players with all-zero embedded stats)
3. `_empty_player_stats()` — zeros plus `fg_pct = 0.45` — used when ESPN returns 404

**Aggregation into team features** (`compute_stats_from_roster(players, side)`):

| Team Feature | Aggregation |
|---|---|
| `{side}_fg_pct` | `sum(fgm) / sum(fga)` if `sum(fga) > 0`, else `mean(fg_pct)` |
| `{side}_rebounds` | `sum(rpg)` |
| `{side}_assists` | `sum(apg)` |
| `{side}_steals` | `sum(spg)` |
| `{side}_blocks` | `sum(bpg)` |
| `{side}_turnovers` | `sum(tov)` |
| `{side}_ast_to_tov` | `sum(apg) / sum(tov)` (computed from totals, not averaged from ratios) |

Player season stats are already pre-game information by nature. No leakage in roster mode.

---

## 7. Roster Cache Fields

Stored in `data/rosters/<team_id>.json`.

| Field | Type | Description |
|-------|------|-------------|
| `team_name` | string | ESPN display name |
| `team_id` | string | ESPN internal team ID |
| `players` | list | Player objects (see Section 6) |
| `fetched_at` | string | ISO 8601. Cache valid for `roster.cache_ttl_hours` (default 24h). |

**Team ID cache** (`data/team_ids.json`): `{display_name: espn_id}` flat dict. Built on first lookup from `GET /teams?limit=1000`. Lookup tries exact match first, then case-insensitive substring match.

---

## 8. Model Metrics

| Metric | Type | Range | Description |
|--------|------|-------|-------------|
| `accuracy` | float | 0 – 1 | (TP + TN) / total |
| `precision` | float | 0 – 1 | TP / (TP + FP) |
| `recall` | float | 0 – 1 | TP / (TP + FN) |
| `f1` | float | 0 – 1 | Harmonic mean of precision and recall |
| `roc_auc` | float | 0 – 1 | Area under ROC curve — primary selection metric |
| `cv_roc_auc_mean` | float | 0 – 1 | Mean ROC-AUC across 5 CV folds |
| `cv_roc_auc_std` | float | 0 – 1 | Standard deviation of CV ROC-AUC scores |
| `confusion_matrix` | list | — | [[TN, FP], [FN, TP]] |
| `feature_importances` | dict | — | {feature_name: importance} for tree models |

**Expected ranges after v2.5 leakage fix:** ROC-AUC 0.60-0.75. Previously 0.95-0.97. The lower number is correct — it reflects genuine pre-game predictive power rather than circular in-game statistics.

**TP/TN/FP/FN:** Defined relative to the positive class (Home Win = 1):
- TP: predicted Home Win, actual Home Win
- TN: predicted Away Win, actual Away Win
- FP: predicted Home Win, actual Away Win
- FN: predicted Away Win, actual Home Win

---

## 9. Registry Entry Fields

Each version entry in `models/registry.json`:

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | "v1", "v2", etc. |
| `model_name` | string | "Gradient Boosting", "Neural Network (MLP)", etc. |
| `filename` | string | .pkl filename including version and MD5 hash fragment |
| `metrics` | dict | All model metrics (see Section 8) |
| `feature_names` | list | Ordered feature list this model was trained on |
| `training_size` | int | Number of samples in the training set |
| `trained_at` | string | ISO 8601 timestamp |
| `hash` | string | First 8 hex chars of MD5 of the serialized model |

Top-level registry fields:

| Field | Type | Description |
|-------|------|-------------|
| `active_version` | string | Version currently served by /predict |
| `versions` | list | All version entries, oldest first |

---

## 10. Learning Log Entry Fields

Each entry in `data/learning_log.json`:

| Field | Type | Always Present | Description |
|-------|------|---------------|-------------|
| `timestamp` | string | Yes | ISO 8601 |
| `triggered_by` | string | Yes | "manual", "new_data", "scheduler", "manual_trigger" |
| `result` | string | Yes | "promoted" or "skipped" |
| `version` | string | Promoted | Registry version assigned |
| `model_name` | string | Promoted | Name of the promoted model |
| `roc_auc` | float | Promoted | ROC-AUC of the promoted model |
| `f1` | float | Promoted | F1-score |
| `accuracy` | float | Promoted | Accuracy |
| `dataset_size` | int | Yes | Total games in dataset at training time |
| `reason` | string | Skipped | Why promotion was skipped |
| `new_auc` | float | Skipped | ROC-AUC of the candidate model |
| `current_auc` | float | Skipped | ROC-AUC of the model that was kept |
| `best_model` | string | Skipped | Name of the candidate that was not promoted |

---

## 11. Config Keys (config.yaml)

### app

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `app.name` | string | "Basketball Game Outcome Predictor" | Display name |
| `app.version` | string | "2.5.1" | App version string |
| `app.debug` | bool | true | Flask debug mode |
| `app.port` | int | 5000 | Flask server port |
| `app.host` | string | "0.0.0.0" | Flask bind address |

### home_team

| Key | Type | Description |
|-----|------|-------------|
| `home_team.name` | string | Home team display name — must match ESPN displayName |
| `home_team.espn_id` | string | ESPN team ID (informational) |
| `home_team.court_name` | string | Arena name |

### data

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `data.dir` | string | "data" | Directory for data files |
| `data.local_file` | string | "data/games.json" | Path to local game records |
| `data.test_size` | float | 0.2 | Fraction held out for testing |
| `data.random_state` | int | 42 | Seed for reproducibility |
| `data.min_games_required` | int | 50 | Minimum records to attempt training |
| `data.features` | list | See Section 1 | Ordered feature column names (14 features) |
| `data.label` | string | "outcome" | Target column name |
| `data.pregame_window` | int | 10 | Number of prior games to average for pre-game features |
| `data.pregame_min_games` | int | 1 | Min prior games before a record is training-eligible |
| `data.leakage_correlation_threshold` | float | 0.70 | Feature-outcome correlation above this triggers a warning |

### ncaa_api

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ncaa_api.provider` | string | "espn" | "espn" or "custom" |
| `ncaa_api.season` | int | 2024 | Legacy single-season fallback |
| `ncaa_api.seasons` | list | [2022, 2023, 2024] | All seasons to fetch |
| `ncaa_api.max_games` | int | 3000 | Total game cap across all seasons |
| `ncaa_api.rate_limit_delay` | float | 0.4 | Seconds between ESPN requests |
| `ncaa_api.espn.base_url` | string | ESPN API base | Base URL |
| `ncaa_api.espn.scoreboard_path` | string | "/scoreboard" | Daily game IDs |
| `ncaa_api.espn.summary_path` | string | "/summary" | Box score details |
| `ncaa_api.espn.teams_path` | string | "/teams" | Team list and roster lookups |
| `ncaa_api.espn.athletes_path` | string | "/athletes" | Per-player statistics fallback |
| `ncaa_api.espn.page_size` | int | 25 | Games per scoreboard request |
| `ncaa_api.custom.base_url` | string | "" | Custom API provider base URL |
| `ncaa_api.custom.api_key` | string | "" | API key (or NCAA_API_KEY env var) |
| `ncaa_api.custom.field_map` | dict | — | Maps our field names to theirs |

### snowflake

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `snowflake.enabled` | bool | false | Enable Snowflake storage |
| `snowflake.user` | string | "" | Set via SNOWFLAKE_USER env var |
| `snowflake.password` | string | "" | Set via SNOWFLAKE_PASSWORD env var |
| `snowflake.account` | string | "" | Snowflake account identifier |
| `snowflake.warehouse` | string | "" | Compute warehouse |
| `snowflake.database` | string | "" | Database |
| `snowflake.schema` | string | "" | Schema |
| `snowflake.table` | string | "BASKETBALL_GAMES" | Table name |

### models

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `models.dir` | string | "models" | Directory for model files |
| `models.registry_file` | string | "models/registry.json" | Version registry path |
| `models.selection_metric` | string | "roc_auc" | Metric used to pick best model |
| `models.keep_top_n` | int | 10 | Max versions before pruning |
| `models.enabled` | list | All 5 plus XGBoost | Models to include each run |

Per-model hyperparameters (all under `models.<key>`). Values below reflect v2.5.1 defaults after fairness restoration:

| Model | Key | Default | Notes |
|-------|-----|---------|-------|
| gradient_boosting | n_estimators | 300 | Raised from 200 |
| gradient_boosting | learning_rate | 0.05 | |
| gradient_boosting | max_depth | 4 | Further capped by adaptive_depth at runtime |
| gradient_boosting | subsample | 0.8 | |
| gradient_boosting | min_samples_split | 5 | Restored from 10 |
| gradient_boosting | min_samples_leaf | 2 | Added explicitly |
| random_forest | n_estimators | 300 | Raised from 200 |
| random_forest | max_depth | 10 | Adaptive ceiling is 4 at current dataset size |
| random_forest | min_samples_split | 5 | Restored from 10 |
| random_forest | min_samples_leaf | 2 | Restored from 4 |
| extra_trees | n_estimators | 300 | |
| extra_trees | max_depth | 10 | |
| extra_trees | min_samples_split | 5 | |
| extra_trees | min_samples_leaf | 2 | |
| svm | kernel | "rbf" | |
| svm | C | 2.0 | Raised from 1.0 |
| svm | gamma | "scale" | |
| mlp | hidden_layer_sizes | [128, 64, 32] | Restored from [64, 32] |
| mlp | activation | "relu" | |
| mlp | max_iter | 500 | |
| mlp | early_stopping | true | |
| mlp | validation_fraction | 0.15 | |
| xgboost | n_estimators | 300 | |
| xgboost | learning_rate | 0.05 | |
| xgboost | max_depth | 5 | |
| xgboost | subsample | 0.8 | |
| xgboost | colsample_bytree | 0.8 | |
| xgboost | min_child_weight | 3 | Added in v2.5.1 |

All models also accept `random_state: 42`.

### auto_learn

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `auto_learn.enabled` | bool | true | Start scheduler on --serve |
| `auto_learn.fetch_interval_hours` | int | 6 | Hours between fetch attempts |
| `auto_learn.retrain_interval_hours` | int | 24 | Hours between forced retrains |
| `auto_learn.min_new_games_to_retrain` | int | 15 | New games to trigger immediate retrain |
| `auto_learn.promote_threshold` | float | 0.002 | Minimum AUC improvement to promote |
| `auto_learn.learning_log_file` | string | "data/learning_log.json" | |

### roster

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `roster.cache_dir` | string | "data/rosters" | Per-team roster cache directory |
| `roster.cache_ttl_hours` | int | 24 | Cache validity window |
| `roster.team_id_cache` | string | "data/team_ids.json" | ESPN name-to-ID lookup cache |

### rolling

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rolling.available_windows` | list | [5, 10, 15, 20] | Window sizes for dashboard rolling selector |
| `rolling.default_window` | int | 12 | Default window when no selection is made |

---

## 12. Environment Variables

| Variable | Used By | Description |
|----------|---------|-------------|
| `SNOWFLAKE_USER` | `_sf_conn()` | Snowflake username. Overrides `snowflake.user` in config. |
| `SNOWFLAKE_PASSWORD` | `_sf_conn()` | Snowflake password. |
| `NCAA_API_KEY` | `CustomAPIFetcher` | API key for custom provider. |

Never put credentials in `config.yaml`. On Windows PowerShell: `$env:SNOWFLAKE_USER = "youruser"`. On macOS/Linux: `export SNOWFLAKE_USER=youruser`.

---

## 13. API Response Fields

### POST /predict

Request: all 14 feature fields, names matching `feature_names` in the active model's registry entry.

Response:

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | "Home Win" or "Away Win" |
| `prediction_value` | int | 1 or 0 |
| `confidence` | float or null | max(predict_proba), range 0.5 to 1.0 |
| `model_name` | string | Name of the active model |
| `version` | string | Registry version |

### POST /predict/from_roster

Response:

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | "Home Win" or "Away Win" |
| `prediction_value` | int | 1 or 0 |
| `confidence` | float or null | max(predict_proba) |
| `model_name` | string | |
| `version` | string | |
| `computed_stats` | dict | The 14 model features actually used (aggregated from player selections) |
| `insights` | dict | insight_* fields: ppg totals, eff_score, scoring_dominance, depth_signal, role_balance, efficiency_spread |
| `home_count` | int | Number of home players selected |
| `away_count` | int | Number of away players selected |

### GET /analytics

| Field | Type | Description |
|-------|------|-------------|
| `total_games` | int | Total records in games.json |
| `home_wins` | int | Records with outcome == 1 |
| `away_wins` | int | Records with outcome == 0 |
| `home_win_rate` | float | home_wins / total_games |
| `enriched_games` | int | Records with pregame_enriched == True |
| `enrichment_rate` | float | enriched_games / total_games |
| `feature_stats` | dict | {home_win: {feature: avg}, away_win: {feature: avg}} |
| `model_comparison` | dict | {model_name: {accuracy, precision, recall, f1, roc_auc, ...}} |
| `feature_importances` | dict | {model_name: {feature: importance}} |
| `data_sources` | dict | {source_name: count} |

### GET /debug

| Field | Type | Description |
|-------|------|-------------|
| `game_count` | int | Total game records |
| `enriched_count` | int | Records with pregame_enriched == True |
| `enrichment_rate` | string | Percentage string e.g. "96.5%" |
| `active_model` | string or null | Active model name |
| `active_version` | string or null | Active version |
| `configured_seasons` | list | Seasons list from config |
| `max_games_cap` | int | max_games from config |
| `pregame_window` | int | pregame_window from config |
| `training_features` | list | Feature names from config |
| `cwd` | string | Current working directory |

### GET /roster/<team_name>

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "loading", "ready", "error", or "not_started" |
| `players` | list | Players fetched so far |
| `done` | int | Players whose stats have been resolved |
| `total` | int | Total players on roster |
| `message` | string | Present only when status == "error" |

### GET /autolearn/status

| Field | Type | Description |
|-------|------|-------------|
| `enabled` | bool | Whether scheduler is enabled |
| `status` | string | "idle", "fetching", or "training" |
| `fetch_interval_h` | int | Hours between fetches |
| `retrain_interval_h` | int | Hours between forced retrains |
| `min_new_games` | int | Threshold for immediate retrain |
| `promote_threshold` | float | Required AUC improvement |
| `next_fetch_in` | string | Countdown e.g. "5h 42m" |
| `next_retrain_in` | string | Countdown |
| `last_fetch` | string or null | ISO 8601, null if never |
| `last_retrain` | string or null | ISO 8601, null if never |

---

## 14. CLI Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--fetch` | flag | Fetch real NCAA games from ESPN (all configured seasons) |
| `--enrich` | flag | Back-fill pre-game rolling averages into existing games.json. Run once after upgrading to v2.5, then retrain. |
| `--fetch-rosters` | flag | Pre-fetch and cache rosters for every team in the dataset. Blocking. |
| `--generate-synthetic` | flag | Generate synthetic games as offline fallback |
| `--train` | flag | Train all enabled models, register best by selection metric |
| `--serve` | flag | Start Flask server and auto-learn scheduler |
| `--list-models` | flag | Print table of all registered versions |
| `--activate` | string | Version to set active, e.g. --activate v2 |
| `--storage` | choice | "local" (default) or "snowflake" |
| `--config` | string | Path to config file (default: config.yaml) |
| `--max-games` | int | Override config max_games cap for one fetch run |

---

## 15. Runtime State Variables

Module-level variables that live in memory for the duration of the server process.

| Variable | Module | Type | Description |
|----------|--------|------|-------------|
| `_roster_progress` | `app/roster.py` | `dict[str, dict]` | Shared state between Flask request threads and RosterFetcher background threads. Keyed by team display name. Written by fetch threads; read by /roster/progress/ and /roster/ endpoints. |
| `_scheduler` | `app/api.py` | `AutoLearnScheduler` | The singleton scheduler instance. Started by main.py --serve. Exposes get_state() and stop(). |

---

*All types reflect Python runtime values. Floats in JSON responses are rounded to 4 decimal places. Null values appear as JSON null and Python None. Config defaults above reflect v2.5.1.*
