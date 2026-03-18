# 📋 Variable List

> *Complete reference for every feature, config key, API field, and runtime variable in the system*

---

## Table of Contents

1. [Game Features (Model Input)](#1-game-features-model-input)
2. [Target Variable](#2-target-variable)
3. [Game Metadata Fields](#3-game-metadata-fields)
4. [Team Stats Fields (Dashboard)](#4-team-stats-fields-dashboard)
5. [Player Stats Fields (Roster)](#5-player-stats-fields-roster)
6. [Roster Cache Fields](#6-roster-cache-fields)
7. [Model Metrics](#7-model-metrics)
8. [Registry Entry Fields](#8-registry-entry-fields)
9. [Learning Log Entry Fields](#9-learning-log-entry-fields)
10. [Config Keys (config.yaml)](#10-config-keys-configyaml)
11. [Environment Variables](#11-environment-variables)
12. [API Response Fields](#12-api-response-fields)
13. [CLI Arguments](#13-cli-arguments)
14. [Runtime State Variables](#14-runtime-state-variables)

---

## 1. Game Features (Model Input)

These 14 fields form the feature matrix **X** passed to every model. Order is fixed by `config.yaml → data.features`. All are continuous floating-point values.

| Variable | Type | Source | Description | Typical Range |
|----------|------|--------|-------------|---------------|
| `home_ppg` | float | ESPN score | Points scored by the home team in this game | 40 – 120 |
| `away_ppg` | float | ESPN score | Points scored by the away team in this game | 40 – 120 |
| `home_fg_pct` | float | ESPN box score | Home team field goal percentage (makes / attempts) | 0.28 – 0.65 |
| `away_fg_pct` | float | ESPN box score | Away team field goal percentage (makes / attempts) | 0.28 – 0.65 |
| `home_rebounds` | float | ESPN box score | Total rebounds (offensive + defensive) by home team | 20 – 60 |
| `away_rebounds` | float | ESPN box score | Total rebounds by away team | 20 – 60 |
| `home_assists` | float | ESPN box score | Assists by home team | 5 – 35 |
| `away_assists` | float | ESPN box score | Assists by away team | 5 – 35 |
| `home_turnovers` | float | ESPN box score | Turnovers committed by home team | 4 – 25 |
| `away_turnovers` | float | ESPN box score | Turnovers committed by away team | 4 – 25 |
| `home_steals` | float | ESPN box score | Steals by home team | 0 – 18 |
| `away_steals` | float | ESPN box score | Steals by away team | 0 – 18 |
| `home_blocks` | float | ESPN box score | Blocks by home team | 0 – 16 |
| `away_blocks` | float | ESPN box score | Blocks by away team | 0 – 16 |

**Notes:**
- `home_ppg` and `away_ppg` represent the **actual score of that specific game**, not a season rolling average. ESPN provides the final score; we use it as the points feature.
- `home_fg_pct` and `away_fg_pct` arrive from ESPN as percentage strings (e.g. `"45.5"`). The fetcher normalizes these to decimals (e.g. `0.455`).
- `3p_pct` (three-point percentage) was removed from the feature list because ESPN does not reliably expose it in the box score statistics array — it returned `0.0` for all games. Removing it improved model signal quality.
- All features are passed through `StandardScaler` inside the Pipeline before reaching the classifier. The scaler is fit only on the training set.

---

## 2. Target Variable

| Variable | Type | Values | Description |
|----------|------|--------|-------------|
| `outcome` | int | `1` or `0` | `1` = Home Win, `0` = Away Win |

Derived from final scores in `get_box_score()`:
```python
outcome = 1 if home_score > away_score else 0
```

Games where both scores are 0 are discarded (game not yet played or ESPN parse failure).

**Class distribution (real ESPN data, 2024-25 season):** approximately 73% Home Win, 27% Away Win. This imbalance is why ROC-AUC is used as the primary model selection metric rather than accuracy.

---

## 3. Game Metadata Fields

Stored in `data/games.json` alongside features. Not used as model inputs.

| Field | Type | Description |
|-------|------|-------------|
| `game_id` | string | Unique identifier. Format: `ESPN_<event_id>` for real data, `SYN_<n>` for synthetic |
| `home_team` | string | Full team display name from ESPN (e.g. `"Duke Blue Devils"`) |
| `away_team` | string | Full team display name from ESPN |
| `home_score` | int | Final score of the home team |
| `away_score` | int | Final score of the away team |
| `source` | string | `"espn"`, `"custom"`, or `"synthetic"` — indicates data origin |
| `fetched_at` | string | ISO 8601 timestamp of when the record was fetched |

---

## 4. Team Stats Fields (Dashboard)

Returned by `/teams` and `/team_stats/<name>`. These are season or rolling-window averages computed by `build_team_stats(data, window)` across all games a team appears in.

Both endpoints accept an optional `?window=N` query parameter. When set, only the team's most recent N games (sorted by `fetched_at` descending) are used to compute averages. When omitted, all games are used (full season average).

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Team display name |
| `home_ppg` | float | Average points scored per game (regardless of home/away) |
| `away_ppg` | float | Average points conceded per game (opponent's scoring) |
| `home_fg_pct` | float | Average field goal percentage across all games |
| `away_fg_pct` | float | Average opponent FG% across all games |
| `home_rebounds` | float | Average rebounds per game |
| `away_rebounds` | float | Average opponent rebounds per game |
| `home_assists` | float | Average assists per game |
| `away_assists` | float | Average opponent assists per game |
| `home_turnovers` | float | Average turnovers committed per game |
| `away_turnovers` | float | Average opponent turnovers per game |
| `home_steals` | float | Average steals per game |
| `away_steals` | float | Average opponent steals per game |
| `home_blocks` | float | Average blocks per game |
| `away_blocks` | float | Average opponent blocks per game |
| `games_played` | int | Total games this team appears in across the entire dataset (not windowed) |
| `games_in_window` | int | Number of games actually used to compute the averages (≤ games_played when window is set) |
| `wins` | int | Total wins across all games (not windowed) |

**Why `home_*` / `away_*` naming for a team's own stats:**
The feature names match the model's training columns. When a team fills the "home slot" in a prediction, their season average `home_ppg` goes into the `home_ppg` feature. The naming convention is positional (home slot vs away slot), not directional (home venue vs away venue).

---

## 5. Player Stats Fields (Roster)

Individual player records returned inside `/roster/<team>` and `/roster/progress/<team>`. Also the format expected by `/predict/from_roster` in the `home_players` and `away_players` arrays.

| Field | Type | Source | Description | Typical Range |
|-------|------|--------|-------------|---------------|
| `id` | string | ESPN athlete ID | ESPN's internal player identifier — used for the `/statistics` fallback call | — |
| `name` | string | ESPN roster | Player's full display name (e.g. `"Kyle Filipowski"`) | — |
| `position` | string | ESPN roster | Position abbreviation (e.g. `"C"`, `"G"`, `"F"`) | — |
| `jersey` | string | ESPN roster | Jersey number as a string (e.g. `"30"`) | — |
| `ppg` | float | ESPN stats | Points per game (season average) | 0 – 35 |
| `rpg` | float | ESPN stats | Rebounds per game | 0 – 15 |
| `apg` | float | ESPN stats | Assists per game | 0 – 12 |
| `spg` | float | ESPN stats | Steals per game | 0 – 4 |
| `bpg` | float | ESPN stats | Blocks per game | 0 – 4 |
| `tov` | float | ESPN stats | Turnovers per game | 0 – 6 |
| `fg_pct` | float | ESPN stats | Field goal percentage (decimal, e.g. `0.48`). Defaults to `0.45` if unavailable | 0.20 – 0.75 |
| `fgm` | float | ESPN stats | Field goals made per game — used for FGA-weighted team fg_pct aggregation | 0 – 15 |
| `fga` | float | ESPN stats | Field goals attempted per game — used for FGA-weighted team fg_pct aggregation | 0 – 25 |

**Stat extraction priority** (highest to lowest):
1. `athlete.statistics[]` and `athlete.displayStats[]` embedded directly in the roster response
2. `GET /athletes/{id}/statistics` — called only when embedded stats are all zero
3. `_empty_player_stats()` — all zeros, `fg_pct = 0.45` — used when ESPN returns 404

**Aggregation into team features** (performed by `compute_stats_from_roster(players, side)`):

| Team Feature | Aggregation Method |
|---|---|
| `{side}_ppg` | Sum of all players' `ppg` |
| `{side}_rebounds` | Sum of all players' `rpg` |
| `{side}_assists` | Sum of all players' `apg` |
| `{side}_steals` | Sum of all players' `spg` |
| `{side}_blocks` | Sum of all players' `bpg` |
| `{side}_turnovers` | Sum of all players' `tov` |
| `{side}_fg_pct` | `sum(fgm) / sum(fga)` if `sum(fga) > 0`, else `mean(fg_pct)` |

---

## 6. Roster Cache Fields

Stored in `data/rosters/<team_id>.json`. One file per team. Written by `RosterFetcher._save_cached()`, read by `_load_cached()`.

| Field | Type | Description |
|-------|------|-------------|
| `team_name` | string | ESPN display name used to look up this roster |
| `team_id` | string | ESPN internal team ID |
| `players` | list[dict] | List of player objects — see Section 5 for field definitions |
| `fetched_at` | string | ISO 8601 timestamp. Cache is considered valid for `roster.cache_ttl_hours` (default 24h) after this |

**Team ID cache** (`data/team_ids.json`): A flat `{display_name: espn_id}` dict. Built on first lookup from `GET /teams?limit=1000` and reused on all subsequent calls. Lookup tries exact match first, then case-insensitive substring match.

---

## 7. Model Metrics

Computed by `compute_metrics()` and stored in `models/registry.json` and `models/latest_comparison.json`.

| Metric | Type | Range | Description |
|--------|------|-------|-------------|
| `accuracy` | float | 0.0 – 1.0 | (TP + TN) / total predictions |
| `precision` | float | 0.0 – 1.0 | TP / (TP + FP) — of predicted home wins, how many were correct |
| `recall` | float | 0.0 – 1.0 | TP / (TP + FN) — of actual home wins, how many were found |
| `f1` | float | 0.0 – 1.0 | Harmonic mean of precision and recall |
| `roc_auc` | float | 0.0 – 1.0 | Area under ROC curve — primary selection metric |
| `cv_roc_auc_mean` | float | 0.0 – 1.0 | Mean ROC-AUC across 5 cross-validation folds |
| `cv_roc_auc_std` | float | 0.0 – 1.0 | Standard deviation of CV ROC-AUC scores |
| `confusion_matrix` | list[list[int]] | — | [[TN, FP], [FN, TP]] on the test set |
| `feature_importances` | dict | — | `{feature_name: importance_value}` — tree models only (via `feature_importances_`); linear models use `abs(coef_)` |

**Metric definitions:**
- TP = True Positive: predicted Home Win, actual Home Win
- TN = True Negative: predicted Away Win, actual Away Win
- FP = False Positive: predicted Home Win, actual Away Win
- FN = False Negative: predicted Away Win, actual Home Win

**`roc_auc` vs `accuracy`:** With 73% home wins, a naive classifier that always predicts Home Win achieves 73% accuracy but ROC-AUC of 0.5. ROC-AUC correctly penalizes this. ROC-AUC of 1.0 = perfect model; 0.5 = random.

**`cv_roc_auc_std`:** A high value (e.g. > 0.05) indicates the model is sensitive to which games end up in the test fold — a sign of instability on the current dataset size.

---

## 8. Registry Entry Fields

Each version entry in `models/registry.json`:

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Version identifier, e.g. `"v1"`, `"v2"` |
| `model_name` | string | Display name of the best model, e.g. `"Gradient Boosting"` |
| `filename` | string | `.pkl` filename, e.g. `"gradient_boosting_v1_a3f2c1d4.pkl"` |
| `metrics` | dict | All model metrics (see Section 7) |
| `feature_names` | list[string] | Ordered feature list this model was trained on |
| `training_size` | int | Number of samples in the training set |
| `trained_at` | string | ISO 8601 timestamp |
| `hash` | string | First 8 hex chars of MD5 hash of the serialized model — integrity check |

**Top-level registry fields:**

| Field | Type | Description |
|-------|------|-------------|
| `active_version` | string | Version currently served by `/predict` |
| `versions` | list[dict] | All registered version entries, oldest first |

---

## 9. Learning Log Entry Fields

Each entry in `data/learning_log.json`:

| Field | Type | Always Present | Description |
|-------|------|---------------|-------------|
| `timestamp` | string | Yes | ISO 8601 timestamp of when training completed |
| `triggered_by` | string | Yes | `"manual"`, `"new_data"`, `"scheduler"`, or `"manual_trigger"` |
| `result` | string | Yes | `"promoted"` or `"skipped"` |
| `version` | string | Promoted only | Registry version assigned, e.g. `"v3"` |
| `model_name` | string | Promoted only | Name of the promoted model |
| `roc_auc` | float | Promoted only | ROC-AUC of the promoted model |
| `f1` | float | Promoted only | F1-score of the promoted model |
| `accuracy` | float | Promoted only | Accuracy of the promoted model |
| `dataset_size` | int | Yes | Total games in dataset at time of training |
| `reason` | string | Skipped only | Human-readable explanation of why promotion was skipped |
| `new_auc` | float | Skipped only | ROC-AUC of the candidate model |
| `current_auc` | float | Skipped only | ROC-AUC of the model that was kept |
| `best_model` | string | Skipped only | Name of the candidate model that was not promoted |

---

## 10. Config Keys (config.yaml)

### app

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `app.name` | string | `"Basketball Game Outcome Predictor"` | Display name |
| `app.version` | string | `"2.2.0"` | App version string |
| `app.debug` | bool | `true` | Flask debug mode |
| `app.port` | int | `5000` | Flask server port |
| `app.host` | string | `"0.0.0.0"` | Flask bind address |

### home_team

| Key | Type | Description |
|-----|------|-------------|
| `home_team.name` | string | Home team display name — must match ESPN displayName exactly or close enough for fuzzy match |
| `home_team.espn_id` | string | ESPN team ID (informational, not used in fetching) |
| `home_team.court_name` | string | Arena name (displayed in dashboard) |

### data

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `data.dir` | string | `"data"` | Directory for data files |
| `data.local_file` | string | `"data/games.json"` | Path to local game records |
| `data.test_size` | float | `0.2` | Fraction of data held out for testing |
| `data.random_state` | int | `42` | Seed for train-test split reproducibility |
| `data.min_games_required` | int | `50` | Minimum records needed to attempt training |
| `data.features` | list[string] | See Section 1 | Ordered feature column names |
| `data.label` | string | `"outcome"` | Target column name |

### ncaa_api

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ncaa_api.provider` | string | `"espn"` | `espn` or `custom` |
| `ncaa_api.season` | int | `2024` | Season start year (2024 = 2024-25 season) |
| `ncaa_api.max_games` | int | `500` | Maximum games to fetch per `--fetch` run |
| `ncaa_api.rate_limit_delay` | float | `0.4` | Seconds to sleep between ESPN requests |
| `ncaa_api.espn.base_url` | string | ESPN API base | Base URL for ESPN unofficial API |
| `ncaa_api.espn.scoreboard_path` | string | `"/scoreboard"` | Path for daily game IDs |
| `ncaa_api.espn.summary_path` | string | `"/summary"` | Path for box score details |
| `ncaa_api.espn.teams_path` | string | `"/teams"` | Path for team list and roster lookups |
| `ncaa_api.espn.athletes_path` | string | `"/athletes"` | Base path for per-player statistics fallback |
| `ncaa_api.espn.page_size` | int | `25` | Games per scoreboard request |
| `ncaa_api.custom.base_url` | string | `""` | Base URL for custom API provider |
| `ncaa_api.custom.api_key` | string | `""` | API key (or set `NCAA_API_KEY` env var) |
| `ncaa_api.custom.games_endpoint` | string | `"/games"` | Endpoint path |
| `ncaa_api.custom.season_param` | string | `"season"` | Query param name for season |
| `ncaa_api.custom.field_map` | dict | — | Maps our field names to their field names |

### snowflake

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `snowflake.enabled` | bool | `false` | Enable Snowflake storage |
| `snowflake.user` | string | `""` | Username (or set `SNOWFLAKE_USER` env var) |
| `snowflake.password` | string | `""` | Password (or set `SNOWFLAKE_PASSWORD` env var) |
| `snowflake.account` | string | `""` | Snowflake account identifier |
| `snowflake.warehouse` | string | `""` | Compute warehouse name |
| `snowflake.database` | string | `""` | Database name |
| `snowflake.schema` | string | `""` | Schema name |
| `snowflake.table` | string | `"BASKETBALL_GAMES"` | Table name |

### models

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `models.dir` | string | `"models"` | Directory for model files |
| `models.registry_file` | string | `"models/registry.json"` | Path to version registry |
| `models.selection_metric` | string | `"roc_auc"` | Metric used to pick best model |
| `models.keep_top_n` | int | `10` | Max versions retained before pruning |
| `models.enabled` | list[string] | All 5 + optional XGBoost | Which models to include in each training run |

**Per-model hyperparameters** (all under `models.<model_key>`):

| Model | Key | Default | Description |
|-------|-----|---------|-------------|
| gradient_boosting | `n_estimators` | `200` | Number of boosting rounds |
| gradient_boosting | `learning_rate` | `0.05` | Shrinkage applied to each tree |
| gradient_boosting | `max_depth` | `4` | Maximum depth of each tree |
| gradient_boosting | `subsample` | `0.8` | Fraction of samples per tree (stochastic) |
| gradient_boosting | `min_samples_split` | `5` | Min samples to split an internal node |
| random_forest | `n_estimators` | `200` | Number of trees |
| random_forest | `max_depth` | `12` | Maximum tree depth |
| random_forest | `min_samples_split` | `5` | Min samples to split a node |
| random_forest | `min_samples_leaf` | `2` | Min samples in a leaf node |
| extra_trees | `n_estimators` | `200` | Number of trees |
| extra_trees | `max_depth` | `12` | Maximum tree depth |
| extra_trees | `min_samples_split` | `5` | Min samples to split a node |
| svm | `kernel` | `"rbf"` | Kernel function |
| svm | `C` | `1.0` | Regularization — higher = tighter fit to training data |
| svm | `gamma` | `"scale"` | RBF kernel width — `"scale"` = 1/(n_features × Var(X)) |
| mlp | `hidden_layer_sizes` | `[128, 64, 32]` | Neurons per hidden layer |
| mlp | `activation` | `"relu"` | Activation function |
| mlp | `max_iter` | `500` | Maximum training epochs |
| mlp | `early_stopping` | `true` | Stop if validation loss stops improving |
| mlp | `validation_fraction` | `0.1` | Fraction of training data used for early stopping |
| xgboost | `n_estimators` | `200` | Number of boosting rounds |
| xgboost | `learning_rate` | `0.05` | Shrinkage per round |
| xgboost | `max_depth` | `4` | Maximum tree depth |
| xgboost | `subsample` | `0.8` | Row subsampling per tree |
| xgboost | `colsample_bytree` | `0.8` | Feature subsampling per tree |
| xgboost | `eval_metric` | `"logloss"` | Evaluation metric during training |

All models also take `random_state: 42` for reproducibility.

### auto_learn

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `auto_learn.enabled` | bool | `true` | Start scheduler when `--serve` runs |
| `auto_learn.fetch_interval_hours` | int | `6` | Hours between ESPN fetch attempts |
| `auto_learn.retrain_interval_hours` | int | `24` | Hours between forced full retrains |
| `auto_learn.min_new_games_to_retrain` | int | `15` | New games needed to trigger immediate retrain |
| `auto_learn.promote_threshold` | float | `0.002` | Minimum AUC improvement to promote new model |
| `auto_learn.learning_log_file` | string | `"data/learning_log.json"` | Path to learning log |

### roster

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `roster.cache_dir` | string | `"data/rosters"` | Directory for per-team roster cache files |
| `roster.cache_ttl_hours` | int | `24` | Hours before a cached roster is considered stale |
| `roster.team_id_cache` | string | `"data/team_ids.json"` | Path to the ESPN team name → ID lookup cache |

### rolling

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rolling.available_windows` | list[int] | `[5, 10, 15, 20]` | Window sizes exposed in the `/features` endpoint and the dashboard rolling selector |
| `rolling.default_window` | int | `12` | Default window used in the dashboard when no explicit selection is made |

---

## 11. Environment Variables

| Variable | Used By | Description |
|----------|---------|-------------|
| `SNOWFLAKE_USER` | `_sf_conn()` | Snowflake username — overrides `snowflake.user` in config |
| `SNOWFLAKE_PASSWORD` | `_sf_conn()` | Snowflake password — overrides `snowflake.password` in config |
| `NCAA_API_KEY` | `CustomAPIFetcher` | API key for custom provider — overrides `ncaa_api.custom.api_key` |

Never put credentials directly in `config.yaml`. Set environment variables instead. On Windows PowerShell: `$env:SNOWFLAKE_USER="youruser"`. On macOS/Linux: `export SNOWFLAKE_USER=youruser`.

---

## 12. API Response Fields

### POST /predict

Request body: all 14 feature fields required. Names must match `feature_names` stored in the active model's registry entry.

```json
{ "home_ppg": 84.3, "away_ppg": 72.1, "home_fg_pct": 0.492, ... }
```

Response:

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | `"Home Win"` or `"Away Win"` |
| `prediction_value` | int | `1` (Home Win) or `0` (Away Win) |
| `confidence` | float or null | `max(predict_proba)` — range 0.5 to 1.0. null if model has no `predict_proba` |
| `model_name` | string | Name of the model that made the prediction |
| `version` | string | Registry version of the active model |

### POST /predict/from_roster

Request body:

```json
{
  "home_players": [
    { "ppg": 18.4, "rpg": 5.1, "apg": 3.2, "spg": 1.1,
      "bpg": 0.4, "tov": 2.1, "fg_pct": 0.48, "fgm": 6.2, "fga": 12.9 }
  ],
  "away_players": [ { ... } ]
}
```

Response:

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | `"Home Win"` or `"Away Win"` |
| `prediction_value` | int | `1` or `0` |
| `confidence` | float or null | `max(predict_proba)` |
| `model_name` | string | Name of the active model |
| `version` | string | Registry version |
| `computed_stats` | dict | The aggregated team-level features actually passed to the model (all 14 feature fields) |
| `home_count` | int | Number of home players included in the prediction |
| `away_count` | int | Number of away players included in the prediction |

### GET /analytics

| Field | Type | Description |
|-------|------|-------------|
| `total_games` | int | Total records in games.json |
| `home_wins` | int | Count of games where outcome == 1 |
| `away_wins` | int | Count of games where outcome == 0 |
| `home_win_rate` | float | home_wins / total_games |
| `feature_stats` | dict | `{home_win: {feature: avg}, away_win: {feature: avg}}` |
| `model_comparison` | dict | `{model_name: {accuracy, precision, recall, f1, roc_auc, ...}}` |
| `feature_importances` | dict | `{model_name: {feature: importance}}` |
| `data_sources` | dict | `{source_name: count}` e.g. `{"espn": 500}` |

### GET /features

| Field | Type | Description |
|-------|------|-------------|
| `features` | list[string] | Ordered feature names from `data.features` in config |
| `rolling_windows` | list[int] | Available window sizes from `rolling.available_windows` |
| `default_window` | int | Default window size from `rolling.default_window` |

### GET /teams and GET /team_stats/\<name\>

Both accept optional `?window=N` query parameter. Response fields per team are the same as Section 4. `/teams` wraps results in:

| Field | Type | Description |
|-------|------|-------------|
| `teams` | list[dict] | List of team stat objects (only teams with ≥ 3 games played) |
| `count` | int | Number of teams returned |
| `window` | int or null | Window value used, null if not specified |

### GET /roster/\<team_name\>

Kicks off an async background fetch and returns the current progress state immediately. Pass `?force=1` to bypass the cache.

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"loading"`, `"ready"`, `"error"`, or `"not_started"` |
| `players` | list[dict] | Players fetched so far (see Section 5 for fields) — populated incrementally |
| `done` | int | Number of players whose stats have been resolved |
| `total` | int | Total players on roster (0 until roster endpoint responds) |
| `message` | string | Only present when `status == "error"` — human-readable failure reason |

### GET /roster/progress/\<team_name\>

Same response shape as `/roster/<team_name>`. Returns `{"status": "not_started", ...}` if no fetch has been initiated for this team in the current server session.

### GET /autolearn/status

| Field | Type | Description |
|-------|------|-------------|
| `enabled` | bool | Whether scheduler is enabled in config |
| `status` | string | `"idle"`, `"fetching"`, or `"training"` |
| `fetch_interval_h` | int | Configured fetch interval in hours |
| `retrain_interval_h` | int | Configured retrain interval in hours |
| `min_new_games` | int | Configured minimum new games threshold |
| `promote_threshold` | float | Configured AUC improvement threshold |
| `next_fetch_in` | string | Human-readable countdown e.g. `"5h 42m"` |
| `next_retrain_in` | string | Human-readable countdown |
| `last_fetch` | string or null | ISO 8601 timestamp of last fetch, null if never run |
| `last_retrain` | string or null | ISO 8601 timestamp of last retrain, null if never run |

---

## 13. CLI Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--fetch` | flag | Fetch real NCAA data from ESPN and append to local store |
| `--fetch-rosters` | flag | Pre-fetch and cache rosters for every team currently in the dataset. Blocking — processes all teams sequentially. Useful to warm the cache before serving |
| `--generate-synthetic` | flag | Generate 500 synthetic games (offline fallback) |
| `--train` | flag | Train all enabled models, register best by selection metric |
| `--serve` | flag | Start Flask server and auto-learn scheduler |
| `--list-models` | flag | Print table of all registered versions to console |
| `--activate` | string | Version to set as active, e.g. `--activate v2` |
| `--storage` | choice | `local` (default) or `snowflake` |
| `--config` | string | Path to config file (default: `config.yaml`) |

---

## 14. Runtime State Variables

Module-level variables that live in memory for the duration of the server process. Not persisted to disk.

| Variable | Type | Description |
|----------|------|-------------|
| `_roster_progress` | `dict[str, dict]` | Shared state between Flask request threads and `RosterFetcher` background threads. Keyed by team display name. Each value is a progress dict with `status`, `players`, `done`, `total` (and optionally `message`). Written by roster fetch threads; read by `/roster/progress/<team>` and `/roster/<team>` endpoints. |
| `_scheduler` | `AutoLearnScheduler` | The singleton scheduler instance. Started by `--serve`. Exposes `get_state()` for `/autolearn/status` and `stop()` for clean shutdown. |

---

*All types reflect Python runtime values. Floats in JSON responses are rounded to 4 decimal places. Null values appear as JSON `null` in responses and `None` in Python.*
