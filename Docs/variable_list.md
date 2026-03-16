# 📋 Variable List

> *Complete reference for every feature, config key, API field, and runtime variable in the system*

---

## Table of Contents

1. [Game Features (Model Input)](#1-game-features-model-input)
2. [Target Variable](#2-target-variable)
3. [Game Metadata Fields](#3-game-metadata-fields)
4. [Team Stats Fields (Dashboard)](#4-team-stats-fields-dashboard)
5. [Model Metrics](#5-model-metrics)
6. [Registry Entry Fields](#6-registry-entry-fields)
7. [Learning Log Entry Fields](#7-learning-log-entry-fields)
8. [Config Keys (config.yaml)](#8-config-keys-configyaml)
9. [Environment Variables](#9-environment-variables)
10. [API Response Fields](#10-api-response-fields)
11. [CLI Arguments](#11-cli-arguments)

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
| `source` | string | `"espn"` or `"synthetic"` — indicates data origin |
| `fetched_at` | string | ISO 8601 timestamp of when the record was fetched |

---

## 4. Team Stats Fields (Dashboard)

Returned by `/teams` and `/team_stats/<name>`. These are season averages computed by `build_team_stats()` across all games a team appears in.

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
| `games_played` | int | Total games this team appears in across the dataset |
| `wins` | int | Total wins across all games |

**Why `home_*` / `away_*` naming for a team's own stats:**
The feature names match the model's training columns. When a team fills the "home slot" in a prediction, their season average `home_ppg` goes into the `home_ppg` feature. The naming convention is positional (home slot vs away slot), not directional (home venue vs away venue).

---

## 5. Model Metrics

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
| `feature_importances` | dict | — | {feature_name: importance_value} — only for tree models |

**Metric definitions:**
- TP = True Positive: predicted Home Win, actual Home Win
- TN = True Negative: predicted Away Win, actual Away Win
- FP = False Positive: predicted Home Win, actual Away Win
- FN = False Negative: predicted Away Win, actual Home Win

**`roc_auc` vs `accuracy`:** With 73% home wins, a naive classifier that always predicts Home Win achieves 73% accuracy but ROC-AUC of 0.5. ROC-AUC correctly penalizes this. ROC-AUC of 1.0 = perfect model; 0.5 = random.

**`cv_roc_auc_std`:** A high value (e.g. > 0.05) indicates the model is sensitive to which games end up in the test fold — a sign of instability on the current dataset size.

---

## 6. Registry Entry Fields

Each version entry in `models/registry.json`:

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Version identifier, e.g. `"v1"`, `"v2"` |
| `model_name` | string | Display name of the best model, e.g. `"Gradient Boosting"` |
| `filename` | string | `.pkl` filename, e.g. `"gradient_boosting_v1_a3f2c1d4.pkl"` |
| `metrics` | dict | All model metrics (see Section 5) |
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

## 7. Learning Log Entry Fields

Each entry in `data/learning_log.json`:

| Field | Type | Always Present | Description |
|-------|------|---------------|-------------|
| `timestamp` | string | Yes | ISO 8601 timestamp of when training completed |
| `triggered_by` | string | Yes | `"manual"`, `"new_data"`, `"scheduler"`, `"manual_trigger"` |
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

## 8. Config Keys (config.yaml)

### app

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `app.name` | string | `"Basketball Game Outcome Predictor"` | Display name |
| `app.version` | string | `"2.1.0"` | App version string |
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
| `models.enabled` | list[string] | All 6 | Which models to include in each training run |

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

---

## 9. Environment Variables

| Variable | Used By | Description |
|----------|---------|-------------|
| `SNOWFLAKE_USER` | `_sf_conn()` | Snowflake username — overrides `snowflake.user` in config |
| `SNOWFLAKE_PASSWORD` | `_sf_conn()` | Snowflake password — overrides `snowflake.password` in config |
| `NCAA_API_KEY` | `CustomAPIFetcher` | API key for custom provider — overrides `ncaa_api.custom.api_key` |

Never put credentials directly in `config.yaml`. Set environment variables instead. On Windows PowerShell: `$env:SNOWFLAKE_USER="youruser"`. On macOS/Linux: `export SNOWFLAKE_USER=youruser`.

---

## 10. API Response Fields

### POST /predict

Request body:
```
{ "home_ppg": float, "away_ppg": float, "home_fg_pct": float, ... }
```
All 14 feature fields required. Names must match `feature_names` stored in the active model's registry entry.

Response:
| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | `"Home Win"` or `"Away Win"` |
| `prediction_value` | int | `1` (Home Win) or `0` (Away Win) |
| `confidence` | float or null | max(predict_proba) — range 0.5 to 1.0. null if model has no predict_proba |
| `model_name` | string | Name of the model that made the prediction |
| `version` | string | Registry version of the active model |

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

## 11. CLI Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--fetch` | flag | Fetch real NCAA data from ESPN and append to local store |
| `--generate-synthetic` | flag | Generate 500 synthetic games (offline fallback) |
| `--train` | flag | Train all enabled models, register best by selection metric |
| `--serve` | flag | Start Flask server and auto-learn scheduler |
| `--list-models` | flag | Print table of all registered versions to console |
| `--activate` | string | Version to set as active, e.g. `--activate v2` |
| `--storage` | choice | `local` (default) or `snowflake` |
| `--config` | string | Path to config file (default: `config.yaml`) |

---

*All types reflect Python runtime values. floats in JSON responses are rounded to 4 decimal places. Null values appear as JSON null in responses and None in Python.*
