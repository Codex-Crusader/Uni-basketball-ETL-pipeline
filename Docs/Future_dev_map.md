# 🚀 Future Development Map

> *Current system state and genuine remaining enhancement opportunities*

---

## Table of Contents

1. [Current System State](#current-system-state)
2. [What Was Planned vs What Got Built](#what-was-planned-vs-what-got-built)
3. [Genuine Remaining Work](#genuine-remaining-work)
4. [Out of Scope](#out-of-scope)

---

## 📊 Current System State

### ✅ What's Working

**Data Ingestion:**
- Real NCAA game data via ESPN unofficial API (no key required)
- 500+ games fetched per run across the full 2024-25 season
- Automatic deduplication by `game_id` — safe to run `--fetch` repeatedly
- Synthetic data fallback (`--generate-synthetic`) for offline use
- Custom API stub (`CustomAPIFetcher`) for plugging in any external provider via config

**Storage:**
- Local JSON (`data/games.json`) as default — zero setup
- Snowflake fully provisioned — disabled by default, enabled via config flag + env vars
- Both backends share the same interface; switching is a single CLI flag

**Models:**
- Five models trained and compared every run: Gradient Boosting, Random Forest, Extra Trees, SVM (RBF), Neural Network (MLP) — plus XGBoost as an optional sixth if installed
- All wrapped in `StandardScaler → estimator` Pipeline — no data leakage
- 5-fold cross-validated ROC-AUC as selection metric
- Per-model feature importances extracted where available

**Model Registry & Versioning:**
- Every training run registers a versioned `.pkl` (v1, v2, v3…)
- `models/registry.json` tracks metrics, feature list, training size, timestamp, and MD5 hash per version
- Activate any version from dashboard or CLI (`--activate v2`)
- Automatic pruning of oldest versions beyond `keep_top_n` (default 10)

**Auto-Learning:**
- Background daemon thread starts automatically with `--serve`
- Fetches new games every 6 hours, retrains if ≥15 new games added
- Forces full retrain every 24 hours regardless
- Promote-only gate: new model must beat current AUC by 0.002 to be promoted
- Every run (promoted or skipped + reason) logged to `data/learning_log.json`

**Configuration:**
- Zero hardcoded values in Python — everything in `config.yaml`
- Snowflake credentials via environment variables (`SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`)
- Feature list, model hyperparameters, thresholds, intervals, team name all configurable

**Rolling Form Window:**
- `build_team_stats(data, window=N)` filters each team to their most recent N games by `fetched_at`
- Available windows and default configurable in `config.yaml` under `rolling`
- `/teams?window=N` and `/team_stats/<name>?window=N` endpoints expose this to the dashboard
- `/features` endpoint returns available windows and the default so the dashboard can build its selector dynamically
- `games_in_window` field returned alongside `games_played` so the dashboard can show how many games were actually used

**Roster System:**
- `--fetch-rosters` CLI flag pre-warms the cache for all teams in the dataset before serving
- `RosterFetcher` resolves team display names to ESPN IDs, caches the mapping to `data/team_ids.json`
- Per-team rosters cached in `data/rosters/<team_id>.json` with a configurable TTL (default 24h)
- Embedded stats in the roster response used first; per-player `/statistics` endpoint called only for players with no embedded stats; graceful fallback to empty stats on 404
- Roster fetches run in a background thread; dashboard polls `/roster/progress/<team>` every second and renders players incrementally as they arrive
- `/predict/from_roster` endpoint accepts selected player lists, aggregates them into team features (FGA-weighted FG%, summed counting stats), and runs through the same model pipeline as stats mode
- `computed_stats` returned in the response so the dashboard can show exactly what numbers the model used

**Dashboard (6 tabs):**
- **Predict:** Home team fixed (Duke), opponent picker with season-average or rolling-window auto-fill, confidence bar; roster mode lets you select individual players and aggregates their stats live before predicting
- **Overview:** Stats cards, outcome donut, active model radar, AUC progression over versions
- **Model Comparison:** Metrics table with inline bars, grouped bar chart, multi-model radar
- **Feature Analysis:** Home-win vs away-win averages, per-model importance chart with selector
- **Registry:** All versions with metrics, one-click Activate, rollback support
- **Auto-Learn:** Live scheduler status, next-run countdowns, full learning log, manual trigger

**Engineering:**
- Script-based (no notebooks)
- `_sanitize()` handles NaN/Inf from XGBoost CV before JSON serialization
- Lazy chart rendering — drawn on tab switch, not page load, to avoid 0×0 canvas bug
- `/debug` health check endpoint for diagnosing blank dashboard issues
- Module-level `_roster_progress` dict shared between Flask and background roster threads for live progress reporting

---

### ⚠️ Remaining Limitations

**Data:**
- `home_ppg` / `away_ppg` = actual game score, not season rolling average (ESPN gives us the score, not a pre-computed PPG stat)
- ESPN unofficial API has no SLA — could change structure without notice
- Roster embedded stats are not always present; the per-player `/statistics` fallback endpoint returns 404 for many players

**Models:**
- No automated hyperparameter tuning (GridSearchCV / Optuna)
- No SHAP values for explainability — importances are raw impurity-based, not Shapley
- No model stacking or ensembling across the trained models

**Deployment:**
- Single-instance Flask — not production-hardened (no gunicorn, no auth, no HTTPS)
- No executable distribution (PyInstaller / Docker)
- Localhost only

---

## 🎯 What Was Planned vs What Got Built

| Planned (Original Roadmap) | Status | Notes |
|----------------------------|--------|-------|
| Real NCAA API integration | ✅ Done | ESPN unofficial API, no key needed |
| Custom API provider slot | ✅ Done | `CustomAPIFetcher` + `config.yaml` field_map |
| Auto-fetch on train | ✅ Done | Auto-learn scheduler handles this automatically |
| Model versioning system | ✅ Done | Registry with v1/v2/… + activate/rollback |
| Config-driven architecture | ✅ Done | `config.yaml` — zero hardcoded values |
| Env vars for secrets | ✅ Done | `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD` |
| Multi-model comparison dashboard | ✅ Done | Full comparison tab with table + charts |
| Radar chart | ✅ Done | Single-model and multi-model radar |
| Historical AUC progression | ✅ Done | Line chart in Overview tab |
| Feature importance visualization | ✅ Done | Per-model horizontal bar in Features tab |
| Last N matches / recent form | ✅ Done | `build_team_stats(window=N)`, `/teams?window=N`, `/team_stats/<name>?window=N`, configurable windows in `config.yaml` |
| Player roster predictions | ✅ Done | `RosterFetcher`, async fetch + progress polling, `/predict/from_roster`, FGA-weighted aggregation |
| PyInstaller executable | ❌ Not done | Out of scope — see below |
| Docker container | ❌ Not done | Out of scope — see below |

---

## 🔮 Genuine Remaining Work

These are real improvements that would meaningfully extend the system. Unlike the original roadmap items, all of these require new work.

### 1. Hyperparameter Tuning

**What:** Automated search over model hyperparameters using cross-validation.

**Why it matters:** Current hyperparameters in `config.yaml` are sensible defaults, not optimized values. With 500 real games, there's enough data for a tuning pass to meaningfully improve AUC.

**What's needed:**
- Add `--tune` CLI flag
- `tune_models()` function running `RandomizedSearchCV` or `Optuna` per enabled model
- Write winning hyperparams back to `config.yaml` automatically
- Integrate into auto-learn: tune every N retrains

---

### 2. SHAP Explainability

**What:** Replace raw feature importances with Shapley values.

**Why it matters:** Current importances (impurity reduction for trees, `abs(coef_)` for SVM) don't account for feature interactions and can be misleading. SHAP gives per-prediction explanations — "this prediction was +12% because home FG% was high."

**What's needed:**
- `pip install shap`
- `compute_shap(model, X_test, feature_names)` using `shap.TreeExplainer` for tree models, `shap.KernelExplainer` for SVM/MLP
- New dashboard chart: SHAP beeswarm or bar plot in Features tab
- Per-prediction SHAP waterfall shown alongside prediction result

---

### 3. Executable Distribution

**What:** Package as a single `.exe` / binary so the system runs without Python installed.

**Why it matters:** Professors and reviewers should not need to set up a Python environment to run a demo.

**What's needed:**
```bash
pip install pyinstaller
pyinstaller --onefile --name basketball-predictor \
  --add-data "dashboard.html:." \
  --add-data "config.yaml:." \
  main.py
```

**Known challenges:**
- Binary size ~80MB with scikit-learn + numpy
- `snowflake-connector-python` has known PyInstaller issues — may need to be excluded from the bundled build
- XGBoost requires separate handling

---

### 4. Docker Container

**What:** Containerized deployment so the system runs identically anywhere.

**Why it matters:** Removes all environment setup for anyone running the project.

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py dashboard.html config.yaml ./
EXPOSE 5000
CMD ["python", "main.py", "--serve"]
```

```bash
docker build -t basketball-predictor .
docker run -p 5000:5000 \
  -e SNOWFLAKE_USER=... \
  -e SNOWFLAKE_PASSWORD=... \
  basketball-predictor
```

---

### 5. Production Hardening

**What:** Make the Flask server suitable for anything beyond localhost demos.

**What's needed:**
- Switch from `app.run()` to `gunicorn` with multiple workers
- Add basic API key authentication on `/predict`
- HTTPS via self-signed cert or reverse proxy (nginx)
- Rate limiting on prediction endpoint

This is explicitly not needed for academic demonstration but documents the gap between current state and production deployment.

---

## 🚫 Out of Scope

These remain explicitly out of scope for the same reasons as the original roadmap:

- ❌ Deep learning models — dataset size does not justify it
- ❌ Mobile app — separate project
- ❌ User authentication — not needed for demo
- ❌ GraphQL API — REST is sufficient
- ❌ WebSockets — polling every 15s is adequate for scheduler status; roster progress polling every 1s is adequate for fetch feedback
- ❌ Microservices — overkill for single-file architecture
- ❌ Database migrations — Snowflake schema auto-created from config

---

## 💡 Key Principles (Unchanged)

1. **Config over code** — if it might change, it belongs in `config.yaml`
2. **Improve only** — new models replace current only if they're actually better
3. **Fail gracefully** — missing XGBoost, disabled Snowflake, blank ESPN response, 404 roster endpoints all handled cleanly
4. **Document the why** — code comments explain decisions, not just what the code does

---

*The goal was never perfection — it was a system that works, explains itself, and keeps getting better.*
