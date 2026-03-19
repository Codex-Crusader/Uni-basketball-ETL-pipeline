# Future Development Map

Current system state and genuine remaining enhancement opportunities. Version 2.5.1.

---

## Table of Contents

1. [Current System State](#current-system-state)
2. [What Was Planned vs What Got Built](#what-was-planned-vs-what-got-built)
3. [Genuine Remaining Work](#genuine-remaining-work)
4. [Out of Scope](#out-of-scope)

---

## Current System State

### What is Working

**Data Ingestion:**
- Real NCAA game data via ESPN unofficial API, no key required
- Multi-season fetch: seasons 2022, 2023, 2024 — approximately 2900 games total
- Automatic deduplication by game_id — safe to run --fetch repeatedly
- game_date extracted from ESPN event metadata for correct chronological ordering
- Synthetic data fallback for offline use
- Custom API stub for plugging in any external provider via config

**Pre-Game Enrichment:**
- `enrich_with_pregame_averages()` replaces in-game box score statistics with pre-game rolling averages
- Feature fields in game records now contain what was knowable before tipoff
- Original in-game stats preserved under `home_game_*` / `away_game_*` for analytics display
- `pregame_enriched` flag on every record; training code filters to True-only records
- `--enrich` CLI command back-fills existing games.json without re-fetching
- Cold-start records (teams with no prior history) correctly excluded from training
- `pregame_window` (default 10) and `pregame_min_games` (default 1) configurable in config.yaml

**Storage:**
- Local JSON as default — zero setup
- Snowflake fully provisioned — disabled by default, enabled via config flag and env vars
- Both backends share the same interface; switching is a single CLI flag

**Models:**
- Five models trained and compared every run: Gradient Boosting, Random Forest, Extra Trees, SVM (RBF), Neural Network (MLP), plus XGBoost as optional sixth
- All wrapped in StandardScaler -> estimator Pipeline
- 5-fold cross-validated ROC-AUC as selection metric
- Adaptive tree depth: depth capped at log2(n_samples / (10 x n_features)) to prevent overfitting
- Per-model feature importances extracted where available
- Hyperparameters calibrated for fairness: all models given proportionate capacity for the current dataset size

**Model Registry and Versioning:**
- Every training run registers a versioned .pkl (v1, v2, v3...)
- models/registry.json tracks metrics, feature list, training size, timestamp, and MD5 hash per version
- Activate any version from dashboard or CLI (--activate v2)
- Automatic pruning of oldest versions beyond keep_top_n (default 10)

**Auto-Learning:**
- Background daemon thread starts automatically with --serve
- Fetches new games every 6 hours, retrains if 15 or more new games added
- Forces full retrain every 24 hours regardless
- Promote-only gate: new model must beat current AUC by 0.002 to be promoted
- Every run (promoted or skipped with reason) logged to data/learning_log.json

**Logging:**
- All print() replaced with Python logging module
- RotatingFileHandler: data/app.log, 10 MB per file, 2 backups (30 MB ceiling)
- Console: INFO and above; file: DEBUG and above
- Module-level loggers: bball.app.fetcher, bball.app.models, etc.
- setup_logging() called before all other imports in main.py

**Configuration:**
- Zero hardcoded values in Python — everything in config.yaml
- Snowflake credentials via environment variables
- Feature list, model hyperparameters, thresholds, intervals, season list all configurable

**Rolling Form Window:**
- build_team_stats(data, window=N) uses most recent N games per team
- Available windows and default configurable in config.yaml
- /teams?window=N and /team_stats/<n>?window=N endpoints
- /features endpoint returns available windows so the dashboard builds its selector dynamically

**Roster System:**
- --fetch-rosters pre-warms cache for all teams before serving
- RosterFetcher resolves team names to ESPN IDs, caches to data/team_ids.json
- Per-team rosters cached with configurable TTL (default 24h)
- Embedded stats used first; per-player /statistics called only as fallback; graceful 404 handling
- Async fetch with incremental progress polling at /roster/progress/<team>
- /predict/from_roster aggregates selected players into team features with FGA-weighted FG%
- computed_stats and insights returned in prediction response

**Dashboard (6 tabs):**
- Predict: stats mode with season or rolling window auto-fill; roster mode with live player aggregation
- Overview: stats cards, outcome donut, active model radar, AUC progression over versions
- Model Comparison: metrics table with inline bars, grouped bar chart, multi-model radar
- Feature Analysis: home-win vs away-win averages, per-model importance chart with selector
- Registry: all versions with metrics, one-click Activate, rollback support
- Auto-Learn: live scheduler status, countdowns, full learning log, manual trigger

**Code Structure:**
- Single main.py (2000 lines) refactored into app/ package with 10 modules
- No circular imports — dependency chain verified
- main.py is now CLI entry point only (~200 lines); all logic in app/
- dashboard.html served from project root via absolute path resolution in api.py

---

### Remaining Limitations

**Data:**
- ESPN unofficial API has no SLA — could change structure without notice
- Rolling average cold-start: first game of each team's season excluded from training (no prior history to average). Small fraction of total records.
- Roster embedded stats not always present; /statistics fallback returns 404 for many college players

**Models:**
- No automated hyperparameter tuning (GridSearchCV / Optuna)
- No SHAP values for explainability — importances are raw impurity-based, not Shapley values
- No model stacking or ensembling across the trained models

**Deployment:**
- Single-instance Flask — not production-hardened (no gunicorn, no auth, no HTTPS)
- No executable distribution (PyInstaller / Docker)
- Localhost only

---

## What Was Planned vs What Got Built

| Planned | Status | Notes |
|---------|--------|-------|
| Real NCAA API integration | Done | ESPN unofficial API, no key needed |
| Multi-season data | Done | seasons: [2022, 2023, 2024], ~2900 games |
| Custom API provider slot | Done | CustomAPIFetcher plus config field_map |
| Auto-fetch on train | Done | Auto-learn scheduler handles this |
| Model versioning system | Done | Registry with v1/v2/... plus activate/rollback |
| Config-driven architecture | Done | config.yaml — zero hardcoded values |
| Env vars for secrets | Done | SNOWFLAKE_USER, SNOWFLAKE_PASSWORD |
| Multi-model comparison dashboard | Done | Full comparison tab with table and charts |
| Radar chart | Done | Single-model and multi-model radar |
| Historical AUC progression | Done | Line chart in Overview tab |
| Feature importance visualization | Done | Per-model horizontal bar in Features tab |
| Rolling form window | Done | build_team_stats(window=N), configurable windows |
| Player roster predictions | Done | RosterFetcher, async progress, FGA-weighted aggregation |
| Pre-game features (no leakage) | Done | enrich_with_pregame_averages, --enrich CLI command |
| Structured logging | Done | RotatingFileHandler, module-level loggers |
| Modular codebase | Done | app/ package, 10 modules, no circular imports |
| PyInstaller executable | Not done | Out of scope |
| Docker container | Not done | Out of scope |

---

## Genuine Remaining Work

These are real improvements that would meaningfully extend the system.

---

### 1. Hyperparameter Tuning

**What:** Automated search over model hyperparameters using cross-validation.

**Why it matters:** Current hyperparameters in config.yaml are calibrated defaults, not optimized values. With 2900 real games, there is enough data for a tuning pass to meaningfully improve AUC.

**What is needed:**
- Add `--tune` CLI flag
- `tune_models()` function running `RandomizedSearchCV` or Optuna per enabled model
- Write winning hyperparameters back to config.yaml automatically
- Integrate into auto-learn: tune every N retrains

---

### 2. SHAP Explainability

**What:** Replace raw feature importances with Shapley values.

**Why it matters:** Current importances (impurity reduction for trees, abs(coef_) for SVM) do not account for feature interactions and can be misleading. SHAP gives per-prediction explanations: "this prediction was +12% because home FG% was high."

**What is needed:**
- `pip install shap`
- `compute_shap(model, X_test, feature_names)` using shap.TreeExplainer for tree models, shap.KernelExplainer for SVM/MLP
- New dashboard chart: SHAP beeswarm or bar in Features tab
- Per-prediction SHAP waterfall shown alongside prediction result

---

### 3. Executable Distribution

**What:** Package as a single .exe / binary so the system runs without Python installed.

**Why it matters:** Reviewers should not need to set up a Python environment to run a demo.

**What is needed:**
```bash
pip install pyinstaller
pyinstaller --onefile --name basketball-predictor \
  --add-data "dashboard.html:." \
  --add-data "config.yaml:." \
  main.py
```

**Known challenges:**
- Binary size approximately 80 MB with scikit-learn and numpy
- snowflake-connector-python has known PyInstaller issues — may need to be excluded from the bundled build
- XGBoost requires separate handling

---

### 4. Docker Container

**What:** Containerised deployment so the system runs identically anywhere.

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py dashboard.html config.yaml ./
COPY app/ ./app/
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

**What is needed:**
- Switch from app.run() to gunicorn with multiple workers
- Add basic API key authentication on /predict
- HTTPS via self-signed cert or reverse proxy (nginx)
- Rate limiting on prediction endpoint

Explicitly not needed for academic demonstration but documents the gap between current state and production deployment.

---

## Out of Scope

These remain explicitly out of scope:

- Deep learning models — dataset size does not justify it
- Mobile app — separate project
- User authentication — not needed for demo
- GraphQL API — REST is sufficient
- WebSockets — polling every 15 seconds is adequate for scheduler status; roster progress polling every 1 second is adequate for fetch feedback
- Microservices — overkill for this architecture
- Database migrations — Snowflake schema auto-created from config
- Per-game opponent strength adjustment — meaningful improvement but would require a separate team rating system (Elo, KenPom) and external data source

---

## Key Principles (Unchanged)

1. Config over code — if it might change, it belongs in config.yaml
2. Improve only — new models replace current only if they are genuinely better
3. Fail gracefully — missing XGBoost, disabled Snowflake, blank ESPN response, 404 roster endpoints all handled cleanly
4. Pre-game only — features must represent information knowable before tipoff
5. Document the why — code comments explain decisions, not just what the code does
6. Log everything — decisions, warnings, promotions, skips all go to data/app.log

---

The goal was never perfection. It was a system that works, explains itself, trains on honest data, and keeps getting better.
