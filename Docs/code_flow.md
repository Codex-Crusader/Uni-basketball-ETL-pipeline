# Code Flow Documentation

![Version](https://img.shields.io/badge/Version-2.5.1-7F77DD?style=flat-square)
![Architecture](https://img.shields.io/badge/Architecture-Modular%20ETL%20Pipeline-1D9E75?style=flat-square)
![No Leakage](https://img.shields.io/badge/Leakage-Eliminated%20in%20v2.5-E24B4A?style=flat-square)
![Models](https://img.shields.io/badge/Models-6%20in%20Competition-BA7517?style=flat-square)

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

The basketball predictor is a config-driven, self-improving ML pipeline with a modular package structure, a background auto-learn scheduler, and a pre-game data enrichment step that eliminates the leakage present in earlier versions.

### Module Dependency Chain

No circular imports. Each module only imports from modules above it.

```mermaid
flowchart TD
    A([config.py]):::cfg --> B([logger.py]):::log
    B --> C([storage.py]):::store
    C --> D([enrichment.py]):::enrich
    D --> E([fetcher.py]):::fetch
    D --> F([roster.py]):::fetch
    D --> G([preprocessing.py]):::prep
    G --> H([models.py]):::model
    H --> I([scheduler.py]):::sched
    I --> J([api.py]):::api
    J --> K([main.py]):::main

    classDef cfg    fill:#eeedfe,stroke:#7F77DD,color:#3c3489
    classDef log    fill:#f1efe8,stroke:#888780,color:#2c2c2a
    classDef store  fill:#e1f5ee,stroke:#1D9E75,color:#04342c
    classDef enrich fill:#fbeaf0,stroke:#D4537E,color:#4b1528
    classDef fetch  fill:#e6f1fb,stroke:#378ADD,color:#042c53
    classDef prep   fill:#faeeda,stroke:#BA7517,color:#412402
    classDef model  fill:#faece7,stroke:#E8593C,color:#4a1b0c
    classDef sched  fill:#eaf3de,stroke:#639922,color:#173404
    classDef api    fill:#eeedfe,stroke:#534AB7,color:#26215c
    classDef main   fill:#f1efe8,stroke:#5F5E5A,color:#2c2c2a
```

### Design Principles

| Principle | Implementation |
|-----------|---------------|
| ![Config](https://img.shields.io/badge/Config--Driven-YAML-7F77DD?style=flat-square) | All settings in `config.yaml`. No hardcoded values in Python. |
| ![Modular](https://img.shields.io/badge/Modular-No%20Circular%20Imports-1D9E75?style=flat-square) | Each module has exactly one responsibility. |
| ![Pregame](https://img.shields.io/badge/Pre--game%20Only-Rolling%20Averages-D4537E?style=flat-square) | Feature vector contains only data knowable before tipoff. |
| ![Promote](https://img.shields.io/badge/Promote--Only-AUC%20Must%20Improve-E8593C?style=flat-square) | New model replaces active only if it beats current AUC + 0.002. |
| ![Logging](https://img.shields.io/badge/Structured-Rotating%20Logs-639922?style=flat-square) | Every module logs to `bball.<module>`. Console + file simultaneously. |

---

## Architecture Components

### Layer 0: Config Loading

![Layer](https://img.shields.io/badge/Layer-0%20%7C%20config.py-7F77DD?style=flat-square)

Runs at module import time. All other modules import constants from here. Nobody reads `config.yaml` directly.

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

### Layer 1: Logging

![Layer](https://img.shields.io/badge/Layer-1%20%7C%20logger.py-888780?style=flat-square)

Must be initialised before any other module is imported. `main.py` calls `setup_logging()` as its very first action.

```python
# main.py
from app.logger import setup_logging
setup_logging()
# now import everything else
```

Child loggers are obtained by name in each module:

```python
log = get_logger(__name__)
# produces: bball.app.fetcher, bball.app.models, etc.
```

The `_configured` guard makes `setup_logging()` idempotent. Safe to call multiple times.

---

### Layer 2: Data Ingestion

![Layer](https://img.shields.io/badge/Layer-2%20%7C%20fetcher.py-378ADD?style=flat-square)

Real NCAA data. No API key required. v2.5 adds multi-season support and `game_date` extraction.

```mermaid
flowchart LR
    A[fetch_ncaa_data]:::entry --> B{For each season\n2022, 2023, 2024}:::decision
    B --> C[get_game_ids\nday by day]:::step
    C --> D[GET /scoreboard\n?dates=YYYYMMDD]:::api
    D --> E[get_box_score\nper event ID]:::step
    E --> F[GET /summary\n?event=ID]:::api
    F --> G[Parse boxscore\nstats + score]:::step
    G --> H[pregame_enriched\n= False]:::flag
    H --> I[enrich_with_pregame\n_averages]:::enrich
    I --> J[append_to_json\ndeduplicated]:::store

    classDef entry    fill:#e6f1fb,stroke:#378ADD,color:#042c53
    classDef decision fill:#eeedfe,stroke:#7F77DD,color:#3c3489
    classDef step     fill:#e1f5ee,stroke:#1D9E75,color:#04342c
    classDef api      fill:#faeeda,stroke:#BA7517,color:#412402
    classDef flag     fill:#fbeaf0,stroke:#D4537E,color:#4b1528
    classDef enrich   fill:#fcebeb,stroke:#E24B4A,color:#501313
    classDef store    fill:#eaf3de,stroke:#639922,color:#173404
```

The returned record stores in-game statistics at this stage. The enrichment step replaces the feature fields with rolling averages before the record is used for training.

---

### Layer 3: Pre-Game Enrichment

![Layer](https://img.shields.io/badge/Layer-3%20%7C%20enrichment.py-D4537E?style=flat-square)
![Core Fix](https://img.shields.io/badge/This%20is%20the%20core%20fix-v2.5-E24B4A?style=flat-square)

This is the most important step in the pipeline. It converts raw in-game box scores into genuinely predictive pre-game features.

```mermaid
flowchart TD
    A([enrich_with_pregame_averages]):::entry --> B[Separate already-enriched\npass through unchanged]:::pass
    A --> C[Sort remaining games\nchronologically by game_date]:::sort

    C --> D{Either team has\nfewer than min_games\nof history?}:::decision

    D -- Yes --> E[Record game in\nteam_history\nSkip from output]:::skip
    D -- No --> F[Save original stats\nunder home_game_*\naway_game_*]:::save

    F --> G[Replace feature fields\nwith rolling averages\nfrom prior games]:::replace
    G --> H[Recompute ast_to_tov\nfrom rolling avg ast\ndivided by rolling avg tov]:::derive
    H --> I[pregame_enriched = True\npregame_window_used = N]:::flag
    I --> J[Add game to output]:::out

    E --> K[Add game to\nteam_history anyway]:::hist
    J --> K

    B --> L([Return enriched records]):::result
    K --> L

    classDef entry    fill:#fbeaf0,stroke:#D4537E,color:#4b1528
    classDef pass     fill:#e1f5ee,stroke:#1D9E75,color:#04342c
    classDef sort     fill:#e6f1fb,stroke:#378ADD,color:#042c53
    classDef decision fill:#eeedfe,stroke:#7F77DD,color:#3c3489
    classDef skip     fill:#fcebeb,stroke:#E24B4A,color:#501313
    classDef save     fill:#faeeda,stroke:#BA7517,color:#412402
    classDef replace  fill:#fbeaf0,stroke:#D4537E,color:#4b1528
    classDef derive   fill:#faeeda,stroke:#BA7517,color:#412402
    classDef flag     fill:#e1f5ee,stroke:#1D9E75,color:#04342c
    classDef out      fill:#eaf3de,stroke:#639922,color:#173404
    classDef hist     fill:#e6f1fb,stroke:#378ADD,color:#042c53
    classDef result   fill:#eaf3de,stroke:#3B6D11,color:#173404
```

> **Before (v2.4):** `home_fg_pct = 0.52` meant what the team shot during the game. The model learned "teams that shot well won" which is circular. AUC was 0.9666 and every bit of it was leakage.
>
> **After (v2.5):** `home_fg_pct = 0.47` means what the team averaged over their last 10 games. That is knowable before tipoff.

`ast_to_tov` is computed as `rolling_avg_assists / rolling_avg_turnovers`, not as an average of per-game ratios. The ratio of averages is more accurate when denominators vary across games.

---

### Layer 4: Storage

![Layer](https://img.shields.io/badge/Layer-4%20%7C%20storage.py-1D9E75?style=flat-square)

Common interface for both backends. The rest of the codebase calls `load_data(storage)` and never knows which backend is active.

```mermaid
flowchart LR
    A([load_data\nstorage=local]):::entry --> B{Which\nbackend?}:::decision
    B -- local --> C[load_from_json\ngames.json]:::local
    B -- snowflake --> D[load_from_snowflake\nenv var creds]:::snow

    E([append_to_json\nnew_data]):::entry2 --> F[load existing]:::local
    F --> G[Build existing_ids set\nby game_id]:::dedup
    G --> H[Filter new_data\nto new_unique only]:::dedup
    H --> I[save_to_json\nexisting + new_unique]:::local

    classDef entry    fill:#e1f5ee,stroke:#1D9E75,color:#04342c
    classDef entry2   fill:#e1f5ee,stroke:#1D9E75,color:#04342c
    classDef decision fill:#eeedfe,stroke:#7F77DD,color:#3c3489
    classDef local    fill:#eaf3de,stroke:#639922,color:#173404
    classDef snow     fill:#e6f1fb,stroke:#378ADD,color:#042c53
    classDef dedup    fill:#faeeda,stroke:#BA7517,color:#412402
```

`_sanitize(obj)` recursively replaces `float('nan')` and `float('inf')` with `None`. Required because XGBoost cross-validation occasionally produces NaN scores, and `json.dumps` writes NaN literally, which is invalid JSON that crashes the browser.

---

### Layer 5: Model Training

![Layer](https://img.shields.io/badge/Layer-5%20%7C%20models.py-E8593C?style=flat-square)

```mermaid
flowchart TD
    A([train_and_evaluate]):::entry --> B[load_data]:::load
    B --> C[prepare_data\nfilter pregame_enriched=True]:::prep
    C --> D[_validate_training_data]:::validate
    D --> E[train_test_split\nstratify=y, 80/20]:::split

    E --> F[build_models\nn_samples, n_features]:::build

    F --> G1[Gradient Boosting]:::model
    F --> G2[Random Forest]:::model
    F --> G3[Extra Trees]:::model
    F --> G4[SVM RBF]:::model
    F --> G5[MLP Neural Net]:::model
    F --> G6[XGBoost\nif installed]:::model

    G1 & G2 & G3 & G4 & G5 & G6 --> H[compute_metrics\nper model]:::metrics
    H --> I[5-fold cross_val_score]:::cv
    I --> J{Best by\nROC-AUC}:::decision

    J --> K[AUC sanity check\n0.80+ warns leakage\n0.52- warns weak signal]:::check

    K --> L{triggered_by\n= manual?}:::decision
    L -- Yes --> M[register_model\nalways promote]:::register
    L -- No --> N{new_auc >=\ncurrent + 0.002?}:::decision
    N -- Yes --> M
    N -- No --> O[log skipped\nreturn None]:::skip
    M --> P[Save comparison JSON\nappend learning log]:::save

    classDef entry    fill:#faece7,stroke:#E8593C,color:#4a1b0c
    classDef load     fill:#e1f5ee,stroke:#1D9E75,color:#04342c
    classDef prep     fill:#faeeda,stroke:#BA7517,color:#412402
    classDef validate fill:#eeedfe,stroke:#7F77DD,color:#3c3489
    classDef split    fill:#e6f1fb,stroke:#378ADD,color:#042c53
    classDef build    fill:#fbeaf0,stroke:#D4537E,color:#4b1528
    classDef model    fill:#eaf3de,stroke:#639922,color:#173404
    classDef metrics  fill:#faeeda,stroke:#BA7517,color:#412402
    classDef cv       fill:#e6f1fb,stroke:#378ADD,color:#042c53
    classDef decision fill:#eeedfe,stroke:#7F77DD,color:#3c3489
    classDef check    fill:#faeeda,stroke:#BA7517,color:#412402
    classDef register fill:#e1f5ee,stroke:#1D9E75,color:#04342c
    classDef skip     fill:#fcebeb,stroke:#E24B4A,color:#501313
    classDef save     fill:#eaf3de,stroke:#3B6D11,color:#173404
```

**Validation checks run before any model sees data:**

| Check | Threshold | What it catches |
|-------|-----------|----------------|
| Leakage detection | Correlation > 0.70 with outcome | Score-derived features sneaking into the feature vector |
| Zero variance | std < 0.001 | Constant features with no predictive value |
| Class balance | Home win rate outside 40-70% | Heavy imbalance that biases all predictions |
| Sample ratio | n / p < 20 | Too few samples per feature, guaranteed overfit |

**Why ROC-AUC and not accuracy:** with ~69% home wins, a model that always predicts "Home Win" hits 69% accuracy but AUC = 0.5. ROC-AUC measures whether probability estimates correctly rank home wins above away wins. It penalises models that win through class imbalance, and it is the right metric here.

**Why adaptive depth:** at n=2300 and p=14, `log2(2300 / (10 * 14)) = 4.04`. Configuring tree depth at 10 gets overridden to 4 at runtime. As more data arrives, the ceiling rises automatically.

---

## Command Execution Flows

### `python main.py --fetch`

```mermaid
flowchart LR
    A([--fetch]):::cmd --> B[fetch_ncaa_data]:::fn
    B --> C[Per season:\nget_game_ids]:::step
    C --> D[Per ID:\nget_box_score]:::step
    D --> E[enrich_with\npregame_averages]:::enrich
    E --> F[append_to_json\ndeduplicate by game_id]:::store

    classDef cmd    fill:#e6f1fb,stroke:#378ADD,color:#042c53
    classDef fn     fill:#eeedfe,stroke:#7F77DD,color:#3c3489
    classDef step   fill:#e1f5ee,stroke:#1D9E75,color:#04342c
    classDef enrich fill:#fbeaf0,stroke:#D4537E,color:#4b1528
    classDef store  fill:#eaf3de,stroke:#639922,color:#173404
```

### `python main.py --enrich`

Applies enrichment to an existing `games.json` fetched before v2.5. Avoids a full re-fetch.

```mermaid
flowchart LR
    A([--enrich]):::cmd --> B[load_from_json]:::load
    B --> C[enrich_with_pregame_averages\nalready-enriched pass through unchanged\nnew: sort by ESPN ID as chronological proxy]:::enrich
    C --> D[save_to_json\noverwrite]:::store
    D --> E([run --train next]):::next

    classDef cmd    fill:#e6f1fb,stroke:#378ADD,color:#042c53
    classDef load   fill:#e1f5ee,stroke:#1D9E75,color:#04342c
    classDef enrich fill:#fbeaf0,stroke:#D4537E,color:#4b1528
    classDef store  fill:#eaf3de,stroke:#639922,color:#173404
    classDef next   fill:#eeedfe,stroke:#7F77DD,color:#3c3489
```

### `python main.py --train`

```mermaid
flowchart LR
    A([--train]):::cmd --> B[train_and_evaluate\ntriggered_by=manual]:::fn
    B --> C[load and prepare\n~2800 enriched games\nX: 2800x14]:::prep
    C --> D[validate\n+ split 80/20]:::validate
    D --> E[build and fit\n5-6 models]:::train
    E --> F[Best by ROC-AUC\nregister_model]:::register
    F --> G[latest_comparison.json\nlearning_log.json]:::save

    classDef cmd      fill:#e6f1fb,stroke:#378ADD,color:#042c53
    classDef fn       fill:#eeedfe,stroke:#7F77DD,color:#3c3489
    classDef prep     fill:#faeeda,stroke:#BA7517,color:#412402
    classDef validate fill:#fbeaf0,stroke:#D4537E,color:#4b1528
    classDef train    fill:#faece7,stroke:#E8593C,color:#4a1b0c
    classDef register fill:#e1f5ee,stroke:#1D9E75,color:#04342c
    classDef save     fill:#eaf3de,stroke:#639922,color:#173404
```

### `python main.py --serve`

```mermaid
flowchart TD
    A([--serve]):::cmd --> B{No games.json\nor no model?}:::decision
    B -- Yes --> C[Generate 2000\nsynthetic games]:::synth
    C --> D[Train on synthetic\nbootstrap only]:::train
    B -- No --> E
    D --> E[scheduler.start\ndaemon thread]:::sched
    E --> F[app.run\nhost=0.0.0.0\nport from PORT env var]:::flask

    classDef cmd      fill:#e6f1fb,stroke:#378ADD,color:#042c53
    classDef decision fill:#eeedfe,stroke:#7F77DD,color:#3c3489
    classDef synth    fill:#faeeda,stroke:#BA7517,color:#412402
    classDef train    fill:#faece7,stroke:#E8593C,color:#4a1b0c
    classDef sched    fill:#eaf3de,stroke:#639922,color:#173404
    classDef flask    fill:#fbeaf0,stroke:#D4537E,color:#4b1528
```

`use_reloader=False` prevents the scheduler from starting twice during Flask's debug reload cycle.

---

## Auto-Learning Pipeline

```mermaid
flowchart TD
    A([_loop starts\nsleep 60s]):::start --> B{fetch due?\nevery 6h}:::decision

    B -- Yes --> C[fetch_ncaa_data\nfull pipeline]:::fetch
    C --> D[append_to_json\ncount added]:::store
    D --> E{added >=\n15 games?}:::decision
    E -- Yes --> F[train_and_evaluate\ntriggered_by=new_data]:::train
    F --> G{new_auc >= current\n+ 0.002?}:::decision
    G -- Yes --> H[register_model\npromote to active\nlog: promoted]:::promote
    G -- No --> I[log: skipped\nwith reason]:::skip
    E -- No --> J

    B -- No --> K{retrain due?\nevery 24h}:::decision
    K -- Yes --> L[train_and_evaluate\ntriggered_by=scheduler]:::train
    L --> G
    K -- No --> J

    H --> J([sleep 60s chunks\ncheck stop event]):::sleep
    I --> J
    J --> B

    classDef start    fill:#e6f1fb,stroke:#378ADD,color:#042c53
    classDef decision fill:#eeedfe,stroke:#7F77DD,color:#3c3489
    classDef fetch    fill:#e6f1fb,stroke:#378ADD,color:#042c53
    classDef store    fill:#eaf3de,stroke:#639922,color:#173404
    classDef train    fill:#faece7,stroke:#E8593C,color:#4a1b0c
    classDef promote  fill:#e1f5ee,stroke:#1D9E75,color:#04342c
    classDef skip     fill:#fcebeb,stroke:#E24B4A,color:#501313
    classDef sleep    fill:#f1efe8,stroke:#888780,color:#2c2c2a
```

The active model's AUC is monotonically non-decreasing over time. The model can only improve or stay the same.

---

## Data Flow

### Stats Mode: End-to-End Prediction

```mermaid
sequenceDiagram
    participant B as Browser
    participant F as Flask
    participant M as models.py
    participant P as preprocessing.py
    participant S as storage.py

    B->>F: GET /
    F-->>B: dashboard.html

    B->>F: GET /features
    B->>F: GET /home_team
    B->>F: GET /teams
    B->>F: GET /model_info
    F-->>B: init data for all tabs

    B->>F: GET /team_stats/Kansas Jayhawks?window=12
    F->>P: build_team_stats(data, window=12)
    P-->>F: rolling averages
    F-->>B: auto-fill away stat fields

    B->>F: POST /predict {home_fg_pct, away_fg_pct, ...}
    F->>M: load_active_model()
    M->>S: read registry.json + .pkl
    S-->>M: Pipeline object
    M-->>F: model, feature_names
    F->>F: model.predict + predict_proba
    F-->>B: prediction, confidence, version
```

### Roster Mode: End-to-End Prediction

```mermaid
sequenceDiagram
    participant B as Browser
    participant F as Flask
    participant R as roster.py
    participant M as models.py

    B->>F: GET /roster/Duke Blue Devils
    F->>R: fetch_team_async(team_name)
    R-->>F: status: loading (thread started)
    F-->>B: current progress state

    loop Every 1 second
        B->>F: GET /roster/progress/Duke Blue Devils
        F-->>B: {status, players so far, done/total}
    end

    B->>F: POST /predict/from_roster\n{home_players, away_players}
    F->>F: compute_stats_from_roster\nFGA-weighted fg_pct\nderived ast_to_tov
    F->>M: load_active_model
    M-->>F: Pipeline
    F->>F: predict + predict_proba
    F-->>B: prediction, confidence,\ncomputed_stats, insights
```

---

## Storage Architecture

### Game Record Structure (v2.5)

```mermaid
block-beta
    columns 3
    block:training["Training Features (pre-game rolling avgs)"]:1
        A["home_fg_pct: 0.4670\naway_fg_pct: 0.4480\nhome_rebounds: 36.4\naway_rebounds: 32.1\nhome_assists: 15.8\naway_assists: 13.2\nhome_turnovers: 11.9\naway_turnovers: 13.7\nhome_steals: 7.1\naway_steals: 6.4\nhome_blocks: 4.2\naway_blocks: 3.1\nhome_ast_to_tov: 1.328\naway_ast_to_tov: 0.964"]
    end
    block:analytics["Analytics Only (in-game stats)"]:1
        B["home_game_fg_pct: 0.521\naway_game_fg_pct: 0.385\nhome_ppg: 84.0\naway_ppg: 72.0\nhome_eff_score: 35.77\naway_eff_score: 27.75"]
    end
    block:meta["Metadata"]:1
        C["game_id: ESPN_401703521\ngame_date: 2024-01-15\nhome_team: Duke Blue Devils\naway_team: Kansas Jayhawks\noutcome: 1\nsource: espn\npregame_enriched: true\npregame_window_used: 10"]
    end
```

### Model Registry Structure

Each `.pkl` stores `{"model": Pipeline, "feature_names": list}`. Feature names travel with the model to prevent silent mismatches if the feature list changes between versions.

```json
{
  "active_version": "v3",
  "versions": [
    {
      "version": "v1",
      "model_name": "Gradient Boosting",
      "filename": "gradient_boosting_v1_a3f2c1d4.pkl",
      "metrics": { "roc_auc": 0.7441, "f1": 0.8161 },
      "feature_names": ["home_fg_pct", "away_fg_pct", "..."],
      "training_size": 2328,
      "trained_at": "2026-03-19T14:22:11",
      "hash": "a3f2c1d4"
    }
  ]
}
```

Pruning: when `len(versions) > keep_top_n` (default 10), oldest `.pkl` files are deleted from disk.

---

## Roster System

```mermaid
flowchart TD
    A([GET /roster/team_name]):::api --> B{Cache valid?\nfetched_at < 24h}:::decision
    B -- Yes --> C[Write progress:\nstatus=ready]:::ready
    B -- No --> D[Set progress:\nstatus=loading]:::loading
    D --> E[Start background thread\nfetch_team]:::thread

    E --> F[get_team_id\nESPN lookup or cache]:::step
    F --> G[GET /teams/id/roster]:::http
    G --> H[_parse_embedded_stats\nper athlete]:::parse
    H --> I{Player missing\nstats?}:::decision
    I -- Yes --> J[GET /athletes/id/statistics\ngraceful 404 handling]:::http
    I -- No --> K
    J --> K[Save to\ndata/rosters/team_id.json]:::store
    K --> L[Write progress:\nstatus=ready\nplayers complete]:::ready

    C --> M([Browser polls\n/roster/progress\nevery 1 second]):::poll
    L --> M

    classDef api      fill:#eeedfe,stroke:#7F77DD,color:#3c3489
    classDef decision fill:#eeedfe,stroke:#7F77DD,color:#3c3489
    classDef ready    fill:#e1f5ee,stroke:#1D9E75,color:#04342c
    classDef loading  fill:#faeeda,stroke:#BA7517,color:#412402
    classDef thread   fill:#e6f1fb,stroke:#378ADD,color:#042c53
    classDef step     fill:#eaf3de,stroke:#639922,color:#173404
    classDef http     fill:#fbeaf0,stroke:#D4537E,color:#4b1528
    classDef parse    fill:#faeeda,stroke:#BA7517,color:#412402
    classDef store    fill:#eaf3de,stroke:#3B6D11,color:#173404
    classDef poll     fill:#f1efe8,stroke:#888780,color:#2c2c2a
```

Player stats aggregation for roster-mode prediction:

- `ppg`, `rpg`, `apg`, `spg`, `bpg`, `tov` are summed across selected players
- `fg_pct` uses FGA-weighted average: `total_fgm / total_fga` (more accurate than a simple mean when players have unequal shot volume)
- `ast_to_tov` is derived from summed totals, not averaged per-player

---

## Dashboard Integration

### Tab Rendering Strategy

Charts are not drawn on page load. They are drawn fresh each time a tab becomes visible via `switchTab()`. This solves the Chart.js 0x0 canvas problem: `display:none` tabs have no dimensions at render time.

`requestAnimationFrame` defers execution by one paint cycle, ensuring the browser has applied `display:block` before Chart.js measures the canvas.

`loadAnalytics()` fetches data and updates DOM stat cards immediately but does not draw charts. This separation means a slow analytics fetch never blocks tab switching.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Config-driven via YAML | No secrets or tunables in Python source |
| Pre-game rolling averages as features | Eliminates leakage. Model trains on knowable pre-game data. |
| `game_date` on every record | Correct chronological ordering for enrichment |
| Enrichment as a separate pipeline step | Can back-fill existing data without re-fetching |
| ROC-AUC for model selection | Robust to class imbalance (~69% home wins) |
| Stratified train-test split | Preserves class ratio in both sets |
| `Pipeline(Scaler + clf)` | Scaler trained on train set only. No leakage. |
| Adaptive tree depth | Prevents overfitting. Scales automatically with dataset size. |
| `Path(__file__).parent.parent / "dashboard.html"` | Resolves correctly from `app/api.py` to project root regardless of working directory |
| `use_reloader=False` | Prevents scheduler starting twice in Flask debug mode |
| `_sanitize()` before JSON serialisation | XGBoost CV can produce NaN. Invalid JSON crashes the browser. |
| Lazy chart rendering | Chart.js cannot render into 0x0 hidden canvases |
| Module-level `_roster_progress` dict | Shared state between Flask and background roster threads |
| FGA-weighted `fg_pct` aggregation | More accurate than simple average when players have unequal shot volume |
| `window=None` default in `build_team_stats` | Full season average unless caller explicitly requests a rolling window |
| `RotatingFileHandler` 10 MB x 2 | Bounded disk usage. Always retains recent history. |
| `setup_logging()` before all other imports | Ensures all module-level loggers attach to the configured handler ||
