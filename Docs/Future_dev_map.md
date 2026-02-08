# 🚀 Future Development Map

> *Planned enhancements and evolution path for the basketball predictor system*

---

## Table of Contents

1. [Current System State](#current-system-state)
2. [Planned Enhancements](#planned-enhancements)
3. [Real NCAA API Integration](#real-ncaa-api-integration)
4. [Model Organization](#model-organization)
5. [Configuration Management](#configuration-management)
6. [Executable Distribution](#executable-distribution)
7. [Implementation Timeline](#implementation-timeline)

---

## 📊 Current System State

### ✅ What's Working

**Core Functionality:**
- Synthetic data generation with realistic statistics
- SQL database storage (Snowflake)
- Local development mode (JSON)
- Three-model comparison pipeline
- Automated best model selection
- Flask web server with REST API
- Interactive dashboard with analytics
- Model serialization and loading
- Prediction with confidence scores

**Engineering Practices:**
- Script-based architecture (no notebooks)
- Command-line interface
- Clear separation of concerns
- Comprehensive documentation
- Git-ready structure

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-score
- Model comparison visualization
- Dataset statistics

---

### ⚠️ Current Limitations

**Data:**
- Synthetic data (not real game statistics)
- Only 500-600 games generated
- No real-time updates

**Models:**
- No hyperparameter tuning
- Single train-test split (no cross-validation)
- No model versioning history

**Code Organization:**
- Some hardcoded values (API keys, paths)
- All logic in single file (main.py)
- No organized model storage

**Deployment:**
- Localhost only
- No executable distribution
- Manual server restart needed

---

## 🎯 Planned Enhancements

### Priority 1: Real NCAA API Integration ⭐⭐⭐

**Goal:** Replace synthetic data with real NCAA basketball statistics

**Implementation Plan:**

#### Step 1: API Selection & Testing
You mentioned you have a **free NCAAB API** ready. Key considerations:

**API Requirements:**
- Historical game data (last season minimum)
- Team season averages (PPG, FG%, rebounds, assists, turnovers)
- Game outcomes
- Reasonable rate limits

**Common NCAA APIs:**
- **SportsDataIO** (has free tier)
- **The Odds API** (has free tier, 500 calls/month)
- **ESPN Hidden API** (unofficial, free but unstable)
- **NCAA.com** (may require scraping)

#### Step 2: API Integration Module

Create `api_config.json`:
```json
{
  "ncaab_api": {
    "provider": "SportsDataIO",
    "base_url": "https://api.sportsdata.io/v3/cbb",
    "api_key": "YOUR_API_KEY_HERE",
    "endpoints": {
      "games": "/scores/json/Games/{season}",
      "team_stats": "/scores/json/TeamSeasonStats/{season}"
    }
  },
  "data_config": {
    "historical_years": 1,
    "last_n_matches": 12,
    "update_on_train": true
  }
}
```

#### Step 3: Data Fetching Logic

Add to `main.py`:
```python
def fetch_real_ncaab_data(config):
    """
    Fetch real NCAA basketball data from API
    
    Strategy:
    1. Fetch last season's games (training data)
    2. Fetch current season's last 12 matches (recent analytics)
    3. Extract relevant features (PPG, FG%, etc.)
    4. Store in Snowflake
    """
    
    # Load API config
    with open('api_config.json') as f:
        api_config = json.load(f)
    
    base_url = api_config['ncaab_api']['base_url']
    api_key = api_config['ncaab_api']['api_key']
    
    # Fetch team season stats
    # Cherry-pick: PPG, FG%, Rebounds, Assists, Turnovers
    
    # Fetch game results
    # Match team stats to game outcomes
    
    # Store to Snowflake
    pass
```

#### Step 4: Automatic Data Updates

**Trigger:** When `--train` is run, automatically fetch latest data

```python
if args.train:
    # First, update data
    print("Fetching latest NCAA data...")
    fetch_real_ncaab_data(config)
    
    # Then, train on updated data
    train_and_evaluate_models(args.storage)
```

**Benefits:**
- Always train on latest data
- No separate update command needed
- Ensures model stays current

#### Step 5: Last 12 Matches Analytics

**Use Case:** Dashboard shows recent trends (last 12 games)

```python
def get_recent_analytics(team_id, n_matches=12):
    """
    Calculate rolling statistics from last N matches
    Shows recent form vs season averages
    """
    
    recent_games = fetch_last_n_games(team_id, n_matches)
    
    recent_stats = {
        'ppg_recent': calculate_avg(recent_games, 'points'),
        'ppg_season': get_season_avg(team_id, 'points'),
        'form': calculate_win_percentage(recent_games),
        # ... other stats
    }
    
    return recent_stats
```

**Dashboard Display:**
- "Team's last 12 games: 8-4 (67% win rate)"
- "Recent PPG: 82.3 (Season: 78.5) ↑"
- "Recent FG%: 46.2% (Season: 44.8%) ↑"

---

### Priority 2: Organized Model Storage ⭐⭐

**Goal:** Create structured model versioning system

**Current State:**
```
project/
├── best_model.pkl          # Overwritten each training
├── model_info.json         # Overwritten each training
└── model_comparison.json   # Overwritten each training
```

**Proposed Structure:**
```
project/
├── models/
│   ├── 2024-02-08_14-23-45/          # Timestamp folder
│   │   ├── best_model.pkl
│   │   ├── model_info.json
│   │   ├── model_comparison.json
│   │   └── training_data_info.json   # Dataset size, date range
│   │
│   ├── 2024-02-15_09-12-33/          # Another training run
│   │   ├── best_model.pkl
│   │   └── ...
│   │
│   └── latest/                        # Symlink to most recent
│       └── → 2024-02-15_09-12-33/
│
└── main.py
```

**Implementation:**

```python
def save_model_with_version(model, model_info, comparison_data):
    """
    Save model with timestamp-based versioning
    """
    from datetime import datetime
    
    # Create timestamp folder
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_dir = f'models/{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save all artifacts
    with open(f'{model_dir}/best_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(f'{model_dir}/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    with open(f'{model_dir}/model_comparison.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    # Update 'latest' symlink
    latest_link = 'models/latest'
    if os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(timestamp, latest_link)
    
    print(f"Model saved to: {model_dir}")
    print(f"Latest model: {latest_link}")
```

**Benefits:**
- Model history preserved
- Easy rollback to previous versions
- Compare performance over time
- Track model evolution

**Dashboard Enhancement:**
```
Model History:
┌─────────────────────┬──────────┬─────────┐
│ Date                │ F1-Score │ Model   │
├─────────────────────┼──────────┼─────────┤
│ 2024-02-15 09:12:33 │ 0.8720   │ RF      │ ← Current
│ 2024-02-08 14:23:45 │ 0.8580   │ RF      │
│ 2024-02-01 11:45:12 │ 0.8420   │ LR      │
└─────────────────────┴──────────┴─────────┘
```

---

### Priority 3: Configuration Management ⭐⭐

**Goal:** Eliminate hardcoded values, use config files

**Current Hardcoded Values:**
- Snowflake credentials
- API keys
- File paths
- Model hyperparameters
- Data generation parameters

**Proposed: `config.json`**

```json
{
  "storage": {
    "default_mode": "local",
    "snowflake": {
      "user": "${SNOWFLAKE_USER}",
      "password": "${SNOWFLAKE_PASSWORD}",
      "account": "${SNOWFLAKE_ACCOUNT}",
      "warehouse": "COMPUTE_WH",
      "database": "BASKETBALL_DB",
      "schema": "PUBLIC",
      "table": "GAMES"
    },
    "local": {
      "data_file": "data.json"
    }
  },
  
  "models": {
    "storage_dir": "models/",
    "use_versioning": true,
    "logistic_regression": {
      "max_iter": 1000,
      "random_state": 42
    },
    "random_forest": {
      "n_estimators": 100,
      "max_depth": null,
      "random_state": 42
    },
    "training": {
      "test_size": 0.2,
      "random_state": 42
    }
  },
  
  "data_generation": {
    "num_initial_games": 500,
    "num_api_fetch_games": 100,
    "feature_ranges": {
      "ppg": [65, 95],
      "fg_pct": [0.40, 0.55],
      "rebounds": [30, 50],
      "assists": [12, 25],
      "turnovers": [8, 18]
    }
  },
  
  "server": {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": true
  },
  
  "ncaab_api": {
    "enabled": false,
    "provider": "SportsDataIO",
    "api_key": "${NCAAB_API_KEY}",
    "base_url": "https://api.sportsdata.io/v3/cbb",
    "rate_limit_calls": 100,
    "rate_limit_period": 3600
  }
}
```

**Environment Variables (for secrets):**

Create `.env` file (not committed to Git):
```bash
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ACCOUNT=your_account
NCAAB_API_KEY=your_api_key
```

**Load configuration:**
```python
import json
import os
from string import Template

def load_config():
    """Load configuration with environment variable substitution"""
    with open('config.json') as f:
        config_str = f.read()
    
    # Substitute environment variables
    config_str = Template(config_str).safe_substitute(os.environ)
    
    config = json.loads(config_str)
    return config

# Usage
config = load_config()
SNOWFLAKE_USER = config['storage']['snowflake']['user']
NUM_GAMES = config['data_generation']['num_initial_games']
```

**Benefits:**
- No secrets in code
- Easy to change settings
- Different configs for dev/prod
- Shareable without exposing credentials

---

### Priority 4: Multi-Model Comparison Dashboard ⭐

**Goal:** Make model comparison more visible and interactive

**Current:** Model comparison data exists but is underutilized

**Enhancement 1: Comparison Table**

Add to dashboard:
```html
<div class="model-comparison-table">
  <h3>Model Performance Comparison</h3>
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Accuracy</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>F1-Score</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody>
      <tr class="best-model">
        <td>Random Forest</td>
        <td>85.8%</td>
        <td>86.9%</td>
        <td>84.7%</td>
        <td>85.8%</td>
        <td>✓ Active</td>
      </tr>
      <tr>
        <td>Logistic Regression</td>
        <td>82.5%</td>
        <td>83.3%</td>
        <td>81.7%</td>
        <td>82.5%</td>
        <td></td>
      </tr>
      <tr>
        <td>Linear Regression</td>
        <td>80.8%</td>
        <td>81.8%</td>
        <td>79.5%</td>
        <td>80.6%</td>
        <td></td>
      </tr>
    </tbody>
  </table>
</div>
```

**Enhancement 2: Radar Chart**

Add radar chart comparing models across all metrics:
```javascript
new Chart(ctx, {
  type: 'radar',
  data: {
    labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    datasets: [
      {
        label: 'Random Forest',
        data: [85.8, 86.9, 84.7, 85.8],
        borderColor: 'green'
      },
      {
        label: 'Logistic Regression',
        data: [82.5, 83.3, 81.7, 82.5],
        borderColor: 'blue'
      },
      {
        label: 'Linear Regression',
        data: [80.8, 81.8, 79.5, 80.6],
        borderColor: 'orange'
      }
    ]
  }
});
```

**Enhancement 3: Historical Performance**

Track and display model performance over time:
```
Random Forest F1-Score History:
┌──────────────────────────────────────┐
│                             ●        │ 87.2%
│                       ●              │
│                 ●                    │ 85.8%
│           ●                          │
│     ●                                │ 84.2%
└──────────────────────────────────────┘
 Feb 1   Feb 8   Feb 15   Feb 22   Mar 1
```

---

### Priority 5: Executable Distribution ⭐

**Goal:** Package as standalone executable for easy distribution

**Why?**
- Professors/reviewers can run without Python installation
- Easy to share on GitHub releases
- Professional presentation

**Implementation: PyInstaller**

**Step 1: Create executable spec**

```bash
pip install pyinstaller
```

**Step 2: Build executable**

```bash
pyinstaller --onefile \
            --name basketball-predictor \
            --add-data "dashboard.html:." \
            --add-data "config.json:." \
            main.py
```

**Step 3: Test executable**

```bash
./dist/basketball-predictor --generate --storage local
./dist/basketball-predictor --train --storage local
./dist/basketball-predictor --serve --storage local
```

**Step 4: Create GitHub release**

```bash
# Tag version
git tag v1.0.0
git push origin v1.0.0

# Upload to GitHub releases
# - Windows: basketball-predictor.exe
# - macOS: basketball-predictor
# - Linux: basketball-predictor
```

**Challenges:**
- Large file size (~50MB with dependencies)
- Snowflake connector may have issues
- May need to include data files

**Alternative: Docker Container**

```dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py dashboard.html config.json ./

CMD ["python", "main.py", "--serve", "--storage", "local"]
```

**Run with:**
```bash
docker build -t basketball-predictor .
docker run -p 5000:5000 basketball-predictor
```

---

## 📅 Implementation Timeline

### Phase 1: Configuration & Organization (Week 1-2)
- ✅ Create `config.json`
- ✅ Implement environment variable loading
- ✅ Create `models/` directory structure
- ✅ Implement model versioning
- ✅ Update documentation

**Deliverable:** Cleaner, more professional codebase

---

### Phase 2: NCAA API Integration (Week 3-4)
- ✅ Set up API credentials
- ✅ Implement data fetching functions
- ✅ Map API data to existing features
- ✅ Store in Snowflake
- ✅ Test with real data

**Deliverable:** System using real NCAA data

---

### Phase 3: Dashboard Enhancements (Week 5)
- ✅ Add model comparison table
- ✅ Add radar chart
- ✅ Add recent match analytics (last 12 games)
- ✅ Improve UI/UX

**Deliverable:** More informative dashboard

---

### Phase 4: Distribution (Week 6)
- ✅ Create executable with PyInstaller
- ✅ Test on different systems
- ✅ Create GitHub release
- ✅ Write deployment guide

**Deliverable:** Distributable application

---

## 🎯 Success Metrics

**For Each Enhancement:**

1. **Real API Integration**
   - ✅ Successfully fetch data from API
   - ✅ Store in Snowflake
   - ✅ Model accuracy comparable or better than synthetic

2. **Model Organization**
   - ✅ 5+ historical model versions saved
   - ✅ Easy to load any previous version
   - ✅ Performance tracking over time

3. **Configuration Management**
   - ✅ No hardcoded secrets in code
   - ✅ Config changes without code changes
   - ✅ Environment-specific configs work

4. **Dashboard Improvements**
   - ✅ Model comparison clearly visible
   - ✅ Recent match analytics displayed
   - ✅ Interactive and informative

5. **Executable Distribution**
   - ✅ Runs without Python installation
   - ✅ GitHub release with downloads
   - ✅ Clear usage instructions

---

## 🚫 What We're NOT Doing (Avoiding Over-Engineering)

These are explicitly OUT of scope:

- ❌ Deep learning models (not needed for this dataset size)
- ❌ Real-time predictions during live games (too complex)
- ❌ Mobile app (separate project)
- ❌ Cloud deployment (Docker container is enough)
- ❌ User authentication (not needed for demo)
- ❌ Database migrations (Snowflake handles schema)
- ❌ Microservices architecture (overkill for this scale)
- ❌ GraphQL API (REST is sufficient)
- ❌ WebSockets (no real-time updates needed)

**Keep it simple, keep it working.**

---

## 💡 Key Principles

1. **Incremental Enhancement:** Add features one at a time, test thoroughly
2. **Maintain Simplicity:** Don't add complexity for its own sake
3. **Documentation First:** Update docs before implementing
4. **Backward Compatibility:** New features shouldn't break existing functionality
5. **Test as You Go:** Verify each enhancement works before moving to next

---

## 🎓 Learning Outcomes

By implementing these enhancements, you'll demonstrate:

**Technical Skills:**
- REST API integration
- Configuration management
- File system organization
- Application packaging
- Version control best practices

**Engineering Mindset:**
- Planning before coding
- Incremental development
- Trade-offs between features and complexity
- Production-readiness considerations

**Professional Practices:**
- Clean code organization
- Comprehensive documentation
- Realistic project scoping
- Maintainable architecture

---

## 📝 Final Notes

**This roadmap is realistic and achievable.** Each enhancement:
- ✅ Adds clear value
- ✅ Is technically feasible
- ✅ Has defined success criteria
- ✅ Builds on existing work

**The goal isn't perfection—it's continuous improvement.**

Start with what works, enhance thoughtfully, document clearly, and you'll have a project that stands out not because it does everything, but because it does what it promises extremely well.

---

*This future development map provides a clear, achievable path forward without over-promising or over-engineering.*