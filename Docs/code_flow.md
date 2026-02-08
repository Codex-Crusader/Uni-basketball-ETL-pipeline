# 🔄 Code Flow Documentation

> *Complete architecture and data flow documentation for the basketball predictor system*

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Components](#architecture-components)
3. [Command Execution Flows](#command-execution-flows)
4. [Data Flow](#data-flow)
5. [Storage Architecture](#storage-architecture)
6. [Model Training Pipeline](#model-training-pipeline)
7. [Prediction Serving](#prediction-serving)
8. [Dashboard Integration](#dashboard-integration)

---

## 🏛️ System Overview

The basketball predictor is built as a **script-based ML pipeline** with four main operational modes:

```
┌─────────────────────────────────────────────────────────────┐
│                     main.py (655 lines)                      │
│                  Single Script Architecture                  │
└────────────────────┬────────────────────────────────────────┘
                     │
          Command Line Arguments Parser
                     │
         ┌───────────┼───────────┬──────────┐
         │           │           │          │
         ▼           ▼           ▼          ▼
    [GENERATE]  [FETCH-API]  [TRAIN]   [SERVE]
```

### Design Philosophy

**Single Script Architecture:** All logic resides in `main.py` to keep the system:
- Easy to understand (no jumping between files)
- Easy to debug (everything in one place)
- Easy to deploy (single file + dependencies)
- Easy to modify (change one file)

**Separation via Functions:** Within `main.py`, concerns are separated into logical sections:
- Data Generation (lines 38-106)
- Storage Layer (lines 113-275)
- Model Training (lines 282-413)
- Prediction Server (lines 419-585)
- Command Interface (lines 591-655)

---

## 📊 Architecture Components

### Layer 1: Command Line Interface

**Location:** `main()` function (lines 591-655)

**Purpose:** Parse user commands and route to appropriate functions

**Commands Supported:**
```bash
--generate      # Create initial dataset
--fetch-api     # Simulate fetching new data
--train         # Train and evaluate models
--serve         # Start web server
--storage       # Choose storage backend (local/snowflake)
```

**Design Pattern:** Command pattern with argument parser

```python
parser = argparse.ArgumentParser()
parser.add_argument('--generate', action='store_true')
parser.add_argument('--fetch-api', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--serve', action='store_true')
parser.add_argument('--storage', choices=['local', 'snowflake'])
```

---

### Layer 2: Data Generation

**Functions:**
- `generate_synthetic_data(num_games=500)` (lines 38-90)
- `simulate_api_fetch(num_new_games=100)` (lines 97-106)

**Data Generation Algorithm:**

```
For each game:
  1. Generate random feature values from uniform distributions
     • home_ppg ~ U(65, 95)
     • home_fg_pct ~ U(0.40, 0.55)
     • ... (10 features total)
  
  2. Calculate strength scores
     home_strength = weighted_sum(home_features) + 3  # +3 = home advantage
     away_strength = weighted_sum(away_features)
  
  3. Determine outcome probabilistically
     if home_strength > away_strength:
       outcome = 1 with 85% probability (home wins)
     else:
       outcome = 0 with 85% probability (away wins)
  
  4. Round and format as JSON object
```

**Why Synthetic Data:**
- Demonstrates the system architecture
- No API dependencies for basic demo
- Reproducible results
- Known ground truth (we control outcome generation)

**Future:** Can be replaced with real API calls without changing rest of system.

---

### Layer 3: Storage Abstraction

**Local Storage (JSON):**
- `save_to_json(data)` (lines 113-118)
- `load_from_json()` (lines 121-131)
- `append_to_json(new_data)` (lines 134-139)

**Cloud Storage (Snowflake):**
- `get_snowflake_connection()` (lines 146-159)
- `create_snowflake_table(conn)` (lines 162-183)
- `save_to_snowflake(data)` (lines 186-220)
- `load_from_snowflake()` (lines 223-242)
- `append_to_snowflake(new_data)` (lines 245-275)

**Storage Pattern:**

```
Application Code
       │
       ├─ if storage == 'local':
       │    └─ save_to_json()
       │
       └─ elif storage == 'snowflake':
            └─ save_to_snowflake()

Common Interface → Different Implementations
```

**Benefits:**
- Swap storage without changing training code
- Local mode for development
- Snowflake mode for production
- Same data format regardless of backend

---

### Layer 4: Feature Engineering

**Function:** `prepare_data(data)` (lines 282-298)

**Transformation Pipeline:**

```
Input: List of game dictionaries
       [
         {"game_id": "GAME_0001", "home_ppg": 78.45, ...},
         {"game_id": "GAME_0002", "home_ppg": 82.31, ...},
         ...
       ]

Step 1: Extract feature names (fixed order)
        feature_names = ['home_ppg', 'away_ppg', 'home_fg_pct', ...]

Step 2: Build feature matrix
        features = [[78.45, 72.31, 0.478, ...],
                   [82.31, 76.54, 0.491, ...],
                   ...]

Step 3: Extract labels
        labels = [1, 0, 1, ...]

Step 4: Convert to NumPy
        X = np.array(features)  # shape: (n_games, 10)
        y = np.array(labels)    # shape: (n_games,)

Output: (X, y) tuple ready for sklearn
```

**Critical Constraint:** Feature order must be consistent across training and prediction!

---

### Layer 5: Model Training

**Function:** `train_and_evaluate_models(storage_mode)` (lines 301-413)

**Training Pipeline:**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. LOAD DATA                                                │
│    Load from storage (JSON or Snowflake)                    │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PREPARE DATA                                             │
│    Convert to (X, y) numpy arrays                           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. TRAIN-TEST SPLIT                                         │
│    80% training, 20% testing (random_state=42)              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. TRAIN MULTIPLE MODELS                                    │
│    ┌──────────────────────────────────────────────┐        │
│    │ For each model in models:                    │        │
│    │   • Initialize model                          │        │
│    │   • Train: model.fit(X_train, y_train)       │        │
│    │   • Predict: y_pred = model.predict(X_test)  │        │
│    │   • Calculate metrics (accuracy, precision, recall, F1)│
│    │   • Store results                            │        │
│    └──────────────────────────────────────────────┘        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. SELECT BEST MODEL                                        │
│    best = max(results, key=lambda k: results[k]['f1'])      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. SAVE ARTIFACTS                                           │
│    • best_model.pkl (pickle serialization)                  │
│    • model_info.json (metadata)                             │
│    • model_comparison.json (all models' metrics)            │
└─────────────────────────────────────────────────────────────┘
```

**Models Compared:**

1. **Logistic Regression**
   - `max_iter=1000` (ensure convergence)
   - `random_state=42` (reproducibility)

2. **Random Forest**
   - `n_estimators=100` (100 trees)
   - `random_state=42` (reproducibility)

3. **Linear Regression (Thresholded)**
   - Default parameters
   - Output ≥ 0.5 → class 1, else class 0

**Model Selection Logic:**
```python
best_model_name = max(results, key=lambda k: results[k]['f1'])
```
F1-score balances precision and recall, making it ideal for binary classification.

---

### Layer 6: Model Persistence

**Serialization:**

```python
# Save model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save metadata
model_info = {
    'model_name': best_model_name,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'trained_at': datetime.now().isoformat()
}
with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)
```

**Files Created:**

| File | Format | Contents |
|------|--------|----------|
| `best_model.pkl` | Binary (pickle) | Complete sklearn model object |
| `model_info.json` | JSON | Metadata about selected model |
| `model_comparison.json` | JSON | Metrics for all three models |

---

### Layer 7: Flask Web Server

**Function:** Flask app instance (lines 419-585)

**Routes:**

```
GET  /              → dashboard.html
POST /predict       → Make prediction
GET  /model_info    → Get model metadata
GET  /analytics     → Get dataset statistics
```

**Server Architecture:**

```
Browser
   │
   │ HTTP Request
   │
   ▼
Flask (@app.route)
   │
   ├─ Route: /
   │  └─ send_file('dashboard.html')
   │
   ├─ Route: /predict (POST)
   │  │
   │  ├─ Load model (best_model.pkl)
   │  ├─ Extract features from request
   │  ├─ model.predict(features)
   │  ├─ Calculate confidence
   │  └─ Return JSON response
   │
   ├─ Route: /model_info (GET)
   │  │
   │  ├─ Load model_info.json
   │  └─ Return as JSON
   │
   └─ Route: /analytics (GET)
      │
      ├─ Load data.json
      ├─ Calculate statistics
      ├─ Load model_comparison.json
      └─ Return combined JSON
```

---

## 🚀 Command Execution Flows

### Flow 1: `python main.py --generate --storage local`

**Goal:** Create initial synthetic dataset

```
main()
  │
  ├─ argparse parses: args.generate=True, args.storage='local'
  │
  ├─ Call generate_synthetic_data(500)
  │   │
  │   ├─ Loop 500 times:
  │   │   │
  │   │   ├─ Random features: U(65,95), U(0.4,0.55), etc.
  │   │   │
  │   │   ├─ Calculate strengths:
  │   │   │   home_strength = home_ppg*0.3 + home_fg_pct*100 + ... + 3
  │   │   │   away_strength = away_ppg*0.3 + away_fg_pct*100 + ...
  │   │   │
  │   │   ├─ Determine outcome:
  │   │   │   if home_strength > away_strength:
  │   │   │     outcome = 1 (85% prob) or 0 (15% prob)
  │   │   │   else:
  │   │   │     outcome = 0 (85% prob) or 1 (15% prob)
  │   │   │
  │   │   └─ Create game dict with 11 fields
  │   │
  │   └─ Return list of 500 games
  │
  ├─ Call save_to_json(data)
  │   │
  │   ├─ Open 'data.json' for writing
  │   ├─ json.dump(data, f, indent=2)
  │   └─ Close file
  │
  └─ Exit (status 0)
```

**Output:**
- Console: "Generated 500 games. Saved 500 records to data.json."
- File: `data.json` (approx 150 KB)

---

### Flow 2: `python main.py --fetch-api --storage local`

**Goal:** Append new season data

```
main()
  │
  ├─ argparse parses: args.fetch_api=True, args.storage='local'
  │
  ├─ Call simulate_api_fetch(100)
  │   │
  │   └─ Call generate_synthetic_data(100)
  │       └─ (Same process as --generate, but 100 games)
  │
  ├─ Call append_to_json(new_data)
  │   │
  │   ├─ Call load_from_json()
  │   │   └─ Returns existing 500 games
  │   │
  │   ├─ Combine: existing_data + new_data → 600 games
  │   │
  │   └─ Call save_to_json(combined_data)
  │       └─ Overwrites data.json with 600 games
  │
  └─ Exit
```

**Output:**
- Console: "Fetched 100 new games. Appended 100 new records. Total: 600"
- File: `data.json` (approx 180 KB)

---

### Flow 3: `python main.py --train --storage local`

**Goal:** Train models and select best

```
main()
  │
  ├─ argparse parses: args.train=True, args.storage='local'
  │
  └─ Call train_and_evaluate_models('local')
      │
      ├─ Load data: load_from_json() → 600 games
      │
      ├─ Prepare data: prepare_data(data) → (X, y)
      │   X.shape = (600, 10)
      │   y.shape = (600,)
      │
      ├─ Train-test split (80/20):
      │   X_train.shape = (480, 10)
      │   X_test.shape = (120, 10)
      │
      ├─ Initialize models:
      │   models = {
      │     'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
      │     'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
      │     'Linear Regression (Thresholded)': LinearRegression()
      │   }
      │
      ├─ For each model:
      │   │
      │   ├─ model.fit(X_train, y_train)
      │   │
      │   ├─ y_pred = model.predict(X_test)
      │   │   (Special handling for Linear Regression: threshold at 0.5)
      │   │
      │   ├─ Calculate metrics:
      │   │   accuracy = accuracy_score(y_test, y_pred)
      │   │   precision = precision_score(y_test, y_pred)
      │   │   recall = recall_score(y_test, y_pred)
      │   │   f1 = f1_score(y_test, y_pred)
      │   │
      │   ├─ Print metrics to console
      │   │
      │   └─ Store in results dict
      │
      ├─ Select best model:
      │   best_name = max(results, key=lambda k: results[k]['f1'])
      │   best_model = results[best_name]['model']
      │
      ├─ Save best model:
      │   pickle.dump(best_model, open('best_model.pkl', 'wb'))
      │
      ├─ Save metadata:
      │   json.dump(model_info, open('model_info.json', 'w'))
      │   json.dump(comparison, open('model_comparison.json', 'w'))
      │
      └─ Exit
```

**Output:**
- Console: Training progress, metrics for all models, best model announcement
- Files: `best_model.pkl`, `model_info.json`, `model_comparison.json`

---

### Flow 4: `python main.py --serve --storage local`

**Goal:** Start prediction web server

```
main()
  │
  ├─ argparse parses: args.serve=True, args.storage='local'
  │
  ├─ Print startup banner
  │
  └─ app.run(debug=True, port=5000)
      │
      └─ Flask starts listening on localhost:5000
          │
          └─ Wait for HTTP requests...
              │
              ├─ User visits http://localhost:5000
              │  │
              │  └─ Route: @app.route('/')
              │      └─ return send_file('dashboard.html')
              │
              ├─ User POSTs to /predict
              │  │
              │  └─ Route: @app.route('/predict', methods=['POST'])
              │      │
              │      ├─ load_model() → (model, model_info)
              │      │
              │      ├─ Extract features from request.json (10 values)
              │      │
              │      ├─ features = np.array([[f1, f2, ..., f10]])
              │      │
              │      ├─ prediction = model.predict(features)
              │      │
              │      ├─ confidence = calculate_confidence(model, features)
              │      │
              │      └─ return jsonify({
              │            'prediction': 'Home Win' or 'Away Win',
              │            'confidence': 0.87,
              │            'model_name': 'Random Forest'
              │          })
              │
              ├─ User GETs /model_info
              │  │
              │  └─ Route: @app.route('/model_info', methods=['GET'])
              │      │
              │      ├─ load_model() → (model, model_info)
              │      │
              │      └─ return jsonify(model_info)
              │
              └─ User GETs /analytics
                 │
                 └─ Route: @app.route('/analytics', methods=['GET'])
                     │
                     ├─ Load data.json
                     │
                     ├─ Calculate statistics:
                     │   • total_games
                     │   • home_wins, away_wins
                     │   • home_win_rate
                     │   • feature_stats (grouped by outcome)
                     │
                     ├─ Load model_comparison.json
                     │
                     └─ return jsonify({
                           'total_games': 600,
                           'home_wins': 327,
                           'away_wins': 273,
                           'home_win_rate': 0.545,
                           'feature_stats': {...},
                           'model_comparison': {...}
                         })
```

**Server runs until Ctrl+C**

---

## 🔄 Data Flow

### End-to-End Prediction Flow

```
User Opens Browser
        │
        │ HTTP GET /
        │
        ▼
Flask: send_file('dashboard.html')
        │
        ▼
Browser Renders HTML
        │
        ├─────────────────────┐
        │                     │
        ▼                     ▼
JavaScript Calls       User Fills Form
GET /model_info       (10 feature inputs)
GET /analytics              │
        │                   │
        ▼                   ▼
Display Charts        User Clicks "Predict"
                            │
                            │ HTTP POST /predict
                            │ Body: { home_ppg: 82.5, ... }
                            │
                            ▼
Flask: @app.route('/predict')
        │
        ├─ Load: best_model.pkl
        │
        ├─ Extract: 10 feature values
        │
        ├─ Create: np.array([[f1, f2, ..., f10]])
        │
        ├─ Predict: model.predict(features)
        │
        ├─ Calculate: confidence score
        │
        └─ Return: JSON response
                │
                ▼
Browser Receives JSON
        │
        ▼
JavaScript Updates DOM
        │
        ├─ Show prediction ("Home Win")
        ├─ Show confidence (87%)
        ├─ Apply styling (green/red)
        └─ Display model name
```

---

## 💾 Storage Architecture

### Local Storage (JSON)

**File:** `data.json`

**Structure:**
```json
[
  {
    "game_id": "GAME_0001",
    "home_ppg": 78.45,
    "away_ppg": 72.31,
    "home_fg_pct": 0.478,
    "away_fg_pct": 0.441,
    "home_rebounds": 38.20,
    "away_rebounds": 35.67,
    "home_assists": 18.90,
    "away_assists": 16.45,
    "home_turnovers": 12.30,
    "away_turnovers": 14.20,
    "outcome": 1
  },
  ...
]
```

**Operations:**
- **Create:** `json.dump(data, file)`
- **Read:** `json.load(file)`
- **Append:** Load → Extend → Save

**Pros:**
- Simple, no dependencies
- Human-readable
- Version-control friendly

**Cons:**
- Not scalable to millions of records
- No query optimization
- Must load entire file

---

### Cloud Storage (Snowflake)

**Table:** `BASKETBALL_GAMES`

**Schema:**
```sql
CREATE TABLE BASKETBALL_GAMES (
    game_id VARCHAR(50),
    home_ppg FLOAT,
    away_ppg FLOAT,
    home_fg_pct FLOAT,
    away_fg_pct FLOAT,
    home_rebounds FLOAT,
    away_rebounds FLOAT,
    home_assists FLOAT,
    away_assists FLOAT,
    home_turnovers FLOAT,
    away_turnovers FLOAT,
    outcome INT
);
```

**Operations:**
- **Create:** `INSERT INTO ... VALUES (...)`
- **Read:** `SELECT * FROM BASKETBALL_GAMES`
- **Append:** `INSERT INTO ... VALUES (...)` (additional rows)

**Pros:**
- Scalable to millions of records
- SQL query capabilities
- Cloud-accessible
- Production-grade

**Cons:**
- Requires Snowflake account
- Credentials management
- Network dependency

---

## 🎯 Model Training Pipeline

### Training Sequence Diagram

```
┌──────┐      ┌──────────┐      ┌─────────┐      ┌───────┐
│ Data │      │ Feature  │      │  Train  │      │ Save  │
│ Load │─────▶│ Engineer │─────▶│  Models │─────▶│ Best  │
└──────┘      └──────────┘      └─────────┘      └───────┘
   │              │                   │               │
   │              │                   │               │
   ▼              ▼                   ▼               ▼
Storage       (X, y)           3 Trained         .pkl + .json
(JSON/SQL)    arrays            Models            files
```

### Detailed Training Steps

**Step 1: Data Loading**
```python
if storage_mode == 'local':
    data = load_from_json()  # List of dicts
else:
    data = load_from_snowflake()  # List of dicts
```

**Step 2: Feature Engineering**
```python
X, y = prepare_data(data)
# X: (n_games, 10) - feature matrix
# y: (n_games,) - outcome labels
```

**Step 3: Train-Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 80% training, 20% testing
# random_state ensures reproducibility
```

**Step 4: Model Training Loop**
```python
for model_name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    metrics = calculate_metrics(y_test, y_pred)
    
    # Store
    results[model_name] = {'model': model, **metrics}
```

**Step 5: Model Selection**
```python
best_model_name = max(results, key=lambda k: results[k]['f1'])
best_model = results[best_model_name]['model']
```

**Step 6: Persistence**
```python
pickle.dump(best_model, open('best_model.pkl', 'wb'))
json.dump(model_info, open('model_info.json', 'w'))
json.dump(comparison, open('model_comparison.json', 'w'))
```

---

## 🌐 Prediction Serving

### Flask Route Handlers

**Route: `/predict`**

```python
@app.route('/predict', methods=['POST'])
def predict():
    # 1. Load model
    model, model_info = load_model()
    
    # 2. Extract features from request
    data = request.json
    features = [
        float(data['home_ppg']),
        float(data['away_ppg']),
        # ... (10 features)
    ]
    
    # 3. Create numpy array
    X = np.array([features])
    
    # 4. Make prediction
    if 'Linear Regression' in model_info['model_name']:
        # Threshold continuous output
        output = model.predict(X)[0]
        prediction = int(output >= 0.5)
        confidence = abs(output - 0.5) * 2
    else:
        # Direct classification
        prediction = int(model.predict(X)[0])
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            confidence = float(max(proba))
        else:
            confidence = None
    
    # 5. Format response
    result = {
        'prediction': 'Home Win' if prediction == 1 else 'Away Win',
        'confidence': confidence,
        'model_name': model_info['model_name']
    }
    
    return jsonify(result)
```

---

## 🎨 Dashboard Integration

### Dashboard Loading Sequence

```
Page Load
    │
    ├─ HTML/CSS renders structure
    │
    └─ JavaScript executes
        │
        ├─ Call loadModelInfo()
        │   │
        │   └─ fetch('/model_info')
        │       │
        │       └─ Display current model name
        │
        └─ Call loadAnalytics()
            │
            └─ fetch('/analytics')
                │
                ├─ Display stats (total games, win rate)
                │
                ├─ Create model comparison chart (Chart.js)
                │
                ├─ Create outcome distribution chart
                │
                └─ Create feature analysis chart
```

### User Interaction Flow

```
User Fills Form (10 inputs)
    │
    └─ User clicks "Predict" button
        │
        └─ JavaScript: form submit event
            │
            ├─ Collect form data
            │
            ├─ Show loading indicator
            │
            └─ fetch('/predict', {method: 'POST', body: formData})
                │
                └─ Response received
                    │
                    ├─ Hide loading indicator
                    │
                    └─ Display prediction result
                        │
                        ├─ Prediction text ("Home Win")
                        ├─ Confidence (87%)
                        ├─ Model name
                        └─ Color styling (green/red)
```

---

## 🔧 Error Handling

### Storage Layer
```python
if not os.path.exists(filename):
    print(f"File {filename} not found.")
    return []
```

### Model Loading
```python
if model is None:
    return jsonify({'error': 'No trained model found'}), 400
```

### Prediction
```python
try:
    features = extract_features(request.json)
    prediction = model.predict(features)
except Exception as e:
    return jsonify({'error': str(e)}), 400
```

---

## 📊 Key Design Decisions

1. **Single Script:** Simplicity over modularity (for project scale)
2. **Fixed Random Seed:** Reproducibility over randomness
3. **F1-Score Selection:** Balance over single metric
4. **80/20 Split:** Standard practice, no cross-validation (yet)
5. **Pickle Serialization:** Standard sklearn approach
6. **Flask Debug Mode:** Development convenience

---

*This document provides a complete technical reference for understanding how the system operates at every level.*