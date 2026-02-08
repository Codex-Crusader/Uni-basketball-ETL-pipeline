# Import Dumpyard
import json
import os
import argparse
import pickle
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# CONFIGURATION

LOCAL_DATA_FILE = "data.json"
MODEL_FILE = "best_model.pkl"
MODEL_INFO_FILE = "model_info.json"
MODEL_COMPARISON_FILE = "model_comparison.json"

# Snowflake configuration (update with your credentials)
# Will be transferred to Json file in future versions
SNOWFLAKE_CONFIG = {
    "user": "YOUR_USERNAME",
    "password": "YOUR_PASSWORD",
    "account": "YOUR_ACCOUNT",
    "warehouse": "YOUR_WAREHOUSE",
    "database": "YOUR_DATABASE",
    "schema": "YOUR_SCHEMA",
    "table": "BASKETBALL_GAMES"
} # don't want to give my credentials to hackers so easily... at least they should take me to dinner first.



# DATA GENERATION

def generate_synthetic_data(num_games=500):
    """Generate synthetic basketball game data."""
    print(f"Generating {num_games} synthetic basketball games...")

    data = []
    for i in range(num_games):
        # Generate features with some correlation to outcome
        home_ppg = np.random.uniform(65, 95)
        away_ppg = np.random.uniform(65, 95)
        home_fg_pct = np.random.uniform(0.40, 0.55)
        away_fg_pct = np.random.uniform(0.40, 0.55)
        home_rebounds = np.random.uniform(30, 50)
        away_rebounds = np.random.uniform(30, 50)
        home_assists = np.random.uniform(12, 25)
        away_assists = np.random.uniform(12, 25)
        home_turnovers = np.random.uniform(8, 18)
        away_turnovers = np.random.uniform(8, 18)
        # FBI type of calculation

        # Calculate a "strength" score to determine winner
        home_strength = (home_ppg * 0.3 + home_fg_pct * 100 +
                         home_rebounds * 0.5 + home_assists * 0.8 -
                         home_turnovers * 0.5)
        away_strength = (away_ppg * 0.3 + away_fg_pct * 100 +
                         away_rebounds * 0.5 + away_assists * 0.8 -
                         away_turnovers * 0.5)

        # Add home court advantage
        home_strength += 3

        # Determine winner (1 = Home Win, 0 = Away Win) with some level of randomness
        if home_strength > away_strength:
            outcome = 1 if np.random.random() > 0.15 else 0 # 85% chance home wins if stronger
        else:
            outcome = 0 if np.random.random() > 0.15 else 1

        game = {
            "game_id": f"GAME_{i + 1:04d}",
            "home_ppg": round(home_ppg, 2),
            "away_ppg": round(away_ppg, 2),
            "home_fg_pct": round(home_fg_pct, 3),
            "away_fg_pct": round(away_fg_pct, 3),
            "home_rebounds": round(home_rebounds, 2),
            "away_rebounds": round(away_rebounds, 2),
            "home_assists": round(home_assists, 2),
            "away_assists": round(away_assists, 2),
            "home_turnovers": round(home_turnovers, 2),
            "away_turnovers": round(away_turnovers, 2),
            # future metrics can be added here
            # remember to update model training accordingly
            "outcome": outcome
        }
        data.append(game)

    print(f"Generated {len(data)} games.")
    return data


# API SIMULATION

def simulate_api_fetch(num_new_games=100):
    """Simulate fetching new season data from an API."""
    print(f"Simulating API fetch of {num_new_games} new games...")

    # In a real system, this would call an external API
    # For this project, we generate new synthetic data
    # Will add future API integration here
    new_data = generate_synthetic_data(num_new_games)

    print(f"Fetched {len(new_data)} new games from API (simulated).")
    return new_data



# STORAGE: LOCAL JSON

# Are ya winning JSON?
def save_to_json(data, filename=LOCAL_DATA_FILE):
    """Save data to a local JSON file."""
    print(f"Saving data to {filename}...")
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} records to {filename}.")


def load_from_json(filename=LOCAL_DATA_FILE):
    """Load data from a local JSON file."""
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return []

    print(f"Loading data from {filename}...")
    with open(filename, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records from {filename}.")
    return data


def append_to_json(new_data, filename=LOCAL_DATA_FILE):
    """Append new data to existing JSON file."""
    existing_data = load_from_json(filename)
    combined_data = existing_data + new_data
    save_to_json(combined_data, filename)
    print(f"Appended {len(new_data)} new records. Total: {len(combined_data)}")



# STORAGE: SNOWFLAKE

# For now it is this way, data storage via API to snowflake will be added in future versions
# still working on it. should API go local or snowflake first? hmmm
# maybe snowflake first, then local backup? for security reasons
# But keeping both options for now
def get_snowflake_connection():
    """Create a Snowflake connection."""
    try:
        import snowflake.connector
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        print("Connected to Snowflake.")
        return conn
    except ImportError:
        print("ERROR: snowflake-connector-python not installed.")
        print("Install with: pip install snowflake-connector-python")
        return None
    except Exception as e:
        print(f"ERROR connecting to Snowflake: {e}")
        return None


def create_snowflake_table(conn):
    """Create the basketball games table in Snowflake if it doesn't exist."""
    cursor = conn.cursor()
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {SNOWFLAKE_CONFIG['table']} (
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
    )
    """
    cursor.execute(create_table_sql)
    print(f"Table {SNOWFLAKE_CONFIG['table']} is ready.")
    cursor.close()


def save_to_snowflake(data):
    """Save data to Snowflake."""
    conn = get_snowflake_connection()
    if not conn:
        return

    create_snowflake_table(conn)
    cursor = conn.cursor()

    # Clear existing data
    cursor.execute(f"DELETE FROM {SNOWFLAKE_CONFIG['table']}")

    # Insert new data
    insert_sql = f"""
    INSERT INTO {SNOWFLAKE_CONFIG['table']} 
    (game_id, home_ppg, away_ppg, home_fg_pct, away_fg_pct, 
     home_rebounds, away_rebounds, home_assists, away_assists, 
     home_turnovers, away_turnovers, outcome)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    for game in data:
        cursor.execute(insert_sql, (
            game['game_id'], game['home_ppg'], game['away_ppg'],
            game['home_fg_pct'], game['away_fg_pct'],
            game['home_rebounds'], game['away_rebounds'],
            game['home_assists'], game['away_assists'],
            game['home_turnovers'], game['away_turnovers'],
            game['outcome']
        ))

    conn.commit()
    print(f"Saved {len(data)} records to Snowflake.")
    cursor.close()
    conn.close()


def load_from_snowflake():
    """Load data from Snowflake."""
    conn = get_snowflake_connection()
    if not conn:
        return []

    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {SNOWFLAKE_CONFIG['table']}")

    columns = [col[0].lower() for col in cursor.description]
    data = []

    for row in cursor:
        game = dict(zip(columns, row))
        data.append(game)

    print(f"Loaded {len(data)} records from Snowflake.")
    cursor.close()
    conn.close()
    return data


def append_to_snowflake(new_data):
    """Append new data to Snowflake."""
    conn = get_snowflake_connection()
    if not conn:
        return

    create_snowflake_table(conn)
    cursor = conn.cursor()

    insert_sql = f"""
    INSERT INTO {SNOWFLAKE_CONFIG['table']} 
    (game_id, home_ppg, away_ppg, home_fg_pct, away_fg_pct, 
     home_rebounds, away_rebounds, home_assists, away_assists, 
     home_turnovers, away_turnovers, outcome)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    for game in new_data:
        cursor.execute(insert_sql, (
            game['game_id'], game['home_ppg'], game['away_ppg'],
            game['home_fg_pct'], game['away_fg_pct'],
            game['home_rebounds'], game['away_rebounds'],
            game['home_assists'], game['away_assists'],
            game['home_turnovers'], game['away_turnovers'],
            game['outcome']
        ))

    conn.commit()
    print(f"Appended {len(new_data)} new records to Snowflake.")
    cursor.close()
    conn.close()



# MODEL TRAINING

# 3 models: Logistic Regression, Random Forest, Linear Regression (thresholded)
# problem is binary classification, so linear regression will be thresholded at 0.5
# another problem: 3 models to compare but one being pkl-ed, simultaneous calc required. or 9-10 times linear is chosen?
def prepare_data(data):
    """Convert data to numpy arrays for training."""
    features = []
    labels = []

    feature_names = [
        'home_ppg', 'away_ppg', 'home_fg_pct', 'away_fg_pct',
        'home_rebounds', 'away_rebounds', 'home_assists', 'away_assists',
        'home_turnovers', 'away_turnovers'
    ]

    for game in data:
        feature_vector = [game[fname] for fname in feature_names]
        features.append(feature_vector)
        labels.append(game['outcome'])

    return np.array(features), np.array(labels)


def train_and_evaluate_models(storage_mode):
    """Train all three models and select the best one."""
    print("\n" + "=" * 70)
    print("TRAINING AND EVALUATION")
    print("=" * 70)

    # Load data
    if storage_mode == 'local':
        data = load_from_json()
    else:
        data = load_from_snowflake()

    if len(data) == 0:
        print("ERROR: No data available. Run --generate first.")
        return

    # Prepare data
    ex, y = prepare_data(data)
    ex_train, ex_test, y_train, y_test = train_test_split(
        ex, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {len(ex_train)} samples")
    print(f"Test set: {len(ex_test)} samples")
    print()

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Linear Regression (Thresholded)': LinearRegression()
    }

    results = {}

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(ex_train, y_train)

        # Make predictions
        if model_name == 'Linear Regression (Thresholded)':
            # Convert continuous output to binary classification
            y_pred_continuous = model.predict(ex_test)
            y_pred = (y_pred_continuous >= 0.5).astype(int)
        else:
            y_pred = model.predict(ex_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print()

    # Select best model based on F1-score
    # linear regression may be chosen more often due to thresholding, consider that in future versions.
    # all models are different in nature, so f1 may not be the best metric for all.
    best_model_name = max(results, key=lambda k: results[k]['f1'])
    best_model = results[best_model_name]['model']

    print("=" * 70)
    print(f"BEST MODEL: {best_model_name}")
    print(f"F1-Score: {results[best_model_name]['f1']:.4f}")
    print("=" * 70)
    print()

    # Save best model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(best_model, f)  # type: ignore

    # Save model info
    model_info = {
        'model_name': best_model_name,
        'accuracy': results[best_model_name]['accuracy'],
        'precision': results[best_model_name]['precision'],
        'recall': results[best_model_name]['recall'],
        'f1': results[best_model_name]['f1'],
        'trained_at': datetime.now().isoformat()
    }

    with open(MODEL_INFO_FILE, 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"Model saved to {MODEL_FILE}")
    print(f"Model info saved to {MODEL_INFO_FILE}")

    # Save model comparison data for dashboard
    comparison_data = {}
    for model_name, result in results.items():
        comparison_data[model_name] = {
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1']
        }

    with open(MODEL_COMPARISON_FILE, 'w') as f:
        json.dump(comparison_data, f, indent=2)

    print(f"Model comparison saved to {MODEL_COMPARISON_FILE}")



# PREDICTION SERVER

# dashboard will be a simple HTML file served at root
# API endpoint /predict will accept JSON input and return prediction
app = Flask(__name__)


def load_model():
    """Load the trained model."""
    if not os.path.exists(MODEL_FILE):
        return None, None

    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)

    if os.path.exists(MODEL_INFO_FILE):
        with open(MODEL_INFO_FILE, 'r') as f:
            model_info = json.load(f)
    else:
        model_info = {'model_name': 'Unknown'}

    return model, model_info


@app.route('/')
def home():
    """Serve the dashboard."""
    return send_file('dashboard.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction."""
    model, model_info = load_model()

    if model is None:
        return jsonify({
            'error': 'No trained model found. Run --train first.'
        }), 400

    try:
        data = request.json

        # Extract features
        features = [
            float(data['home_ppg']),
            float(data['away_ppg']),
            float(data['home_fg_pct']),
            float(data['away_fg_pct']),
            float(data['home_rebounds']),
            float(data['away_rebounds']),
            float(data['home_assists']),
            float(data['away_assists']),
            float(data['home_turnovers']),
            float(data['away_turnovers'])
        ]

        ex = np.array([features])

        # Make prediction
        if 'Linear Regression' in model_info['model_name']:
            prediction_continuous = model.predict(ex)[0]
            prediction = int(prediction_continuous >= 0.5)
            confidence = abs(prediction_continuous - 0.5) * 2  # Scale to 0-1
        else:
            prediction = int(model.predict(ex)[0])

            # Get probability if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(ex)[0]
                confidence = float(max(probabilities))
            else:
                confidence = None

        result = {
            'prediction': 'Home Win' if prediction == 1 else 'Away Win',
            'prediction_value': prediction,
            'confidence': confidence,
            'model_name': model_info['model_name']
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/analytics', methods=['GET'])
def get_analytics():
    """Get analytics data for dashboard visualizations."""
    try:
        # Load data from local JSON only
        if not os.path.exists(LOCAL_DATA_FILE):
            return jsonify({'error': 'No data available. Run: python main.py --generate --storage local'}), 400

        with open(LOCAL_DATA_FILE, 'r') as f:
            data = json.load(f)

        if len(data) == 0:
            return jsonify({'error': 'Data file is empty'}), 400

        # Calculate statistics
        total_games = len(data)
        home_wins = sum(1 for game in data if game['outcome'] == 1)
        away_wins = total_games - home_wins
        home_win_rate = home_wins / total_games if total_games > 0 else 0

        # Calculate feature averages by outcome
        home_win_games = [game for game in data if game['outcome'] == 1]
        away_win_games = [game for game in data if game['outcome'] == 0]

        features = ['home_ppg', 'away_ppg', 'home_fg_pct', 'away_fg_pct',
                    'home_rebounds', 'away_rebounds', 'home_assists', 'away_assists',
                    'home_turnovers', 'away_turnovers']

        feature_stats = {'home_win': {}, 'away_win': {}}

        for feature in features:
            if home_win_games:
                feature_stats['home_win'][feature] = round(
                    sum(g[feature] for g in home_win_games) / len(home_win_games), 2
                )
            else:
                feature_stats['home_win'][feature] = 0

            if away_win_games:
                feature_stats['away_win'][feature] = round(
                    sum(g[feature] for g in away_win_games) / len(away_win_games), 2
                )
            else:
                feature_stats['away_win'][feature] = 0

        # Load model comparison data
        model_comparison = {}
        if os.path.exists(MODEL_COMPARISON_FILE):
            with open(MODEL_COMPARISON_FILE, 'r') as f:
                model_comparison = json.load(f)
        else:
            # Placeholder data
            model_comparison = {
                'Logistic Regression': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0},
                'Random Forest': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0},
                'Linear Regression (Thresholded)': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
            }

        return jsonify({
            'total_games': total_games,
            'home_wins': home_wins,
            'away_wins': away_wins,
            'home_win_rate': round(home_win_rate, 4),
            'feature_stats': feature_stats,
            'model_comparison': model_comparison
        })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Analytics error: {error_details}")
        return jsonify({'error': str(e)}), 400


@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get information about the current model."""
    _, model_info = load_model()

    if model_info is None:
        return jsonify({'error': 'No model info available'}), 400

    return jsonify(model_info)



# MAIN

# Command-line interface to run different parts of the system
# Various options: generate data, fetch API data, train models, start server
# improve CLI in future versions with subcommands but for now this is sufficient
# inefficient but works for now
def main():
    parser = argparse.ArgumentParser(
        description='Basketball Game Outcome Prediction System'
    )
    parser.add_argument(
        '--generate',
        action='store_true',
        help='Generate initial synthetic data'
    )
    parser.add_argument(
        '--fetch-api',
        action='store_true',
        help='Fetch/simulate new season data and append'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train and evaluate all models, select best'
    )
    parser.add_argument(
        '--serve',
        action='store_true',
        help='Start the prediction web server'
    )
    parser.add_argument(
        '--storage',
        choices=['local', 'snowflake'],
        default='local',
        help='Storage backend (local JSON or Snowflake)'
    )

    args = parser.parse_args()

    if args.generate:
        data = generate_synthetic_data(num_games=500)
        if args.storage == 'local':
            save_to_json(data)
        else:
            save_to_snowflake(data)

    elif args.fetch_api:
        new_data = simulate_api_fetch(num_new_games=100)
        if args.storage == 'local':
            append_to_json(new_data)
        else:
            append_to_snowflake(new_data)

    elif args.train:
        train_and_evaluate_models(args.storage)

    elif args.serve:
        print("\n" + "=" * 70)
        print("STARTING PREDICTION SERVER")
        print("=" * 70)
        print("Dashboard available at: http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        print("=" * 70 + "\n") # start server and listen on port 5000
        app.run(debug=True, port=5000)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
# Current status: "Monke see monke do" level.
# live data to make monke smart is remaining.

# coffee log 4 -> 5