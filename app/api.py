"""
app/api.py
──────────
Flask application and all HTTP routes.

Route groups:
  /                       → dashboard.html
  /predict                → stats-mode prediction
  /predict/from_roster    → roster-mode prediction
  /analytics              → overview stats for dashboard charts
  /model_info             → active model metadata
  /registry               → all registered versions
  /registry/activate/<v>  → promote a version
  /debug                  → system state dump
  /features               → feature list + rolling window config
  /teams                  → all teams with stats
  /team_stats/<name>      → stats for one team
  /home_team              → configured home team stats
  /roster/<name>          → kick off async roster fetch
  /roster/progress/<name> → poll fetch progress
  /roster/refresh/<name>  → force re-fetch
  /autolearn/status       → scheduler state
  /autolearn/trigger      → manual retrain
  /learning_log           → training history

keep the user informed of what's going on in the mysterious training dungeon.
you would not think with how pretty the dashboard looks
the code would be a "pretty princess too".... well shoot call this code buster
and dip it in the oil
"""

import os
import json
import threading

import numpy as np
from flask import Flask, request, jsonify, send_file

from app.config import DATA_CFG, MODELS_DIR, HT_CFG, ROLLING_CFG, API_CFG
from app.logger import get_logger
from app.models import (
    load_active_model, load_registry, set_active_version,
    train_and_evaluate, _load_log,
)
from app.preprocessing import build_team_stats, get_home_team_stats
from app.roster import RosterFetcher, compute_stats_from_roster, _roster_progress
from app.scheduler import AutoLearnScheduler
from app.storage import load_from_json, _sanitize

log = get_logger(__name__)

app  = Flask(__name__)

# Single shared scheduler — started by main.py --serve
scheduler = AutoLearnScheduler()


# ── CORE ──────────────────────────────────────────────────────────────────────
@app.route("/")
def serve_dashboard():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return send_file(os.path.join(root, "dashboard.html"))



@app.route("/predict", methods=["POST"])
def predict():
    """Predict from raw feature values (stats mode)."""
    model, entry = load_active_model()
    if model is None:
        return jsonify({"error": "No trained model. Run --train first."}), 400
    try:
        payload       = request.json
        feature_names = entry.get("feature_names", DATA_CFG["features"])
        X             = np.array([[float(payload[f]) for f in feature_names]])
        pred          = int(model.predict(X)[0])
        conf          = float(max(model.predict_proba(X)[0])) if hasattr(model, "predict_proba") else None
        return jsonify({
            "prediction":       "Home Win" if pred == 1 else "Away Win",
            "prediction_value": pred,
            "confidence":       conf,
            "model_name":       entry["model_name"],
            "version":          entry["version"],
        })
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400
    except Exception as e:
        log.error("[API /predict] %s", e)
        return jsonify({"error": str(e)}), 400


@app.route("/predict/from_roster", methods=["POST"])
def predict_from_roster():
    """
    Predict from selected player lists (roster mode).

    Request body:
      { "home_players": [{ppg, rpg, apg, spg, bpg, tov, fg_pct, fgm, fga}, ...],
        "away_players": [{...}, ...] }

    Player season averages are already pre-game values — no leakage.
    Returns model features (what the model saw) + insights (display context).
    """
    model, entry = load_active_model()
    if model is None:
        return jsonify({"error": "No trained model. Run --train first."}), 400
    try:
        payload      = request.json
        home_players = payload.get("home_players", [])
        away_players = payload.get("away_players", [])

        if not home_players:
            return jsonify({"error": "No home players selected."}), 400
        if not away_players:
            return jsonify({"error": "No away players selected."}), 400

        home_stats = compute_stats_from_roster(home_players, "home")
        away_stats = compute_stats_from_roster(away_players, "away")
        combined   = {**home_stats, **away_stats}

        # Separate model features from insight features — insight_* keys are
        # display context and were never in the training data.
        model_features = {k: v for k, v in combined.items() if not k.startswith("insight_")}
        insights       = {k: v for k, v in combined.items() if k.startswith("insight_")}

        feature_names = entry.get("feature_names", DATA_CFG["features"])
        # Fill any missing features with 0 — shouldn't happen but graceful as F#@K
        X    = np.array([[float(model_features.get(f, 0)) for f in feature_names]])
        pred = int(model.predict(X)[0])
        conf = float(max(model.predict_proba(X)[0])) if hasattr(model, "predict_proba") else None

        return jsonify({
            "prediction":       "Home Win" if pred == 1 else "Away Win",
            "prediction_value": pred,
            "confidence":       conf,
            "model_name":       entry["model_name"],
            "version":          entry["version"],
            "computed_stats":   model_features,
            "insights":         insights,
            "home_count":       len(home_players),
            "away_count":       len(away_players),
        })
    except Exception as e:
        log.error("[API /predict/from_roster] %s", e)
        return jsonify({"error": str(e)}), 400


# ── ANALYTICS ─────────────────────────────────────────────────────────────────

@app.route("/analytics")
def analytics():
    try:
        data = load_from_json()
        if not data:
            return jsonify({"error": "No data."}), 400

        total     = len(data)
        home_wins = sum(1 for g in data if g.get("outcome") == 1)
        cfg_feats = DATA_CFG["features"]
        hw_games  = [g for g in data if g.get("outcome") == 1]
        aw_games  = [g for g in data if g.get("outcome") == 0]

        def avg(games, field):
            v = [g[field] for g in games if field in g]
            return round(sum(v) / len(v), 4) if v else 0

        comp_file = MODELS_DIR / "latest_comparison.json"
        mc = {}
        fi = {}
        if comp_file.exists():
            with open(comp_file, encoding="utf-8") as f:
                c = json.load(f)
            mc = c.get("results", {})
            fi = c.get("feature_importances", {})

        sources = {}
        for g in data:
            src = g.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1

        enriched_count = sum(1 for g in data if g.get("pregame_enriched") is True)

        return jsonify(_sanitize({
            "total_games":     total,
            "home_wins":       home_wins,
            "away_wins":       total - home_wins,
            "home_win_rate":   round(home_wins / total, 4) if total else 0,
            "enriched_games":  enriched_count,
            "enrichment_rate": round(enriched_count / total, 4) if total else 0,
            "feature_stats": {
                "home_win": {feat: avg(hw_games, feat) for feat in cfg_feats},
                "away_win": {feat: avg(aw_games, feat) for feat in cfg_feats},
            },
            "model_comparison":   mc,
            "feature_importances": fi,
            "data_sources":        sources,
        }))
    except Exception as e:
        log.error("[API /analytics] %s", e)
        return jsonify({"error": str(e)}), 500


# ── MODEL INFO / REGISTRY ─────────────────────────────────────────────────────

@app.route("/model_info")
def model_info():
    _, entry = load_active_model()
    if entry is None:
        return jsonify({"error": "No active model."}), 400
    return jsonify(entry)


@app.route("/registry")
def registry():
    return jsonify(load_registry())


@app.route("/registry/activate/<version>", methods=["POST"])
def activate_version(version):
    if set_active_version(version):
        return jsonify({"status": "ok", "active_version": version})
    return jsonify({"error": f"Version {version} not found."}), 404


@app.route("/debug")
def debug():
    data     = load_from_json()
    _, entry = load_active_model()
    enriched = sum(1 for g in data if g.get("pregame_enriched") is True)
    import os
    return jsonify({
        "data_file":          str(DATA_CFG["local_file"]),
        "data_file_exists":   (DATA_CFG["local_file"] and
                               __import__("pathlib").Path(DATA_CFG["local_file"]).exists()),
        "game_count":         len(data),
        "enriched_count":     enriched,
        "enrichment_rate":    f"{enriched/len(data)*100:.1f}%" if data else "0%",
        "active_model":       entry.get("model_name") if entry else None,
        "active_version":     entry.get("version")    if entry else None,
        "configured_seasons": API_CFG.get("seasons", [API_CFG.get("season", 2024)]),
        "max_games_cap":      API_CFG.get("max_games", 3000),
        "pregame_window":     DATA_CFG.get("pregame_window", 10),
        "training_features":  DATA_CFG["features"],
        "cwd":                os.getcwd(),
    })


@app.route("/features")
def features():
    return jsonify({
        "features":        DATA_CFG["features"],
        "rolling_windows": ROLLING_CFG.get("available_windows", [5, 10, 15, 20]),
        "default_window":  ROLLING_CFG.get("default_window", 12),
    })


# ── TEAMS & ROLLING AVERAGES ──────────────────────────────────────────────────
# i know this is a bit of a mess but it was the quickest way to get rolling
# stats without re-processing everything on the dashboard side.
# I promise I'll refactor this into a proper API layer later.

@app.route("/teams")
def teams():
    data = load_from_json()
    if not data:
        return jsonify({"error": "No data."}), 400
    window     = int(request.args["window"]) if "window" in request.args else None
    ts         = build_team_stats(data, window=window)
    teams_list = sorted(
        [{"name": n, **s} for n, s in ts.items() if s.get("games_played", 0) >= 3],
        key=lambda x: x["name"],
    )
    return jsonify({"teams": teams_list, "count": len(teams_list), "window": window})


@app.route("/team_stats/<path:team_name>")
def team_stats(team_name):
    data = load_from_json()
    if not data:
        return jsonify({"error": "No data."}), 400
    window = int(request.args["window"]) if "window" in request.args else None
    ts     = build_team_stats(data, window=window)
    if team_name in ts:
        return jsonify({"name": team_name, "stats": ts[team_name], "window": window})
    matches = [k for k in ts if team_name.lower() in k.lower()]
    if matches:
        return jsonify({"name": matches[0], "stats": ts[matches[0]], "window": window})
    return jsonify({"error": f"Team '{team_name}' not found."}), 404
    # basically to dumb it down for the dashboard search box, which does a simple
    # substring match. This way it handles minor typos without needing an exact match.
    # I know I am awesome


@app.route("/home_team")
def home_team_endpoint():
    data   = load_from_json()
    window = int(request.args["window"]) if "window" in request.args else None
    cfg    = {
        "name":     HT_CFG["name"],
        "court":    HT_CFG.get("court_name", ""),
        "espn_id":  HT_CFG.get("espn_id",   ""),
    }
    ht = get_home_team_stats(data, window=window) if data else None
    return jsonify({"config": cfg, "stats": ht, "window": window})


# ── ROSTERS ───────────────────────────────────────────────────────────────────
# This is a bit more complex due to async fetching and caching of rosters.
# very last moment inspiration here

@app.route("/roster/<path:team_name>")
def get_roster_route(team_name):
    """
    Kicks off a background fetch and returns immediately with status 'loading'.
    Dashboard polls /roster/progress/<team> every second.
    """
    force   = request.args.get("force", "0") == "1"
    fetcher = RosterFetcher()
    fetcher.fetch_team_async(team_name, force=force)
    prog = _roster_progress.get(
        team_name,
        {"status": "loading", "players": [], "done": 0, "total": 0},
    )
    return jsonify(prog)


@app.route("/roster/progress/<path:team_name>")
def roster_progress(team_name):
    """
    Returns current fetch progress.
    { status: "loading"|"ready"|"error", players: [...], done: N, total: M }
    """
    prog = _roster_progress.get(team_name)
    if prog is None:
        return jsonify({"status": "not_started", "players": [], "done": 0, "total": 0})
    return jsonify(prog)


@app.route("/roster/refresh/<path:team_name>", methods=["POST"])
def refresh_roster(team_name):
    """Force a fresh ESPN fetch for a team's roster."""
    fetcher = RosterFetcher()
    fetcher.fetch_team_async(team_name, force=True)
    return jsonify({"status": "started", "team": team_name})


# ── AUTO-LEARN ────────────────────────────────────────────────────────────────

@app.route("/autolearn/status")
def autolearn_status():
    return jsonify(scheduler.get_state())


@app.route("/autolearn/trigger", methods=["POST"])
def autolearn_trigger():
    def _run():
        train_and_evaluate("local", triggered_by="manual_trigger")
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/learning_log")
def learning_log():
    n   = request.args.get("n", 50, type=int)
    log_data = _load_log()
    return jsonify({"log": log_data[-n:], "total": len(log_data)})