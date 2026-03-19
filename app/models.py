"""
app/models.py
─────────────
Everything related to model lifecycle:
  - Registry (version tracking, pickle I/O)
  - Learning log (auto-learn history)
  - build_models()       — construct all enabled sklearn pipelines
  - train_and_evaluate() — full training run with validation and promotion logic
  - compute_metrics()    — accuracy, F1, ROC-AUC, confusion matrix
  - get_feature_importances()

call me gustav for I am cooking here.
"""

import json
import pickle
import hashlib
import shutil
import warnings
from datetime import datetime

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from app.config import DATA_CFG, MODEL_CFG, AL_CFG, MODELS_DIR, REGISTRY_FILE, LEARN_LOG
from app.logger import get_logger
from app.storage import load_data, _sanitize
from app.preprocessing import prepare_data, _validate_training_data, _adaptive_depth

warnings.filterwarnings("ignore")
log = get_logger(__name__)


# ── MODEL REGISTRY ────────────────────────────────────────────────────────────

def load_registry() -> dict:  # this is a modeling registry not a modeling agency
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"versions": [], "active_version": None}


def _save_registry(reg: dict):
    REGISTRY_FILE.parent.mkdir(exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)


def register_model(
    model_name:    str,
    model_obj,
    metrics:       dict,
    feature_names: list,
    training_size: int,
) -> str:
    reg      = load_registry()
    existing = [int(v["version"].lstrip("v")) for v in reg["versions"]]
    ver_num  = (max(existing) + 1) if existing else 1
    version  = f"v{ver_num}"

    model_hash = hashlib.md5(pickle.dumps(model_obj)).hexdigest()[:8]
    filename   = f"{model_name.lower().replace(' ','_')}_{version}_{model_hash}.pkl"

    model_path = MODELS_DIR / filename
    model_path.write_bytes(
        pickle.dumps({"model": model_obj, "feature_names": feature_names})
    )
    # all my lovely models are supposed to be well documented
    entry = {
        "version":       version,
        "model_name":    model_name,
        "filename":      filename,
        "metrics":       metrics,
        "feature_names": feature_names,
        "training_size": training_size,
        "trained_at":    datetime.now().isoformat(),
        "hash":          model_hash,
    }
    reg["versions"].append(entry)
    reg["active_version"] = version

    keep = MODEL_CFG.get("keep_top_n", 10)
    if len(reg["versions"]) > keep:
        for old in reg["versions"][:-keep]:
            p = MODELS_DIR / old["filename"]
            if p.exists():
                p.unlink()
        reg["versions"] = reg["versions"][-keep:]

    _save_registry(reg)
    log.info("[Registry] %s → %s", model_name, version)
    return version


def load_active_model():
    reg = load_registry()
    if not reg["active_version"] or not reg["versions"]:
        return None, None
    entry = next(
        (v for v in reg["versions"] if v["version"] == reg["active_version"]), None
    )
    if not entry:
        return None, None
    path = MODELS_DIR / entry["filename"]
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], entry


def set_active_version(version: str) -> bool:
    reg = load_registry()
    if any(v["version"] == version for v in reg["versions"]):
        reg["active_version"] = version
        _save_registry(reg)
        return True
    return False


# ── LEARNING LOG   got to log the EE hours somehow.... ───────────────────────

def _load_log() -> list:
    if LEARN_LOG.exists():
        with open(LEARN_LOG, encoding="utf-8") as f:
            return json.load(f)
    return []


def _append_log(entry: dict):
    log_data = _load_log()
    log_data.append(entry)
    LEARN_LOG.parent.mkdir(exist_ok=True)
    with open(LEARN_LOG, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)


# ── MODEL BUILDER ─────────────────────────────────────────────────────────────

def build_models(n_samples: int = 1000, n_features: int = 14) -> dict:
    """
    Build all enabled model pipelines with adaptive depth scaling.
    n_samples / n_features are used to cap tree depth for small datasets
    to prevent memorisation.
    """
    enabled = MODEL_CFG.get("enabled", [])
    mc      = MODEL_CFG
    pipe    = {}

    if "gradient_boosting" in enabled:
        c     = mc.get("gradient_boosting", {})
        depth = _adaptive_depth(c.get("max_depth", 4), n_samples, n_features)
        pipe["Gradient Boosting"] = Pipeline([
            ("s",   StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators     = c.get("n_estimators",    300),
                learning_rate    = c.get("learning_rate",   0.05),
                max_depth        = depth,
                subsample        = c.get("subsample",       0.8),
                min_samples_split= c.get("min_samples_split", 5),
                min_samples_leaf = c.get("min_samples_leaf",  2),
                random_state     = c.get("random_state",    42),
            )),
        ])

    if "random_forest" in enabled:
        c     = mc.get("random_forest", {})
        depth = _adaptive_depth(c.get("max_depth", 10), n_samples, n_features)
        pipe["Random Forest"] = Pipeline([
            ("s",   StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators     = c.get("n_estimators",    300),
                max_depth        = depth,
                min_samples_split= c.get("min_samples_split", 5),
                min_samples_leaf = c.get("min_samples_leaf",  2),
                random_state     = c.get("random_state",    42),
            )),
        ])

    if "extra_trees" in enabled:
        c     = mc.get("extra_trees", {})
        depth = _adaptive_depth(c.get("max_depth", 10), n_samples, n_features)
        pipe["Extra Trees"] = Pipeline([
            ("s",   StandardScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators     = c.get("n_estimators",    300),
                max_depth        = depth,
                min_samples_split= c.get("min_samples_split", 5),
                min_samples_leaf = c.get("min_samples_leaf",  2),
                random_state     = c.get("random_state",    42),
            )),
        ])

    if "svm" in enabled:
        c = mc.get("svm", {})
        pipe["SVM (RBF)"] = Pipeline([
            ("s",   StandardScaler()),
            ("clf", SVC(
                kernel      = c.get("kernel",  "rbf"),
                C           = c.get("C",        2.0),
                gamma       = c.get("gamma",    "scale"),
                probability = True,
                random_state= c.get("random_state", 42),
            )),
        ])

    if "mlp" in enabled:
        c = mc.get("mlp", {})
        pipe["Neural Network (MLP)"] = Pipeline([
            ("s",   StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes  = tuple(c.get("hidden_layer_sizes", [128, 64, 32])),
                activation          = c.get("activation",         "relu"),
                max_iter            = c.get("max_iter",            500),
                early_stopping      = True,
                validation_fraction = c.get("validation_fraction", 0.15),
                random_state        = c.get("random_state",        42),
            )),
        ])

    if "xgboost" in enabled:
        try:
            from xgboost import XGBClassifier  # noqa: PLC0415
            c     = mc.get("xgboost", {})
            depth = _adaptive_depth(c.get("max_depth", 5), n_samples, n_features)
            pipe["XGBoost"] = Pipeline([
                ("s",   StandardScaler()),
                ("clf", XGBClassifier(
                    n_estimators     = c.get("n_estimators",    300),
                    learning_rate    = c.get("learning_rate",   0.05),
                    max_depth        = depth,
                    subsample        = c.get("subsample",       0.8),
                    colsample_bytree = c.get("colsample_bytree",0.8),
                    min_child_weight = c.get("min_child_weight", 3),
                    eval_metric      = "logloss",
                    random_state     = c.get("random_state",    42),
                    verbosity        = 0,
                )),
            ])
        except ImportError:
            XGBClassifier = None  # not installed — skip gracefully, satisfy type checker
            log.warning("[Models] XGBoost not installed — skipping.")

    return pipe  # I do not know what I am cooking here


# ── METRICS ───────────────────────────────────────────────────────────────────

def compute_metrics(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    m = {
        "accuracy":         round(accuracy_score (y_test, y_pred),                    4),
        "precision":        round(precision_score(y_test, y_pred, zero_division=0),   4),
        "recall":           round(recall_score   (y_test, y_pred, zero_division=0),   4),
        "f1":               round(f1_score       (y_test, y_pred, zero_division=0),   4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    if hasattr(model, "predict_proba"):
        m["roc_auc"] = round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 4)
    else:
        m["roc_auc"] = m["accuracy"]
    return m


def get_feature_importances(model, feature_names: list) -> dict | None:
    clf = model.named_steps.get("clf")
    if clf is None:
        return None
    if hasattr(clf, "feature_importances_"):
        imps = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imps = np.abs(clf.coef_[0]) if clf.coef_.ndim > 1 else np.abs(clf.coef_)
    else:
        return None
    return dict(zip(feature_names, [round(float(v), 6) for v in imps]))


# ── TRAINING ──────────────────────────────────────────────────────────────────

def train_and_evaluate(storage: str = "local", triggered_by: str = "manual"):
    log.info("=" * 70)
    log.info("TRAINING (%s)", triggered_by)
    log.info("=" * 70)

    data = load_data(storage)
    if len(data) < DATA_CFG.get("min_games_required", 50):
        log.warning("[Train] Not enough data (%d). Skipping.", len(data))
        return None

    X, y, feature_names = prepare_data(data)
    _validate_training_data(X, y, feature_names)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size    = DATA_CFG["test_size"],
        random_state = DATA_CFG["random_state"],
        stratify     = y,
    )
    log.info("Dataset: %d games  Train: %d  Test: %d", len(data), len(X_tr), len(X_te))
    log.info("Home win rate: %.2f%%", y.mean() * 100)

    models     = build_models(n_samples=len(X_tr), n_features=X.shape[1])
    results    = {}
    sel_metric = MODEL_CFG.get("selection_metric", "roc_auc")

    for name, model in models.items():
        log.info("▶ %s...", name)
        model.fit(X_tr, y_tr)
        m  = compute_metrics(model, X_te, y_te)
        fi = get_feature_importances(model, feature_names)
        if fi:
            m["feature_importances"] = fi
        cv = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
        m["cv_roc_auc_mean"] = round(float(cv.mean()), 4)
        m["cv_roc_auc_std"]  = round(float(cv.std()),  4)
        results[name] = {"model": model, "metrics": m}
        log.info(
            "  Acc:%.4f  F1:%.4f  AUC:%.4f  CV-AUC:%.4f±%.4f",
            m["accuracy"], m["f1"], m["roc_auc"],
            m["cv_roc_auc_mean"], m["cv_roc_auc_std"],
        )
        # ........................ bruh

    best_name = max(results, key=lambda k: results[k]["metrics"].get(sel_metric, 0))
    best      = results[best_name]
    log.info("=" * 70)
    log.info("BEST: %s  %s=%.4f", best_name, sel_metric,
             best["metrics"].get(sel_metric, 0))
    log.info("=" * 70)

    best_auc = best["metrics"].get("roc_auc", 0)
    if best_auc > 0.80:
        log.warning("[Train] AUC=%.4f is high. Are records enriched with pre-game averages?", best_auc)
        log.warning("[Train] If not, run: python main.py --enrich  then retrain.")
    elif best_auc < 0.52:
        log.info("[Train] AUC=%.4f barely above chance. Fetch more data.", best_auc)
    else:
        log.info("[Train] ✓  AUC=%.4f — honest pre-game prediction range.", best_auc)

    if triggered_by != "manual":
        _, active_entry = load_active_model()
        threshold = AL_CFG.get("promote_threshold", 0.002)
        if active_entry:
            current_auc = active_entry["metrics"].get("roc_auc", 0)
            new_auc     = best["metrics"].get("roc_auc", 0)
            if new_auc < current_auc + threshold:
                msg = (f"New AUC {new_auc:.4f} vs current {current_auc:.4f} "
                       f"(threshold +{threshold}). Skipping promotion.")
                log.info("[AutoLearn] %s", msg)
                _append_log({
                    "timestamp":    datetime.now().isoformat(),
                    "triggered_by": triggered_by,
                    "result":       "skipped",
                    "reason":       msg,
                    "best_model":   best_name,
                    "new_auc":      new_auc,
                    "current_auc":  current_auc,
                    "dataset_size": len(data),
                })
                return None  # don't promote, but still log the attempt

    version = register_model(
        model_name    = best_name,
        model_obj     = best["model"],
        metrics       = best["metrics"],
        feature_names = feature_names,
        training_size = len(X_tr),
    )

    comparison = {
        n: {k: v for k, v in r["metrics"].items() if k != "feature_importances"}
        for n, r in results.items()
    }
    snap = {
        "version":             version,
        "trained_at":          datetime.now().isoformat(),
        "best_model":          best_name,
        "selection_metric":    sel_metric,
        "results":             comparison,
        "feature_importances": {
            n: r["metrics"].get("feature_importances", {}) for n, r in results.items()
        },
        "triggered_by": triggered_by,
        "dataset_size":  len(data),
    }
    comp_file = MODELS_DIR / f"comparison_{version}.json"
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(_sanitize(snap), f, indent=2)
    shutil.copy(comp_file, MODELS_DIR / "latest_comparison.json")

    _append_log({
        "timestamp":    datetime.now().isoformat(),
        "triggered_by": triggered_by,
        "result":       "promoted",
        "version":      version,
        "model_name":   best_name,
        "roc_auc":      best["metrics"].get("roc_auc", 0),
        "f1":           best["metrics"]["f1"],
        "accuracy":     best["metrics"]["accuracy"],
        "dataset_size": len(data),
    })
    log.info("[Train] Registered & promoted → %s", version)
    return {"version": version, "model_name": best_name, "metrics": best["metrics"]}