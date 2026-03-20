"""
main.py
───────
CLI entry point. Parses arguments and delegates to the appropriate module.
This file should stay thin — business logic lives in app/.

Available commands:
  --fetch             Fetch real NCAA games from ESPN (multi-season)
  --enrich            Back-fill pre-game rolling averages into existing games.json
  --fetch-rosters     Pre-fetch rosters for all teams in dataset
  --generate-synthetic Generate synthetic fallback data (5 000 games)
  --train             Train all models, register best
  --list-models       Print registered model versions
  --activate VERSION  Promote a specific version to active
  --serve             Start Flask server + auto-learn scheduler
  --max-games N       Override config max_games cap for one fetch run
  --storage           local (default) | snowflake

v2.5 recommended workflow for existing data:
  python main.py --enrich        ← back-fill pre-game averages (once)
  python main.py --train         ← retrain on honest data
  python main.py --serve         ← dashboard at http://localhost:5000
"""

import argparse
from pathlib import Path
import threading

# Setup logging FIRST — before any app module is imported so every
# module's get_logger() call attaches to the already-configured handler.
from app.logger import setup_logging
setup_logging()

from app.config    import APP_CFG, HT_CFG, AL_CFG, API_CFG, DATA_CFG
from app.enrichment import enrich_with_pregame_averages
from app.fetcher   import fetch_ncaa_data
from app.models    import train_and_evaluate, load_registry, set_active_version
from app.roster    import RosterFetcher
from app.storage   import (
    load_from_json, save_to_json, append_to_json,
    save_to_snowflake, append_to_snowflake,
)
from app.api       import app, scheduler
from app.logger    import get_logger

log = get_logger(__name__)


# ── SYNTHETIC FALLBACK ────────────────────────────────────────────────────────
# you fools, you thought depriving me of my API access would stop me?
# You have activated my trap card.
# I summon from my deck: THE SYNTHETIC DATA GENERATOR!

def _generate_synthetic(num_games: int = 5000):
    import numpy as np

    rng       = np.random.default_rng(42)
    home_name = HT_CFG["name"]
    pool      = [home_name] + [f"Team_{i}" for i in range(1, 65)]
    data      = []

    for i in range(num_games):
        ht = pool[i % len(pool)]
        at = pool[(i * 7 + 3) % len(pool)]
        if ht == at:
            at = pool[(i * 7 + 4) % len(pool)]

        hp, ap   = rng.uniform(60, 95, 2)
        hfg, afg = rng.uniform(0.38, 0.55, 2)
        hrb, arb = rng.uniform(28, 48, 2)
        ha, aa   = rng.uniform(10, 22, 2)
        ht_, at_ = rng.uniform(8,  17, 2)
        hst, ast = rng.uniform(4,  10, 2)
        hbl, abl = rng.uniform(2,   8, 2)

        # Outcome driven by efficiency, not scores — same logic as real enrichment
        h_str = hfg*100 + hrb*0.5 + ha*0.8 - ht_*0.6 + hst*0.4 + hbl*0.3 + 3
        a_str = afg*100 + arb*0.5 + aa*0.8 - at_*0.6 + ast*0.4 + abl*0.3
        outcome = (1 if h_str > a_str else 0) if rng.random() > 0.15 else (0 if h_str > a_str else 1)

        game = {
            "game_id":   f"SYN_{i+1:05d}",
            "game_date": f"2024-{(i//90)+1:02d}-{(i%28)+1:02d}",
            "home_team": ht, "away_team": at,
            "home_ppg":       round(float(hp),  2),  "away_ppg":       round(float(ap),  2),
            "home_fg_pct":    round(float(hfg), 4),  "away_fg_pct":    round(float(afg), 4),
            "home_rebounds":  round(float(hrb), 2),  "away_rebounds":  round(float(arb), 2),
            "home_assists":   round(float(ha),  2),  "away_assists":   round(float(aa),  2),
            "home_turnovers": round(float(ht_), 2),  "away_turnovers": round(float(at_), 2),
            "home_steals":    round(float(hst), 2),  "away_steals":    round(float(ast), 2),
            "home_blocks":    round(float(hbl), 2),  "away_blocks":    round(float(abl), 2),
            "home_ast_to_tov": round(float(ha)  / (float(ht_) if float(ht_) > 0 else 0.1), 3),
            "away_ast_to_tov": round(float(aa)  / (float(at_) if float(at_) > 0 else 0.1), 3),
            "home_eff_score":  round(float(hp)  * float(hfg), 2),
            "away_eff_score":  round(float(ap)  * float(afg), 2),
            "outcome": int(outcome), "source": "synthetic",
            "pregame_enriched": False,
        }
        data.append(game)

    log.info("[Synthetic] Generated %d games. Running enrich pipeline...", len(data))
    window    = DATA_CFG.get("pregame_window",    10)
    min_games = DATA_CFG.get("pregame_min_games",  1)
    enriched  = enrich_with_pregame_averages(data, window=window, min_games=min_games)
    log.info("[Synthetic] %d enriched synthetic games ready.", len(enriched))
    return enriched  # very linear data — but at least it's honest now


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Basketball Predictor v2.5")
    parser.add_argument("--fetch",            action="store_true",
                        help="Fetch real NCAA games from ESPN (multi-season)")
    parser.add_argument("--enrich",           action="store_true",
                        help="Back-fill pre-game rolling averages into existing games.json. "
                             "Run once after upgrading to v2.5, then retrain.")
    parser.add_argument("--fetch-rosters",    action="store_true",
                        help="Pre-fetch rosters for all teams in dataset")
    parser.add_argument("--generate-synthetic", action="store_true",
                        help="Generate synthetic fallback data (5 000 games)")
    parser.add_argument("--train",            action="store_true",
                        help="Train all models, register best")
    parser.add_argument("--serve",            action="store_true",
                        help="Start Flask server + auto-learn scheduler")
    parser.add_argument("--storage",          choices=["local","snowflake"], default="local")
    parser.add_argument("--activate",         metavar="VERSION")
    parser.add_argument("--list-models",      action="store_true")
    parser.add_argument("--config",           default="config.yaml")
    parser.add_argument("--max-games",        type=int, default=None,
                        help="Override config max_games cap for this fetch run")
    args = parser.parse_args()
    # I love you

    # ── list-models ──────────────────────────────────────────────────────────
    if args.list_models:
        reg = load_registry()
        if not reg["versions"]:
            print("No registered models.")
            return
        print(f"\n{'Ver':<6} {'Model':<28} {'AUC':<8} {'F1':<8} Trained At")
        print("-" * 70)
        for v in reg["versions"]:
            m      = v["metrics"]
            active = " ◀ ACTIVE" if v["version"] == reg["active_version"] else ""
            print(f"{v['version']:<6} {v['model_name']:<28} "
                  f"{m.get('roc_auc',0):.4f}   {m.get('f1',0):.4f}   "
                  f"{v['trained_at'][:19]}{active}")
        return

    # ── activate ─────────────────────────────────────────────────────────────
    if args.activate:
        ok = set_active_version(args.activate)
        print(f"Active → {args.activate}" if ok else f"Version {args.activate} not found.")
        return

    # ── enrich ───────────────────────────────────────────────────────────────
    if args.enrich:
        # Back-fill pre-game rolling averages into an existing games.json.
        # Makes your existing games training-ready without re-fetching.
        # Games are sorted by ESPN ID (sequential = chronological) as proxy
        # for game_date when older records lack that field.
        # Run once, then retrain: python main.py --enrich && python main.py --train
        data = load_from_json()
        if not data:
            log.warning("[Enrich] No data found. Run --fetch first.")
            return
        already = sum(1 for g in data if g.get("pregame_enriched") is True)
        log.info("[Enrich] %d games loaded. %d already enriched.", len(data), already)
        window    = DATA_CFG.get("pregame_window",    10)
        min_games = DATA_CFG.get("pregame_min_games",  1)
        enriched  = enrich_with_pregame_averages(data, window=window, min_games=min_games)
        save_to_json(enriched)
        new_enriched = sum(1 for g in enriched if g.get("pregame_enriched") is True)
        log.info("[Enrich] Done. %d/%d games now have pre-game features.",
                 new_enriched, len(enriched))
        log.info("[Enrich] Now run: python main.py --train")
        return

    # ── fetch ────────────────────────────────────────────────────────────────
    if args.fetch:
        games = fetch_ncaa_data(max_games=args.max_games)
        if games:
            if args.storage == "snowflake":
                append_to_snowflake(games)
            else:
                append_to_json(games)
        return

    # ── fetch-rosters ────────────────────────────────────────────────────────
    if args.fetch_rosters:
        data = load_from_json()
        if not data:
            log.warning("[Roster] No game data found. Run --fetch first.")
            return
        teams_seen: set = set()
        for g in data:
            teams_seen.add(g.get("home_team", "").strip())
            teams_seen.add(g.get("away_team", "").strip())
        teams_seen.discard("")
        log.info("[Roster] Pre-fetching rosters for %d teams...", len(teams_seen))
        fetcher = RosterFetcher()
        ok = fail = 0
        for name in sorted(teams_seen):
            result = fetcher.fetch_team(name)
            if result: ok   += 1
            else:      fail += 1
        log.info("[Roster] Done. Success: %d  Failed: %d", ok, fail)
        return

    # ── generate-synthetic ───────────────────────────────────────────────────
    if args.generate_synthetic:
        data = _generate_synthetic(5000)
        if args.storage == "snowflake":
            save_to_snowflake(data)
        else:
            save_to_json(data)
        return

    # ── train ────────────────────────────────────────────────────────────────
    if args.train:
        train_and_evaluate(args.storage, triggered_by="manual")
        return

    # ── serve ────────────────────────────────────────────────────────────────
    if args.serve:

        # Render (and other cloud platforms) wipe the disk on every cold start.
        # If no data exists, bootstrap with synthetic data and train immediately
        # so the dashboard is never served with a missing model.
        # On local runs the files will already be there and this is skipped.
        if not Path("data/games.json").exists():
            def _bootstrap():
                log.info("[Startup] No data found — generating synthetic data for cold start...")
                data = _generate_synthetic(2000)
                save_to_json(data)
                log.info("[Startup] Synthetic data ready. Training initial model...")
                train_and_evaluate(args.storage, triggered_by="startup")
                log.info("[Startup] Bootstrap complete.")
            threading.Thread(target=_bootstrap, daemon=True).start()

        seasons = API_CFG.get("seasons", [API_CFG.get("season", 2024)])
        log.info("=" * 70)
        log.info("  %s  v%s", APP_CFG["name"], APP_CFG["version"])
        log.info("  Home team  : %s", HT_CFG["name"])
        log.info("  Auto-learn : %s", "ON" if AL_CFG.get("enabled") else "OFF")
        log.info("  Seasons    : %s  (cap: %d games)", seasons, API_CFG.get("max_games", 3000))
        log.info("  Pre-game window: %d games", DATA_CFG.get("pregame_window", 10))
        log.info("  Log file   : data/app.log  (10 MB × 2 backups)")
        log.info("=" * 70)
        scheduler.storage = args.storage
        scheduler.start()

        # PORT: Render injects the PORT env var. Falls back to config for local use.
        import os
        port = int(os.environ.get("PORT", APP_CFG.get("port", 5000)))
        app.run(
            debug        = False,       # never debug mode in production
            port         = port,
            host         = "0.0.0.0",
            use_reloader = False,
        )
        return # at your service milord

    parser.print_help()


if __name__ == "__main__":
    main()

# Current status: v2.5 — modular, logged, production-ish
# main.py is now ~200 lines. used to be 2000.
# each module has one job. no circular imports. no print() anywhere.
# logging goes to data/app.log (10MB rotating) AND console simultaneously.

# coffee log 26 -> 27
