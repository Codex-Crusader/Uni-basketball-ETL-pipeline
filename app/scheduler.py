"""
app/scheduler.py
────────────────
AutoLearnScheduler — background thread that fetches new games on a timer
and retrains when enough new data has accumulated.

No clanker of mine will go into this world without the ability to learn
from its mistakes.

Two independent intervals:
  fetch_interval   — how often to check for new games (default 6h)
  retrain_interval — how often to force a full retrain regardless of new games (24h)

Promotion guard: new model must beat the current one by promote_threshold
(default 0.002 ROC-AUC) to be promoted. Small improvements get logged but
not promoted — stops the registry filling with marginal versions.
"""

import threading
import time
import traceback
from datetime import datetime

from app.config import AL_CFG
from app.fetcher import fetch_ncaa_data
from app.logger import get_logger
from app.models import train_and_evaluate
from app.storage import append_to_json

log = get_logger(__name__)


class AutoLearnScheduler:

    def __init__(self, storage: str = "local"):
        self.storage          = storage
        self.fetch_interval   = AL_CFG.get("fetch_interval_hours",    6)  * 3600
        self.retrain_interval = AL_CFG.get("retrain_interval_hours", 24)  * 3600
        self.min_new_games    = AL_CFG.get("min_new_games_to_retrain", 15)
        self._thread:  threading.Thread | None = None
        self._stop     = threading.Event()
        self._last_fetch   = 0.0
        self._last_retrain = 0.0
        self._status       = "idle"

    def start(self):
        if not AL_CFG.get("enabled", True):
            log.info("[AutoLearn] Disabled in config.")
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info(
            "[AutoLearn] Scheduler started — fetch every %dh, retrain every %dh.",
            self.fetch_interval   // 3600,
            self.retrain_interval // 3600,
        )

    def stop(self):
        self._stop.set()

    def _loop(self):
        time.sleep(60)  # short initial delay so the server fully starts first
        while not self._stop.is_set():
            now = time.time()
            try:
                if now - self._last_fetch >= self.fetch_interval:
                    self._status = "fetching"
                    log.info("[AutoLearn] Fetching new games...")
                    new_games = fetch_ncaa_data()
                    added = append_to_json(new_games) if new_games else 0
                    self._last_fetch = time.time()
                    log.info("[AutoLearn] %d new games added.", added)

                    if added >= self.min_new_games:
                        self._status = "training"
                        log.info("[AutoLearn] %d new games → retraining...", added)
                        train_and_evaluate(self.storage, triggered_by="new_data")
                        self._last_retrain = time.time()

                elif now - self._last_retrain >= self.retrain_interval:
                    self._status = "training"
                    log.info("[AutoLearn] Scheduled retrain...")
                    train_and_evaluate(self.storage, triggered_by="scheduler")
                    self._last_retrain = time.time()

            except Exception as e:
                log.error("[AutoLearn] Error: %s", e)
                log.debug(traceback.format_exc())
            finally:
                self._status = "idle"

            # Sleep in 60-second chunks so stop() is responsive
            for _ in range(60):
                if self._stop.is_set():
                    break
                time.sleep(60)

    def get_state(self) -> dict:
        def countdown(last: float, interval: float) -> str:
            rem = max(0, int(last + interval - time.time()))
            h, m = divmod(rem // 60, 60)
            return f"{h}h {m}m" if h else f"{m}m"

        return {
            "enabled":            AL_CFG.get("enabled", True),
            "status":             self._status,
            "fetch_interval_h":   self.fetch_interval   // 3600,
            "retrain_interval_h": self.retrain_interval // 3600,
            "min_new_games":      self.min_new_games,
            "promote_threshold":  AL_CFG.get("promote_threshold", 0.002),
            "next_fetch_in":      countdown(self._last_fetch,   self.fetch_interval),
            "next_retrain_in":    countdown(self._last_retrain, self.retrain_interval),
            "last_fetch":   (datetime.fromtimestamp(self._last_fetch).isoformat()
                             if self._last_fetch   else None),
            "last_retrain": (datetime.fromtimestamp(self._last_retrain).isoformat()
                             if self._last_retrain else None),
        }