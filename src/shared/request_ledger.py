import os
import json
import time
import logging

logger = logging.getLogger(__name__)

# Ledger path is set by orchestrator dynamically if provided, or defaults to a local path.
_ledger_path = os.path.join("artifacts", "round19", "ledger.jsonl")

def set_ledger_path(path: str):
    global _ledger_path
    _ledger_path = path

def log_request(
    job_id: str,
    purpose: str,
    model: str,
    live_or_cache: str,  # "live" or "cache"
    request_id_or_hash: str,
    status: str,         # "success" or "error"
    tokens: int = 0,
    error_msg: str = ""
):
    try:
        os.makedirs(os.path.dirname(_ledger_path), exist_ok=True)
        entry = {
            "timestamp": time.time(),
            "job_id": job_id,
            "purpose": purpose,
            "model": model,
            "live_or_cache": live_or_cache,
            "request_id_or_hash": request_id_or_hash,
            "status": status,
            "tokens": tokens,
            "error_msg": error_msg
        }
        with open(_ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error("Failed to write to request ledger: %s", str(e))
