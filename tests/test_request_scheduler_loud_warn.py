import logging

from src.infrastructure.gemini.request_scheduler import RequestScheduler


def test_scheduler_logs_ledger_write_failure(monkeypatch, tmp_path, caplog):
    scheduler = RequestScheduler(ledger_path=str(tmp_path / "ledger.json"))

    def fail_replace(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("src.infrastructure.gemini.request_scheduler.os.replace", fail_replace)
    with caplog.at_level(logging.WARNING):
        scheduler._save_ledger()

    assert "LOUD-WARN: RequestScheduler could not persist ledger" in caplog.text
