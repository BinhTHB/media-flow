"""Tests for session storage and the browser-publisher contract.

No browser is launched: the Playwright layer is exercised only through its
error paths and the publisher template method is tested with a fake page, so
these run in CI without a browser install.
"""

from __future__ import annotations

import json

import pytest

from src.core.contracts import TransientPublishError
from src.core.models.account import AccountIdentity, PostingWindow, SessionState
from src.core.models.publication import PublishItem, VideoMetadata
from src.infrastructure.publishing.browser_base import BrowserPublisher
from src.infrastructure.publishing.session_store import FileSessionStore


# --------------------------------------------------------------------------
# Account models
# --------------------------------------------------------------------------

def test_account_identity_requires_all_fields() -> None:
    with pytest.raises(ValueError, match="platform"):
        AccountIdentity("", "acc", "REF")
    with pytest.raises(ValueError, match="account_id"):
        AccountIdentity("tiktok", "", "REF")
    with pytest.raises(ValueError, match="credential_ref"):
        AccountIdentity("tiktok", "acc", "")


def test_account_key_is_platform_scoped() -> None:
    assert AccountIdentity("youtube", "a1", "R").key == "youtube:a1"


def test_posting_window_validates_bounds() -> None:
    with pytest.raises(ValueError, match="start_hour"):
        PostingWindow(start_hour=23, end_hour=2)
    with pytest.raises(ValueError, match="start_hour"):
        PostingWindow(start_hour=24)
    with pytest.raises(ValueError, match="max_posts_per_day"):
        PostingWindow(max_posts_per_day=0)


# --------------------------------------------------------------------------
# FileSessionStore
# --------------------------------------------------------------------------

def _account(platform: str = "tiktok", account_id: str = "a1") -> AccountIdentity:
    return AccountIdentity(platform, account_id, f"{platform.upper()}_COOKIE")


def _session_file(tmp_path, name: str = "session.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps({"cookies": [], "origins": []}), encoding="utf-8")
    return str(path)


def test_store_returns_none_before_any_session(tmp_path) -> None:
    assert FileSessionStore(str(tmp_path / "sessions")).load(_account()) is None


def test_store_saves_and_loads_a_session(tmp_path) -> None:
    store = FileSessionStore(str(tmp_path / "sessions"))
    saved = store.save(_account(), _session_file(tmp_path))

    assert store.load(_account()) == saved
    # The copy lives inside the store root, named from the account key.
    assert saved.endswith("tiktok_a1.json")


def test_store_copies_the_session_instead_of_referencing_it(tmp_path) -> None:
    store = FileSessionStore(str(tmp_path / "sessions"))
    source = _session_file(tmp_path)
    saved = store.save(_account(), source)

    # Mutating the original must not change what the store hands back.
    with open(source, "w", encoding="utf-8") as handle:
        handle.write('{"cookies": ["changed"]}')
    assert json.loads(open(saved, encoding="utf-8").read())["cookies"] == []


def test_invalidate_makes_load_return_none(tmp_path) -> None:
    store = FileSessionStore(str(tmp_path / "sessions"))
    store.save(_account(), _session_file(tmp_path))

    store.invalidate(_account(), "expired")

    assert store.load(_account()) is None
    states = store.list_states()
    assert states[0]["valid"] is False
    assert states[0]["note"] == "expired"


def test_store_is_per_account(tmp_path) -> None:
    store = FileSessionStore(str(tmp_path / "sessions"))
    store.save(_account("tiktok", "a1"), _session_file(tmp_path, "one.json"))
    store.save(_account("tiktok", "a2"), _session_file(tmp_path, "two.json"))

    assert store.load(_account("tiktok", "a1")) != store.load(_account("tiktok", "a2"))
    assert store.load(_account("youtube", "a1")) is None


def test_missing_session_file_after_save_reads_as_none(tmp_path) -> None:
    store = FileSessionStore(str(tmp_path / "sessions"))
    saved = store.save(_account(), _session_file(tmp_path))

    from pathlib import Path

    Path(saved).unlink()

    assert store.load(_account()) is None


def test_corrupt_index_does_not_raise(tmp_path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    (root / "sessions.json").write_text("{not json", encoding="utf-8")

    store = FileSessionStore(str(root))

    assert store.load(_account()) is None
    assert store.list_states() == []


def test_session_state_serializes_metadata_only(tmp_path) -> None:
    state = SessionState(
        account_key="tiktok:a1", session_path="tiktok_a1.json", created_at="2026-01-01T00:00:00+00:00"
    )
    payload = state.to_dict()
    assert set(payload) == {
        "account_key", "session_path", "created_at", "last_used_at", "valid", "note",
    }


# --------------------------------------------------------------------------
# BrowserPublisher template method
# --------------------------------------------------------------------------

class _FakePage:
    """Just enough page surface for the template method."""

    def __init__(self, logged_in: bool = True) -> None:
        self.logged_in = logged_in
        self.submitted: list[str] = []


class _FakePublisher(BrowserPublisher):
    def __init__(self, *args, logged_in: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.page = _FakePage(logged_in=logged_in)
        self.opened = False

    def _open_upload_page(self, page) -> None:
        self.opened = True

    def _is_logged_in(self, page) -> bool:
        return page.logged_in

    def _submit(self, page, item) -> str:
        page.submitted.append(item.video_path)
        return "post-42"

    # Replace the real Playwright context with the fake page.
    def _session(self, session_path):
        publisher = self

        class _Ctx:
            def __enter__(self_inner):
                return publisher.page

            def __exit__(self_inner, *exc):
                return False

        return _Ctx()


def _item(tmp_path) -> PublishItem:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"video")
    return PublishItem("item-1", str(video), "tiktok", VideoMetadata("Title"))


def test_publisher_requires_a_stored_session(tmp_path) -> None:
    store = FileSessionStore(str(tmp_path / "sessions"))
    publisher = _FakePublisher(_account(), store)

    with pytest.raises(TransientPublishError, match="no stored session"):
        publisher.publish(_item(tmp_path))


def test_publisher_invalidates_session_on_login_wall(tmp_path) -> None:
    store = FileSessionStore(str(tmp_path / "sessions"))
    store.save(_account(), _session_file(tmp_path))
    publisher = _FakePublisher(_account(), store, logged_in=False)

    with pytest.raises(TransientPublishError, match="expired"):
        publisher.publish(_item(tmp_path))

    # The dead session must not be retried forever.
    assert store.load(_account()) is None


def test_publisher_returns_published_record(tmp_path) -> None:
    store = FileSessionStore(str(tmp_path / "sessions"))
    store.save(_account(), _session_file(tmp_path))
    publisher = _FakePublisher(_account(), store)

    record = publisher.publish(_item(tmp_path))

    assert record.status == "PUBLISHED"
    assert record.platform_video_id == "post-42"
    assert record.platform == "tiktok"
    assert publisher.opened is True


def test_publisher_surfaces_missing_playwright_as_configuration_error(tmp_path) -> None:
    """Without Playwright the error must say how to fix it, not ImportError."""
    store = FileSessionStore(str(tmp_path / "sessions"))
    store.save(_account(), _session_file(tmp_path))

    class _RealSessionPublisher(BrowserPublisher):
        def _open_upload_page(self, page): ...
        def _submit(self, page, item): return "x"

    publisher = _RealSessionPublisher(_account(), store)

    try:
        import playwright  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(RuntimeError, match="playwright is required"):
            publisher.publish(_item(tmp_path))
    else:  # pragma: no cover - only when Playwright happens to be installed
        pytest.skip("Playwright is installed; the lazy-import guard is unreachable")
