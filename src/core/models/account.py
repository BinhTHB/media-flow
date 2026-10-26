"""Publication account and session contracts.

Re-upload automation at scale needs more than one identity per platform: one
account gets rate limited, flagged, or capped, and the operator switches. This
module models *who* is publishing, separately from *where*.

Credentials never live in these models — only a reference to a secret the
operator supplies through the environment or a credentials file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AccountIdentity:
    """One publishing identity on one platform.

    ``credential_ref`` names an environment variable or a key in the secrets
    store; the model itself never carries a token.
    """

    platform: str
    account_id: str
    credential_ref: str
    #: Optional per-account hints (timezone, preferred posting window, ...).
    settings: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.platform or not self.platform.strip():
            raise ValueError("AccountIdentity.platform must not be empty")
        if not self.account_id or not self.account_id.strip():
            raise ValueError("AccountIdentity.account_id must not be empty")
        if not self.credential_ref or not self.credential_ref.strip():
            raise ValueError("AccountIdentity.credential_ref must not be empty")

    @property
    def key(self) -> str:
        """Stable identity used for state keys and rate-limit bookkeeping."""
        return f"{self.platform}:{self.account_id}"


@dataclass
class PostingWindow:
    """A daily posting window in local wall-clock time.

    Re-upload tools spread posts across the day instead of dumping a batch at
    once: a burst is the clearest automation signal an account can emit.
    """

    start_hour: int = 9
    end_hour: int = 22
    max_posts_per_day: int = 3
    min_gap_minutes: int = 45

    def __post_init__(self) -> None:
        if not (0 <= int(self.start_hour) <= 23):
            raise ValueError("PostingWindow.start_hour must be within 0..23")
        if not (0 <= int(self.end_hour) <= 24):
            raise ValueError("PostingWindow.end_hour must be within 0..24")
        if int(self.start_hour) >= int(self.end_hour):
            raise ValueError("PostingWindow.start_hour must be before end_hour")
        if int(self.max_posts_per_day) < 1:
            raise ValueError("PostingWindow.max_posts_per_day must be >= 1")
        if int(self.min_gap_minutes) < 0:
            raise ValueError("PostingWindow.min_gap_minutes must not be negative")

    def to_dict(self) -> Dict[str, int]:
        return {
            "start_hour": int(self.start_hour),
            "end_hour": int(self.end_hour),
            "max_posts_per_day": int(self.max_posts_per_day),
            "min_gap_minutes": int(self.min_gap_minutes),
        }


@dataclass
class SessionState:
    """Last known state of one stored browser session.

    Only metadata: the cookie jar itself lives in the session file, never in
    the publication state or the logs.
    """

    account_key: str
    session_path: str
    created_at: str
    last_used_at: Optional[str] = None
    valid: bool = True
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "account_key": self.account_key,
            "session_path": self.session_path,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "valid": bool(self.valid),
            "note": self.note,
        }
