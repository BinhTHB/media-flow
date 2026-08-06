"""Project root and model path resolution.

Path resolution rules (in priority order):

1. Environment variable ``MEDIA_FLOW_YOLO_WEIGHTS``.
2. A value supplied explicitly in config.
3. The default ``models/yolo/best.pt`` relative to the project root.

Everything is resolved with :mod:`pathlib.Path` so the code works identically on
Windows and Linux. If the resolved model does not exist the code raises a clear
:class:`ModelNotFoundError` that tells the operator what path was resolved,
which config was used, and how to place the model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

#: Environment variable that may override the YOLO weights path.
MEDIA_FLOW_YOLO_WEIGHTS_ENV = "MEDIA_FLOW_YOLO_WEIGHTS"

#: Default relative location of the fine-tuned YOLO model inside the repo.
DEFAULT_YOLO_WEIGHTS_REL = Path("models") / "yolo" / "best.pt"


class ModelNotFoundError(FileNotFoundError):
    """Raised when the YOLO weights file cannot be resolved or is missing."""

    def __init__(
        self,
        resolved_path: Path,
        *,
        config_path: Optional[str] = None,
        env_value: Optional[str] = None,
    ) -> None:
        self.resolved_path = Path(resolved_path)
        self.config_path = config_path
        self.env_value = env_value
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        lines = [
            "YOLO fine-tuned weights file was not found.",
            "",
            f"Resolved path : {self.resolved_path}",
            f"Exists       : {self.resolved_path.exists()}",
        ]
        if self.env_value is not None:
            lines.append(f"{MEDIA_FLOW_YOLO_WEIGHTS_ENV} : {self.env_value!r}")
        else:
            lines.append(f"{MEDIA_FLOW_YOLO_WEIGHTS_ENV} : (not set)")
        lines.append(f"Config path   : {self.config_path or '(default)'}")
        lines.append("")
        lines.append(
            "Place the fine-tuned model at the resolved path above, or set the "
            f"{MEDIA_FLOW_YOLO_WEIGHTS_ENV} environment variable to an absolute "
            "path. The project does NOT download a pretrained model automatically."
        )
        return "\n".join(lines)


def resolve_project_root() -> Path:
    """Return the project root.

    The project root is located by walking up from this file until we find the
    ``AGENTS.md`` marker (or the ``models`` directory). This keeps the code
    agnostic to the current working directory.
    """
    here = Path(__file__).resolve().parent
    # src/infrastructure/detection -> climb until AGENTS.md is present.
    for candidate in [here, *here.parents]:
        if (candidate / "AGENTS.md").exists():
            return candidate
    # Fallback: the repo root contains the top-level `config` directory.
    for candidate in [here, *here.parents]:
        if (candidate / "config").is_dir() and (candidate / "src").is_dir():
            return candidate
    # Last resort: three levels up from this package.
    return here.parents[2]


def resolve_yolo_weights(
    *,
    config_value: Optional[str] = None,
    env_value: Optional[str] = None,
    project_root: Optional[Path] = None,
    allow_missing: bool = False,
) -> Path:
    """Resolve the YOLO weights path.

    Priority: explicit ``env_value`` > ``config_value`` > default relative path.

    If the resolved file does not exist and ``allow_missing`` is False, a
    :class:`ModelNotFoundError` is raised.
    """
    root = project_root or resolve_project_root()

    env = env_value if env_value is not None else os.environ.get(MEDIA_FLOW_YOLO_WEIGHTS_ENV)
    if env:
        resolved = Path(env).expanduser()
        if not resolved.is_absolute():
            resolved = root / resolved
    elif config_value:
        candidate = Path(config_value).expanduser()
        resolved = candidate if candidate.is_absolute() else root / candidate
    else:
        resolved = root / DEFAULT_YOLO_WEIGHTS_REL

    resolved = resolved.resolve()
    if not allow_missing and not resolved.is_file():
        raise ModelNotFoundError(
            resolved,
            config_path=str(config_value) if config_value else None,
            env_value=env if env else None,
        )
    return resolved
