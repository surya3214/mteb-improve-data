"""Repository path helpers."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root containing configs/ and pyproject.toml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    # Fallback for editable installs where package lives under src/
    return here.parents[2]
