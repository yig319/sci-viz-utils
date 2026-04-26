"""Tiny path helpers for notebooks and lightweight analysis packages."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the nearest parent containing ``.git`` or ``pyproject.toml``."""

    path = Path.cwd() if start is None else Path(start)
    path = path.resolve()
    if path.is_file():
        path = path.parent
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as a resolved ``Path``."""

    out = Path(path).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def list_files(root: str | Path, patterns: str | list[str], recursive: bool = True) -> list[Path]:
    """List files under ``root`` matching one or more glob patterns."""

    root = Path(root)
    if isinstance(patterns, str):
        patterns = [patterns]
    if not root.exists():
        return []
    out: list[Path] = []
    for pattern in patterns:
        iterator = root.rglob(pattern) if recursive else root.glob(pattern)
        out.extend(path for path in iterator if path.is_file())
    return sorted(set(out), key=lambda path: str(path).lower())


__all__ = ["ensure_dir", "find_repo_root", "list_files"]
