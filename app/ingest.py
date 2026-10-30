"""Repository ingestion: accept a folder or an archive, filter out noise, return source files."""
from __future__ import annotations

import fnmatch
import shutil
import tarfile
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pathspec

from . import config


@dataclass
class SourceFile:
    path: str          # repo-relative, forward slashes
    abs_path: Path
    language: str
    size: int


@dataclass
class IngestReport:
    root: str
    indexed: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)   # {"path", "reason"}

    def summary(self) -> dict:
        reasons: dict[str, int] = {}
        for s in self.skipped:
            reasons[s["reason"]] = reasons.get(s["reason"], 0) + 1
        return {"indexed": len(self.indexed), "skipped": len(self.skipped), "skipped_by_reason": reasons}


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract zip/tar while blocking path traversal (zip-slip)."""
    dest = dest.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for m in zf.infolist():
                target = (dest / m.filename).resolve()
                if not str(target).startswith(str(dest)):
                    raise ValueError(f"Unsafe path in archive: {m.filename}")
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            for m in tf.getmembers():
                target = (dest / m.name).resolve()
                if not str(target).startswith(str(dest)) or m.issym() or m.islnk():
                    raise ValueError(f"Unsafe member in archive: {m.name}")
            tf.extractall(dest)
    else:
        raise ValueError("Unsupported archive (use .zip, .tar, .tar.gz, .tgz)")


def prepare_root(source: str | Path, workdir: Path) -> Path:
    """Return a directory to index. Archives are extracted into workdir."""
    src = Path(source).expanduser().resolve()
    if src.is_dir():
        return src
    if not src.is_file():
        raise FileNotFoundError(f"No such file or folder: {source}")
    out = workdir / f"src-{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True)
    _safe_extract(src, out)
    # Unwrap single top-level folder (typical for GitHub zips)
    entries = [p for p in out.iterdir() if not p.name.startswith("__MACOSX")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return out


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\0" in f.read(4096)
    except OSError:
        return True


def _load_gitignore(root: Path):
    gi = root / ".gitignore"
    if gi.exists():
        try:
            return pathspec.PathSpec.from_lines("gitwildmatch", gi.read_text(errors="ignore").splitlines())
        except Exception:
            return None
    return None


def collect_files(root: Path, max_file_kb: int | None = None,
                  extra_excludes: list[str] | None = None) -> tuple[list[SourceFile], IngestReport]:
    max_bytes = (max_file_kb or config.MAX_FILE_KB) * 1024
    extra_excludes = [e.strip() for e in (extra_excludes or []) if e.strip()]
    gitignore = _load_gitignore(root)
    report = IngestReport(root=str(root))
    files: list[SourceFile] = []

    def skip(rel: str, reason: str):
        report.skipped.append({"path": rel, "reason": reason})

    stack = [root]
    while stack:
        d = stack.pop()
        try:
            entries = sorted(d.iterdir())
        except OSError:
            continue
        for p in entries:
            rel = p.relative_to(root).as_posix()
            if p.is_symlink():
                skip(rel, "symlink")
                continue
            if p.is_dir():
                if p.name in config.EXCLUDED_DIRS or p.name.endswith(".egg-info"):
                    skip(rel + "/", "excluded directory")
                elif gitignore and gitignore.match_file(rel + "/"):
                    skip(rel + "/", "gitignored")
                elif any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel + "/", pat) for pat in extra_excludes):
                    skip(rel + "/", "user exclude")
                else:
                    stack.append(p)
                continue
            lang = config.LANG_BY_EXT.get(p.suffix.lower())
            if lang is None:
                skip(rel, "unsupported language")
                continue
            if any(fnmatch.fnmatch(p.name, pat) for pat in config.EXCLUDED_FILE_PATTERNS):
                skip(rel, "generated/minified")
                continue
            if gitignore and gitignore.match_file(rel):
                skip(rel, "gitignored")
                continue
            if any(fnmatch.fnmatch(rel, pat) for pat in extra_excludes):
                skip(rel, "user exclude")
                continue
            size = p.stat().st_size
            if size > max_bytes:
                skip(rel, f"over size limit ({max_bytes // 1024} KB)")
                continue
            if size == 0:
                skip(rel, "empty")
                continue
            if _is_binary(p):
                skip(rel, "binary")
                continue
            files.append(SourceFile(rel, p, lang, size))
            report.indexed.append(rel)
    return files, report


def cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
