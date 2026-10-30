"""Builds and persists the per-repository index: chunks, symbols, BM25, embeddings, import graph."""
from __future__ import annotations

import hashlib
import json
import pickle
import posixpath
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .embeddings import make_embedder
from .ingest import IngestReport, SourceFile
from .parser import Chunk, ParsedFile, parse_file
from .text import split_identifier, tokenize


@dataclass
class RepoIndex:
    repo_id: str
    name: str
    root: str
    created: float
    report: dict
    chunks: list[Chunk] = field(default_factory=list)
    files: dict[str, dict] = field(default_factory=dict)          # path -> {language, n_lines, symbols, imports}
    graph_out: dict[str, list[str]] = field(default_factory=dict)  # file -> files it imports
    graph_in: dict[str, list[str]] = field(default_factory=dict)   # file -> files importing it
    bm25: BM25Okapi | None = None
    chunk_tokens: list[list[str]] = field(default_factory=list)
    symbol_tokens: list[set[str]] = field(default_factory=list)   # name + path tokens per chunk
    name_tokens: list[set[str]] = field(default_factory=list)     # symbol-name tokens per chunk
    embedder: object | None = None
    vectors: np.ndarray | None = None

    # ----------------------------------------------------------------- build
    @classmethod
    def build(cls, name: str, root: Path, files: list[SourceFile], report: IngestReport,
              embed_backend: str | None = None, progress=None) -> "RepoIndex":
        rid = hashlib.sha1(f"{root}{time.time()}".encode()).hexdigest()[:10]
        idx = cls(repo_id=rid, name=name, root=str(root), created=time.time(), report={})
        parsed: list[ParsedFile] = []
        lang_count: dict[str, int] = defaultdict(int)
        for i, f in enumerate(files):
            try:
                src = f.abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            pf = parse_file(f.path, f.language, src)
            parsed.append(pf)
            lang_count[f.language] += 1
            idx.files[f.path] = {
                "language": f.language, "n_lines": pf.n_lines, "imports": pf.imports,
                "symbols": [{"name": s.qualname, "kind": s.kind, "start": s.start_line, "end": s.end_line}
                            for s in pf.symbols],
            }
            idx.chunks.extend(pf.chunks)
            if progress and i % 50 == 0:
                progress(f"parsed {i}/{len(files)} files")

        idx._build_graph(parsed)
        idx._build_lexical()
        if progress:
            progress(f"embedding {len(idx.chunks)} chunks")
        idx.embedder = make_embedder(embed_backend)
        idx.vectors = idx.embedder.fit([idx.embed_text(c) for c in idx.chunks]) if idx.chunks else None

        idx.report = {
            **report.summary(),
            "languages": dict(lang_count),
            "chunks": len(idx.chunks),
            "symbols": sum(len(v["symbols"]) for v in idx.files.values()),
            "embedding_backend": getattr(idx.embedder, "name", "none"),
            "skipped_files": report.skipped[:500],
        }
        return idx

    @staticmethod
    def embed_text(c: Chunk) -> str:
        head = f"{c.kind} {c.symbol or ''} in {c.path}\n"
        if c.symbol:
            head += " ".join(split_identifier(c.symbol.replace(".", "_"))) + "\n"
        return head + (c.doc + "\n" if c.doc else "") + c.text[:2000]

    def _build_lexical(self):
        self.chunk_tokens, self.symbol_tokens, self.name_tokens = [], [], []
        for c in self.chunks:
            path_toks = tokenize(c.path.replace("/", " ").replace(".", " "))
            sym_toks = tokenize(c.symbol.replace(".", " ")) if c.symbol else []
            # Field weighting by repetition: symbol name x3, path x2, docstring x2, body x1
            toks = sym_toks * 3 + path_toks * 2 + tokenize(c.doc) * 2 + tokenize(c.text)
            self.chunk_tokens.append(toks)
            self.symbol_tokens.append(set(sym_toks) | set(path_toks))
            self.name_tokens.append(set(sym_toks))
        self.bm25 = BM25Okapi(self.chunk_tokens) if self.chunks else None

    # ---------------------------------------------------------- import graph
    def _build_graph(self, parsed: list[ParsedFile]):
        paths = set(self.files)
        py_modules: dict[str, str] = {}
        for p in paths:
            if p.endswith(".py"):
                mod = p[:-3].replace("/", ".")
                if mod.endswith(".__init__"):
                    mod = mod[: -len(".__init__")]
                parts = mod.split(".")
                for i in range(len(parts)):              # allow "src.flask.app" -> "flask.app"
                    py_modules.setdefault(".".join(parts[i:]), p)

        out: dict[str, set[str]] = defaultdict(set)
        for pf in parsed:
            for imp in pf.imports:
                for target in self._resolve(pf.path, pf.language, imp, paths, py_modules):
                    if target != pf.path:
                        out[pf.path].add(target)
        self.graph_out = {k: sorted(v) for k, v in out.items()}
        gin: dict[str, set[str]] = defaultdict(set)
        for src, tgts in out.items():
            for t in tgts:
                gin[t].add(src)
        self.graph_in = {k: sorted(v) for k, v in gin.items()}

    @staticmethod
    def _resolve(path, lang, imp, paths, py_modules) -> list[str]:
        res = []
        if lang == "python":
            m = re.match(r"from\s+(\.*)([\w.]*)\s+import\s+(.+)", imp, re.S)
            if m:
                dots, mod, names = m.groups()
                if dots:
                    pkg = path.rsplit("/", 1)[0].replace("/", ".") if "/" in path else ""
                    for _ in range(len(dots) - 1):
                        pkg = pkg.rsplit(".", 1)[0] if "." in pkg else ""
                    mod = f"{pkg}.{mod}".strip(".") if mod else pkg
                cands = [mod] + [f"{mod}.{n.strip(' ()')}".strip(".")
                                 for n in names.replace("\n", " ").split(",")]
            else:
                cands = [x.strip().split(" as ")[0] for x in imp.replace("import", "", 1).split(",")]
            for c in cands:
                if c in py_modules:
                    res.append(py_modules[c])
        elif lang in ("javascript", "typescript", "tsx"):
            m = re.search(r"""from\s+['"]([^'"]+)['"]|import\s+['"]([^'"]+)['"]""", imp)
            spec = (m.group(1) or m.group(2)) if m else None
            if spec and spec.startswith("."):
                base = posixpath.normpath(posixpath.join(posixpath.dirname(path), spec))
                for ext in ("", ".ts", ".tsx", ".js", ".jsx", ".mjs", "/index.ts", "/index.tsx", "/index.js"):
                    if base + ext in paths:
                        res.append(base + ext)
                        break
        elif lang == "java":
            m = re.search(r"import\s+(?:static\s+)?([\w.]+)", imp)
            if m:
                suffix = m.group(1).replace(".", "/") + ".java"
                res += [p for p in paths if p.endswith(suffix)]
        elif lang == "go":
            m = re.search(r'"([^"]+)"', imp)
            if m:
                pkg_dir = m.group(1).rstrip("/").split("/")
                for p in paths:
                    d = p.rsplit("/", 1)[0] if "/" in p else ""
                    if d and d.split("/")[-2:] == pkg_dir[-2:]:
                        res.append(p)
        return res

    # ------------------------------------------------------------ persistence
    def save(self, base: Path = config.DATA_DIR):
        d = base / self.repo_id
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "index.pkl", "wb") as f:
            pickle.dump(self, f)
        (d / "meta.json").write_text(json.dumps(self.meta(), indent=2))

    @staticmethod
    def load(repo_id: str, base: Path = config.DATA_DIR) -> "RepoIndex":
        with open(base / repo_id / "index.pkl", "rb") as f:
            return pickle.load(f)

    def meta(self) -> dict:
        r = {k: v for k, v in self.report.items() if k != "skipped_files"}
        return {"repo_id": self.repo_id, "name": self.name, "root": self.root, "created": self.created, **r}

    def read_lines(self, path: str, start: int, end: int) -> str:
        p = (Path(self.root) / path).resolve()
        if not str(p).startswith(str(Path(self.root).resolve())) or path not in self.files:
            raise FileNotFoundError(path)
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[max(0, start - 1):end])
