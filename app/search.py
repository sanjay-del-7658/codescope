"""Hybrid retrieval: BM25 + dense embeddings + symbol matching, fused with weighted RRF.

Every result carries an explanation of *why* it ranked (per-retriever rank/score, matched terms),
so the UI can show an explainable ranking.
"""
from __future__ import annotations

import math
import re

import numpy as np

from . import config
from .index import RepoIndex
from .text import looks_like_identifier, query_terms, tokenize

MODES = ("hybrid", "bm25", "dense", "symbol")
TEST_PATH = re.compile(r"(^|/)(tests?|__tests__|spec)(/|$)|(_test|\.test|\.spec|test_)[^/]*$")


def _top(scores: np.ndarray, n: int) -> list[int]:
    if scores.size == 0:
        return []
    n = min(n, scores.size)
    idx = np.argpartition(-scores, n - 1)[:n]
    idx = idx[np.argsort(-scores[idx])]
    return [int(i) for i in idx if scores[i] > 0]


def _bm25(ix: RepoIndex, terms: list[str]) -> tuple[list[int], np.ndarray]:
    if not terms or ix.bm25 is None:
        return [], np.zeros(len(ix.chunks))
    s = np.asarray(ix.bm25.get_scores(terms))
    return _top(s, config.CANDIDATES_PER_RETRIEVER), s


def _dense(ix: RepoIndex, query: str) -> tuple[list[int], np.ndarray]:
    if ix.vectors is None:
        return [], np.zeros(len(ix.chunks))
    qv = ix.embedder.encode([query])
    s = (ix.vectors @ qv[0]).astype(float)
    return _top(np.clip(s, 0, None), config.CANDIDATES_PER_RETRIEVER), s


def _symbol(ix: RepoIndex, query: str, terms: list[str]) -> tuple[list[int], np.ndarray]:
    """Score chunks whose *names* (symbol, then path) contain query terms, IDF-weighted."""
    s = np.zeros(len(ix.chunks))
    if not terms:
        return [], s
    idf = ix.bm25.idf if ix.bm25 is not None else {}
    w = {t: max(idf.get(t, 0.0), 0.1) for t in terms}
    total = sum(w.values()) or 1.0
    ident = query.strip().rstrip("()") if looks_like_identifier(query) else None
    raw_words = {x.lower() for x in re.findall(r"[A-Za-z_][\w]*", query) if len(x) > 2}
    for i, c in enumerate(ix.chunks):
        name_toks = ix.name_tokens[i]
        path_toks = ix.symbol_tokens[i] - name_toks
        sc = sum(w[t] for t in terms if t in name_toks) / total
        sc += 0.3 * sum(w[t] for t in terms if t in path_toks) / total
        if c.symbol:
            name = c.symbol.split(".")[-1].lower()
            if ident and (c.symbol.lower() == ident.lower() or c.symbol.lower().endswith("." + ident.lower())
                          or name == ident.lower()):
                sc += 3.0                          # exact identifier hit
            elif name in raw_words:
                sc += 1.0                          # query literally names this symbol
        s[i] = sc
    return _top(s, config.CANDIDATES_PER_RETRIEVER), s


def _norm(scores: np.ndarray, cand: list[int]) -> dict[int, float]:
    """Min-max normalise a retriever's scores over its own candidate list."""
    if not cand:
        return {}
    vals = scores[cand]
    hi, lo = float(vals.max()), float(vals.min())
    rng = hi - lo or 1.0
    return {ci: (float(scores[ci]) - lo) / rng if hi > lo else 1.0 for ci in cand}


def search(ix: RepoIndex, query: str, top_k: int = 10, mode: str = "hybrid",
           extra_terms: list[str] | None = None, include_tests: bool | None = None) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    terms = query_terms(query)
    exp = [t for t in query_terms(" ".join(extra_terms or [])) if t not in terms]
    all_terms = terms + exp
    wants_tests = include_tests if include_tests is not None else bool(re.search(r"\btest", query, re.I))

    rankings: dict[str, list[int]] = {}
    raw: dict[str, np.ndarray] = {}
    if mode in ("hybrid", "bm25"):
        rankings["bm25"], raw["bm25"] = _bm25(ix, all_terms)
    if mode in ("hybrid", "dense"):
        rankings["dense"], raw["dense"] = _dense(ix, query)
    if mode in ("hybrid", "symbol"):
        rankings["symbol"], raw["symbol"] = _symbol(ix, query, all_terms)

    fused: dict[int, float] = {}
    rank_of: dict[str, dict[int, int]] = {}
    weights = config.weights_for(getattr(ix.embedder, "name", "lsa"))
    for name, ranked in rankings.items():
        rank_of[name] = {ci: r + 1 for r, ci in enumerate(ranked)}
    if config.FUSION == "rrf" or len(rankings) == 1:
        for name, ranked in rankings.items():
            for r, ci in enumerate(ranked):
                fused[ci] = fused.get(ci, 0.0) + weights[name] / (config.RRF_K + r + 1)
    else:  # linear: weighted sum of min-max normalised scores
        for name, ranked in rankings.items():
            for ci, v in _norm(raw[name], ranked).items():
                fused[ci] = fused.get(ci, 0.0) + weights[name] * v

    # Structural priors (small, explainable multipliers)
    boosts: dict[int, dict[str, float]] = {}
    for ci in fused:
        c = ix.chunks[ci]
        b = {}
        if not wants_tests and TEST_PATH.search(c.path):
            b["test_file"] = 0.8
        if c.kind == "module":
            b["module_level"] = 0.9
        indeg = len(ix.graph_in.get(c.path, []))
        if indeg:
            b["imported_by"] = round(1 + 0.03 * math.log1p(indeg), 3)
        for v in b.values():
            fused[ci] *= v
        boosts[ci] = b

    order = sorted(fused, key=lambda i: -fused[i])

    # De-duplicate overlapping windows of the same symbol/file region
    results, seen = [], []
    for ci in order:
        c = ix.chunks[ci]
        if any(c.path == p and not (c.end_line < s or c.start_line > e) for p, s, e in seen):
            continue
        seen.append((c.path, c.start_line, c.end_line))
        results.append(_result(ix, ci, fused[ci], terms, exp, rank_of, raw, boosts[ci]))
        if len(results) >= top_k:
            break

    for i, r in enumerate(results):
        r["id"] = f"S{i + 1}"

    return {
        "query": query,
        "mode": mode,
        "terms": terms,
        "expanded_terms": exp,
        "evidence": _overall_strength(results),
        "results": results,
        "files": _group_by_file(results),
    }


def _result(ix, ci, score, terms, exp, rank_of, raw, boosts) -> dict:
    c = ix.chunks[ci]
    ctoks = set(ix.chunk_tokens[ci])
    matched = [t for t in terms if t in ctoks]
    matched_exp = [t for t in exp if t in ctoks]
    coverage = len(matched) / len(terms) if terms else 0.0
    dense_sim = float(raw["dense"][ci]) if "dense" in raw else None

    # Lines containing query terms (for highlighting in the UI)
    want = set(terms) | set(exp)
    hl = [c.start_line + i for i, line in enumerate(c.text.splitlines()) if want & set(tokenize(line))]

    strength = "weak"
    hits = sum(1 for r in rank_of.values() if ci in r and r[ci] <= 10)
    if coverage >= 0.5 and (hits >= 2 or len(rank_of) == 1):
        strength = "strong"
    elif coverage >= 0.25 or (dense_sim is not None and dense_sim >= 0.35) or hits >= 2:
        strength = "medium"

    return {
        "path": c.path,
        "start_line": c.start_line,
        "end_line": c.end_line,
        "symbol": c.symbol,
        "kind": c.kind,
        "language": c.language,
        "signature": c.signature,
        "snippet": c.text,
        "highlight_lines": hl[:50],
        "score": round(score, 5),
        "evidence_strength": strength,
        "why": {
            "ranks": {k: v.get(ci) for k, v in rank_of.items()},
            "scores": {k: round(float(raw[k][ci]), 4) for k in raw},
            "matched_terms": matched,
            "matched_expanded_terms": matched_exp,
            "term_coverage": round(coverage, 2),
            "boosts": boosts,
            "imported_by": len(ix.graph_in.get(c.path, [])),
        },
    }


def _overall_strength(results: list[dict]) -> dict:
    if not results:
        return {"level": "none", "message": "No matching source code found."}
    best = results[0]["evidence_strength"]
    strong = sum(r["evidence_strength"] == "strong" for r in results)
    if best == "weak" and strong == 0:
        return {"level": "weak",
                "message": "No strong evidence found. Results are loose matches; verify before relying on them."}
    return {"level": "strong" if strong else "medium", "strong_results": strong}


def _group_by_file(results: list[dict]) -> list[dict]:
    groups: dict[str, dict] = {}
    for r in results:
        g = groups.setdefault(r["path"], {"path": r["path"], "best_score": r["score"], "regions": []})
        g["regions"].append({"id": r["id"], "symbol": r["symbol"],
                             "lines": f"L{r['start_line']}-L{r['end_line']}"})
    return list(groups.values())
