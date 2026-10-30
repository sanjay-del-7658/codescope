"""Featherless (OpenAI-compatible) chat completions for query rewriting and grounded explanations.

The model only ever sees the retrieved snippets (never the whole repo). Every citation it produces
is validated against the actual result set; anything that doesn't map to a retrieved snippet is
stripped and reported.
"""
from __future__ import annotations

import json
import re

import httpx

from . import config


def available() -> bool:
    return bool(config.LLM_API_KEY)


def _chat(messages: list[dict], max_tokens: int = 700, temperature: float = 0.1) -> str:
    r = httpx.post(
        f"{config.LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
        json={"model": config.LLM_MODEL, "messages": messages,
              "max_tokens": max_tokens, "temperature": temperature},
        timeout=config.LLM_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


REWRITE_PROMPT = """You help search a source-code repository. Given a developer question, list up to 10
short code search terms (identifiers, API names, keywords) likely to appear in the relevant code.
Prefer concrete words: function/class name fragments, library calls, config keys.
Respond with ONLY a JSON array of strings."""


def rewrite_query(question: str) -> list[str]:
    if not available():
        return []
    try:
        out = _chat([{"role": "system", "content": REWRITE_PROMPT},
                     {"role": "user", "content": question}], max_tokens=120, temperature=0.0)
        m = re.search(r"\[.*\]", out, re.S)
        terms = json.loads(m.group(0)) if m else []
        return [str(t)[:40] for t in terms if isinstance(t, (str, int))][:10]
    except Exception:
        return []


EXPLAIN_PROMPT = """You are a code navigation assistant. Answer the developer's question using ONLY the
numbered source snippets provided. Rules:
- Cite every claim with the snippet id in square brackets, e.g. [S2]. Use only ids that are given.
- Never mention a file, function, or line that is not in the snippets.
- If the snippets do not clearly answer the question, say so explicitly and say what is missing.
- Be concise: 3-8 sentences, then a short "Start here:" line naming the best snippet id(s) to read first."""

CITE = re.compile(r"\[(S\d+)(?:[^\]]*)\]")


def explain(question: str, results: list[dict], max_snippets: int = 6, max_lines: int = 60) -> dict:
    if not available():
        return {"status": "disabled", "text": None,
                "message": "Set FEATHERLESS_API_KEY to enable AI explanations."}
    if not results:
        return {"status": "no_evidence", "text": None, "message": "Nothing retrieved to explain."}

    used = results[:max_snippets]
    blocks = []
    for r in used:
        lines = r["snippet"].splitlines()[:max_lines]
        blocks.append(f"[{r['id']}] {r['path']} L{r['start_line']}-L{r['start_line'] + len(lines) - 1}"
                      f" ({r['kind']} {r['symbol'] or ''})\n```{r['language']}\n" + "\n".join(lines) + "\n```")
    try:
        text = _chat([{"role": "system", "content": EXPLAIN_PROMPT},
                      {"role": "user", "content": f"Question: {question}\n\nSnippets:\n\n" + "\n\n".join(blocks)}])
    except Exception as e:
        return {"status": "error", "text": None, "message": f"LLM call failed: {e}"}
    return validate_citations(text, used)


def validate_citations(text: str, used: list[dict]) -> dict:
    valid_ids = {r["id"]: r for r in used}
    cited, invalid = [], []
    for m in CITE.finditer(text):
        sid = m.group(1)
        (cited if sid in valid_ids else invalid).append(sid)
    clean = CITE.sub(lambda m: m.group(0) if m.group(1) in valid_ids else "[unverified]", text)

    # Flag file paths the model mentions that are not among the retrieved snippets
    known_paths = {r["path"] for r in used}
    mentioned = set(re.findall(r"[\w./-]+\.(?:py|js|jsx|ts|tsx|java|go)\b", text))
    unknown_paths = sorted(p for p in mentioned if not any(k.endswith(p) or p.endswith(k) for k in known_paths))

    return {
        "status": "ok" if cited and not invalid and not unknown_paths else "needs_review",
        "text": clean,
        "citations": [{"id": s, "path": valid_ids[s]["path"],
                       "lines": f"L{valid_ids[s]['start_line']}-L{valid_ids[s]['end_line']}"}
                      for s in dict.fromkeys(cited)],
        "invalid_citations": sorted(set(invalid)),
        "unsupported_paths": unknown_paths,
        "uncited": not cited,
        "label": "AI-generated explanation. Verify against the source evidence.",
    }
