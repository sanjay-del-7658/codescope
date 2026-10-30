# CodeScope: backend

Repository-exploration engine for the *Helping New Contributors Find the Right Code* problem.
It indexes a bounded repo and answers natural-language or keyword questions with **real source evidence**
(file path, exact line range, snippet), ranked by a hybrid, explainable retriever.

## Quick start

```bash
pip install -r requirements.txt
pip install sentence-transformers          # optional but recommended (better semantic search)
cp .env.example .env && export $(grep -v '^#' .env | xargs)   # add FEATHERLESS_API_KEY

# CLI
python cli.py index path/to/repo-or.zip --exclude "docs/*,examples/*"
python cli.py search <repo_id> "where is the session cookie signed" -k 5 --explain
python cli.py eval <repo_id> eval/flask_questions.json

# API (Swagger UI at http://localhost:8000/docs)
uvicorn app.main:app --reload
pytest -q
```

## How it maps to the requirements

| # | Requirement | Where |
|---|---|---|
| 1 | Local folder or archive | `ingest.prepare_root`: folder, `.zip`, `.tar.gz` (zip-slip safe) |
| 2 | Exclude generated/build/deps/large | `ingest.collect_files` + `config.EXCLUDED_*`, `.gitignore`, size limit, binary check, user globs; skip report with reasons |
| 3 | Index paths, symbols, snippets | `parser.py` (tree-sitter: Python, JS, TS/TSX, Java, Go): functions/classes/methods, docstrings, imports, calls; function-level chunks |
| 4 | NL + keyword queries | `text.py` code-aware tokenizer (camelCase/snake_case, stemming); identifier detection; optional LLM query expansion |
| 5 | Hybrid ranking | `search.py`: BM25 + embeddings + symbol-name match → weighted fusion (`linear` or `rrf`) + structural priors |
| 6 | Snippet, path, line range | every result: `path`, `start_line`, `end_line`, `snippet`, `highlight_lines`; `/file` endpoint for full context |
| 7 | Evidence vs generated text | `llm.py`: LLM sees only top-k snippets, must cite `[S#]`; citations and file paths are validated; weak evidence → no explanation |
| 8 | ≥10 verified questions | `eval/flask_questions.json` (12 Qs, symbols checked in source) + `evaluate.py` (Hit@1, Recall@k, MRR, per-mode ablation) |

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/repos` | form: `path` **or** `archive` upload, `max_file_kb`, `exclude`, `embed_backend` → index |
| GET | `/repos` | list indexed repos |
| GET | `/repos/{id}/report` | ingestion report incl. skipped files + reasons |
| GET | `/repos/{id}/overview` | most-imported files, largest files |
| POST | `/repos/{id}/search` | `{query, top_k, mode, rewrite, explain}` → results + explanation |
| GET | `/repos/{id}/file?path=&start=&end=` | exact source lines |
| GET | `/repos/{id}/graph?path=` | imports / imported-by |
| GET | `/repos/{id}/symbols?q=` | symbol lookup |
| POST | `/repos/{id}/eval` | upload questions JSON → metrics per mode |

### Search response (abridged)
```json
{
  "evidence": {"level": "strong", "strong_results": 3},
  "results": [{
    "id": "S1", "path": "src/flask/sessions.py", "start_line": 307, "end_line": 318,
    "symbol": "SecureCookieSessionInterface.get_signing_serializer", "kind": "method",
    "snippet": "def get_signing_serializer(...)...", "highlight_lines": [307],
    "evidence_strength": "strong",
    "why": {"ranks": {"bm25": 2, "dense": 32, "symbol": 1}, "matched_terms": ["session","cooki","sign"],
            "term_coverage": 1.0, "boosts": {"imported_by": 1.058}}
  }],
  "files": [{"path": "src/flask/sessions.py", "regions": [{"id": "S1", "lines": "L307-L318"}]}],
  "explanation": {"status": "ok", "text": "... [S1] ...", "citations": [...],
                  "invalid_citations": [], "unsupported_paths": [],
                  "label": "AI-generated explanation. Verify against the source evidence."}
}
```

## Evaluation (Flask 3.0.3, 12 questions, k=5, offline LSA embeddings)

| Mode | Hit@1 | Recall@5 | MRR |
|---|---|---|---|
| **hybrid** | **0.667** | **0.778** | **0.799** |
| bm25 | 0.583 | 0.764 | 0.764 |
| dense (LSA) | 0.417 | 0.569 | 0.552 |
| symbol | 0.250 | 0.389 | 0.372 |

Install `sentence-transformers` and re-index to use a real embedding model, which should widen the hybrid gap.
Fusion weights were tuned on these same 12 questions, so add held-out questions before quoting final numbers.

## Design notes
- **Function-level chunks**: classes are indexed by their header only, so methods aren't duplicated. Long functions are split into overlapping 120-line windows, and top-level code gets its own `module` chunks.
- **Field weighting**: symbol name ×3, path ×2, docstring ×2 and body ×1 in the BM25 document.
- **Structural priors**: test files ×0.8 (unless the query mentions tests), module-level chunks ×0.9, and a small boost for files imported by many others. All of these show up in `why.boosts`.
- **Grounding guard**: invalid `[S#]` citations are replaced with `[unverified]`, and file paths not among the retrieved snippets are listed in `unsupported_paths`. Weak or absent evidence means no LLM call.
