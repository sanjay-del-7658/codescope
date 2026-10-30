"""FastAPI service.

Run:  uvicorn app.main:app --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import config, llm
from .evaluate import evaluate
from .index import RepoIndex
from .ingest import collect_files, prepare_root
from .search import MODES, search

app = FastAPI(title="CodeScope — repository exploration API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_CACHE: dict[str, RepoIndex] = {}
UPLOADS = config.DATA_DIR / "_uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)


def get_index(repo_id: str) -> RepoIndex:
    if repo_id not in _CACHE:
        try:
            _CACHE[repo_id] = RepoIndex.load(repo_id)
        except FileNotFoundError:
            raise HTTPException(404, f"Unknown repo_id {repo_id}")
    return _CACHE[repo_id]


@app.get("/health")
def health():
    return {"ok": True, "llm_enabled": llm.available(), "embed_backend_setting": config.EMBED_BACKEND}


@app.post("/repos")
async def create_repo(
    path: str | None = Form(None, description="Local folder or archive path on the server"),
    archive: UploadFile | None = File(None, description=".zip / .tar.gz upload"),
    name: str | None = Form(None),
    max_file_kb: int = Form(config.MAX_FILE_KB),
    exclude: str = Form("", description="Comma-separated extra glob excludes, e.g. docs/*,examples/*"),
    embed_backend: str | None = Form(None, description="auto | st | api | lsa"),
):
    if not path and not archive:
        raise HTTPException(400, "Provide either `path` or an `archive` upload")
    work = UPLOADS / uuid.uuid4().hex[:8]
    work.mkdir(parents=True)
    try:
        if archive is not None:
            dest = work / Path(archive.filename or "upload.zip").name
            with open(dest, "wb") as f:
                shutil.copyfileobj(archive.file, f)
            root = prepare_root(dest, work)
            dest.unlink(missing_ok=True)
        else:
            root = prepare_root(path, work)
    except (ValueError, FileNotFoundError) as e:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(400, str(e))

    files, report = collect_files(root, max_file_kb, exclude.split(","))
    if not files:
        raise HTTPException(422, {"message": "No supported source files found", "report": report.summary()})
    ix = RepoIndex.build(name or root.name, root, files, report, embed_backend)
    ix.save()
    _CACHE[ix.repo_id] = ix
    return ix.meta()


@app.get("/repos")
def list_repos():
    out = []
    for d in sorted(config.DATA_DIR.iterdir()):
        m = d / "meta.json"
        if m.exists():
            out.append(json.loads(m.read_text()))
    return out


@app.get("/repos/{repo_id}/report")
def repo_report(repo_id: str):
    ix = get_index(repo_id)
    return {**ix.meta(), "skipped_files": ix.report.get("skipped_files", [])}


@app.get("/repos/{repo_id}/overview")
def repo_overview(repo_id: str):
    ix = get_index(repo_id)
    most_imported = sorted(ix.graph_in.items(), key=lambda kv: -len(kv[1]))[:10]
    largest = sorted(ix.files.items(), key=lambda kv: -kv[1]["n_lines"])[:10]
    return {
        **ix.meta(),
        "most_imported": [{"path": p, "imported_by": len(v)} for p, v in most_imported],
        "largest_files": [{"path": p, "lines": v["n_lines"], "symbols": len(v["symbols"])} for p, v in largest],
    }


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = Field(10, ge=1, le=50)
    mode: str = Field("hybrid", description=" | ".join(MODES))
    rewrite: bool = Field(True, description="Expand the query with LLM-suggested code terms")
    explain: bool = Field(True, description="Generate a grounded, cited explanation")
    include_tests: bool | None = None


@app.post("/repos/{repo_id}/search")
def search_repo(repo_id: str, req: SearchRequest):
    ix = get_index(repo_id)
    extra = llm.rewrite_query(req.query) if req.rewrite else []
    try:
        out = search(ix, req.query, req.top_k, req.mode, extra, req.include_tests)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if req.explain:
        if out["evidence"]["level"] in ("none", "weak"):
            out["explanation"] = {"status": "insufficient_evidence", "text": None,
                                  "message": out["evidence"]["message"]}
        else:
            out["explanation"] = llm.explain(req.query, out["results"])
    return out


@app.get("/repos/{repo_id}/file")
def get_file(repo_id: str, path: str, start: int = 1, end: int | None = None):
    ix = get_index(repo_id)
    if path not in ix.files:
        raise HTTPException(404, "File not in index")
    end = end or ix.files[path]["n_lines"]
    return {"path": path, "start_line": start, "end_line": end,
            "language": ix.files[path]["language"], "content": ix.read_lines(path, start, end),
            "symbols": ix.files[path]["symbols"]}


@app.get("/repos/{repo_id}/graph")
def file_graph(repo_id: str, path: str):
    ix = get_index(repo_id)
    if path not in ix.files:
        raise HTTPException(404, "File not in index")
    return {"path": path, "imports": ix.graph_out.get(path, []), "imported_by": ix.graph_in.get(path, [])}


@app.get("/repos/{repo_id}/symbols")
def find_symbols(repo_id: str, q: str, limit: int = 30):
    ix = get_index(repo_id)
    ql = q.lower()
    hits = [{"path": p, **s} for p, f in ix.files.items() for s in f["symbols"] if ql in s["name"].lower()]
    hits.sort(key=lambda h: (h["name"].lower().split(".")[-1] != ql, len(h["name"])))
    return hits[:limit]


@app.post("/repos/{repo_id}/eval")
async def run_eval(repo_id: str, questions: UploadFile = File(...), k: int = Form(5)):
    ix = get_index(repo_id)
    try:
        qs = json.loads(await questions.read())
    except json.JSONDecodeError:
        raise HTTPException(400, "questions must be a JSON array")
    return evaluate(ix, qs, k)
