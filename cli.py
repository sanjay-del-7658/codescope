"""Command-line helper.

  python cli.py index <folder-or-archive> [--exclude "docs/*,examples/*"] [--max-kb 500] [--embed lsa]
  python cli.py search <repo_id> "how are routes registered?" [--mode hybrid] [-k 5] [--explain]
  python cli.py eval <repo_id> eval/flask_questions.json [-k 5]
  python cli.py list
"""
import argparse
import json
import tempfile
from pathlib import Path

from app import config, llm
from app.evaluate import evaluate, load_questions
from app.index import RepoIndex
from app.ingest import collect_files, prepare_root
from app.search import MODES, search


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index"); p.add_argument("source"); p.add_argument("--exclude", default="")
    p.add_argument("--max-kb", type=int, default=config.MAX_FILE_KB); p.add_argument("--embed", default=None)
    p.add_argument("--name", default=None)
    s = sub.add_parser("search"); s.add_argument("repo_id"); s.add_argument("query")
    s.add_argument("--mode", default="hybrid", choices=MODES); s.add_argument("-k", type=int, default=5)
    s.add_argument("--explain", action="store_true")
    e = sub.add_parser("eval"); e.add_argument("repo_id"); e.add_argument("questions"); e.add_argument("-k", type=int, default=5)
    sub.add_parser("list")
    a = ap.parse_args()

    if a.cmd == "index":
        work = config.DATA_DIR / "_uploads" / Path(tempfile.mkdtemp()).name
        work.mkdir(parents=True, exist_ok=True)
        root = prepare_root(a.source, work)
        files, report = collect_files(root, a.max_kb, a.exclude.split(","))
        ix = RepoIndex.build(a.name or root.name, root, files, report, a.embed, progress=print)
        ix.save()
        print(json.dumps(ix.meta(), indent=2))
    elif a.cmd == "search":
        ix = RepoIndex.load(a.repo_id)
        extra = llm.rewrite_query(a.query) if a.explain else []
        out = search(ix, a.query, a.k, a.mode, extra)
        print(f"Evidence: {out['evidence']}  terms={out['terms']} expanded={out['expanded_terms']}\n")
        for r in out["results"]:
            print(f"[{r['id']}] {r['path']}  L{r['start_line']}-L{r['end_line']}  {r['kind']} {r['symbol'] or ''}"
                  f"  score={r['score']}  ({r['evidence_strength']})")
            print(f"     why: ranks={r['why']['ranks']} matched={r['why']['matched_terms']} boosts={r['why']['boosts']}")
        if a.explain:
            print("\n--- AI explanation (generated; verify against evidence) ---")
            print(json.dumps(llm.explain(a.query, out["results"]), indent=2))
    elif a.cmd == "eval":
        ix = RepoIndex.load(a.repo_id)
        rep = evaluate(ix, load_questions(a.questions), a.k)
        print(json.dumps(rep["modes"], indent=2))
        for q in rep["per_question"]:
            print(f"{q['id']}: " + "  ".join(f"{m}={q[m]['first_relevant_rank']}" for m in MODES))
    elif a.cmd == "list":
        for d in sorted(config.DATA_DIR.iterdir()):
            if (d / "meta.json").exists():
                m = json.loads((d / "meta.json").read_text())
                print(m["repo_id"], m["name"], m["indexed"], "files", m["chunks"], "chunks", m["embedding_backend"])


if __name__ == "__main__":
    main()
