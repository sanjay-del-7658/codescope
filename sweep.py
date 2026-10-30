import sys
from app import config
from app.index import RepoIndex
from app.evaluate import evaluate, load_questions

ix = RepoIndex.load(sys.argv[1])
qs = load_questions("eval/flask_questions.json")
rows = []
for b in (0.2, 0.35, 0.5, 0.7):
    for s in (0.1, 0.2, 0.3):
        config.WEIGHTS["st"] = {"bm25": b, "dense": 1.0, "symbol": s}
        m = evaluate(ix, qs, 5, modes=("hybrid",))["modes"]["hybrid"]
        rows.append((m["mrr"], m["recall@5"], m["hit@1"], b, s))
for r in sorted(rows, reverse=True)[:5]:
    print(f"MRR={r[0]}  Recall@5={r[1]}  Hit@1={r[2]}  ->  bm25={r[3]} symbol={r[4]}")