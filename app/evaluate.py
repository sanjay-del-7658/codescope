"""Retrieval evaluation: Hit@1, Recall@k, MRR per question and per retrieval mode (ablation)."""
from __future__ import annotations

import json
from pathlib import Path

from .index import RepoIndex
from .search import MODES, search


def _matches(result: dict, gold: dict) -> bool:
    if not result["path"].endswith(gold["path"]):
        return False
    sym = gold.get("symbol")
    if not sym:
        return True
    rs = result.get("symbol") or ""
    return rs == sym or rs.endswith("." + sym) or rs.startswith(sym + ".")


def evaluate(ix: RepoIndex, questions: list[dict], k: int = 5, modes=MODES) -> dict:
    report = {"k": k, "n_questions": len(questions), "modes": {}, "per_question": []}
    per_q = {q["id"]: {"id": q["id"], "question": q["question"]} for q in questions}
    for mode in modes:
        hit1 = rec = mrr = 0.0
        for q in questions:
            res = search(ix, q["question"], top_k=max(k, 10), mode=mode)["results"]
            gold = q["relevant"]
            found_ranks = []
            for g in gold:
                rank = next((i + 1 for i, r in enumerate(res) if _matches(r, g)), None)
                found_ranks.append(rank)
            first = min([r for r in found_ranks if r], default=None)
            recall = sum(1 for r in found_ranks if r and r <= k) / len(gold)
            hit1 += 1 if first == 1 else 0
            rec += recall
            mrr += 1 / first if first else 0
            per_q[q["id"]][mode] = {"first_relevant_rank": first, f"recall@{k}": round(recall, 2),
                                    "top3": [f"{r['path']}:{r['symbol'] or 'module'} L{r['start_line']}"
                                             for r in res[:3]]}
        n = len(questions) or 1
        report["modes"][mode] = {"hit@1": round(hit1 / n, 3), f"recall@{k}": round(rec / n, 3),
                                 "mrr": round(mrr / n, 3)}
    report["per_question"] = list(per_q.values())
    return report


def load_questions(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())
