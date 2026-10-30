"""Run: pytest -q"""
from app.llm import validate_citations
from app.parser import parse_file
from app.text import tokenize


def test_tokenize_splits_identifiers():
    toks = tokenize("getUserById add_url_rule")
    assert "getuserbyid" in toks and "user" in toks and "add_url_rule" in toks and "rule" in toks


def test_parser_extracts_symbols_and_lines():
    src = 'import os\n\nclass A:\n    """Doc."""\n    def run(self):\n        return go()\n\ndef helper():\n    pass\n'
    pf = parse_file("m.py", "python", src)
    names = {s.qualname: (s.kind, s.start_line, s.end_line) for s in pf.symbols}
    assert names["A"][0] == "class" and names["A.run"] == ("method", 5, 6) and names["helper"] == ("function", 8, 9)
    assert pf.imports == ["import os"]
    assert any(c.symbol == "A.run" and c.start_line == 5 for c in pf.chunks)


def test_js_symbols():
    src = "import x from './x'\nexport const handler = async (req) => {\n  return y()\n}\nclass Svc { start() { } }\n"
    pf = parse_file("a.js", "javascript", src)
    q = {s.qualname for s in pf.symbols}
    assert {"handler", "Svc", "Svc.start"} <= q


def test_citation_validator_flags_fabrications():
    used = [{"id": "S1", "path": "src/a.py", "start_line": 1, "end_line": 9}]
    out = validate_citations("Sessions are signed in get_signer [S1]. Also see [S7] and src/zzz.py.", used)
    assert out["status"] == "needs_review"
    assert out["invalid_citations"] == ["S7"] and "[unverified]" in out["text"]
    assert out["unsupported_paths"] == ["src/zzz.py"]
    ok = validate_citations("Signed in [S1].", used)
    assert ok["status"] == "ok" and ok["citations"][0]["lines"] == "L1-L9"


def test_llm_flow_with_mocked_featherless(monkeypatch):
    import httpx
    from app import config, llm
    monkeypatch.setattr(config, "LLM_API_KEY", "test")
    sent = {}

    class R:
        def __init__(self, content): self.content = content
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": self.content}}]}

    def fake_post(url, headers, json, timeout):
        sent["url"], sent["body"] = url, json
        if "JSON array" in json["messages"][0]["content"]:
            return R('["get_signing_serializer", "itsdangerous"]')
        return R("The cookie is signed by a URLSafeTimedSerializer [S1].\nStart here: [S1]")

    monkeypatch.setattr(httpx, "post", fake_post)
    assert llm.rewrite_query("where is the cookie signed") == ["get_signing_serializer", "itsdangerous"]
    res = [{"id": "S1", "path": "src/flask/sessions.py", "start_line": 307, "end_line": 318,
            "kind": "method", "symbol": "X.get_signing_serializer", "language": "python", "snippet": "def f():\n  pass"}]
    out = llm.explain("where is the cookie signed", res)
    assert out["status"] == "ok" and sent["url"].endswith("/chat/completions")
    assert "src/flask/sessions.py L307" in sent["body"]["messages"][1]["content"]
