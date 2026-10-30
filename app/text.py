"""Code-aware tokenization: splits camelCase / snake_case, light stemming, stopwords."""
import re

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOPWORDS = set("""
a an the and or not of to in on for with by from at as is are was were be been being it its this that these those
how where what which who whom when why does do did done can could should would will shall may might must i we you
he she they them our your their my me us there here into over under about than then so if else elif also just any
all some each every no yes get gets got show tell explain implemented implement implementation code file files
function functions method methods class classes defined define happen happens work works logic
self none true false return returns def var let const new
""".split())


try:
    import snowballstemmer
    _STEMMER = snowballstemmer.stemmer("english")
except ImportError:  # pragma: no cover
    _STEMMER = None

_STEM_CACHE: dict[str, str] = {}
_IRREGULAR = {"registration": "regist", "registry": "regist", "authentication": "authent",
              "authorization": "author", "configuration": "config", "initialization": "init",
              "initialize": "init", "initial": "init", "serialization": "serial", "validation": "valid", "application": "app", "applications": "app",
              "requests": "request", "responses": "response", "templates": "templat"}


def stem(w: str) -> str:
    """Snowball stemming (routes/routing -> rout), with a few code-specific normalizations."""
    if w in _STEM_CACHE:
        return _STEM_CACHE[w]
    if w in _IRREGULAR:
        out = _IRREGULAR[w]
    elif _STEMMER is not None:
        out = _STEMMER.stemWord(w)
    else:
        out = w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w
    _STEM_CACHE[w] = out
    return out


def split_identifier(tok: str) -> list[str]:
    parts = []
    for piece in tok.split("_"):
        parts.extend(m.group(0) for m in _CAMEL.finditer(piece))
    return [p.lower() for p in parts if p]


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    out: list[str] = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text):
        low = raw.lower().strip("_")
        subs = split_identifier(raw)
        if len(subs) > 1 and low:
            out.append(low)                     # keep the full identifier too
        for s in subs:
            if keep_stopwords or (s not in STOPWORDS and len(s) > 1):
                out.append(stem(s))
    return out


def query_terms(q: str) -> list[str]:
    seen, out = set(), []
    for t in tokenize(q):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def looks_like_identifier(q: str) -> bool:
    q = q.strip()
    return bool(re.fullmatch(r"[A-Za-z_][\w.]*(\(\))?", q)) and (
        "_" in q or "." in q or any(c.isupper() for c in q[1:]) or len(q.split()) == 1)
