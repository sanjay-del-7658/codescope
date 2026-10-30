"""Tree-sitter based structural parsing: symbols, imports, calls, and function-level chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from tree_sitter import Language, Parser

from . import config


def _load_languages() -> dict[str, Language]:
    langs: dict[str, Language] = {}
    try:
        import tree_sitter_python as m
        langs["python"] = Language(m.language())
    except ImportError:
        pass
    try:
        import tree_sitter_javascript as m
        langs["javascript"] = Language(m.language())
    except ImportError:
        pass
    try:
        import tree_sitter_typescript as m
        langs["typescript"] = Language(m.language_typescript())
        langs["tsx"] = Language(m.language_tsx())
    except ImportError:
        pass
    try:
        import tree_sitter_java as m
        langs["java"] = Language(m.language())
    except ImportError:
        pass
    try:
        import tree_sitter_go as m
        langs["go"] = Language(m.language())
    except ImportError:
        pass
    return langs


LANGUAGES = _load_languages()
_PARSERS: dict[str, Parser] = {}

# node type -> symbol kind
SYMBOL_NODES = {
    "python": {"function_definition": "function", "class_definition": "class"},
    "javascript": {"function_declaration": "function", "generator_function_declaration": "function",
                   "class_declaration": "class", "method_definition": "method"},
    "java": {"class_declaration": "class", "interface_declaration": "interface", "enum_declaration": "enum",
             "record_declaration": "class", "method_declaration": "method", "constructor_declaration": "method"},
    "go": {"function_declaration": "function", "method_declaration": "method", "type_spec": "type"},
}
SYMBOL_NODES["typescript"] = {**SYMBOL_NODES["javascript"], "interface_declaration": "interface",
                              "abstract_class_declaration": "class", "type_alias_declaration": "type",
                              "enum_declaration": "enum"}
SYMBOL_NODES["tsx"] = SYMBOL_NODES["typescript"]

CONTAINER_KINDS = {"class", "interface", "enum"}

IMPORT_NODES = {
    "python": {"import_statement", "import_from_statement"},
    "javascript": {"import_statement"},
    "typescript": {"import_statement"},
    "tsx": {"import_statement"},
    "java": {"import_declaration"},
    "go": {"import_spec"},
}
CALL_NODES = {
    "python": ("call", "function"),
    "javascript": ("call_expression", "function"),
    "typescript": ("call_expression", "function"),
    "tsx": ("call_expression", "function"),
    "java": ("method_invocation", "name"),
    "go": ("call_expression", "function"),
}
COMMENT_TYPES = {"comment", "line_comment", "block_comment"}


@dataclass
class Symbol:
    name: str
    qualname: str
    kind: str
    start_line: int     # 1-based inclusive
    end_line: int
    signature: str
    doc: str
    parent: str | None
    calls: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    path: str
    language: str
    start_line: int
    end_line: int
    text: str
    symbol: str | None      # qualified name, e.g. "Flask.add_url_rule"
    kind: str               # function | method | class | module | ...
    signature: str = ""
    doc: str = ""
    calls: list[str] = field(default_factory=list)


@dataclass
class ParsedFile:
    path: str
    language: str
    symbols: list[Symbol]
    imports: list[str]
    chunks: list[Chunk]
    n_lines: int


def _parser(lang: str) -> Parser | None:
    if lang not in LANGUAGES:
        return None
    if lang not in _PARSERS:
        _PARSERS[lang] = Parser(LANGUAGES[lang])
    return _PARSERS[lang]


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _name_of(node, lang: str, src: bytes) -> str | None:
    n = node.child_by_field_name("name")
    if n is not None:
        return _text(n, src)
    return None


def _js_arrow_name(node, src: bytes) -> str | None:
    """`const handler = (req) => {...}` -> 'handler'."""
    if node.type == "variable_declarator":
        val = node.child_by_field_name("value")
        if val is not None and val.type in ("arrow_function", "function_expression", "function"):
            n = node.child_by_field_name("name")
            if n is not None:
                return _text(n, src)
    return None


def _docstring(node, lang: str, src: bytes) -> str:
    if lang == "python":
        body = node.child_by_field_name("body")
        if body is not None and body.named_child_count:
            first = body.named_children[0]
            if first.type == "expression_statement" and first.named_child_count and \
                    first.named_children[0].type == "string":
                return _text(first.named_children[0], src).strip("\"' \n")[:600]
        return ""
    # Other languages: comments immediately above the declaration (or above its `export`)
    if node.parent is not None and node.parent.type in ("export_statement", "decorator"):
        node = node.parent
    comments = []
    prev = node.prev_sibling
    while prev is not None and prev.type in COMMENT_TYPES and \
            (node.start_point[0] - prev.end_point[0]) <= 1 + len(comments):
        comments.insert(0, _text(prev, src))
        prev = prev.prev_sibling
    return "\n".join(comments)[:600]


def _signature(node, src: bytes) -> str:
    text = _text(node, src)
    first = text.split("{", 1)[0] if "{" in text.split("\n", 1)[0] else text.split("\n", 1)[0]
    return first.strip()[:200]


def _collect_calls(node, lang: str, src: bytes, limit: int = 60) -> list[str]:
    ctype, field_name = CALL_NODES.get(lang, (None, None))
    if ctype is None:
        return []
    out: list[str] = []
    stack = [node]
    while stack and len(out) < limit:
        n = stack.pop()
        if n.type == ctype:
            f = n.child_by_field_name(field_name)
            if f is not None:
                name = _text(f, src).split("(")[0]
                name = name.replace("?.", ".").split(".")[-1].strip()
                if name and name.isidentifier() and name not in out:
                    out.append(name)
        stack.extend(reversed(n.children))
    return out


def parse_file(path: str, language: str, source: str) -> ParsedFile:
    lines = source.splitlines()
    n_lines = len(lines)
    p = _parser(language)
    symbols: list[Symbol] = []
    imports: list[str] = []

    if p is not None:
        src = source.encode("utf-8", errors="replace")
        tree = p.parse(src)
        sym_types = SYMBOL_NODES.get(language, {})
        imp_types = IMPORT_NODES.get(language, set())

        def walk(node, parents: list[str], parent_kind: str | None = None):
            for child in node.children:
                kind = sym_types.get(child.type)
                name = None
                if kind:
                    name = _name_of(child, language, src)
                elif language in ("javascript", "typescript", "tsx"):
                    name = _js_arrow_name(child, src)
                    if name:
                        kind = "function"
                if child.type in imp_types:
                    imports.append(_text(child, src).strip())
                if kind and name:
                    # Include decorators (python) in the symbol range
                    outer = child.parent if child.parent is not None and child.parent.type == "decorated_definition" else child
                    if language == "python" and kind == "function" and parent_kind in CONTAINER_KINDS:
                        kind = "method"
                    scope = parents
                    if language == "go" and child.type == "method_declaration":
                        recv = child.child_by_field_name("receiver")
                        rt = re.search(r"\*?\s*([A-Za-z_]\w*)\s*\)?\s*$", _text(recv, src).rstrip(")")) if recv else None
                        if rt:
                            scope = parents + [rt.group(1)]
                    qual = ".".join(scope + [name])
                    symbols.append(Symbol(
                        name=name, qualname=qual, kind=kind,
                        start_line=outer.start_point[0] + 1, end_line=child.end_point[0] + 1,
                        signature=_signature(child, src), doc=_docstring(child, language, src),
                        parent=".".join(scope) or None,
                        calls=_collect_calls(child, language, src) if kind not in CONTAINER_KINDS else [],
                    ))
                    walk(child, scope + [name], kind)
                else:
                    walk(child, parents, parent_kind)

        walk(tree.root_node, [])

    chunks = _make_chunks(path, language, lines, symbols)
    return ParsedFile(path, language, symbols, imports, chunks, n_lines)


def _slice(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start - 1:end])


def _make_chunks(path: str, language: str, lines: list[str], symbols: list[Symbol]) -> list[Chunk]:
    chunks: list[Chunk] = []
    covered = [False] * (len(lines) + 2)
    syms = sorted(symbols, key=lambda s: (s.start_line, -s.end_line))

    for s in syms:
        children = [c for c in syms if c.parent == s.qualname]
        if s.kind in CONTAINER_KINDS and children:
            # Container: index only its header (up to first member) so methods aren't duplicated
            end = max(s.start_line, min(c.start_line for c in children) - 1)
        else:
            end = s.end_line
        # Split very long symbols into overlapping windows
        start = s.start_line
        while start <= end:
            stop = min(end, start + config.MAX_CHUNK_LINES - 1)
            chunks.append(Chunk(path, language, start, stop, _slice(lines, start, stop),
                                s.qualname, s.kind, s.signature, s.doc, s.calls))
            for i in range(start, stop + 1):
                covered[i] = True
            if stop == end:
                break
            start = stop - config.CHUNK_OVERLAP + 1

    # Module-level code not inside any symbol (imports, constants, route tables, ...)
    i = 1
    while i <= len(lines):
        if covered[i]:
            i += 1
            continue
        j = i
        while j <= len(lines) and not covered[j] and j - i < config.MODULE_WINDOW:
            j += 1
        text = _slice(lines, i, j - 1)
        if text.strip():
            chunks.append(Chunk(path, language, i, j - 1, text, None, "module"))
        i = j
    return chunks
