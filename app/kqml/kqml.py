from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterator


class KQMLError(ValueError):
    pass


Atom = str | int | float | bool | None
Sexp = Atom | list["Sexp"]


_WS = re.compile(r"\s+")


def _tokenize(text: str) -> Iterator[str]:
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in ("(", ")"):
            yield ch
            i += 1
            continue
        if ch == '"':
            i += 1
            out = []
            while i < n:
                ch = text[i]
                if ch == "\\":
                    i += 1
                    if i >= n:
                        raise KQMLError("Unterminated escape in string.")
                    out.append(text[i])
                    i += 1
                    continue
                if ch == '"':
                    i += 1
                    break
                out.append(ch)
                i += 1
            else:
                raise KQMLError("Unterminated string.")
            yield '"' + "".join(out) + '"'
            continue

        j = i
        while j < n and (not text[j].isspace()) and text[j] not in ("(", ")"):
            j += 1
        yield text[i:j]
        i = j


def _parse_atom(tok: str) -> Atom:
    if tok.startswith('"') and tok.endswith('"') and len(tok) >= 2:
        return tok[1:-1]
    low = tok.lower()
    if low == "nil":
        return None
    if low == "t":
        return True
    if low == "f":
        return False
    try:
        if "." in tok:
            return float(tok)
        return int(tok)
    except ValueError:
        return tok


def parse_sexp(text: str) -> Sexp:
    tokens = list(_tokenize(text))
    pos = 0

    def parse_one() -> Sexp:
        nonlocal pos
        if pos >= len(tokens):
            raise KQMLError("Unexpected end of input.")
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            items: list[Sexp] = []
            while True:
                if pos >= len(tokens):
                    raise KQMLError("Unterminated list.")
                if tokens[pos] == ")":
                    pos += 1
                    return items
                items.append(parse_one())
        if tok == ")":
            raise KQMLError("Unexpected ')'.")
        return _parse_atom(tok)

    expr = parse_one()
    if pos != len(tokens):
        raise KQMLError("Trailing tokens after expression.")
    return expr


def _needs_quotes(s: str) -> bool:
    if s == "":
        return True
    if _WS.search(s):
        return True
    if any(ch in s for ch in ('(', ')', '"')):
        return True
    return False


def dump_sexp(expr: Sexp) -> str:
    if isinstance(expr, list):
        return "(" + " ".join(dump_sexp(x) for x in expr) + ")"
    if expr is None:
        return "nil"
    if expr is True:
        return "t"
    if expr is False:
        return "f"
    if isinstance(expr, (int, float)):
        return str(expr)
    s = str(expr)
    if _needs_quotes(s) or s.startswith(":"):
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return '"' + escaped + '"'
    return s


@dataclass(frozen=True)
class KQMLMessage:
    performative: str
    slots: dict[str, Sexp]

    def dump(self) -> str:
        parts: list[Sexp] = [self.performative]
        for k, v in self.slots.items():
            key = k if k.startswith(":") else ":" + k
            parts.append(key)
            parts.append(v)
        return dump_sexp(parts)


def parse_message(text: str) -> KQMLMessage:
    expr = parse_sexp(text)
    if not isinstance(expr, list) or not expr:
        raise KQMLError("KQML message must be a non-empty list.")
    perf = expr[0]
    if not isinstance(perf, str):
        raise KQMLError("Performative must be a symbol.")

    slots: dict[str, Sexp] = {}
    rest = expr[1:]
    if len(rest) % 2 != 0:
        raise KQMLError("Slots must be key/value pairs.")
    for i in range(0, len(rest), 2):
        key = rest[i]
        val = rest[i + 1]
        if not isinstance(key, str) or not key.startswith(":"):
            raise KQMLError("Slot keys must be keywords starting with ':'.")
        slots[key.lower()] = val
    return KQMLMessage(performative=str(perf).lower(), slots=slots)


def plain_obj_to_sexp(obj: Any) -> Sexp:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [plain_obj_to_sexp(x) for x in obj]
    if isinstance(obj, dict):
        out: list[Sexp] = ["dict"]
        for k, v in obj.items():
            out.append(":" + str(k))
            out.append(plain_obj_to_sexp(v))
        return out
    return str(obj)


def json_to_sexp(value: Any) -> Sexp:
    return plain_obj_to_sexp(json.loads(json.dumps(value)))
