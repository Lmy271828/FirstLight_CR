"""Static contagion checker for @invasive (see invasive.py).

Walks every function/class-method definition under native_runner, builds a
caller -> callee graph from AST call expressions, and computes the
transitive closure of "infected" functions (those that reach an
@invasive-decorated leaf).

Modes:
  check   -- exit 1 if any function reaches an invasive leaf without being
             itself marked (async-style: callers must become async);
  report  -- print the infected set and the reasons flowing into each.

Known limits (documented, same as any static analysis): dynamic dispatch
(``getattr``/monkeypatching) and calls through variables holding functions
are not traced; decorate such indirection points explicitly.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def _iter_defs(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _qualified(module: str, node: ast.FunctionDef) -> str:
    return f"{module}.{node.name}"


def analyze(root: Path = PACKAGE_ROOT):
    """Return (defined, calls, decorated) over all .py files under root."""
    defined: dict[str, dict] = {}   # qualname -> {module, node, file}
    calls: dict[str, set[str]] = {}  # caller qualname -> callee names (raw)
    decorated: set[str] = set()

    for file in sorted(root.rglob("*.py")):
        module = ".".join(file.relative_to(root.parent).with_suffix("").parts)
        tree = ast.parse(file.read_text(encoding="utf-8"))

        # pass 1: collect function defs and @invasive marks
        scope: list[str] = []
        parents: dict[int, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[id(child)] = parent

        for node in _iter_defs(tree):
            qual = _qualified(module, node)
            defined[qual] = {"module": module, "node": node, "file": file}
            for dec in node.decorator_list:
                name = dec.func.id if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) else (
                    dec.id if isinstance(dec, ast.Name) else "")
                if name == "invasive":
                    decorated.add(qual)

        # pass 2: calls inside each function body.  Callees are recorded as
        # raw names: method calls lose their class (self.set_speed ->
        # 'set_speed'), which is exactly the granularity the audit needs.
        for node in _iter_defs(tree):
            caller = _qualified(module, node)
            calls.setdefault(caller, set())
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    if isinstance(fn, ast.Name):
                        calls[caller].add(fn.id)
                    elif isinstance(fn, ast.Attribute):
                        calls[caller].add(fn.attr)  # self.set_speed -> 'set_speed'
    return defined, calls, decorated


def infect(defined, calls, decorated):
    """Transitive closure: leaf decorated set + anyone calling an infected name."""
    by_name: dict[str, set[str]] = {}
    for qual in defined:
        by_name.setdefault(qual.rsplit(".", 1)[-1], set()).add(qual)

    infected = set(decorated)
    changed = True
    while changed:
        changed = False
        for caller, callees in calls.items():
            if caller in infected:
                continue
            for c in callees:
                # a raw name matches any defined function with that short name
                if c in by_name and (by_name[c] & infected):
                    infected.add(caller)
                    changed = True
                    break
    return infected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["check", "report", "freeze"])
    ap.add_argument("--allow", type=Path, default=PACKAGE_ROOT / "invasive_allowlist.txt",
                    help="allowlist of accepted infected callers (check mode)")
    args = ap.parse_args()

    defined, calls, decorated = analyze()
    infected = infect(defined, calls, decorated)
    unmarked = sorted(infected - decorated)

    if args.mode == "report":
        print(f"decorated leaves: {len(decorated)}")
        for qual in sorted(decorated):
            print(f"  leaf   {qual}")
        print(f"infected (transitive callers): {len(infected - decorated)}")
        for qual in unmarked:
            print(f"  taint  {qual}  ({defined[qual]['file'].name})")
        return

    if args.mode == "freeze":
        args.allow.write_text("\n".join(unmarked) + "\n", encoding="utf-8")
        print(f"froze {len(unmarked)} accepted callers into {args.allow}")
        return

    allowed = set()
    if args.allow.exists():
        allowed = {line.strip() for line in args.allow.read_text(encoding="utf-8").splitlines()
                   if line.strip() and not line.startswith("#")}
    new_taint = [qual for qual in unmarked if qual not in allowed]
    if new_taint:
        print(f"invasive contagion violated: {len(new_taint)} caller(s) reach an "
              f"@invasive leaf without being marked or allowlisted:")
        for qual in new_taint:
            print(f"  {qual}  ({defined[qual]['file'].name})")
        raise SystemExit(1)
    print(f"ok: {len(decorated)} invasive leaves, {len(unmarked)} infected callers "
          f"({len(unmarked) - len(new_taint)} allowlisted, 0 new)")


if __name__ == "__main__":
    main()
