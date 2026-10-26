"""Invasive-capability marking, audit and enforcement.

``@invasive("reason")`` is the async-def analogue for probe-dependent
capabilities: reads or controls that exist in the offline engine/probe but
not on a real phone (elixirRaw, set_speed, ability runtime states, ...).

Three mechanisms mirror the async contagion:

1. Declaration: decorating a leaf records it in INVASIVE registry.
2. Runtime enforcement: in deploy mode any call raises InvasiveAccessError;
   in train/dev mode calls are counted for auditing.
3. Static contagion: ``python -m native_runner.check_invasive check`` walks
   the AST call graph and requires every transitive caller to be marked
   too -- exactly how async forces callers to become async.  Anything the
   serving entrypoints can reach must therefore be clean or explicitly
   accepted as documented deployment debt.
"""

from __future__ import annotations

import os
from collections import Counter
from functools import wraps

INVASIVE_CALLS: Counter[str] = Counter()
INVASIVE_REGISTRY: dict[str, str] = {}  # qualname -> reason

_POLICY_MODE = os.environ.get("CR_POLICY_MODE", "train")


class InvasiveAccessError(PermissionError):
    """An invasive capability was used in a mode that forbids it."""


def policy_mode() -> str:
    return os.environ.get("CR_POLICY_MODE", _POLICY_MODE)


def invasive(reason: str):
    """Mark a function as probe/engine-dependent (unavailable on a phone)."""

    def deco(fn):
        qualname = f"{fn.__module__}.{fn.__qualname__}"
        INVASIVE_REGISTRY[qualname] = reason

        @wraps(fn)
        def wrapper(*args, **kwargs):
            INVASIVE_CALLS[qualname] += 1
            if policy_mode() == "deploy":
                raise InvasiveAccessError(f"{qualname}: {reason}")
            return fn(*args, **kwargs)

        wrapper.__invasive__ = reason
        wrapper.__invasive_qualname__ = qualname
        return wrapper

    return deco
