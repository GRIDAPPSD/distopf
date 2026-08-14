"""Decorator and helpers for recording method call arguments on results.

Used to make analysis runs reproducible for benchmarking: the decorator
captures the exact arguments passed to a method and attaches them to the
result's ``metadata`` attribute as a JSON-serializable dict structured as
``{"call": {...}, "provenance": {...}}``.
"""
from __future__ import annotations
import functools
import inspect
import json
import logging
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("distopf")


@functools.lru_cache(maxsize=1)
def _get_version() -> str | None:
    """Return the installed distopf package version, or None if unavailable."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("distopf")
        except PackageNotFoundError:
            return None
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _get_git_sha() -> str | None:
    """Return the current git short SHA, or None if not in a git checkout."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _is_json_safe(value) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _sanitize(arguments: dict) -> tuple[dict, bool]:
    """Return (sanitized_dict, replayable). Non-serializable values become
    descriptive placeholders and flip the replayable flag to False."""
    replayable = True
    out: dict = {}
    for k, v in arguments.items():
        if _is_json_safe(v):
            out[k] = v
        else:
            replayable = False
            out[k] = {
                "__nonserializable__": True,
                "type": type(v).__name__,
                "repr": repr(v),
            }
    return out, replayable


def _build_provenance() -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "distopf_version": _get_version(),
        "git_sha": _get_git_sha(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
    }


def record_call(fn):
    """Capture call arguments on the result's ``metadata`` attribute.

    Writes ``result.metadata`` as a dict with two keys:

    - ``call``: method name, arguments (defaults applied, **kwargs flattened),
      and a ``replayable`` flag.
    - ``provenance``: timestamp, package version, git SHA, hostname, Python version.

    Non-JSON-serializable values (e.g. a custom Callable objective) are stored
    as descriptive placeholders and ``replayable`` is set False.
    """
    sig = inspect.signature(fn)
    var_kw_name = next(
        (p.name for p in sig.parameters.values() if p.kind == p.VAR_KEYWORD),
        None,
    )

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("self", None)
        if var_kw_name:
            arguments.update(arguments.pop(var_kw_name, {}))

        serialized, replayable = _sanitize(arguments)

        meta = {
            "call": {
                "method": fn.__name__,
                "replayable": replayable,
                "arguments": serialized,
            },
            "provenance": _build_provenance(),
        }

        result = fn(self, *args, **kwargs)

        try:
            existing = getattr(result, "metadata", None) or {}
            existing.update(meta)
            result.metadata = existing
        except (AttributeError, TypeError):
            logger.debug(
                "record_call: result of %s has no settable metadata attribute",
                fn.__name__,
            )
        return result

    return wrapper