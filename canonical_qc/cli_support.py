"""Stable one-line JSON envelopes shared by Canonical command entrypoints."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, NoReturn, TextIO


class CliUsageError(ValueError):
    pass


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CliUsageError(message)


def emit(payload: dict[str, Any], *, stream: TextIO | None = None) -> None:
    target = sys.stdout if stream is None else stream
    target.write(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def error_payload(
    *,
    command: str,
    category: str,
    code: str,
    stage: str,
    field: str | None,
    message: str,
    retryable: bool,
) -> dict[str, Any]:
    return {
        "ok": False,
        "command": command,
        "error": {
            "category": category,
            "code": code,
            "stage": stage,
            "field": field,
            "message": message,
            "retryable": retryable,
        },
    }


__all__ = ["CliUsageError", "JsonArgumentParser", "emit", "error_payload"]
