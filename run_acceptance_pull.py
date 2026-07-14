#!/usr/bin/env python3
"""Run supplier batch sampling and pull workflow."""

from __future__ import annotations

import sys

from acceptance_pull.cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
