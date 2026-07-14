#!/usr/bin/env python3
"""Run no-reference video quality checks for a sampled supplier batch."""

from __future__ import annotations

import sys

from acceptance_pull.video_quality import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
