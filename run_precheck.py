#!/usr/bin/env python3
"""Run independent precheck package."""

import argparse
import logging
import sys
from pathlib import Path

from precheck import PrecheckRunner, load_precheck_config


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data trustworthiness prechecks")
    parser.add_argument("config", type=Path, help="Path to YAML config")
    args = parser.parse_args()

    config = load_precheck_config(args.config)
    setup_logging(config.log_level)
    try:
        PrecheckRunner(config).run()
    except Exception as exc:
        logging.getLogger(__name__).error("Precheck failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
