#!/usr/bin/env python3
"""Run independent annotation verification package.

This entry point validates config and runner wiring. Real clip loading is
intentionally injected by downstream workflow code once the workflow diagram is
available.
"""

import argparse
import logging
import sys
from pathlib import Path

from annotation_verify import AnnotationVerifyRunner, load_annotation_verify_config


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic annotation verification")
    parser.add_argument("config", type=Path, help="Path to YAML config")
    args = parser.parse_args()

    config = load_annotation_verify_config(args.config)
    setup_logging(config.log_level)
    runner = AnnotationVerifyRunner(config)
    try:
        runner.run([])
    except Exception as exc:
        logging.getLogger(__name__).error(
            "Annotation verification failed: %s", exc, exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
