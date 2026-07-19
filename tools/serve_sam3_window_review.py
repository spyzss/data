#!/usr/bin/env python3
"""Serve the standalone SAM3 containment window-review workbench."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from human_qc.sam3_window_review import (  # noqa: E402
    Sam3WindowReviewStore,
    load_review_bundle,
)
from human_qc.sam3_window_review_server import (  # noqa: E402
    create_sam3_window_review_server,
)


LOGGER = logging.getLogger("serve_sam3_window_review")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--review-queue", required=True, type=Path)
    parser.add_argument("--evidence-manifest", required=True, type=Path)
    parser.add_argument("--review-dir", required=True, type=Path)
    parser.add_argument("--save-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8898, type=int)
    parser.add_argument("--log-level", default="INFO")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    bundle = load_review_bundle(
        manifest_path=args.manifest,
        queue_path=args.review_queue,
        evidence_path=args.evidence_manifest,
        review_dir=args.review_dir,
    )
    store = Sam3WindowReviewStore(bundle, args.save_dir)
    server = create_sam3_window_review_server(
        args.host,
        args.port,
        bundle=bundle,
        store=store,
    )
    LOGGER.info("Loaded %d SAM3 review windows", len(bundle.items))
    LOGGER.info("Serving SAM3 window review at http://%s:%s", args.host, server.server_port)
    LOGGER.info("Authoritative review state will be written to %s", args.save_dir.resolve())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
