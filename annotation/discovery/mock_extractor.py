"""Explicit mock discovery extractor for pipeline wiring tests."""

from __future__ import annotations

from .qwen_extractor import QwenExtractor


class MockExtractor(QwenExtractor):
    """Use QwenExtractor's lightweight mock extraction without loading a model."""

    def __init__(self) -> None:
        super().__init__(model_path=None)
