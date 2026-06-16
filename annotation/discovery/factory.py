"""Factory for configurable discovery extractors."""

from __future__ import annotations

from .base import ObjectDiscoverer
from .manual_extractor import ManualExtractor
from .mock_extractor import MockExtractor
from .qwen_extractor import QwenExtractor
from .rule_extractor import RuleBasedExtractor


class NormalizingDiscoverer(ObjectDiscoverer):
    """Apply shared query cleanup around a configured extractor."""

    def __init__(self, inner: ObjectDiscoverer):
        self.inner = inner

    def discover_objects(self, instruction: str, config: dict) -> list[str]:
        queries = self.inner.discover_objects(instruction, config)
        queries.extend(config.get("always_include", []))
        normalized = {
            " ".join(str(query).strip().lower().split())
            for query in queries
            if str(query).strip()
        }
        return sorted(normalized)


def create_discoverer(config: dict) -> ObjectDiscoverer:
    extractor = config.get("extractor", "qwen")
    if extractor == "rule":
        return NormalizingDiscoverer(RuleBasedExtractor())
    if extractor == "mock":
        return NormalizingDiscoverer(MockExtractor())
    if extractor == "manual":
        return NormalizingDiscoverer(ManualExtractor())
    if extractor == "qwen":
        # TODO: Intern should implement real Qwen model loading in QwenExtractor.
        # The interface must remain discover_objects(instruction, config) -> list[str].
        return NormalizingDiscoverer(QwenExtractor(model_path=config.get("qwen_model_path")))
    raise ValueError(f"Unsupported discovery extractor: {extractor}")
