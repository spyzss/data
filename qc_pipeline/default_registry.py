"""Bind configured implementation names to concrete automatic QC runners."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qc_common.config import LoadedQcConfig
from qc_common.contracts import ModuleResult
from qc_common.module_registry import ModuleRegistry, ModuleRunner
from qc_pipeline.context import AssetContext
from qc_pipeline.runners import precheck, sam3_containment, video_quality


def build_default_registry(
    context: AssetContext,
    config: LoadedQcConfig,
    *,
    segmenter_factory: Callable[..., Any] | None = None,
) -> ModuleRegistry:
    """Build a fresh automatic-runner registry for one asset worker."""
    registry = ModuleRegistry()
    precheck_session = precheck.PrecheckSession(context, config)
    runners: dict[str, ModuleRunner] = {
        **{name: precheck_session.runner_for(name) for name in precheck.MODULES},
        "video_quality": video_quality.run,
        "sam3_containment": sam3_containment.runner(segmenter_factory),
    }
    for module_name, runner in runners.items():
        module_config = config.module_config(module_name)
        if not module_config.get("enabled"):
            continue
        implementation = module_config.get("implementation")
        if not isinstance(implementation, str) or not implementation:
            continue

        def asset_runner(
            runner_context: AssetContext,
            loaded_config: LoadedQcConfig,
            *,
            runner: ModuleRunner = runner,
        ) -> ModuleResult:
            if runner_context is not context:
                raise ValueError("per-asset registry cannot be shared across assets")
            return runner(runner_context, loaded_config)

        registry.register(implementation, asset_runner)
    return registry


__all__ = ["build_default_registry"]
