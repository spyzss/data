"""Annotation verification registry."""

from qc_common.registry import CheckRegistry

registry = CheckRegistry()
register = registry.register


def create_check(name: str, config: dict):
    return registry.create(name, config)


def available_checks() -> list[str]:
    return registry.names()
