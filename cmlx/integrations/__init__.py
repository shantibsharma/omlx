"""Integration registry for external coding tools."""

from cmlx.integrations.base import Integration
from cmlx.integrations.codex import CodexIntegration
from cmlx.integrations.opencode import OpenCodeIntegration
from cmlx.integrations.openclaw import OpenClawIntegration

INTEGRATIONS: dict[str, Integration] = {
    "codex": CodexIntegration(),
    "opencode": OpenCodeIntegration(),
    "openclaw": OpenClawIntegration(),
}


def get_integration(name: str) -> Integration | None:
    """Get an integration by name."""
    return INTEGRATIONS.get(name)


def list_integrations() -> list[Integration]:
    """List all available integrations."""
    return list(INTEGRATIONS.values())


__all__ = [
    "Integration",
    "INTEGRATIONS",
    "get_integration",
    "list_integrations",
]
