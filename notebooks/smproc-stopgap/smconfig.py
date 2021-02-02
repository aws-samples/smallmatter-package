"""Configuration for calling SageMaker SDK.

This example disables debugger & profiling, while the rest configs are still the
defaults (e.g., SageMaker's VPC with public internet, no taggings, no KMS keys).

See also the original pattern: https://github.com/awslabs/mlmax/tree/main/notebooks/screening/
"""

from typing import Any, Dict

# typedefs
Kwargs = Dict[str, Any]


class SmKwargs:
    """Default kwargs to whatever in the SageMaker SDK."""

    def __init__(self, role):
        self.d = dict(role=role)

    @property
    def tags(self) -> None:
        return None

    @property
    def train(self) -> Kwargs:
        return {**self.d, "debugger_hook_config": False, "disable_profiler": True}

    @property
    def model(self) -> Kwargs:
        return self.d

    @property
    def bt(self) -> Kwargs:
        return {}

    @property
    def processing(self) -> Kwargs:
        return self.d
