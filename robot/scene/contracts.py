"""Isaac-independent helpers for validating composed scene contracts."""

from typing import Any


def find_missing_prim_paths(
    stage: Any,
    required_prim_paths: tuple[str, ...],
) -> tuple[str, ...]:
    """Return required USD paths that are absent from the composed stage."""

    return tuple(
        prim_path
        for prim_path in required_prim_paths
        if not stage.GetPrimAtPath(prim_path).IsValid()
    )
