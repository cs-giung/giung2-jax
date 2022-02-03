from fvcore.common.config import CfgNode


__all__ = [
    "CfgNode",
    "get_cfg",
]


CfgNode = CfgNode


def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.

    Returns:
        CfgNode instance.
    """
    from .defaults import _C
    return _C.clone()
