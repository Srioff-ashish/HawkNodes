"""HawkNodes -- Atlas Cloud nodes for ComfyUI.

Entry point. ComfyUI discovers custom nodes by importing this module and calling
``comfy_entrypoint()``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("HawkNodes")

__version__ = "0.1.0"

MIN_COMFYUI = "0.26.0"

try:
    from comfy_api.latest import ComfyExtension
except ImportError:  # pragma: no cover
    ComfyExtension = None
    logger.error(
        "HawkNodes needs ComfyUI >= %s for its node API (comfy_api.latest is missing). "
        "Update ComfyUI and restart; no HawkNodes nodes were loaded.",
        MIN_COMFYUI,
    )


if ComfyExtension is not None:
    from .hawknodes.nodes.documents import HawkDocuments
    from .hawknodes.nodes.i2i import HawkAtlasI2I
    from .hawknodes.nodes.llm import HawkAtlasLLM
    from .hawknodes.nodes.t2i import HawkAtlasT2I

    NODES = [HawkAtlasLLM, HawkDocuments, HawkAtlasT2I, HawkAtlasI2I]

    class HawkNodesExtension(ComfyExtension):
        async def get_node_list(self) -> list:
            return NODES

    async def comfy_entrypoint() -> HawkNodesExtension:
        return HawkNodesExtension()

    __all__ = [
        "HawkAtlasLLM",
        "HawkDocuments",
        "HawkAtlasT2I",
        "HawkAtlasI2I",
        "comfy_entrypoint",
    ]
