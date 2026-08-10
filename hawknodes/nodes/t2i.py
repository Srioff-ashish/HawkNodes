"""HawkAtlasT2I -- text to image via Atlas Cloud."""

from __future__ import annotations

from comfy_api.latest import IO

from .. import registry
from ..client import DEFAULT_IMAGE_URL
from .common import CATEGORY
from .image_base import build_payload, run_generation, shared_inputs


class HawkAtlasT2I(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HawkAtlasT2I",
            display_name="Hawk Atlas Text to Image",
            category=CATEGORY,
            description=(
                "Generates an image from a text prompt using an Atlas Cloud image "
                "model. Connect Hawk Atlas LLM to the prompt to expand a rough idea "
                "into a detailed one first."
            ),
            search_aliases=["atlas", "t2i", "text to image", "txt2img"],
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="What to generate. Accepts a connection from another node.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=registry.image_options("t2i"),
                    tooltip=(
                        "Atlas image model. The widgets underneath change to match the "
                        "model, so you only ever see parameters it accepts."
                    ),
                ),
                *shared_inputs(),
            ],
            outputs=[
                IO.Image.Output("images", tooltip="The generated image(s)."),
                IO.String.Output("info", tooltip="The request that was sent, for debugging."),
            ],
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: dict,
        seed: int = 0,
        negative_prompt: str = "",
        n: int = 1,
        output_format: str = "jpeg",
        api_url: str = DEFAULT_IMAGE_URL,
        api_key: str = "",
        poll_interval: float = 2.0,
        timeout: int = 300,
        max_retries: int = 3,
        extra_params: str = "",
    ) -> IO.NodeOutput:
        payload = build_payload(
            model,
            prompt,
            seed=seed,
            negative_prompt=negative_prompt,
            n=n,
            output_format=output_format,
            extra_params=extra_params,
        )
        return await run_generation(
            payload,
            api_url=api_url,
            api_key=api_key,
            poll_interval=poll_interval,
            timeout=timeout,
            max_retries=max_retries,
        )
