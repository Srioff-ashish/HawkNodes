"""HawkAtlasI2I -- image to image / edit via Atlas Cloud."""

from __future__ import annotations

from comfy_api.latest import IO

from .. import registry
from ..client import DEFAULT_IMAGE_URL
from ..images import tensor_batch_to_data_uris
from .common import CATEGORY
from .image_base import build_payload, run_generation, shared_inputs

MAX_REFERENCE_IMAGES = 6


class HawkAtlasI2I(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HawkAtlasI2I",
            display_name="Hawk Atlas Image to Image",
            category=CATEGORY,
            description=(
                "Edits or reimagines reference images with an Atlas Cloud edit model. "
                "Connect one image to transform it, or several to blend and combine them."
            ),
            search_aliases=["atlas", "i2i", "image to image", "img2img", "edit", "inpaint"],
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="The image to edit. Batches are sent as multiple references.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="How to change the image. Accepts a connection from another node.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=registry.image_options("edit"),
                    tooltip=(
                        "Atlas edit model. Only edit-capable models are listed, and the "
                        "widgets underneath change to match the one you pick."
                    ),
                ),
                IO.Autogrow.Input(
                    "extra_images",
                    template=IO.Autogrow.TemplateNames(
                        IO.Image.Input("image"),
                        names=[f"image_{i}" for i in range(2, MAX_REFERENCE_IMAGES + 1)],
                        min=0,
                    ),
                    optional=True,
                    tooltip=(
                        "Additional reference images. A new slot appears each time you "
                        "connect one."
                    ),
                ),
                IO.Int.Input(
                    "reference_max_side",
                    default=1536,
                    min=256,
                    max=4096,
                    step=64,
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "References are downscaled to this longest edge before being "
                        "base64-encoded, to keep the request under size limits."
                    ),
                ),
                *shared_inputs(),
            ],
            outputs=[
                IO.Image.Output("images", tooltip="The edited image(s)."),
                IO.String.Output("info", tooltip="The request that was sent, for debugging."),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image,
        prompt: str,
        model: dict,
        extra_images: dict | None = None,
        reference_max_side: int = 1536,
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
        tensors = [image] + [
            tensor for tensor in (extra_images or {}).values() if tensor is not None
        ]

        image_uris: list[str] = []
        for tensor in tensors:
            image_uris.extend(
                tensor_batch_to_data_uris(
                    tensor, max_side=reference_max_side, image_format=output_format
                )
            )
        if not image_uris:
            raise ValueError("No reference image connected.")

        payload = build_payload(
            model,
            prompt,
            seed=seed,
            negative_prompt=negative_prompt,
            n=n,
            output_format=output_format,
            extra_params=extra_params,
            image_uris=image_uris,
        )
        return await run_generation(
            payload,
            api_url=api_url,
            api_key=api_key,
            poll_interval=poll_interval,
            timeout=timeout,
            max_retries=max_retries,
        )
