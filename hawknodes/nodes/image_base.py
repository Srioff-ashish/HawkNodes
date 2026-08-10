"""Shared plumbing for the two image nodes.

Text-to-image and image-to-image hit the same async ``generateImage`` endpoint;
they differ only in the model list and whether reference images are attached.
"""

from __future__ import annotations

import json

from comfy_api.latest import IO

from .. import registry
from ..client import DEFAULT_IMAGE_URL, generate_image, resolve_api_key
from ..images import stack_images
from .common import connection_inputs, progress_bar


def shared_inputs() -> list:
    """Everything both image nodes expose below the model dropdown."""
    return [
        IO.Int.Input(
            "seed",
            default=0,
            min=0,
            max=2147483647,
            control_after_generate=True,
            tooltip=(
                "0 lets the model pick its own seed -- but ComfyUI then reuses the "
                "cached image when nothing else changes. Set control_after_generate "
                "to randomize for a different image on every run."
            ),
        ),
        IO.String.Input(
            "negative_prompt",
            multiline=True,
            default="",
            optional=True,
            advanced=True,
            tooltip="What to avoid. Sent only when filled in; not every model accepts it.",
        ),
        IO.Int.Input(
            "n",
            default=1,
            min=1,
            max=4,
            optional=True,
            advanced=True,
            tooltip="How many images to request. Sent only when above 1.",
        ),
        IO.Combo.Input(
            "output_format",
            options=["jpeg", "png"],
            default="jpeg",
            optional=True,
            advanced=True,
            tooltip="Format Atlas encodes the result in. png is lossless but much larger.",
        ),
        *connection_inputs(DEFAULT_IMAGE_URL),
        IO.Float.Input(
            "poll_interval",
            default=2.0,
            min=0.5,
            max=30.0,
            step=0.5,
            optional=True,
            advanced=True,
            tooltip="Seconds between status checks while the job runs.",
        ),
        IO.Int.Input(
            "timeout",
            default=300,
            min=30,
            max=3600,
            optional=True,
            advanced=True,
            tooltip="Give up after this many seconds. Raise it for slow, high-resolution models.",
        ),
        IO.Int.Input(
            "max_retries",
            default=3,
            min=0,
            max=10,
            optional=True,
            advanced=True,
            tooltip="Retries for rate limits and server errors. Auth errors are never retried.",
        ),
        IO.String.Input(
            "extra_params",
            multiline=True,
            default="",
            optional=True,
            advanced=True,
            tooltip=(
                'Extra JSON merged into the request body, e.g. {"guidance_scale": 3.5}. '
                "Use this for parameters specific to a model that this node does not expose."
            ),
        ),
    ]


def build_payload(
    model: dict,
    prompt: str,
    *,
    seed: int,
    negative_prompt: str,
    n: int,
    output_format: str,
    extra_params: str,
    image_uris: list[str] | None = None,
) -> dict:
    """Assemble the generateImage body, omitting anything the user did not set.

    Models reject parameters they do not understand, so defaults are never sent
    speculatively -- `size` on 'default (model)' drops out entirely.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is empty. Describe the image you want.")

    payload: dict = {
        "model": registry.resolve_slug(model),
        "prompt": prompt.strip(),
        "output_format": output_format,
        "enable_base64_output": True,
        "enable_sync_mode": False,
    }

    size = model.get("size")
    if size and size != registry.SIZE_DEFAULT:
        payload["size"] = size
    for key in ("quality", "input_fidelity"):
        if model.get(key):
            payload[key] = model[key]

    if seed > 0:
        payload["seed"] = seed
    if negative_prompt and negative_prompt.strip():
        payload["negative_prompt"] = negative_prompt.strip()
    if n > 1:
        payload["n"] = n
    if image_uris:
        payload["images"] = image_uris

    if extra_params and extra_params.strip():
        try:
            extra = json.loads(extra_params)
        except json.JSONDecodeError as exc:
            raise ValueError(f"extra_params is not valid JSON: {exc}") from None
        if not isinstance(extra, dict):
            raise ValueError('extra_params must be a JSON object, e.g. {"guidance_scale": 3.5}')
        payload.update(extra)

    return payload


async def run_generation(
    payload: dict,
    *,
    api_url: str,
    api_key: str,
    poll_interval: float,
    timeout: int,
    max_retries: int,
) -> IO.NodeOutput:
    key = resolve_api_key(api_key)
    progress = progress_bar(max(1, int(timeout / max(poll_interval, 0.5))))

    images = await generate_image(
        api_url,
        key,
        payload,
        poll_interval=poll_interval,
        timeout=timeout,
        max_retries=max_retries,
        progress=progress,
    )

    info = {key_: value for key_, value in payload.items() if key_ != "images"}
    info["images_returned"] = len(images)
    if "images" in payload:
        info["reference_images"] = len(payload["images"])

    return IO.NodeOutput(stack_images(images), json.dumps(info, indent=2, ensure_ascii=False))
