"""Conversions between ComfyUI IMAGE tensors and the wire formats Atlas expects.

A ComfyUI IMAGE is a float tensor of shape ``[B, H, W, C]`` with values in 0..1.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
from PIL import Image

# Vision requests carry images inline as base64. A raw 1024x1024 PNG is roughly
# 1.5 MB once encoded, so a handful of full-size frames will hit request size
# limits -- hence the downscale before encoding.
DEFAULT_MAX_SIDE = 1024


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """One frame of an IMAGE tensor to PIL."""
    if image.dim() == 4:
        image = image[0]
    array = (image.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    if array.shape[2] == 4:
        return Image.fromarray(array, mode="RGBA")
    if array.shape[2] == 3:
        return Image.fromarray(array, mode="RGB")
    return Image.fromarray(array[:, :, 0], mode="L")


def downscale(image: Image.Image, max_side: int) -> Image.Image:
    """Shrink so the longest edge is at most ``max_side``. Never upscales."""
    if max_side <= 0:
        return image
    longest = max(image.size)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(new_size, Image.LANCZOS)


def pil_to_data_uri(image: Image.Image, image_format: str = "png") -> str:
    image_format = image_format.lower()
    if image_format in ("jpg", "jpeg"):
        pil_format, mime = "JPEG", "image/jpeg"
        if image.mode in ("RGBA", "P", "LA"):
            image = image.convert("RGB")  # JPEG has no alpha channel
    else:
        pil_format, mime = "PNG", "image/png"

    buffer = io.BytesIO()
    image.save(buffer, format=pil_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def tensor_batch_to_data_uris(
    image: torch.Tensor,
    *,
    max_side: int = DEFAULT_MAX_SIDE,
    image_format: str = "png",
) -> list[str]:
    """Every frame of a batch becomes its own data URI."""
    if image is None:
        return []
    if image.dim() == 3:
        image = image.unsqueeze(0)

    uris = []
    for index in range(image.shape[0]):
        frame = downscale(tensor_to_pil(image[index]), max_side)
        uris.append(pil_to_data_uri(frame, image_format))
    return uris


def bytes_to_tensor(data: bytes) -> torch.Tensor:
    """Decoded image bytes to a ``[1, H, W, C]`` IMAGE tensor."""
    image = Image.open(io.BytesIO(data))

    if image.mode == "RGBA":
        # Composite onto white; ComfyUI IMAGE is RGB and a bare convert() would
        # drop the alpha channel against black.
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def stack_images(images: list[bytes]) -> torch.Tensor:
    """Build one IMAGE batch out of several results.

    ComfyUI batches must share a resolution; models can return mixed sizes, so
    anything that does not match the first image is resized to it.
    """
    tensors = [bytes_to_tensor(data) for data in images]
    if not tensors:
        raise ValueError("No images to stack.")
    if len(tensors) == 1:
        return tensors[0]

    height, width = tensors[0].shape[1], tensors[0].shape[2]
    normalized = [tensors[0]]
    for tensor in tensors[1:]:
        if tensor.shape[1] != height or tensor.shape[2] != width:
            # [B,H,W,C] -> [B,C,H,W] for interpolate, then back.
            resized = torch.nn.functional.interpolate(
                tensor.permute(0, 3, 1, 2),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            tensor = resized.permute(0, 2, 3, 1)
        normalized.append(tensor)
    return torch.cat(normalized, dim=0)
